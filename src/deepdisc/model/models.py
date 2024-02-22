from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from detectron2.config import instantiate
from detectron2.engine.defaults import create_ddp_model
from detectron2.layers import Conv2d, ShapeSpec, cat, get_norm, nonzero_tuple
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import CascadeROIHeads, StandardROIHeads, select_foreground_proposals
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from torch import nn
from torch.distributions.beta import Beta
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.normal import Normal
from torch.nn import functional as F


def return_lazy_model(cfg, freeze=True):
    """Return a model formed from a LazyConfig with the backbone
    frozen. Only the head layers will be trained.

    Parameters
    ----------
    cfg : .py file
        a LazyConfig

    Returns
    -------
        torch model
    """
    model = instantiate(cfg.model)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        # Phase 1: Unfreeze only the roi_heads
        for param in model.roi_heads.parameters():
            param.requires_grad = True
        # Phase 2: Unfreeze region proposal generator with reduced lr
        for param in model.proposal_generator.parameters():
            param.requires_grad = True

    model.to(cfg.train.device)
    model = create_ddp_model(model, **cfg.train.ddp)

    return model



class WeightedRedshiftPDFCasROIHeads(CascadeROIHeads):
    """CascadeROIHead with added redshift pdf capability.  Follows the detectron2 CascadeROIHead class init, except for

    Parameters
    ----------
    num_components : int
        Number of gaussian components in the Mixture Density Network
    """

    # def __init__(self, cfg, input_shape):
    def __init__(
        self,
        num_components: int,
        zloss_factor: float,
        weights: List[float],
        zbins: List[float],
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_heads: List[nn.Module],
        box_predictors: List[nn.Module],
        proposal_matchers: List[Matcher],
        **kwargs,
    ):
        super().__init__(
            box_in_features=box_in_features,
            box_pooler=box_pooler,
            box_heads=box_heads,
            box_predictors=box_predictors,
            proposal_matchers=proposal_matchers,
            **kwargs,
        )

        self.redshift_pooler = ROIPooler(
            output_size=7,
            scales=tuple(k for k in [0.25, 0.125, 0.0625, 0.03125]),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        )
        
        in_channels = 256
        inshape = ShapeSpec(channels=in_channels, height=7, width=7)

        self._output_size = (inshape.channels, inshape.height, inshape.width)
        self.num_components = num_components
        
        #self.dummy_param = nn.Parameter(torch.empty(0))
        
        self.zloss_factor = zloss_factor
        zbins = torch.tensor(zbins)
        weights = torch.tensor(weights)
        
        self.register_buffer('zbins', zbins, persistent=False)
        self.register_buffer('weights', weights, persistent=False)
        

        self.redshift_fc = nn.Sequential(
            nn.Linear(np.prod(self._output_size), 1024),
            nn.Tanh(),
            nn.Linear(1024, 64),
            nn.Tanh(),
            nn.Linear(64, self.num_components * 3),
            #nn.Softplus()
        )

    def output_pdf(self, inputs):
        pdf = Independent(
            MixtureSameFamily(
                mixture_distribution=Categorical(logits=inputs[..., : self.num_components]),
                component_distribution=Normal(
                    inputs[..., self.num_components : 2 * self.num_components],
                    #F.softplus(inputs[..., 2 * self.num_components :]),
                    torch.exp(inputs[..., 2 * self.num_components :]),
                ),
            ),
            0,
        )
        return pdf

    def _forward_redshift(self, features, instances):
        
        if self.training:
            instances, _ = select_foreground_proposals(instances, self.num_classes)
        
        if self.redshift_pooler is not None:
            features = [features[f] for f in self.box_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.redshift_pooler(features, boxes)

        features = nn.Flatten()(features)
        #ebvs = cat([x.gt_ebv for x in instances])
        #features = torch.cat((features, ebvs.unsqueeze(1)), dim=-1)
        
        num_instances_per_img = [len(i) for i in instances]

        if self.training:
            fcs = self.redshift_fc(features)
            pdfs = self.output_pdf(fcs)

            gt_redshifts = cat([x.gt_redshift for x in instances])

            sample_bins = torch.bucketize(gt_redshifts,self.zbins)
            sample_weights = self.weights[sample_bins]
            nlls = -pdfs.log_prob(gt_redshifts) * self.zloss_factor * sample_weights


            return {"redshift_loss": torch.mean(nlls)}

        else:
            # print(len(instances))
            # print(len(instances[0]))
            if len(instances[0]) == 0:
                return instances
                
            fcs = self.redshift_fc(features)
            pdfs = self.output_pdf(fcs)
            zs = torch.tensor(np.linspace(0, 5, 200)).to(fcs.device)
            nin = torch.as_tensor(np.array([num_instances_per_img]))
            #probs = torch.zeros((num_instances_per_img[0], 200)).to(fcs.device)

            inds = np.cumsum(num_instances_per_img)

            

            probs = torch.zeros((torch.sum(nin), 200)).to(fcs.device)
            for j, z in enumerate(zs):
                probs[:, j] = pdfs.log_prob(z)

            for i, pred_instances in enumerate(instances):
                pred_instances.pred_redshift_pdf = np.split(probs,inds)[i]

            return instances

    def forward(self, images, features, proposals, targets=None):
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        if self.training:
            # Need targets to box head
            losses = self._forward_box(features, proposals, targets)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_redshift(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            pred_instances = self._forward_redshift(features, pred_instances)
            return pred_instances, {}



class CNNRedshiftPDFCasROIHeads(CascadeROIHeads):
    """CascadeROIHead with added redshift pdf capability.  Follows the detectron2 CascadeROIHead class init, except for

    Parameters
    ----------
    num_components : int
        Number of gaussian components in the Mixture Density Network
    """

    # def __init__(self, cfg, input_shape):
    def __init__(
        self,
        num_components: int,
        zloss_factor: float,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_heads: List[nn.Module],
        box_predictors: List[nn.Module],
        proposal_matchers: List[Matcher],
        **kwargs,
    ):
        super().__init__(
            box_in_features=box_in_features,
            box_pooler=box_pooler,
            box_heads=box_heads,
            box_predictors=box_predictors,
            proposal_matchers=proposal_matchers,
            **kwargs,
        )

        self.redshift_pooler = ROIPooler(
            output_size=7,
            scales=tuple(k for k in [0.25, 0.125, 0.0625, 0.03125]),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        )

        in_channels = 256
        inshape = ShapeSpec(channels=in_channels, height=7, width=7)

        self._output_size = (inshape.channels, inshape.height, inshape.width)
        self.num_components = num_components
        self.zloss_factor = zloss_factor

        self.redshift_conv = nn.Sequential(
            nn.Conv2d(in_channels, 512, stride=1, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(512, 1024, stride=1,kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024*3*3, 1024),
            nn.Tanh(),
            nn.Linear(1024, 64),
            nn.Tanh(),
            nn.Linear(64, num_components * 3),

        )

    def output_pdf(self, inputs):
        pdf = Independent(
            MixtureSameFamily(
                mixture_distribution=Categorical(logits=inputs[..., : self.num_components]),
                component_distribution=Normal(
                    inputs[..., self.num_components : 2 * self.num_components],
                    #F.softplus(inputs[..., 2 * self.num_components :]),
                    torch.exp(inputs[..., 2 * self.num_components :]),
                ),
            ),
            0,
        )
        return pdf

    def _forward_redshift(self, features, instances):
        
        if self.training:
            instances, _ = select_foreground_proposals(instances, self.num_classes)
        
        if self.redshift_pooler is not None:
            features = [features[f] for f in self.box_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.redshift_pooler(features, boxes)
        
        #features = nn.Flatten()(features)
        #ebvs = cat([x.gt_ebv for x in instances])
        #features = torch.cat((features, ebvs.unsqueeze(1)), dim=-1)

        
        num_instances_per_img = [len(i) for i in instances]

        if self.training:
            fcs = self.redshift_conv(features)
            pdfs = self.output_pdf(fcs)

            gt_redshifts = cat([x.gt_redshift for x in instances])
            nlls = -pdfs.log_prob(gt_redshifts) * self.zloss_factor

            return {"redshift_loss": torch.mean(nlls)}

        else:
            # print(len(instances))
            # print(len(instances[0]))
            if len(instances[0]) == 0:
                return instances
                
            fcs = self.redshift_conv(features)
            pdfs = self.output_pdf(fcs)
            zs = torch.tensor(np.linspace(0, 3, 300)).to(fcs.device)
            nin = torch.as_tensor(np.array([num_instances_per_img]))
            #probs = torch.zeros((num_instances_per_img[0], 200)).to(fcs.device)

            inds = np.cumsum(num_instances_per_img)

            

            probs = torch.zeros((torch.sum(nin), 300)).to(fcs.device)
            for j, z in enumerate(zs):
                probs[:, j] = pdfs.log_prob(z)

            for i, pred_instances in enumerate(instances):
                pred_instances.pred_redshift_pdf = np.split(probs,inds)[i]
                
            
            #for i, pred_instances in enumerate(instances):
            #    probs = torch.zeros((num_instances_per_img[i], 200)).to(fcs.device)
            #    for j, z in enumerate(zs):
            #        if i<len(num_instances_per_img)-1:
            #            probs[:, j] = pdfs.log_prob(z)[lowi:highi]
            #        else:
            #            probs[:, j] = pdfs.log_prob(z)[lowi:]
            #    pred_instances.pred_redshift_pdf = probs
            #    lowi=highi
            #    highi=num_instances_per_img[i+1]+lowi

            return instances

    def forward(self, images, features, proposals, targets=None):
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        if self.training:
            # Need targets to box head
            losses = self._forward_box(features, proposals, targets)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_redshift(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            pred_instances = self._forward_redshift(features, pred_instances)
            return pred_instances, {}


        
class RedshiftPDFCasROIHeadsGold(CascadeROIHeads):
    """CascadeROIHead with added redshift pdf capability.  Follows the detectron2 CascadeROIHead class init, except for

    Parameters
    ----------
    num_components : int
        Number of gaussian components in the Mixture Density Network
    """

    # def __init__(self, cfg, input_shape):
    def __init__(
        self,
        num_components: int,
        zloss_factor: float,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_heads: List[nn.Module],
        box_predictors: List[nn.Module],
        proposal_matchers: List[Matcher],
        **kwargs,
    ):
        super().__init__(
            box_in_features=box_in_features,
            box_pooler=box_pooler,
            box_heads=box_heads,
            box_predictors=box_predictors,
            proposal_matchers=proposal_matchers,
            **kwargs,
        )

        self.redshift_pooler = ROIPooler(
            output_size=7,
            scales=tuple(k for k in [0.25, 0.125, 0.0625, 0.03125]),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        )

        in_channels = 256
        inshape = ShapeSpec(channels=in_channels, height=7, width=7)

        self._output_size = (inshape.channels, inshape.height, inshape.width)
        self.num_components = num_components
        self.zloss_factor = zloss_factor



        self.redshift_fc = nn.Sequential(
            nn.Linear(np.prod(self._output_size), 1024),
            nn.Tanh(),
            nn.Linear(1024, 64),
            nn.Tanh(),
            nn.Linear(64, self.num_components * 3),
            #nn.Softplus()
        )

    def output_pdf(self, inputs):
        pdf = Independent(
            MixtureSameFamily(
                mixture_distribution=Categorical(logits=inputs[..., : self.num_components]),
                component_distribution=Normal(
                    inputs[..., self.num_components : 2 * self.num_components],
                    #F.softplus(inputs[..., 2 * self.num_components :]),
                    torch.exp(inputs[..., 2 * self.num_components :]),
                ),
            ),
            0,
        )
        return pdf

    def _forward_redshift(self, features, instances):
        
        if self.training:
            finstances, _ = select_foreground_proposals(instances, self.num_classes)
            instances = []
            for x in finstances:
                gold_inst = x[x.gt_magi < 25.3]
                instances.append(gold_inst)
            if len(instances)==0:
                return 0
                
        if self.redshift_pooler is not None:
            features = [features[f] for f in self.box_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.redshift_pooler(features, boxes)
        
        features = nn.Flatten()(features)
        #ebvs = cat([x.gt_ebv for x in instances])
        #features = torch.cat((features, ebvs.unsqueeze(1)), dim=-1)
        
        num_instances_per_img = [len(i) for i in instances]

        if self.training:
            fcs = self.redshift_fc(features)
            pdfs = self.output_pdf(fcs)

            gt_redshifts = cat([x.gt_redshift for x in instances])
            nlls = -pdfs.log_prob(gt_redshifts) * self.zloss_factor

            return {"redshift_loss": torch.mean(nlls)}

        else:
            # print(len(instances))
            # print(len(instances[0]))
            if len(instances[0]) == 0:
                return instances
                
            fcs = self.redshift_fc(features)
            pdfs = self.output_pdf(fcs)
            zs = torch.tensor(np.linspace(0, 5, 200)).to(fcs.device)
            nin = torch.as_tensor(np.array([num_instances_per_img]))
            #probs = torch.zeros((num_instances_per_img[0], 200)).to(fcs.device)

            inds = np.cumsum(num_instances_per_img)

            

            probs = torch.zeros((torch.sum(nin), 200)).to(fcs.device)
            for j, z in enumerate(zs):
                probs[:, j] = pdfs.log_prob(z)

            for i, pred_instances in enumerate(instances):
                pred_instances.pred_redshift_pdf = np.split(probs,inds)[i]

            return instances

    def forward(self, images, features, proposals, targets=None):
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        if self.training:
            # Need targets to box head
            losses = self._forward_box(features, proposals, targets)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_redshift(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            pred_instances = self._forward_redshift(features, pred_instances)
            return pred_instances, {}

        


class RedshiftPDFCasROIHeads(CascadeROIHeads):
    """CascadeROIHead with added redshift pdf capability.  Follows the detectron2 CascadeROIHead class init, except for

    Parameters
    ----------
    num_components : int
        Number of gaussian components in the Mixture Density Network
    """

    # def __init__(self, cfg, input_shape):
    def __init__(
        self,
        num_components: int,
        zloss_factor: float,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_heads: List[nn.Module],
        box_predictors: List[nn.Module],
        proposal_matchers: List[Matcher],
        **kwargs,
    ):
        super().__init__(
            box_in_features=box_in_features,
            box_pooler=box_pooler,
            box_heads=box_heads,
            box_predictors=box_predictors,
            proposal_matchers=proposal_matchers,
            **kwargs,
        )

        self.redshift_pooler = ROIPooler(
            output_size=7,
            scales=tuple(k for k in [0.25, 0.125, 0.0625, 0.03125]),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        )

        in_channels = 256
        inshape = ShapeSpec(channels=in_channels, height=7, width=7)

        self._output_size = (inshape.channels, inshape.height, inshape.width)
        self.num_components = num_components
        self.zloss_factor = zloss_factor



        self.redshift_fc = nn.Sequential(
            nn.Linear(np.prod(self._output_size), 1024),
            nn.Tanh(),
            nn.Linear(1024, 64),
            nn.Tanh(),
            nn.Linear(64, self.num_components * 3),
            #nn.Softplus()
        )

    def output_pdf(self, inputs):
        pdf = Independent(
            MixtureSameFamily(
                mixture_distribution=Categorical(logits=inputs[..., : self.num_components]),
                component_distribution=Normal(
                    inputs[..., self.num_components : 2 * self.num_components],
                    #F.softplus(inputs[..., 2 * self.num_components :]),
                    torch.exp(inputs[..., 2 * self.num_components :]),
                ),
            ),
            0,
        )
        return pdf

    def _forward_redshift(self, features, instances):
        
        if self.training:
            instances, _ = select_foreground_proposals(instances, self.num_classes)
        
        if self.redshift_pooler is not None:
            features = [features[f] for f in self.box_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.redshift_pooler(features, boxes)
        
        features = nn.Flatten()(features)
        #ebvs = cat([x.gt_ebv for x in instances])
        #features = torch.cat((features, ebvs.unsqueeze(1)), dim=-1)
        
        num_instances_per_img = [len(i) for i in instances]

        if self.training:
            fcs = self.redshift_fc(features)
            pdfs = self.output_pdf(fcs)

            gt_redshifts = cat([x.gt_redshift for x in instances])
            nlls = -pdfs.log_prob(gt_redshifts) * self.zloss_factor

            return {"redshift_loss": torch.mean(nlls)}

        else:
            # print(len(instances))
            # print(len(instances[0]))
            if len(instances[0]) == 0:
                return instances
                
            fcs = self.redshift_fc(features)
            pdfs = self.output_pdf(fcs)
            zs = torch.tensor(np.linspace(0, 3, 300)).to(fcs.device)
            nin = torch.as_tensor(np.array([num_instances_per_img]))
            #probs = torch.zeros((num_instances_per_img[0], 200)).to(fcs.device)

            inds = np.cumsum(num_instances_per_img)

            

            probs = torch.zeros((torch.sum(nin), 300)).to(fcs.device)
            for j, z in enumerate(zs):
                probs[:, j] = pdfs.log_prob(z)

            for i, pred_instances in enumerate(instances):
                pred_instances.pred_redshift_pdf = np.split(probs,inds)[i]
                
            
            #for i, pred_instances in enumerate(instances):
            #    probs = torch.zeros((num_instances_per_img[i], 200)).to(fcs.device)
            #    for j, z in enumerate(zs):
            #        if i<len(num_instances_per_img)-1:
            #            probs[:, j] = pdfs.log_prob(z)[lowi:highi]
            #        else:
            #            probs[:, j] = pdfs.log_prob(z)[lowi:]
            #    pred_instances.pred_redshift_pdf = probs
            #    lowi=highi
            #    highi=num_instances_per_img[i+1]+lowi

            return instances

    def forward(self, images, features, proposals, targets=None):
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        if self.training:
            # Need targets to box head
            losses = self._forward_box(features, proposals, targets)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_redshift(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            pred_instances = self._forward_redshift(features, pred_instances)
            return pred_instances, {}

        

class OldRedshiftPDFCasROIHeads(CascadeROIHeads):
    """CascadeROIHead with added redshift pdf capability.  Follows the detectron2 CascadeROIHead class init, except for

    Parameters
    ----------
    num_components : int
        Number of gaussian components in the Mixture Density Network
    """

    # def __init__(self, cfg, input_shape):
    def __init__(
        self,
        num_components: int,
        zloss_factor: float,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_heads: List[nn.Module],
        box_predictors: List[nn.Module],
        proposal_matchers: List[Matcher],
        **kwargs,
    ):
        super().__init__(
            box_in_features=box_in_features,
            box_pooler=box_pooler,
            box_heads=box_heads,
            box_predictors=box_predictors,
            proposal_matchers=proposal_matchers,
            **kwargs,
        )

        self.redshift_pooler = ROIPooler(
            output_size=7,
            scales=tuple(k for k in [0.25, 0.125, 0.0625, 0.03125]),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        )

        in_channels = 256
        inshape = ShapeSpec(channels=in_channels, height=7, width=7)

        self._output_size = (inshape.channels, inshape.height, inshape.width)
        self.num_components = num_components
        self.zloss_factor = zloss_factor



        self.redshift_fc = nn.Sequential(
            nn.Linear(np.prod(self._output_size), 1024),
            nn.Tanh(),
            nn.Linear(1024, 64),
            nn.Tanh(),
            nn.Linear(64, self.num_components * 3),
            #nn.Softplus()
        )
        
        
    def output_pdf(self, inputs):
        pdf = Independent(
            MixtureSameFamily(
                mixture_distribution=Categorical(logits=inputs[..., : self.num_components]),
                component_distribution=Normal(
                    inputs[..., self.num_components : 2 * self.num_components],
                    F.softplus(inputs[..., 2 * self.num_components :]),
                    #torch.exp(inputs[..., 2 * self.num_components :]),
                ),
            ),
            0,
        )
        return pdf

    def _forward_redshift(self, features, instances):
        
        if self.training:
            instances, _ = select_foreground_proposals(instances, self.num_classes)
        
        if self.redshift_pooler is not None:
            features = [features[f] for f in self.box_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.redshift_pooler(features, boxes)
        
        features = nn.Flatten()(features)
        #ebvs = cat([x.gt_ebv for x in instances])
        #features = torch.cat((features, ebvs.unsqueeze(1)), dim=-1)
        
        num_instances_per_img = [len(i) for i in instances]

        if self.training:
            fcs = self.redshift_fc(features)
            pdfs = self.output_pdf(fcs)

            gt_redshifts = cat([x.gt_redshift for x in instances])
            nlls = -pdfs.log_prob(gt_redshifts) * self.zloss_factor

            return {"redshift_loss": torch.mean(nlls)}

        else:
            # print(len(instances))
            # print(len(instances[0]))
            if len(instances[0]) == 0:
                return instances
                
            fcs = self.redshift_fc(features)
            pdfs = self.output_pdf(fcs)
            zs = torch.tensor(np.linspace(-1, 5, 200)).to(fcs.device)
            nin = torch.as_tensor(np.array([num_instances_per_img]))
            #probs = torch.zeros((num_instances_per_img[0], 200)).to(fcs.device)

            inds = np.cumsum(num_instances_per_img)

            

            probs = torch.zeros((torch.sum(nin), 200)).to(fcs.device)
            for j, z in enumerate(zs):
                probs[:, j] = pdfs.log_prob(z)

            for i, pred_instances in enumerate(instances):
                pred_instances.pred_redshift_pdf = np.split(probs,inds)[i]
                
            
            #for i, pred_instances in enumerate(instances):
            #    probs = torch.zeros((num_instances_per_img[i], 200)).to(fcs.device)
            #    for j, z in enumerate(zs):
            #        if i<len(num_instances_per_img)-1:
            #            probs[:, j] = pdfs.log_prob(z)[lowi:highi]
            #        else:
            #            probs[:, j] = pdfs.log_prob(z)[lowi:]
            #    pred_instances.pred_redshift_pdf = probs
            #    lowi=highi
            #    highi=num_instances_per_img[i+1]+lowi

            return instances

    def forward(self, images, features, proposals, targets=None):
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        if self.training:
            # Need targets to box head
            losses = self._forward_box(features, proposals, targets)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_redshift(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            pred_instances = self._forward_redshift(features, pred_instances)
            return pred_instances, {}

        

class RedshiftPointCasROIHeads(CascadeROIHeads):
    """CascadeROIHeads with added redshift point estimate capability.  Follows the detectron2 CascadeROIHeads class init"""

    # def __init__(self, cfg, input_shape):
    def __init__(
        self,
        zloss_factor: float,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_heads: List[nn.Module],
        box_predictors: List[nn.Module],
        proposal_matchers: List[Matcher],
        **kwargs,
    ):
        super().__init__(
            box_in_features=box_in_features,
            box_pooler=box_pooler,
            box_heads=box_heads,
            box_predictors=box_predictors,
            proposal_matchers=proposal_matchers,
            **kwargs,
        )

        # super().__init__(cfg, input_shape, **kwargs)

        self.redshift_pooler = ROIPooler(
            output_size=7,
            scales=tuple(k for k in [0.25, 0.125, 0.0625, 0.03125]),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        )
        # in_channels = [input_shape[f].channels for f in in_features]
        # in_channels = in_channels[0]
        in_channels = 256
        inshape = ShapeSpec(channels=in_channels, height=7, width=7)

        # The input dim should follow from the classification head
        self._output_size = (inshape.channels, inshape.height, inshape.width)

        self.zloss_factor = zloss_factor
        
        # self.redshift_fc = nn.Linear(int(np.prod(self._output_size)), 1)

        self.redshift_fc = nn.Sequential(
            nn.Linear(int(np.prod(self._output_size)), 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        # for l in self.redshift_fc:
        #    if type(l) == nn.Linear:
        #        #nn.init.constant_(l.bias, 0.1)
        #        nn.init.normal_(l.weight,std=0.01)

    def _forward_redshift(self, features, instances):
        if self.redshift_pooler is not None:
            features = [features[f] for f in self.box_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.redshift_pooler(features, boxes)

        features = nn.Flatten()(features)

        prediction = self.redshift_fc(features)[:, 0]
        # prediction = self.redshift_fc(features)

        num_instances_per_img = [len(i) for i in instances]

        if self.training:
            gt_classes = cat([x.gt_classes for x in instances])
            # print('gt_classes')
            # print(gt_classes)
            # print('fg_inds')
            fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]

            gt_redshifts = cat([x.gt_redshift for x in instances])

            diff = (prediction[fg_inds] - gt_redshifts[fg_inds]) * self.zloss_factor
            # $diff = prediction - gt_redshifts

            return {"redshift_loss": torch.square(diff).mean()}
            # return{"redshift_loss": torch.abs(diff).median()}
        else:
            z_pred = torch.split(prediction, num_instances_per_img, dim=0)
            for z, pred_instances in zip(z_pred, instances):
                pred_instances.pred_redshift = z
            return instances

    def forward(self, images, features, proposals, targets=None):
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        if self.training:
            # Need targets to box head
            losses = self._forward_box(features, proposals, targets)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_redshift(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            pred_instances = self._forward_redshift(features, pred_instances)
            return pred_instances, {}


class RedshiftPointROIHeads(StandardROIHeads):
    """ROIHead with added redshift point estimate capability.  Follows the detectron2 StandardROIHead class init"""

    def __init__(
        self,
        zloss_factor: float,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        mask_in_features: Optional[List[str]] = None,
        mask_pooler: Optional[ROIPooler] = None,
        mask_head: Optional[nn.Module] = None,
        keypoint_in_features: Optional[List[str]] = None,
        keypoint_pooler: Optional[ROIPooler] = None,
        keypoint_head: Optional[nn.Module] = None,
        train_on_pred_boxes: bool = False,
        **kwargs,
    ):
        super().__init__(
            box_in_features=box_in_features,
            box_pooler=box_pooler,
            box_head=box_head,
            box_predictor=box_predictor,
            mask_in_features=mask_in_features,
            mask_pooler=mask_pooler,
            mask_head=mask_head,
            keypoint_in_features=keypoint_in_features,
            keypoint_pooler=keypoint_pooler,
            keypoint_head=keypoint_head,
            train_on_pred_boxes=train_on_pred_boxes,
            **kwargs,
        )

        self.redshift_pooler = ROIPooler(
            output_size=7,
            scales=(0.25, 0.125, 0.0625, 0.03125),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        )
        
        slef.zloss_factor = zloss_factor

        in_channels = 256
        inshape = ShapeSpec(channels=in_channels, height=7, width=7)

        # The input dim should follow from the classification head
        self._output_size = (inshape.channels, inshape.height, inshape.width)
        self.redshift_fc = nn.Sequential(
            nn.Linear(int(np.prod(self._output_size)), 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        # self.redshift_fc = nn.Linear(12, 1)

    def _forward_redshift(self, features, instances):
        if self.redshift_pooler is not None:
            features = [features[f] for f in self.box_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.redshift_pooler(features, boxes)

        features = nn.Flatten()(features)

        prediction = self.redshift_fc(features)

        num_instances_per_img = [len(i) for i in instances]

        if self.training:
            gt_classes = cat([x.gt_classes for x in instances])
            fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
            gt_redshifts = cat([x.gt_redshift for x in instances])
            diff = (prediction[fg_inds] - gt_redshifts[fg_inds]) * self.zloss_factor
            # diff = prediction - cat([x.gt_redshift for x in instances])
            return {"redshift_loss": torch.square(diff).mean()}
        else:
            z_pred = torch.split(prediction, num_instances_per_img, dim=0)
            for z, pred_instances in zip(z_pred, instances):
                pred_instances.pred_redshift = z
            return instances

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            losses.update(self._forward_redshift(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            pred_instances = self._forward_redshift(features, pred_instances)
            return pred_instances, {}


class RedshiftPDFROIHeads(StandardROIHeads):
    """ROIHead with added redshift pdf capability.  Follows the detectron2 StandardROIHead class init, except for

    Parameters
    ----------
    num_components : int
        Number of gaussian components in the Mixture Density Network
    """

    def __init__(
        self,
        num_components: int,
        zloss_factor: float,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        mask_in_features: Optional[List[str]] = None,
        mask_pooler: Optional[ROIPooler] = None,
        mask_head: Optional[nn.Module] = None,
        keypoint_in_features: Optional[List[str]] = None,
        keypoint_pooler: Optional[ROIPooler] = None,
        keypoint_head: Optional[nn.Module] = None,
        train_on_pred_boxes: bool = False,
        **kwargs,
    ):
        super().__init__(
            box_in_features=box_in_features,
            box_pooler=box_pooler,
            box_head=box_head,
            box_predictor=box_predictor,
            mask_in_features=mask_in_features,
            mask_pooler=mask_pooler,
            mask_head=mask_head,
            keypoint_in_features=keypoint_in_features,
            keypoint_pooler=keypoint_pooler,
            keypoint_head=keypoint_head,
            train_on_pred_boxes=train_on_pred_boxes,
            **kwargs,
        )

        self.redshift_pooler = ROIPooler(
            output_size=7,
            scales=(0.25, 0.125, 0.0625, 0.03125),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        )

        self.zloss_factor = zloss_factor
        
        in_channels = 256
        inshape = ShapeSpec(channels=in_channels, height=7, width=7)

        # The input dim should follow from the classification head
        self._output_size = (inshape.channels, inshape.height, inshape.width)

        self.num_components = num_components

        self.redshift_fc = nn.Sequential(
            # nn.Linear(int(np.prod(self._output_size)), self.num_components * 3)
            nn.Linear(int(np.prod(self._output_size)), 16),
            nn.Tanh(),
            nn.Linear(16, self.num_components * 3),
        )

    def output_pdf(self, inputs):
        pdf = Independent(
            MixtureSameFamily(
                mixture_distribution=Categorical(logits=inputs[..., : self.num_components]),
                component_distribution=Normal(
                    inputs[..., self.num_components : 2 * self.num_components],
                    F.softplus(inputs[..., 2 * self.num_components :]),
                ),
            ),
            0,
        )
        return pdf

    def _forward_redshift(self, features, instances):
        if self.redshift_pooler is not None:
            features = [features[f] for f in self.box_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.redshift_pooler(features, boxes)

        features = nn.Flatten()(features)

        prediction = self.redshift_fc(features)

        num_instances_per_img = [len(i) for i in instances]

        if self.training:
            fcs = self.redshift_fc(features)
            pdfs = self.output_pdf(fcs)
            gt_classes = cat([x.gt_classes for x in instances])
            fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
            pdfs_fg = self.output_pdf(fcs[fg_inds, ...])

            gt_redshifts = cat([x.gt_redshift for x in instances])
            nlls_fg = -pdfs_fg.log_prob(gt_redshifts[fg_inds])

            nlls = -pdfs.log_prob(gt_redshifts)[fg_inds] * self.zloss_factor
            return {"redshift_loss": torch.mean(nlls)}

        else:
            # print(len(instances))
            # print(len(instances[0]))
            if len(instances[0]) == 0:
                return instances
            # for i, instances in enumerate(instances):
            #    if num_instances_per_img[i] ==0:
            #        continue
            fcs = self.redshift_fc(features)
            pdfs = self.output_pdf(fcs)
            zs = torch.tensor(np.linspace(0, 5, 200)).to(fcs.device)

            probs = torch.zeros((num_instances_per_img[0], 200)).to(fcs.device)
            for i, z in enumerate(zs):
                # probs.append(outputs.log_prob(z))
                probs[:, i] = pdfs.log_prob(z)

            for i, pred_instances in enumerate(instances):
                pred_instances.pred_redshift_pdf = probs

            return instances

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            losses.update(self._forward_redshift(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            pred_instances = self._forward_redshift(features, pred_instances)
            return pred_instances, {}

        
class OldEBVRedshiftPDFCasROIHeads(CascadeROIHeads):
    """CascadeROIHead with added redshift pdf capability.  Follows the detectron2 CascadeROIHead class init, except for

    Parameters
    ----------
    num_components : int
        Number of gaussian components in the Mixture Density Network
    """

    # def __init__(self, cfg, input_shape):
    def __init__(
        self,
        num_components: int,
        zloss_factor: float,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_heads: List[nn.Module],
        box_predictors: List[nn.Module],
        proposal_matchers: List[Matcher],
        **kwargs,
    ):
        super().__init__(
            box_in_features=box_in_features,
            box_pooler=box_pooler,
            box_heads=box_heads,
            box_predictors=box_predictors,
            proposal_matchers=proposal_matchers,
            **kwargs,
        )

        self.redshift_pooler = ROIPooler(
            output_size=7,
            scales=tuple(k for k in [0.25, 0.125, 0.0625, 0.03125]),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        )

        in_channels = 256
        inshape = ShapeSpec(channels=in_channels, height=7, width=7)

        self._output_size = (inshape.channels, inshape.height, inshape.width)
        self.num_components = num_components
        self.zloss_factor = zloss_factor



        self.redshift_fc = nn.Sequential(
            nn.Linear(int(np.prod(self._output_size))+1, 1024),
            nn.Tanh(),
            nn.Linear(1024, 64),
            nn.Tanh(),
            nn.Linear(64, self.num_components * 3),
            #nn.Softplus()
        )

    def output_pdf(self, inputs):
        pdf = Independent(
            MixtureSameFamily(
                mixture_distribution=Categorical(logits=inputs[..., : self.num_components]),
                component_distribution=Normal(
                    inputs[..., self.num_components : 2 * self.num_components],
                    #F.softplus(inputs[..., 2 * self.num_components :]),
                    torch.exp(inputs[..., 2 * self.num_components :]),
                ),
            ),
            0,
        )
        return pdf

    def _forward_redshift(self, features, instances):
        
        if self.training:
            instances, _ = select_foreground_proposals(instances, self.num_classes)
        
        if self.redshift_pooler is not None:
            features = [features[f] for f in self.box_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.redshift_pooler(features, boxes)

        features = nn.Flatten()(features)
        ebvs = cat([x.gt_ebv for x in instances])
        features = torch.cat((features, ebvs.unsqueeze(1)), dim=-1)
        
        num_instances_per_img = [len(i) for i in instances]

        if self.training:
            fcs = self.redshift_conv(features)
            pdfs = self.output_pdf(fcs)

            gt_redshifts = cat([x.gt_redshift for x in instances])
            nlls = -pdfs.log_prob(gt_redshifts) * self.zloss_factor

            return {"redshift_loss": torch.mean(nlls)}

        else:
            # print(len(instances))
            # print(len(instances[0]))
            if len(instances[0]) == 0:
                return instances
            # for i, instances in enumerate(instances):
            #    if num_instances_per_img[i] ==0:
            #        continue
            fcs = self.redshift_fc(features)
            pdfs = self.output_pdf(fcs)
            zs = torch.tensor(np.linspace(0, 5, 200)).to(fcs.device)

            probs = torch.zeros((num_instances_per_img[0], 200)).to(fcs.device)
            for i, z in enumerate(zs):
                # probs.append(outputs.log_prob(z))
                probs[:, i] = pdfs.log_prob(z)

            for i, pred_instances in enumerate(instances):
                pred_instances.pred_redshift_pdf = probs

            return instances

    def forward(self, images, features, proposals, targets=None):
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        if self.training:
            # Need targets to box head
            losses = self._forward_box(features, proposals, targets)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_redshift(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            pred_instances = self._forward_redshift(features, pred_instances)
            return pred_instances, {}

