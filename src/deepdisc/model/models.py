from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from detectron2.config import instantiate
from detectron2.engine.defaults import create_ddp_model
from detectron2.layers import Conv2d, ShapeSpec, cat, get_norm, nonzero_tuple
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import CascadeROIHeads, StandardROIHeads, select_foreground_proposals
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from torch import nn
from torch.distributions.beta import Beta
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.normal import Normal
from torch.nn import functional as F

import dustmaps
from dustmaps.sfd import SFDQuery
from dustmaps.config import config
config['data_dir'] = '/home/shared/hsc/DC2/dustmaps/'

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord



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
            #print('proposals ', len(instances[0]))
            proposals = add_ground_truth_to_proposals(targets, instances)
         
            #Add all gt bounding boxes for redshift regression
            finstances, _ = select_foreground_proposals(proposals, self.num_classes)

            #print('sampled foreground proposals', len(fproposals[0]))

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


        

        
class RedshiftPDFCasROIHeadsJWST(CascadeROIHeads):
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
        zmin: int,
        zmax: int,
        zn: int,
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
        self.zmin = zmin
        self.zmax = zmax
        self.zn = zn
        
        self.redshift_fc = nn.Sequential(
            nn.Linear(np.prod(self._output_size), 1024),
            nn.Tanh(),
            nn.Linear(1024, 64),
            nn.Tanh(),
            nn.Linear(64, self.num_components * 3),
            #nn.Softplus()
        )
        

        '''
        self.redshift_fc = nn.Sequential(
            nn.Linear(np.prod(self._output_size), 64),
            nn.Tanh(),
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Linear(16, self.num_components * 3),
            #nn.Softplus()
        )
        '''

        
        
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

    def _forward_redshift(self, features, instances, targets=None):
        
        if self.training:
            #Add all gt bounding boxes for redshift regression
            proposals = add_ground_truth_to_proposals(targets, instances)
            finstances, _ = select_foreground_proposals(proposals, self.num_classes)

            #finstances, _ = select_foreground_proposals(instances, self.num_classes)

            instances = []
            for x in finstances:
                z_inst = x[x.gt_redshift != -1]
                instances.append(z_inst)
            if len(instances)==0:
                return 0
            
        #print('instances ', len(instances[0]))
        #sz = np.load('/home/g4merz/rail_deepdisc/sampled_zs_gold.npy')
        #sampled_zs = instances[0].gt_redshift.detach().cpu().numpy()
        #szs = np.concatenate([sz,sampled_zs])
        #np.save('/home/g4merz/rail_deepdisc/sampled_zs_gold.npy', szs)
                
        if self.redshift_pooler is not None:
            features = [features[f] for f in self.box_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.redshift_pooler(features, boxes)
        
        features = nn.Flatten()(features)
        
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
            zs = torch.tensor(np.linspace(self.zmin, self.zmax, self.zn)).to(fcs.device)
            nin = torch.as_tensor(np.array([num_instances_per_img]))

            inds = np.cumsum(num_instances_per_img)

            probs = torch.zeros((torch.sum(nin), self.zn)).to(fcs.device)
            for j, z in enumerate(zs):
                probs[:, j] = pdfs.log_prob(z)

            for i, pred_instances in enumerate(instances):
                pred_instances.pred_redshift_pdf = np.split(probs,inds)[i]
                pred_instances.pred_gmm =  np.split(fcs,inds)[i]

            return instances

    def forward(self, images, features, proposals, targets=None):
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        if self.training:
            # Need targets to box head
            losses = self._forward_box(features, proposals, targets)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_redshift(features, proposals, targets))
            #losses.update(self._forward_redshift(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            pred_instances = self._forward_redshift(features, pred_instances)
            return pred_instances, {}


        
        
class RedshiftPDFCasROIHeadsGoldEBVGals(CascadeROIHeads):
    """CascadeROIHead with added redshift pdf capability.  Uses the image wcs and dustmaps to calculate ebv values for each box center 
        Follows the detectron2 CascadeROIHead class init, except for

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
        maglim: float,
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
        self.maglim = maglim
        
        self.redshift_fc = nn.Sequential(
            nn.Linear(np.prod(self._output_size)+1, 1024),
            nn.Tanh(),
            nn.Linear(1024, 64),
            nn.Tanh(),
            nn.Linear(64, self.num_components * 3),
            #nn.Softplus()
        )
        

        '''
        self.redshift_fc = nn.Sequential(
            nn.Linear(np.prod(self._output_size), 64),
            nn.Tanh(),
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Linear(16, self.num_components * 3),
            #nn.Softplus()
        )
        '''
        
        self.sfd = SFDQuery()

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

    def _forward_redshift(self, features, instances, targets=None, image_wcs=None):
        
        if self.training:
            #Add all gt bounding boxes for redshift regression
            proposals = add_ground_truth_to_proposals(targets, instances)
            finstances, _ = select_foreground_proposals(proposals, self.num_classes)

            #finstances, _ = select_foreground_proposals(instances, self.num_classes)

            instances = []
            for x in finstances:
                gold_inst = x[(x.gt_magi < self.maglim) & (x.gt_redshift!=0)]
                instances.append(gold_inst)
            if len(instances)==0:
                return 0
            
        #print('instances ', len(instances[0]))
        #sz = np.load('/home/g4merz/rail_deepdisc/sampled_zs_gold.npy')
        #sampled_zs = instances[0].gt_redshift.detach().cpu().numpy()
        #szs = np.concatenate([sz,sampled_zs])
        #np.save('/home/g4merz/rail_deepdisc/sampled_zs_gold.npy', szs)
                
        if self.redshift_pooler is not None:
            features = [features[f] for f in self.box_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.redshift_pooler(features, boxes)
        
        features = nn.Flatten()(features)
        
        num_instances_per_img = [len(i) for i in instances]
        inds = np.cumsum(num_instances_per_img)
        
        #Add EBV
        centers = cat([box.get_centers().cpu() for box in boxes]) # Center box coords for wcs             
        #calculates coords for box centers in each image. Need to split and cumsum to make sure the box centers get the right wcs  
        coords = [WCS(wcs).pixel_to_world(np.split(centers,inds)[i][:,0],np.split(centers,inds)[i][:,1]) for i, wcs in enumerate(image_wcs)] 
        #use dustamps to get all ebv with the associated coords
        ebvvec = [torch.tensor(self.sfd(coordsi)).to(features.device) for coordsi in coords]
        ebvs = cat(ebvvec)
        #gather into a tensor and add as a feature for the input to the fully connected network
        features = torch.cat((features, ebvs.unsqueeze(1)), dim=-1)

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


            probs = torch.zeros((torch.sum(nin), 300)).to(fcs.device)
            for j, z in enumerate(zs):
                probs[:, j] = pdfs.log_prob(z)

            for i, pred_instances in enumerate(instances):
                pred_instances.pred_redshift_pdf = np.split(probs,inds)[i]
                pred_instances.pred_gmm =  np.split(fcs,inds)[i]

            return instances

    def forward(self, images, features, proposals, targets=None, image_wcs=None):
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        if self.training:
            # Need targets to box head
            losses = self._forward_box(features, proposals, targets)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_redshift(features, proposals, targets, image_wcs))
            #losses.update(self._forward_redshift(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            pred_instances = self._forward_redshift(features, pred_instances, image_wcs=image_wcs)
            return pred_instances, {}

        
        
        
class RedshiftPDFCasROIHeadsGoldEBV(CascadeROIHeads):
    """CascadeROIHead with added redshift pdf capability.  Uses the image wcs and dustmaps to calculate ebv values for each box center 
        Follows the detectron2 CascadeROIHead class init, except for

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
        maglim: float,
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
        self.maglim = maglim
        
        self.redshift_fc = nn.Sequential(
            nn.Linear(np.prod(self._output_size)+1, 1024),
            nn.Tanh(),
            nn.Linear(1024, 64),
            nn.Tanh(),
            nn.Linear(64, self.num_components * 3),
            #nn.Softplus()
        )
        

        '''
        self.redshift_fc = nn.Sequential(
            nn.Linear(np.prod(self._output_size), 64),
            nn.Tanh(),
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Linear(16, self.num_components * 3),
            #nn.Softplus()
        )
        '''
        
        self.sfd = SFDQuery()

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

    def _forward_redshift(self, features, instances, targets=None, image_wcs=None):
        
        if self.training:
            #Add all gt bounding boxes for redshift regression
            proposals = add_ground_truth_to_proposals(targets, instances)
            finstances, _ = select_foreground_proposals(proposals, self.num_classes)

            #finstances, _ = select_foreground_proposals(instances, self.num_classes)

            instances = []
            for x in finstances:
                gold_inst = x[x.gt_magi < self.maglim]
                instances.append(gold_inst)
            if len(instances)==0:
                return 0
            
        #print('instances ', len(instances[0]))
        #sz = np.load('/home/g4merz/rail_deepdisc/sampled_zs_gold.npy')
        #sampled_zs = instances[0].gt_redshift.detach().cpu().numpy()
        #szs = np.concatenate([sz,sampled_zs])
        #np.save('/home/g4merz/rail_deepdisc/sampled_zs_gold.npy', szs)
                
        if self.redshift_pooler is not None:
            features = [features[f] for f in self.box_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.redshift_pooler(features, boxes)
        
        features = nn.Flatten()(features)
        
        num_instances_per_img = [len(i) for i in instances]
        inds = np.cumsum(num_instances_per_img)
        
        #Add EBV
        centers = cat([box.get_centers().cpu() for box in boxes]) # Center box coords for wcs             
        #calculates coords for box centers in each image. Need to split and cumsum to make sure the box centers get the right wcs  
        coords = [WCS(wcs).pixel_to_world(np.split(centers,inds)[i][:,0],np.split(centers,inds)[i][:,1]) for i, wcs in enumerate(image_wcs)] 
        #use dustamps to get all ebv with the associated coords
        ebvvec = [torch.tensor(self.sfd(coordsi)).to(features.device) for coordsi in coords]
        ebvs = cat(ebvvec)
        #gather into a tensor and add as a feature for the input to the fully connected network
        features = torch.cat((features, ebvs.unsqueeze(1)), dim=-1)

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


            probs = torch.zeros((torch.sum(nin), 300)).to(fcs.device)
            for j, z in enumerate(zs):
                probs[:, j] = pdfs.log_prob(z)

            for i, pred_instances in enumerate(instances):
                pred_instances.pred_redshift_pdf = np.split(probs,inds)[i]
                pred_instances.pred_gmm =  np.split(fcs,inds)[i]

            return instances

    def forward(self, images, features, proposals, targets=None, image_wcs=None):
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        if self.training:
            # Need targets to box head
            losses = self._forward_box(features, proposals, targets)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_redshift(features, proposals, targets, image_wcs))
            #losses.update(self._forward_redshift(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            pred_instances = self._forward_redshift(features, pred_instances, image_wcs=image_wcs)
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
        maglim: float,
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
        self.maglim = maglim
        
        self.redshift_fc = nn.Sequential(
            nn.Linear(np.prod(self._output_size), 1024),
            nn.Tanh(),
            nn.Linear(1024, 64),
            nn.Tanh(),
            nn.Linear(64, self.num_components * 3),
            #nn.Softplus()
        )
        

        '''
        self.redshift_fc = nn.Sequential(
            nn.Linear(np.prod(self._output_size), 64),
            nn.Tanh(),
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Linear(16, self.num_components * 3),
            #nn.Softplus()
        )
        '''

        
        
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

    def _forward_redshift(self, features, instances, targets=None):
        
        if self.training:
            #Add all gt bounding boxes for redshift regression
            proposals = add_ground_truth_to_proposals(targets, instances)
            finstances, _ = select_foreground_proposals(proposals, self.num_classes)

            #finstances, _ = select_foreground_proposals(instances, self.num_classes)

            instances = []
            for x in finstances:
                gold_inst = x[x.gt_magi < self.maglim]
                instances.append(gold_inst)
            if len(instances)==0:
                return 0
            
        #print('instances ', len(instances[0]))
        #sz = np.load('/home/g4merz/rail_deepdisc/sampled_zs_gold.npy')
        #sampled_zs = instances[0].gt_redshift.detach().cpu().numpy()
        #szs = np.concatenate([sz,sampled_zs])
        #np.save('/home/g4merz/rail_deepdisc/sampled_zs_gold.npy', szs)
                
        if self.redshift_pooler is not None:
            features = [features[f] for f in self.box_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.redshift_pooler(features, boxes)
        
        features = nn.Flatten()(features)
        
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
                pred_instances.pred_gmm =  np.split(fcs,inds)[i]

            return instances

    def forward(self, images, features, proposals, targets=None):
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        if self.training:
            # Need targets to box head
            losses = self._forward_box(features, proposals, targets)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_redshift(features, proposals, targets))
            #losses.update(self._forward_redshift(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            pred_instances = self._forward_redshift(features, pred_instances)
            return pred_instances, {}


        
class RedshiftPDFCasROIHeadsGoldFC2(CascadeROIHeads):
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
            nn.Linear(np.prod(self._output_size), 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_components * 3),
            nn.ReLU()
            #nn.Softplus()
        )
        

        '''
        self.redshift_fc = nn.Sequential(
            nn.Linear(np.prod(self._output_size), 64),
            nn.Tanh(),
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Linear(16, self.num_components * 3),
            #nn.Softplus()
        )
        '''

        
        
    def output_pdf(self, inputs):
        pdf = Independent(
            MixtureSameFamily(
                mixture_distribution=Categorical(probs=inputs[..., : self.num_components]),
                component_distribution=Normal(
                    inputs[..., self.num_components : 2 * self.num_components],
                    #F.softplus(inputs[..., 2 * self.num_components :]),
                    inputs[..., 2 * self.num_components :],
                ),
            ),
            0,
        )
        return pdf

    def _forward_redshift(self, features, instances, targets=None):
        
        if self.training:
            #Add all gt bounding boxes for redshift regression
            proposals = add_ground_truth_to_proposals(targets, instances)
            finstances, _ = select_foreground_proposals(proposals, self.num_classes)

            #finstances, _ = select_foreground_proposals(instances, self.num_classes)

            instances = []
            for x in finstances:
                gold_inst = x[x.gt_magi < 25.3]
                instances.append(gold_inst)
            if len(instances)==0:
                return 0
            
        #print('instances ', len(instances[0]))
        #sz = np.load('/home/g4merz/rail_deepdisc/sampled_zs_gold.npy')
        #sampled_zs = instances[0].gt_redshift.detach().cpu().numpy()
        #szs = np.concatenate([sz,sampled_zs])
        #np.save('/home/g4merz/rail_deepdisc/sampled_zs_gold.npy', szs)
                
        if self.redshift_pooler is not None:
            features = [features[f] for f in self.box_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.redshift_pooler(features, boxes)
        
        features = nn.Flatten()(features)
        
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

            return instances

    def forward(self, images, features, proposals, targets=None):
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        if self.training:
            # Need targets to box head
            losses = self._forward_box(features, proposals, targets)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_redshift(features, proposals, targets))
            #losses.update(self._forward_redshift(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            pred_instances = self._forward_redshift(features, pred_instances)
            return pred_instances, {}

        
        
class RedshiftPDFCasROIHeadsGoldCRPS(CascadeROIHeads):
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
        

        '''
        self.redshift_fc = nn.Sequential(
            nn.Linear(np.prod(self._output_size), 64),
            nn.Tanh(),
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Linear(16, self.num_components * 3),
            #nn.Softplus()
        )
        '''

        
        
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
    
    def A(self, mu, std, true_zs):
    
        #n = norm(loc=mu, scale=std)
        n = Independent(Normal(loc=mu, scale=std), 0)

        A = (2*std*torch.exp(n.log_prob(true_zs)) 
            + mu*(2*n.cdf(true_zs)-1))

        #A = (2*std*torch.tensor(n.pdf(true_zs)) 
        #    + mu*(2*n.cdf(true_zs)-1))

        return A

    '''
    def CRPS_loss(self, inputs, true_zs):
    
        #loss

        w = inputs[...,:self.num_components]
        w = inputs[...,:self.num_components]
        w = w - w.logsumexp(dim=-1, keepdim=True)
        w = F.softmax(w, dim=-1)
        mu = inputs[...,self.num_components:2*self.num_components]
        std = torch.exp(inputs[...,2*self.num_components:])

        losses = []
        n = Normal(loc=0,scale=1)
        
        for q,x in enumerate(mu):
            muq = mu[q]
            stdq = std[q]
            wq = w[q]

            sx = (true_zs[q]-muq)/stdq
            T1 = torch.sum(wq* ((2*stdq *torch.exp(n.log_prob(sx)))+ ((true_zs[q]-muq)*(2*n.cdf(sx)-1))))

            wi = wq.unsqueeze(1)
            wj = wq
            mud = muq.unsqueeze(1)-muq
            stdd = torch.sqrt(stdq.unsqueeze(1)**2 + stdq**2)
            sx = (mud)/stdd
            T2=torch.sum(wi*wj* ((2*stdd * torch.exp(n.log_prob(sx))) + (mud*(2*n.cdf(sx)-1))))

            loss = torch.tensor([T1 - 0.5 * T2])
            #loss = cat(loss)
            losses.append(loss)
        #print(cat(losses))
        return cat(losses)
    '''
    
    def CRPS_loss(self, inputs, true_zs):
    
        #This uses some tensor reshaping tricks for speedup (for loops are very slow)
        #Based on the equations in D'Isanto et al 2018 Appendix A

        w = inputs[...,:self.num_components]
        w = w - w.logsumexp(dim=-1, keepdim=True)
        w = F.softmax(w, dim=-1)
        mu = inputs[...,self.num_components:2*self.num_components]
        std = torch.exp(inputs[...,2*self.num_components:])

        n = Normal(loc=0,scale=1)
        sx = (true_zs.unsqueeze(1) - mu)/std
        T1 = torch.sum(w* ((2*std *torch.exp(n.log_prob(sx)))+ ((true_zs.unsqueeze(1)-mu)*(2*n.cdf(sx)-1))), dim=-1)

        wi = w.unsqueeze(-1)
        wj = w.unsqueeze(1)
        mud = mu.unsqueeze(-1)-mu.unsqueeze(1)
        stdd = torch.sqrt(std.unsqueeze(1)**2 + std.unsqueeze(-1)**2)
        n = Normal(loc=0,scale=1)
        sx = (mud)/stdd
        T2=torch.sum(wi*wj* ((2*stdd * torch.exp(n.log_prob(sx))) + (mud*(2*n.cdf(sx)-1))),dim=[-2,-1])
        
        losses = T1-0.5*T2
        #print(losses)
        return losses
    
    
    def _forward_redshift(self, features, instances, targets=None):
        
        if self.training:
            #Add all gt bounding boxes for redshift regression
            proposals = add_ground_truth_to_proposals(targets, instances)
            finstances, _ = select_foreground_proposals(proposals, self.num_classes)

            #finstances, _ = select_foreground_proposals(instances, self.num_classes)

            instances = []
            for x in finstances:
                gold_inst = x[x.gt_magi < 25.3]
                instances.append(gold_inst)
            if len(instances)==0:
                return 0
            
        #print('instances ', len(instances[0]))
        #sz = np.load('/home/g4merz/rail_deepdisc/sampled_zs_gold.npy')
        #sampled_zs = instances[0].gt_redshift.detach().cpu().numpy()
        #szs = np.concatenate([sz,sampled_zs])
        #np.save('/home/g4merz/rail_deepdisc/sampled_zs_gold.npy', szs)
                
        if self.redshift_pooler is not None:
            features = [features[f] for f in self.box_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.redshift_pooler(features, boxes)
        
        features = nn.Flatten()(features)
        
        num_instances_per_img = [len(i) for i in instances]

        if self.training:
            fcs = self.redshift_fc(features)
            #pdfs = self.output_pdf(fcs)

            gt_redshifts = cat([x.gt_redshift for x in instances])
            #print(gt_redshifts.size())
            
            #nlls = -pdfs.log_prob(gt_redshifts) * self.zloss_factor
            #return {"redshift_loss": torch.mean(nlls)}
            
            losses = self.CRPS_loss(fcs,gt_redshifts)
            return {"redshift_loss": torch.mean(losses)}

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

            return instances

    def forward(self, images, features, proposals, targets=None):
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        if self.training:
            # Need targets to box head
            losses = self._forward_box(features, proposals, targets)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_redshift(features, proposals, targets))
            #losses.update(self._forward_redshift(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            pred_instances = self._forward_redshift(features, pred_instances)
            return pred_instances, {}
        
        
class RedshiftPDFCasROIHeadsGoldUniform(CascadeROIHeads):
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
        coarse_bins: List[float],
        cmax: int,
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
        self.cmax = cmax
        
        #coarse_bins = np.linspace(0,3,10)
        coarse_bins = torch.tensor(coarse_bins)        
        self.register_buffer('coarse_bins', coarse_bins, persistent=False)

        self.redshift_fc = nn.Sequential(
            nn.Linear(np.prod(self._output_size), 1024),
            nn.Tanh(),
            nn.Linear(1024, 64),
            nn.Tanh(),
            nn.Linear(64, self.num_components * 3),
            #nn.Softplus()
        )

    def _positive_sample_proposals(
        self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        #sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
        #    gt_classes, self.batch_size_per_image, self.positive_fraction, self.num_classes
        #)
        
        positive = nonzero_tuple((gt_classes != -1) & (gt_classes != self.num_classes))[0]
        #positive = nonzero_tuple((gt_classes != -1) & (gt_classes != self.num_classes))[0]
        #sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return positive, gt_classes[positive]
        
    @torch.no_grad()
    def select_all_positive_proposals(
        self, proposals: List[Instances], targets: List[Instances], proposal_append_gt=True
    ) -> List[Instances]:
        """
        Prepare all positive proposals to be used to train the Redshift ROI head. 
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            
            sampled_idxs, gt_classes = self._positive_sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            #num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            #num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        #storage = get_event_storage()
        #storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        #storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))
        #print(len(proposals_with_gt))
        return proposals_with_gt

    @torch.no_grad()
    def uniform_redshift_sample(self, proposals: List[Instances],
    ) -> List[Instances]:
        """
        Resample foreground proposals on a coarse grid to approximate a uniform distribution. 
        Assumes proposals have been matched and assigned redshifts. It returns a sample from the proposals.   

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the resampled proposals
        """
        resampled_proposals = []
        for proposals_per_image in proposals:
            redshifts = proposals_per_image.gt_redshift
            inds = torch.bucketize(redshifts,self.coarse_bins)
            inds_resample = []
            for i in torch.unique(inds):
                ci = torch.where(inds==i)[0]
                cinds = inds[ci]
                indices = torch.randperm(len(cinds))[0:self.cmax]
                cinds = ci[indices]
                inds_resample.append(cinds)

            #inds_resample = nonzero_tuple(inds_resample)[0]
            inds_resample = cat(inds_resample)
            resampled_proposals.append(proposals_per_image[inds_resample])
            
        return resampled_proposals
        
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

    def _forward_redshift(self, features, instances, targets=None):
        
        if self.training:
            #print('proposals ', len(instances[0]))
            #proposals = add_ground_truth_to_proposals(targets, instances)
            finstances = self.select_all_positive_proposals(instances, targets)
            
            #sz = np.load('/home/g4merz/rail_deepdisc/sampled_zs.npy')
            #sampled_zs = proposals[0].gt_redshift.detach().cpu().numpy()
            #szs = np.concatenate([sz,sampled_zs])
            #np.save('/home/g4merz/rail_deepdisc/sampled_zs.npy', szs)
            
            
            #print('positive proposals', len(finstances[0]))
            #proposals = proposals[:100]
            #print('proposals with gt', len(proposals[0]))
            #finstances, _ = select_foreground_proposals(instances, self.num_classes)
            #print('sampled foreground proposals', len(finstances[0]))
         
            #Add all gt bounding boxes for redshift regression
            #finstances, _ = select_foreground_proposals(proposals, self.num_classes)

            #print('sampled foreground proposals', len(finstances[0]))

            instances = []
            for x in finstances:
                gold_inst = x[x.gt_magi < 25.3]
                instances.append(gold_inst)
            if len(instances)==0:
                return 0
            
            instances = self.uniform_redshift_sample(instances)
        
        #print('gold sampled proposals', len(instances[0]))
        #sz = np.load('/home/g4merz/rail_deepdisc/sampled_zs_gold.npy')
        #sampled_zs = instances[0].gt_redshift.detach().cpu().numpy()
        #szs = np.concatenate([sz,sampled_zs])
        #np.save('/home/g4merz/rail_deepdisc/sampled_zs_gold.npy', szs)
        
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
                pred_instances.pred_gmm =  np.split(fcs,inds)[i]

            return instances

    def forward(self, images, features, proposals, targets=None):
        del images
        if self.training:
            sampledproposals = self.label_and_sample_proposals(proposals, targets)
            #zproposals = self.select_uniform_zproposals(proposals)

        if self.training:
            # Need targets to box head
            losses = self._forward_box(features, sampledproposals, targets)
            losses.update(self._forward_mask(features, sampledproposals))
            losses.update(self._forward_redshift(features, proposals, targets))
            losses.update(self._forward_keypoint(features, sampledproposals))
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
        

        '''
        self.redshift_fc = nn.Sequential(
            nn.Linear(np.prod(self._output_size), 64),
            nn.Tanh(),
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Linear(16, self.num_components * 3),
            #nn.Softplus()
        )
        '''

        
        
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

    def _forward_redshift(self, features, instances, targets=None):
        
        if self.training:
            #Add all gt bounding boxes for redshift regression
            proposals = add_ground_truth_to_proposals(targets, instances)
            instances, _ = select_foreground_proposals(proposals, self.num_classes)

            #finstances, _ = select_foreground_proposals(instances, self.num_classes)
            
            if len(instances)==0:
                return 0
            
        #print('instances ', len(instances[0]))
        #sz = np.load('/home/g4merz/rail_deepdisc/sampled_zs_gold.npy')
        #sampled_zs = instances[0].gt_redshift.detach().cpu().numpy()
        #szs = np.concatenate([sz,sampled_zs])
        #np.save('/home/g4merz/rail_deepdisc/sampled_zs_gold.npy', szs)
                
        if self.redshift_pooler is not None:
            features = [features[f] for f in self.box_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.redshift_pooler(features, boxes)
        
        features = nn.Flatten()(features)
        
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
                pred_instances.pred_gmm =  np.split(fcs,inds)[i]

            return instances

    def forward(self, images, features, proposals, targets=None):
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        if self.training:
            # Need targets to box head
            losses = self._forward_box(features, proposals, targets)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_redshift(features, proposals, targets))
            #losses.update(self._forward_redshift(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            pred_instances = self._forward_redshift(features, pred_instances)
            return pred_instances, {}

        
class RedshiftPDFCasROIHeadsUniform(CascadeROIHeads):
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
        coarse_bins: List[float],
        cmax: int,
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
        self.cmax = cmax
        
        #coarse_bins = np.linspace(0,3,10)
        coarse_bins = torch.tensor(coarse_bins)        
        self.register_buffer('coarse_bins', coarse_bins, persistent=False)

        self.redshift_fc = nn.Sequential(
            nn.Linear(np.prod(self._output_size), 1024),
            nn.Tanh(),
            nn.Linear(1024, 64),
            nn.Tanh(),
            nn.Linear(64, self.num_components * 3),
            #nn.Softplus()
        )

    def _positive_sample_proposals(
        self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        #sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
        #    gt_classes, self.batch_size_per_image, self.positive_fraction, self.num_classes
        #)
        
        positive = nonzero_tuple((gt_classes != -1) & (gt_classes != self.num_classes))[0]
        #positive = nonzero_tuple((gt_classes != -1) & (gt_classes != self.num_classes))[0]
        #sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return positive, gt_classes[positive]
        
    @torch.no_grad()
    def select_all_positive_proposals(
        self, proposals: List[Instances], targets: List[Instances], proposal_append_gt=True
    ) -> List[Instances]:
        """
        Prepare all positive proposals to be used to train the Redshift ROI head. 
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            
            sampled_idxs, gt_classes = self._positive_sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            #num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            #num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        #storage = get_event_storage()
        #storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        #storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))
        #print(len(proposals_with_gt))
        return proposals_with_gt

    @torch.no_grad()
    def uniform_redshift_sample(self, proposals: List[Instances],
    ) -> List[Instances]:
        """
        Resample foreground proposals on a coarse grid to approximate a uniform distribution. 
        Assumes proposals have been matched and assigned redshifts. It returns a sample from the proposals.   

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the resampled proposals
        """
        resampled_proposals = []
        for proposals_per_image in proposals:
            redshifts = proposals_per_image.gt_redshift
            inds = torch.bucketize(redshifts,self.coarse_bins)
            inds_resample = []
            for i in torch.unique(inds):
                ci = torch.where(inds==i)[0]
                cinds = inds[ci]
                indices = torch.randperm(len(cinds))[0:self.cmax]
                cinds = ci[indices]
                inds_resample.append(cinds)

            #inds_resample = nonzero_tuple(inds_resample)[0]
            inds_resample = cat(inds_resample)
            resampled_proposals.append(proposals_per_image[inds_resample])
            
        return resampled_proposals
        
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

    def _forward_redshift(self, features, instances, targets=None):
        
        if self.training:
            #print('proposals ', len(instances[0]))
            #proposals = add_ground_truth_to_proposals(targets, instances)
            instances = self.select_all_positive_proposals(instances, targets)
            
            #sz = np.load('/home/g4merz/rail_deepdisc/sampled_zs.npy')
            #sampled_zs = proposals[0].gt_redshift.detach().cpu().numpy()
            #szs = np.concatenate([sz,sampled_zs])
            #np.save('/home/g4merz/rail_deepdisc/sampled_zs.npy', szs)
            
            
            #print('positive proposals', len(finstances[0]))
            #proposals = proposals[:100]
            #print('proposals with gt', len(proposals[0]))
            #finstances, _ = select_foreground_proposals(instances, self.num_classes)
            #print('sampled foreground proposals', len(finstances[0]))
         
            #Add all gt bounding boxes for redshift regression
            #finstances, _ = select_foreground_proposals(proposals, self.num_classes)

            #print('sampled foreground proposals', len(finstances[0]))
            
            instances = self.uniform_redshift_sample(instances)
        
        #print('gold sampled proposals', len(instances[0]))
        #sz = np.load('/home/g4merz/rail_deepdisc/sampled_zs_gold.npy')
        #sampled_zs = instances[0].gt_redshift.detach().cpu().numpy()
        #szs = np.concatenate([sz,sampled_zs])
        #np.save('/home/g4merz/rail_deepdisc/sampled_zs_gold.npy', szs)
        
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

            return instances

    def forward(self, images, features, proposals, targets=None):
        del images
        if self.training:
            sampledproposals = self.label_and_sample_proposals(proposals, targets)
            #zproposals = self.select_uniform_zproposals(proposals)

        if self.training:
            # Need targets to box head
            losses = self._forward_box(features, sampledproposals, targets)
            losses.update(self._forward_mask(features, sampledproposals))
            losses.update(self._forward_redshift(features, proposals, targets))
            losses.update(self._forward_keypoint(features, sampledproposals))
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

        
