import sys, os
import numpy as np
import json
import cv2
from astropy.io import fits
import time
import datetime
import logging


import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()
from detectron2.utils.logger import log_every_n_seconds
from detectron2.utils import comm

import detectron2.data as data
import detectron2.data.transforms as T
from detectron2.data.transforms import Transform
from detectron2.data.transforms import Augmentation


from detectron2.data.transforms import Transform
from detectron2.data.transforms import Augmentation
from fvcore.transforms.transform import Transform, TransformList

from detectron2.data import build_detection_train_loader
from detectron2.structures import BoxMode

# Yufeng Dec21 more import
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import HookBase

from detectron2.utils import comm
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.env import seed_all_rng
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.engine.hooks import LRScheduler
from fvcore.common.param_scheduler import ParamScheduler
import torch
from torch import nn
from torch.nn import functional as F

from typing import Dict, List, Optional, Tuple
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.layers import Conv2d, ShapeSpec, get_norm, cat, nonzero_tuple
from detectron2.modeling import ROI_HEADS_REGISTRY
from detectron2.modeling.roi_heads import StandardROIHeads
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import CascadeROIHeads
from detectron2.modeling.matcher import Matcher

from torch.distributions.independent import Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.beta import Beta
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


def plot_stretch_Q(
    dataset_dicts,
    astro_metadata,
    num=0,
    stretches=[0.01, 0.1, 0.5, 1],
    Qs=[1, 10, 5, 100],
    ceil_percentile=99.5,
):
    """
    Plots different normalizations of your image using the stretch, Q parameters.

    Parameters
    ----------
    dataset_dicts : dict
        detectron dataset dictionary
    num : int
        Dataset number/index to use
    stretches : array
        List of stretch params you want to permutate through to find optimal image normalization.
        Default is [0.01, 0.1, 0.5, 1]
    Qs : array
        List of Q params you want to permutate through to find optimal image normalization.
        Default is [1, 10, 5, 100]

    Code adapted from:
        https://pmelchior.github.io/scarlet/tutorials/display.html

    Returns
    -------
    fig : Figure object

    """

    d = dataset_dicts[num]

    fig, ax = plt.subplots(len(stretches), len(Qs), figsize=(9, 9))
    for i, stretch in enumerate(stretches):
        for j, Q in enumerate(Qs):
            img = read_image(d, normalize="lupton", stretch=stretch, Q=Q, ceil_percentile=ceil_percentile)
            # Scale the RGB channels for the image
            visualizer = Visualizer(img, metadata=astro_metadata)
            out = visualizer.draw_dataset_dict(d)
            ax[i][j].imshow(out.get_image(), origin="lower")
            ax[i][j].set_title("Stretch {}, Q {}".format(stretch, Q), fontsize=10)
            ax[i][j].axis("off")

    return fig


"""Note:SaveHook is in charge of saving the trained model"""


class SaveHook(HookBase):

    """
    This Hook saves the model after training

    """

    output_name = "model_temp"

    def set_output_name(self, name):
        self.output_name = name

    def after_train(self):
        print("saving", self.output_name)
        self.trainer.checkpointer.save(self.output_name)  # Note: Set the name of the output model here


#
class LossEvalHook(HookBase):

    """
    Validation loss code adopted from https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b

    Parameters
    ----------
    eval_period: int
        How many iterations to run before validation loss is calculated
    model: torch.NN.module
        The model being trained
    data_loader: detectron2 DataLoader
        The dataloader that loads in the evaluation dataset
    """

    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader

    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        with torch.no_grad():
            for idx, inputs in enumerate(self._data_loader):
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_compute_time = 0
                start_compute_time = time.perf_counter()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_compute_time += time.perf_counter() - start_compute_time
                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                seconds_per_img = total_compute_time / iters_after_start
                if idx >= num_warmup * 2 or seconds_per_img > 5:
                    total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                    eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                    log_every_n_seconds(
                        logging.INFO,
                        "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                            idx + 1, total, seconds_per_img, str(eta)
                        ),
                        n=5,
                    )
                loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        # print('validation_loss', mean_loss)
        self.trainer.storage.put_scalar("validation_loss", mean_loss)
        self.trainer.add_val_loss(mean_loss)
        self.trainer.valloss = mean_loss
        comm.synchronize()
        return losses

    def _get_loss(self, data):
        # How loss is calculated on train_loop
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):  # or (next_iter == 1):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)


class CustomLRScheduler(HookBase):
    """
    A hook which executes a torch builtin LR scheduler and summarizes the LR.
    It is executed after every iteration.

    Parameters
    ----------
    optimizer: torch optimizer
        the training optimizer
    scheduler: torch scheduler
        The learning rate scheduler

    """

    def __init__(self, optimizer=None, scheduler=None):
        """
        Args:
            optimizer (torch.optim.Optimizer):
            scheduler (torch.optim.LRScheduler or fvcore.common.param_scheduler.ParamScheduler):
                if a :class:`ParamScheduler` object, it defines the multiplier over the base LR
                in the optimizer.

        If any argument is not given, will try to obtain it from the trainer.
        """
        self._optimizer = optimizer
        self._scheduler = scheduler

    def before_train(self):
        # print(self.scheduler)
        self._optimizer = self._optimizer or self.trainer.optimizer
        print(self.scheduler)
        if isinstance(self.scheduler, ParamScheduler):
            self._scheduler = LRMultiplier(
                self._optimizer,
                self.scheduler,
                self.trainer.max_iter,
                last_iter=self.trainer.iter - 1,
            )
        self._best_param_group_id = LRScheduler.get_best_param_group_id(self._optimizer)

    @staticmethod
    def get_best_param_group_id(optimizer):
        # NOTE: some heuristics on what LR to summarize
        # summarize the param group with most parameters
        largest_group = max(len(g["params"]) for g in optimizer.param_groups)

        if largest_group == 1:
            # If all groups have one parameter,
            # then find the most common initial LR, and use it for summary
            lr_count = Counter([g["lr"] for g in optimizer.param_groups])
            lr = lr_count.most_common()[0][0]
            for i, g in enumerate(optimizer.param_groups):
                if g["lr"] == lr:
                    return i
        else:
            for i, g in enumerate(optimizer.param_groups):
                if len(g["params"]) == largest_group:
                    return i

    def after_step(self):
        lr = self._optimizer.param_groups[self._best_param_group_id]["lr"]
        self.trainer.storage.put_scalar("lr", lr, smoothing_hint=False)
        self.scheduler.step()

    @property
    def scheduler(self):
        return self._scheduler or self.trainer.scheduler

    def state_dict(self):
        if isinstance(self.scheduler, _LRScheduler):
            return self.scheduler.state_dict()
        return {}

    def load_state_dict(self, state_dict):
        if isinstance(self.scheduler, _LRScheduler):
            logger = logging.getLogger(__name__)
            logger.info("Loading scheduler from state_dict ...")
            self.scheduler.load_state_dict(state_dict)


class GenericWrapperTransform(Transform):
    """
    Code from Deshwal on Stack Overflow
    Generic wrapper for any transform (for color transform only. You can give functionality to apply_coods, apply_segmentation too)
    """

    def __init__(self, custom_function):
        """
        Args:
            custom_function (Callable): operation to be applied to the image which takes in an ndarray and returns an ndarray.
        """
        if not callable(custom_function):
            raise ValueError("'custom_function' should be callable")

        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img):
        """
        apply transformation to image array based on the `custom_function`
        """
        return self.custom_function(img)

    def apply_coords(self, coords):
        """
        Apply transformations to Bounding Box Coordinates. Currently is won't do anything but we can change this based on our use case
        """
        return coords

    def inverse(self):
        return T.NoOpTransform()

    def apply_segmentation(self, segmentation):
        """
        Apply transformations to segmentation. currently is won't do anything but we can change this based on our use case
        """
        return segmentation


class CustomAug(Augmentation):
    """
    Given a probability and a custom function, return a GenericWrapperTransform object whose `apply_image`
    will be called to perform augmentation
    """

    def __init__(self, custom_function, prob=1.0):
        """
        Args:
            custom_op: Operation to use. Must be a function takes an ndarray and returns an ndarray
            prob (float): probability of applying the function
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        """
        Based on probability, choose whether you want to apply the given function or not
        """
        do = self._rand_range() < self.prob
        if do:
            return GenericWrapperTransform(self.custom_function)
        else:
            return (
                T.NoOpTransform()
            )  # it returns a Transform which just returns the original Image array only


class KRandomAugmentationList(Augmentation):
    """
    Select and Apply "K" augmentations in "RANDOM" order with "Every"  __call__ method invoke
    """

    def __init__(self, augs, k: int = -1, cropaug=None):
        """
        Args:
            augs: list of [Augmentation or Transform]
            k: Number of augment to use from the given list in range [1,len_augs].
            If None, use all. If it is -1, generate K randomly between [1,len_augs]
        """
        super().__init__()
        self.max_range = len(augs)
        self.k = k
        self.augs = augs  # set augs to use as fixed if we have to use same augs everytime
        self.cropaug = cropaug

    def _setup_augs(self, augs, k: int):
        """
        Setup the argument list. Generates the list of argument to use from the given list
        args:
            augs: list of [Augmentation or Transform])
            k: Number of augment to use from the given list in range [1,len_augs].
            If False, use all. If it is -1, generate K randomly between [1,len_augs]
        """
        if k == -1:  # Generate a random number
            k = np.random.randint(1, len(augs) + 1)

        elif k is None:  # use all
            k = self.max_range

        temp = np.random.choice(augs, k, replace=False)  # get k augments randomly
        augl = [_transform_to_aug(x) for x in temp]
        if self.cropaug is None:
            pass
        else:
            augl.insert(0, self.cropaug)

        return augl

    def __call__(self, aug_input) -> Transform:
        tfms = []

        for x in self._setup_augs(self.augs, self.k):  # generate auguments to use randomly on the fly
            # print(x)
            tfm = x(aug_input)
            tfms.append(tfm)
        return TransformList(tfms)

    def __repr__(self):
        msgs = [str(x) for x in self.augs]
        return "AugmentationList[{}]".format(", ".join(msgs))

    __str__ = __repr__


## Added for importing
class _TransformToAug(Augmentation):
    def __init__(self, tfm: Transform):
        self.tfm = tfm

    def get_transform(self, *args):
        return self.tfm

    def __repr__(self):
        return repr(self.tfm)

    __str__ = __repr__


def _transform_to_aug(tfm_or_aug):
    """
    Wrap Transform into Augmentation.
    Private, used internally to implement custom augmentations.
    """
    assert isinstance(tfm_or_aug, (Transform, Augmentation)), tfm_or_aug
    if isinstance(tfm_or_aug, Augmentation):
        return tfm_or_aug
    else:
        return _TransformToAug(tfm_or_aug)


# @ROI_HEADS_REGISTRY.register()
class RedshiftCasROIHeads(CascadeROIHeads):
    # def __init__(self, cfg, input_shape):
    def __init__(
        self,
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

            diff = prediction[fg_inds] - gt_redshifts[fg_inds]
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


# @ROI_HEADS_REGISTRY.register()
class RedshiftPDFCasROIHeads(CascadeROIHeads):
    # def __init__(self, cfg, input_shape):
    def __init__(
        self,
        num_components: int,
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

        self.redshift_fc = nn.Sequential(
            nn.Linear(int(np.prod(self._output_size)), 1024),
            nn.Tanh(),
            nn.Linear(1024, 64),
            nn.Tanh(),
            nn.Linear(64, self.num_components * 3),
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

        fcs = self.redshift_fc(features)

        pdfs = self.output_pdf(fcs)

        num_instances_per_img = [len(i) for i in instances]

        if self.training:
            gt_classes = cat([x.gt_classes for x in instances])
            fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
            pdfs_fg = self.output_pdf(fcs[fg_inds, ...])

            gt_redshifts = cat([x.gt_redshift for x in instances])

            nlls_fg = -pdfs_fg.log_prob(gt_redshifts[fg_inds])

            nlls = -pdfs.log_prob(gt_redshifts)[fg_inds]
            return {"redshift_loss": torch.mean(nlls)}

        else:
            zs = torch.tensor(np.linspace(-1, 5, 200)).to(fcs.device)

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


@ROI_HEADS_REGISTRY.register()
class ConvRedshiftROIHeads(StandardROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.redshift_pooler = ROIPooler(
            output_size=7,
            scales=tuple(1.0 / input_shape[k].stride for k in in_features),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        )

        in_channels = [input_shape[f].channels for f in in_features]
        in_channels = in_channels[0]
        inshape = ShapeSpec(channels=in_channels, height=7, width=7)

        # The input dim should follow from the classification head
        self._output_size = (inshape.channels, inshape.height, inshape.width)
        self.redshift_fc = nn.Linear(int(np.prod(self._output_size)), 1)

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
            diff = prediction - cat([x.gt_redshift for x in instances])
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
