from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2 import model_zoo
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import MultiStepParamScheduler
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.roi_heads import FastRCNNOutputLayers, FastRCNNConvFCHead, CascadeROIHeads
from omegaconf import OmegaConf
import numpy as np
from ..COCO.cascade_mask_rcnn_mvitv2_b_in21k_100ep import dataloader, train, lr_multiplier, optimizer
from ..common.models.mask_rcnn_fpn import model


#import deepdisc.model.models as roiheads
#import deepdisc.model.loaders as loaders
#import deepdisc.model.meta_arch as meta_arch 
#from deepdisc.data_format.augment_image import dc2_train_augs, dc2_train_augs_full
#from deepdisc.data_format.image_readers import DC2ImageReader


# ---------------------------------------------------------------------------- #
# Local variables and metadata
# ---------------------------------------------------------------------------- #
bs = 1

metadata = OmegaConf.create() 
metadata.classes = ["object"]

numclasses = len(metadata.classes)

# Overrides
#dataloader.augs = dc2_train_augs
dataloader.train.total_batch_size = bs

model.proposal_generator.anchor_generator.sizes = [[8], [16], [32], [64], [128]]
model.roi_heads.num_classes = numclasses
model.roi_heads.batch_size_per_image = 512

model.roi_heads.num_classes = numclasses
model.roi_heads.batch_size_per_image = 512
model.proposal_generator.post_nms_topk=[6000,1000]
model.roi_heads.box_predictor.test_topk_per_image = 500
model.roi_heads.box_predictor.test_score_thresh = 0.3
model.roi_heads.box_predictor.test_nms_thresh = 0.5
   

train.init_checkpoint = "detectron2://ImageNetPretrained/MSRA/R-50.pkl" 
#train.init_checkpoint = '/home/g4merz/DC2/model_tests/zoobot/zoobot_GZ2_resnet50_d2.pkl'

optimizer.lr = 0.001
#dataloader.test.mapper = loaders.Dictmapper
#dataloader.train.mapper = loaders.DictMapper
#reader = HSCImageReader()
#dataloader.imagereader = reader

# ---------------------------------------------------------------------------- #
# Yaml-style config (was formerly saved as a .yaml file, loaded to cfg_loader)
# ---------------------------------------------------------------------------- #
# Get values from template
from .yacs_style_defaults import MISC, DATALOADER, DATASETS, GLOBAL, INPUT, MODEL, SOLVER, TEST

# Overrides
SOLVER.IMS_PER_BATCH = bs

DATASETS.TRAIN = "astro_train"
DATASETS.TEST = "astro_val"

SOLVER.BASE_LR = 0.001
SOLVER.CLIP_GRADIENTS.ENABLED = True
# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
# Maximum absolute value used for clipping gradients
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
SOLVER.CLIP_GRADIENTS.NORM_TYPE = 5.0


SOLVER.STEPS = []  # do not decay learning rate for retraining
SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
SOLVER.WARMUP_ITERS = 0
TEST.DETECTIONS_PER_IMAGE = 3000
