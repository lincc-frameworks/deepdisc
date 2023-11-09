""" This is a demo "solo config" file for use in the modified version of test_run_transformers.py"""

# 100

from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling import SwinTransformer

from ..common.coco_loader_lsj import dataloader

# ---------------------------------------------------------------------------- #
# Local variables
# ---------------------------------------------------------------------------- #
bs = 1
classes = ["star", "galaxy"]
numclasses = len(classes)

# ---------------------------------------------------------------------------- #
# Standard config (this has always been the LazyConfig/.py-style config)
# ---------------------------------------------------------------------------- #
# Get values from template
from ..COCO.cascade_mask_rcnn_swin_b_in21k_100ep import dataloader, model, train, lr_multiplier, optimizer

# Overrides
model.proposal_generator.anchor_generator.sizes = [[8], [16], [32], [64], [128]]
dataloader.train.total_batch_size = bs
model.roi_heads.num_classes = numclasses
model.roi_heads.batch_size_per_image = 512

# ---------------------------------------------------------------------------- #
# Dataloader config (was formerly saved as a .yaml file, loaded to cfg_loader)
# ---------------------------------------------------------------------------- #
# Get values from template
from .data_loader_defaults import MISC, DATALOADER, DATASETS, GLOBAL, INPUT, MODEL, SOLVER, TEST

# Overrides
# DATALOADER.NUM_WORKERS = 0 # Check, but I think this was commented out in original testruntransformers code
DATALOADER.PREFETCH_FACTOR = 2

DATASETS.TRAIN = "astro_train"  # Register Metadata
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
SOLVER.IMS_PER_BATCH = bs