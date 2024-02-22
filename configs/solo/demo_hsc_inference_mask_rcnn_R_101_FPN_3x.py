"""solo_demo_hsc_inference_mask_rcnn_R_101_FPN_3x"""

from omegaconf import OmegaConf

# ---------------------------------------------------------------------------- #
# Local variables and metadata
# ---------------------------------------------------------------------------- #
e1 = 200

# ---------------------------------------------------------------------------- #
# Standard, Lazy-Config-style config values
# ---------------------------------------------------------------------------- #
# Baselines

# ---------------------------------------------------------------------------- #
# Yacs-style config values
# ---------------------------------------------------------------------------- #
# Defaults
from .yacs_style_defaults import (
    MISC,
    DATALOADER,
    DATASETS,
    GLOBAL,
    INPUT,
    MODEL,
    SOLVER,
    TEST,
)

# Values from Detectron2's COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
# https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml

from .base_rcnn_fpn import DATASETS, INPUT, MODEL, SOLVER

MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
MODEL.MASK_ON = True
MODEL.RESNETS.DEPTH = 50

SOLVER.STEPS = (210000, 250000)
SOLVER.MAX_ITER = 270000

# Additional Overrides

DATALOADER.NUM_WORKERS = 2

DATASETS.TRAIN = "astro_train"
DATASETS.TEST = "astro_test"

INPUT.MIN_SIZE_TRAIN = 500
INPUT.MAX_SIZE_TRAIN = 525

MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]
MODEL.BACKBONE.FREEZE_AT = 4
MODEL.PIXEL_MEAN = [13.49794151, 9.11051305, 5.42995532]
MODEL.RPN.POST_NMS_TOPK_TRAIN = 6000
MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # faster, and good enough for this toy dataset (default: 512)
MODEL.ROI_HEADS.NUM_CLASSES = 1
MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.33

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
SOLVER.IMS_PER_BATCH = (
    2  # this is images per iteration. 1 epoch is len(images)/(ims_per_batch iterations)
)
SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
SOLVER.MAX_ITER = e1  # for DefaultTrainer
SOLVER.STEPS = []
SOLVER.WARMUP_ITERS = 0

TEST.DETECTIONS_PER_IMAGE = 500
