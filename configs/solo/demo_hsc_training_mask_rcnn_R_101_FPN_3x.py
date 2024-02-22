"""solo_demo_hsc_training_mask_rcnn_R_101_FPN_3x"""

from omegaconf import OmegaConf

# ---------------------------------------------------------------------------- #
# Local variables and metadata
# ---------------------------------------------------------------------------- #
e1 = 200
roi_thresh = 0.2
nms_thresh = 0.3

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

MISC.e1 = e1

MODEL.ROI_MASK_HEAD.NAME = "MaskRCNNConvUpsampleHead"
MODEL.ROI_MASK_HEAD.NUM_CONV = 4
MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14

MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
MODEL.MASK_ON = True
MODEL.RESNETS.DEPTH = 50

SOLVER.STEPS = (210000, 250000)
SOLVER.MAX_ITER = 270000

# Additional Overrides

DATASETS.TRAIN = "astro_train"  # Register Metadata # TODO: Should be TRAIN
DATASETS.TEST = "astro_test"  # Config calls this TEST, but it should be the val dataset

INPUT.MIN_SIZE_TRAIN = 1025
INPUT.MAX_SIZE_TRAIN = 1050

MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]

MODEL.ROI_HEADS.SCORE_THRESH_TEST = roi_thresh  # set a custom testing threshold
MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_thresh
MODEL.ROI_HEADS.NUM_CLASSES = 1

MODEL.RPN.POST_NMS_TOPK_TEST = 6000
MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
MODEL.RPN.BATCH_SIZE_PER_IMAGE = 512

TEST.DETECTIONS_PER_IMAGE = 500