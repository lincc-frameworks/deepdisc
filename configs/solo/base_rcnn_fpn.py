"""Base RCNN FPN

Translated into a LazyConfig-compatible OmegaConf from Detectron2's model zoo:
https://github.com/facebookresearch/detectron2/blob/main/configs/Base-RCNN-FPN.yaml

Include in a config via:
from .base_rcnn_fpn import DATASETS, INPUT, MODEL, SOLVER
"""

from omegaconf import OmegaConf

from .yacs_style_defaults import (
    DATALOADER,
    DATASETS,
    GLOBAL,
    INPUT,
    MISC,
    MODEL,
    SOLVER,
    TEST,
)

DATASETS.TRAIN = ("coco_2017_train",)
DATASETS.TEST = ("coco_2017_val",)

INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)

MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [256], [512]]  # One size for each in feature map
MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]  # Three aspect ratios (same for all in feature maps)

MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"

MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]

MODEL.META_ARCHITECTURE = "GeneralizedRCNN"

MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]

MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead"
MODEL.ROI_BOX_HEAD.NUM_FC = 2
MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7

MODEL.ROI_HEADS.NAME = "StandardROIHeads"
MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]

MODEL.ROI_MASK_HEAD.NAME = "MaskRCNNConvUpsampleHead"
MODEL.ROI_MASK_HEAD.NUM_CONV = 4
MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14

MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
MODEL.RPN.PRE_NMS_TOPK_TRAIN = 2000  # Per FPN level
MODEL.RPN.PRE_NMS_TOPK_TEST = 1000  # Per FPN level
    # Detectron1 uses 2000 proposals per-batch,
    # (See "modeling/rpn/rpn_outputs.py" for details of this legacy issue)
    # which is approximately 1000 proposals per-image since the default batch size for FPN is 2.
MODEL.RPN.POST_NMS_TOPK_TRAIN = 1000
MODEL.RPN.POST_NMS_TOPK_TEST = 1000

SOLVER.IMS_PER_BATCH = 16
SOLVER.BASE_LR = 0.02
SOLVER.STEPS = (60000, 80000)
SOLVER.MAX_ITER = 90000

VERSION = 2