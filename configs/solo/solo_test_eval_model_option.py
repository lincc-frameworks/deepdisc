"""This is a 'solo config' file using various baselines."""

from omegaconf import OmegaConf

# ---------------------------------------------------------------------------- #
# Local variables and metadata
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# Standard, Lazy-Config-style config values
# ---------------------------------------------------------------------------- #
# Baselines
from ..solo.solo_cascade_mask_rcnn_swin_b_in21k_50ep_test_eval import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)

# ---------------------------------------------------------------------------- #
# Yacs-style config values
# ---------------------------------------------------------------------------- #
# Baselines
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

# Overrides
DATASETS.TRAIN = "astro_train"  # Register Metadata
DATASETS.TEST = (
    "astro_test",
)  # Config calls this TEST, but it should be the val dataset
DATALOADER.NUM_WORKERS = 1
SOLVER.IMS_PER_BATCH = 4  # this is images per iteration. 1 epoch is len(images)/(ims_per_batch iterations*num_gpus)
SOLVER.BASE_LR = 0.001
SOLVER.STEPS = []  # do not decay learning rate for retraining
SOLVER.MAX_ITER = 100  # for DefaultTrainer
MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    512  # faster, and good enough for this toy dataset (default: 512)
)
# MODEL.ROI_HEADS.NUM_CLASSES = nc
# OUTPUT_DIR = output_dir
TEST.DETECTIONS_PER_IMAGE = 1000
INPUT.MIN_SIZE_TRAIN = 1025
INPUT.MAX_SIZE_TRAIN = 1050

# Defaults
# PRE_NMS_TOPK_TEST: 6000git st
# POST_NMS_TOPK_TEST: 1000
# PRE_NMS_TOPK_TRAIN: 12000
# POST_NMS_TOPK_TRAIN: 2000

MODEL.RPN.POST_NMS_TOPK_TEST = 6000
MODEL.RPN.PRE_NMS_TOPK_TEST = 6000

MODEL.RPN.BATCH_SIZE_PER_IMAGE = 512
MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]

# MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, run_name)  # path to the model we just trained
# MODEL.ROI_HEADS.SCORE_THRESH_TEST = roi_thresh  # set a custom testing threshold
MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
MODEL.ROI_BOX_HEAD.GAMMA = 1
MODEL.ROI_BOX_HEAD.ALPHAS = None
