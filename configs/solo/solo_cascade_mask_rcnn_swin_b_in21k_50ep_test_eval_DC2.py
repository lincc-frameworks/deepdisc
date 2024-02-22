"""Config used in test_eval_model.

- COCO.cascade_mask_rcnn_swin_b_in21k_50ep
- DC2 data
- no redshifts

"""

from omegaconf import OmegaConf

# ---------------------------------------------------------------------------- #
# Local variables and metadata
# ---------------------------------------------------------------------------- #

classes = ["object"]
roi_thresh = 0.1 #! check default

# ---------------------------------------------------------------------------- #
# Standard, Lazy-Config-style config values
# ---------------------------------------------------------------------------- #
# Baselines
from ..COCO.cascade_mask_rcnn_swin_b_in21k_50ep import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)

# Overrides
dataloader.train.total_batch_size = 4

model.backbone.bottom_up.in_chans = 6
model.pixel_mean = [0.05381286, 0.04986344, 0.07526361, 0.10420945, 0.14229655, 0.21245764]
model.pixel_std = [2.9318833, 1.8443471, 2.581817, 3.5950038, 4.5809164, 7.302009]

model.roi_heads.num_classes = len(classes)
model.roi_heads.batch_size_per_image = 512

for box_predictor in model.roi_heads.box_predictors:
    box_predictor.test_topk_per_image = 1000
    box_predictor.test_score_thresh = roi_thresh

model.proposal_generator.anchor_generator.sizes = [[8], [16], [32], [64], [128]]
model.proposal_generator.pre_nms_topk = [6000, 6000]
model.proposal_generator.post_nms_topk = [6000, 6000]
model.proposal_generator.nms_thresh = 0.3

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
DATALOADER.NUM_WORKERS = 1

DATASETS.TRAIN = "astro_train"  # Register Metadata
DATASETS.TEST = "astro_val"

MISC.classes = classes

SOLVER.BASE_LR = 0.001
SOLVER.IMS_PER_BATCH = 4

TEST.DETECTIONS_PER_IMAGE = 1000
