"""This is a 'solo config' file using various baselines."""

from omegaconf import OmegaConf

# ---------------------------------------------------------------------------- #
# Local variables and metadata
# ---------------------------------------------------------------------------- #
bs = 1

metadata = OmegaConf.create()
metadata.classes = ["object"]

numclasses = len(metadata.classes)

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

from deepdisc.model.models import RedshiftPDFCasROIHeads

# Overrides
model.proposal_generator.anchor_generator.sizes = [[8], [16], [32], [64], [128]]
dataloader.train.total_batch_size = bs
model.roi_heads.num_classes = numclasses
model.roi_heads.batch_size_per_image = 512
model.roi_heads._target_ = RedshiftPDFCasROIHeads
model.roi_heads.zloss_factor=0.1
for box_predictor in model.roi_heads.box_predictors:
    box_predictor.test_topk_per_image = 1000
    box_predictor.test_score_thresh = 0.5
model.roi_heads.num_components = 1

# DC2 overrides
model.backbone.bottom_up.in_chans = 6
model.pixel_mean = [
    0.05381286,
    0.04986344,
    0.07526361,
    0.10420945,
    0.14229655,
    0.21245764,
]
model.pixel_std = [2.9318833, 1.8443471, 2.581817, 3.5950038, 4.5809164, 7.302009]

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
DATALOADER.PREFETCH_FACTOR = 2

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
SOLVER.IMS_PER_BATCH = bs
