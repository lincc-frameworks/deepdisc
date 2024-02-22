"""This is a 'solo config' file using various baselines."""

from omegaconf import OmegaConf

# ---------------------------------------------------------------------------- #
# Local variables and metadata
# ---------------------------------------------------------------------------- #
bs = 1

metadata = OmegaConf.create()
metadata.classes = ["star", "galaxy"]

numclasses = len(metadata.classes)

# ---------------------------------------------------------------------------- #
# Standard, Lazy-Config-style config values
# ---------------------------------------------------------------------------- #
# Baselines
from ..COCO.cascade_mask_rcnn_mvitv2_b_in21k_100ep import (
    constants,
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)

# Overrides
model.proposal_generator.anchor_generator.sizes = [[8], [16], [32], [64], [128]]
dataloader.train.total_batch_size = bs
model.roi_heads.num_classes = numclasses
model.roi_heads.batch_size_per_image = 512

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
