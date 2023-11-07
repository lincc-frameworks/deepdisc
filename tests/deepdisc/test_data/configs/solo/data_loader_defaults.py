"""Defaults translated from _C, a yaml-style config, to a LazyConfig style config"""

from detectron2.config import LazyCall as L
from omegaconf import OmegaConf

# Lazy calls:
from detectron2.data.samplers.distributed_sampler import TrainingSampler

# TODO go through string vals and replace with lazy function calls as needed
# TODO fill in 3 missing lines that came from weird sort-of-array syntax

# ---------------------------------------------------------------------------- #
# Misc
# ---------------------------------------------------------------------------- #

MISC = OmegaConf.create() # for non-objects (otherwise they are ignored)
MISC.CUDNNBENCHMARK = False
MISC.OUTPUT_DIR = "./output"
MISC.SEED = -1
MISC.VERSION = 2
MISC.VIS_PERIOD = 0

# ---------------------------------------------------------------------------- #
# Dataloader
# ---------------------------------------------------------------------------- #

DATALOADER = OmegaConf.create()
DATALOADER.ASPECT_RATIO_GROUPING = True
DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
DATALOADER.NUM_WORKERS = 4
DATALOADER.REPEAT_THRESHOLD = 0.0
DATALOADER.SAMPLER_TRAIN = L(TrainingSampler)() # do not forget the () at the end! even if empty

# ---------------------------------------------------------------------------- #
# Datasets
# ---------------------------------------------------------------------------- #

DATASETS = OmegaConf.create()
DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST = 1000
DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN = 2000
DATASETS.PROPOSAL_FILES_TEST = []
DATASETS.PROPOSAL_FILES_TRAIN = []
DATASETS.TEST =[]
DATASETS.TRAIN = []

# ---------------------------------------------------------------------------- #
# Global
# ---------------------------------------------------------------------------- #

GLOBAL = OmegaConf.create()
GLOBAL.HACK = 1.0

# ---------------------------------------------------------------------------- #
# Input
# ---------------------------------------------------------------------------- #

INPUT = OmegaConf.create()
INPUT.CROP = OmegaConf.create()
INPUT.CROP.ENABLED = False
INPUT.CROP.SIZE = [0.9, 0.9] # is this a correct translation?
INPUT.CROP.TYPE = "relative_range" # is it ok to make these strings? should some be lazy objects?
INPUT.FORMAT = "BGR"
INPUT.MASK_FORMAT = "polygon"
INPUT.MAX_SIZE_TEST = 1333
INPUT.MAX_SIZE_TRAIN = 1333
INPUT.MIN_SIZE_TEST = 800
INPUT.MIN_SIZE_TRAIN = [800] # is this a correct translation?
INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
INPUT.RANDOM_FLIP = "horizontal"

# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #

MODEL = OmegaConf.create()
MODEL.ANCHOR_GENERATOR = OmegaConf.create()
MODEL.ANCHOR_GENERATOR.ANGLES = [[-90, 0, 90]]
MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
MODEL.ANCHOR_GENERATOR.NAME = "DefaultAnchorGenerator"
MODEL.ANCHOR_GENERATOR.OFFSET = 0.0
MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]
MODEL.BACKBONE = OmegaConf.create()
MODEL.BACKBONE.FREEZE_AT = 2
MODEL.BACKBONE.NAME = "build_resnet_backbone"
MODEL.DEVICE = "cuda"
MODEL.FPN = OmegaConf.create()
MODEL.FPN.FUSE_TYPE = "sum"
MODEL.FPN.IN_FEATURES = []
MODEL.FPN.NORM = ""
MODEL.FPN.OUT_CHANNELS = 256
MODEL.KEYPOINT_ON = False
MODEL.LOAD_PROPOSALS = False
MODEL.MASK_ON = False
MODEL.META_ARCHITECTURE = "GeneralizedRCNN"

MODEL.PANOPTIC_FPN = OmegaConf.create()
MODEL.PANOPTIC_FPN.COMBINE = OmegaConf.create()
MODEL.PANOPTIC_FPN.COMBINE.ENABLED = True
MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
MODEL.PANOPTIC_FPN.COMBINE.OVERLAP_THRESH = 0.5
MODEL.PANOPTIC_FPN.COMBINE.STUFF_AREA_LIMIT = 4096
MODEL.PANOPTIC_FPN.INSTANCE_LOSS_WEIGHT = 1.0
MODEL.PIXEL_MEAN = [103.53, 116.28, 123.675]
MODEL.PIXEL_STD = [1.0, 1.0, 1.0]

MODEL.PROPOSAL_GENERATOR = OmegaConf.create()
MODEL.PROPOSAL_GENERATOR.MIN_SIZE = 0
MODEL.PROPOSAL_GENERATOR.NAME = "RPN"

MODEL.RESNETS = OmegaConf.create()
MODEL.RESNETS.DEFORM_MODULATED = False
MODEL.RESNETS.DEFORM_NUM_GROUPS = 1
MODEL.RESNETS.DEFORM_ON_PER_STAGE = [False, False, False, False]
MODEL.RESNETS.DEPTH = 50
MODEL.RESNETS.NORM = "FrozenBN"
MODEL.RESNETS.NUM_GROUPS = 1
MODEL.RESNETS.OUT_FEATURES = ["res4"]
MODEL.RESNETS.RES2_OUT_CHANNELS = 256
MODEL.RESNETS.RES5_DILATION = 1
MODEL.RESNETS.STEM_OUT_CHANNELS = 64
MODEL.RESNETS.STRIDE_IN_1X1 = True
MODEL.RESNETS.WIDTH_PER_GROUP = 64

MODEL.RETINANET = OmegaConf.create()
MODEL.RETINANET.BBOX_REG_LOSS_TYPE = "smooth_l1"
#MODEL.RETINANET.BBOX_REG_WEIGHTS = TODO
MODEL.RETINANET.FOCAL_LOSS_ALPHA = 0.25
MODEL.RETINANET.FOCAL_LOSS_GAMMA = 2.0
MODEL.RETINANET.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
MODEL.RETINANET.IOU_LABELS = [0, -1, 1]
MODEL.RETINANET.IOU_THRESHOLDS = [0.4, 0.5]
MODEL.RETINANET.NMS_THRESH_TEST = 0.5
MODEL.RETINANET.NORM = ""
MODEL.RETINANET.NUM_CLASSES = 80
MODEL.RETINANET.NUM_CONVS = 4
MODEL.RETINANET.PRIOR_PROB = 0.01
MODEL.RETINANET.SCORE_THRESH_TEST = 0.05
MODEL.RETINANET.SMOOTH_L1_LOSS_BETA = 0.1
MODEL.RETINANET.TOPK_CANDIDATES_TEST = 1000

MODEL.ROI_BOX_CASCADE_HEAD = OmegaConf.create()
#MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS = TODO
MODEL.ROI_BOX_CASCADE_HEAD.IOUS = [0.5, 0.6, 0.7]

MODEL.ROI_BOX_HEAD = OmegaConf.create()
MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "smooth_l1"
MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT = 1.0
# MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = TODO
MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = False
MODEL.ROI_BOX_HEAD.CONV_DIM = 256
MODEL.ROI_BOX_HEAD.FC_DIM = 1024
MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT_POWER = 0.5
MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CLASSES = 50
MODEL.ROI_BOX_HEAD.NAME = ""
MODEL.ROI_BOX_HEAD.NORM = ""
MODEL.ROI_BOX_HEAD.NUM_CONV = 0
MODEL.ROI_BOX_HEAD.NUM_FC = 0
MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14
MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0
MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignV2"
MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA = 0.0
MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES = False
MODEL.ROI_BOX_HEAD.USE_FED_LOSS = False
MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE = False

MODEL.ROI_HEADS = OmegaConf.create()
MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
MODEL.ROI_HEADS.IN_FEATURES = ["res4"]
MODEL.ROI_HEADS.IOU_LABELS = [0, 1]
MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
MODEL.ROI_HEADS.NAME = "Res5ROIHeads"
MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
MODEL.ROI_HEADS.NUM_CLASSES = 80
MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25
MODEL.ROI_HEADS.PROPOSAL_APPEND_GT = True
MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05

MODEL.ROI_KEYPOINT_HEAD = OmegaConf.create()
MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS = [512] * 8
MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT = 1.0
MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE = 1
MODEL.ROI_KEYPOINT_HEAD.NAME = "KRCNNConvDeconvUpsampleHead"
MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS = True
MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 17
MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = 14
MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING = ""

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #

SOLVER = OmegaConf.create()
SOLVER.AMP = OmegaConf.create({'ENABLED': False})
SOLVER.BASE_LR = 0.001
SOLVER.BASE_LR_END = 0.0
SOLVER.BIAS_LR_FACTOR = 1.0
SOLVER.CHECKPOINT_PERIOD = 5000
SOLVER.CLIP_GRADIENTS = OmegaConf.create({
    'CLIP_TYPE': 'value',
    'CLIP_VALUE': 1.0,
    'ENABLED': False,
    'NORM_TYPE': 2.0
})
SOLVER.GAMMA = 0.1
SOLVER.IMS_PER_BATCH = 16
SOLVER.LR_SCHEDULER_NAME = 'WarmupMultiStepLR'
SOLVER.MAX_ITER = 40000
SOLVER.MOMENTUM = 0.9
SOLVER.NESTEROV = False
SOLVER.NUM_DECAYS = 3
SOLVER.REFERENCE_WORLD_SIZE = 0
SOLVER.RESCALE_INTERVAL = False
SOLVER.STEPS = [30000]
SOLVER.WARMUP_FACTOR = 0.001
SOLVER.WARMUP_ITERS = 1000
SOLVER.WARMUP_METHOD = 'linear'
SOLVER.WEIGHT_DECAY = 0.0001
SOLVER.WEIGHT_DECAY_BIAS = None
SOLVER.WEIGHT_DECAY_NORM = 0.0

# ---------------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------------- #

TEST = OmegaConf.create()
TEST.AUG = OmegaConf.create({
    'ENABLED': False,
    'FLIP': True,
    'MAX_SIZE': 4000,
    'MIN_SIZES': [400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
})
TEST.DETECTIONS_PER_IMAGE = 100
TEST.EVAL_PERIOD = 0
TEST.EXPECTED_RESULTS = []
TEST.KEYPOINT_OKS_SIGMAS = []
TEST.PRECISE_BN = OmegaConf.create({
    'ENABLED': False,
    'NUM_ITER': 200
})