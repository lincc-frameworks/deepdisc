# Some basic setup:

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

import copy
import json
import logging
import os
import random
import time
import weakref
from typing import Dict, List, Optional

import cv2
import detectron2.checkpoint as checkpointer
import detectron2.data as data
import detectron2.data.transforms as T
import detectron2.modeling as modeler
import detectron2.solver as solver
import imgaug.augmenters as iaa

# from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt

# import some common libraries
import numpy as np
import torch
from astropy.visualization import make_lupton_rgb

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_train_loader,
)
from detectron2.data import detection_utils as utils
from detectron2.engine import DefaultPredictor, DefaultTrainer, HookBase, SimpleTrainer
from detectron2.utils.file_io import PathManager
from detectron2.utils.visualizer import Visualizer
from iopath.common.file_io import file_lock
from PIL import Image, ImageEnhance

from astrodet import astrodet as toolkit
from astrodet import detectron as detectron_addons
from astrodet.detectron import plot_stretch_Q

logger = logging.getLogger(__name__)
import argparse
import glob
import json
import shutil
from pathlib import Path

import pycocotools.mask as mask_util
from astropy.io import fits
from detectron2 import structures
from detectron2.structures import BoxMode

from deepdisc.data_format.file_io import get_data_from_json

"""
This code will read in a trained model and output the classes for predicted objects matched to the ground truth 

"""


parser = argparse.ArgumentParser()


parser.add_argument("--run-name", default="Swin_test.pth", type=str)
parser.add_argument("--output-dir", default="/home/shared/hsc/HSC/HSC_DR3/models/withclasses/", type=str)
parser.add_argument(
    "--savedir",
    default="/home/shared/hsc/HSC/HSC_DR3/models/withclasses/eval/",
    type=str,
)
parser.add_argument("--norm", default="astrolupton", type=str)
parser.add_argument("--scheme", default=2, type=int)
parser.add_argument("--datatype", default=8, type=int)
parser.add_argument("--roi-thresh", default=0.5, type=float)
parser.add_argument(
    "--testfile",
    default="/home/shared/hsc/HSC/HSC_DR3/data/test_sample_new.json",
    type=str,
)

args = parser.parse_args()


dirpath = "/home/shared/hsc/HSC/HSC_DR3/data/"  # Path to dataset
output_dir = "/home/shared/hsc/HSC/HSC_DR3/models/"
roidir = f"roi_thresh{args.roi_thresh}"
savedir = os.path.join(args.savedir, roidir)
print(savedir)
Path(savedir).mkdir(parents=True, exist_ok=True)


def get_data_from_json(file):
    # Opening JSON file
    with open(file, "r") as f:
        data = json.load(f)
    return data


testfile = args.testfile

classes = ["star", "galaxy"]

dataset_names = ["test"]
datadir = "/home/shared/hsc/HSC/HSC_DR3/data/"
t0 = time.time()
dataset_dicts = {}
for i, d in enumerate(dataset_names):
    dataset_dicts[d] = get_data_from_json(testfile)


print("Took ", time.time() - t0, "seconds to load samples")

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:


def return_predictor(
    cfgfile,
    run_name,
    nc=1,
    output_dir="/home/shared/hsc/HSC/HSC_DR3/models/noclass/",
    roi_thresh=0.5,
):

    """
    This function returns a trained model and its config file.
    Used for models that have yacs config files

    Parameters
    ----------
    cfgfile: str
        A path to a model config file, provided by the detectron2 repo
    run_name: str
        Prefix used for the name of the saved model
    nc: int
        Number of classes used in the model
    output_dir: str
        THe directory to save metric outputs
    roi_thresh: float
        Hyperparamter that functions as a detection sensitivity level

    """

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfgfile))  # Get model structure
    cfg.DATASETS.TRAIN = "astro_train"  # Register Metadata
    cfg.DATASETS.TEST = ("astro_test",)  # Config calls this TEST, but it should be the val dataset
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.SOLVER.IMS_PER_BATCH = (
        4  # this is images per iteration. 1 epoch is len(images)/(ims_per_batch iterations*num_gpus)
    )
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.STEPS = []  # do not decay learning rate for retraining
    cfg.SOLVER.MAX_ITER = 100  # for DefaultTrainer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        512  # faster, and good enough for this toy dataset (default: 512)
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = nc
    cfg.OUTPUT_DIR = output_dir
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    cfg.INPUT.MIN_SIZE_TRAIN = 1025
    cfg.INPUT.MAX_SIZE_TRAIN = 1050

    # Defaults
    # PRE_NMS_TOPK_TEST: 6000
    # POST_NMS_TOPK_TEST: 1000
    # PRE_NMS_TOPK_TRAIN: 12000
    # POST_NMS_TOPK_TRAIN: 2000

    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 6000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000

    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, run_name)  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = roi_thresh  # set a custom testing threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
    cfg.MODEL.ROI_BOX_HEAD.GAMMA = 1
    cfg.MODEL.ROI_BOX_HEAD.ALPHAS = None

    predictor = toolkit.AstroPredictor(cfg)

    return predictor, cfg


# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
from detectron2.config import LazyConfig, instantiate
from detectron2.engine.defaults import create_ddp_model


def return_predictor_transformer(
    cfgfile,
    run_name,
    nc=1,
    output_dir="/home/shared/hsc/HSC/HSC_DR3/models/noclass/",
    roi_thresh=0.5,
):

    """
    This function returns a trained model and its config file.
    Used for models with lazy config files.  Also assumes a cascade roi head structure

    Parameters
    ----------
    cfgfile: str
        A path to a model config file, provided by the detectron2 repo
    run_name: str
        Prefix used for the name of the saved model
    nc: int
        Number of classes used in the model
    output_dir: str
        THe directory to save metric outputs
    roi_thresh: float
        Hyperparamter that functions as a detection sensitivity level

    """

    cfg = LazyConfig.load(cfgfile)

    metadata = MetadataCatalog.get(cfg.dataloader.test.dataset.names)  # to get labels from ids
    classes = metadata.thing_classes

    cfg.model.proposal_generator.anchor_generator.sizes = [[8], [16], [32], [64], [128]]
    cfg.dataloader.train.total_batch_size = 4
    cfg.model.roi_heads.num_classes = nc
    cfg.model.roi_heads.batch_size_per_image = 512

    for bp in cfg.model.roi_heads.box_predictors:
        bp.test_score_thresh = roi_thresh
    cfg.model.proposal_generator.pre_nms_topk = [6000, 6000]
    cfg.model.proposal_generator.post_nms_topk = [6000, 6000]
    cfg.model.proposal_generator.nms_thresh = 0.3

    for box_predictor in cfg.model.roi_heads.box_predictors:
        box_predictor.test_topk_per_image = 1000
        box_predictor.test_score_thresh = roi_thresh

    cfg_loader = get_cfg()
    cfg_loader.SOLVER.IMS_PER_BATCH = 4
    cfg_loader.DATASETS.TRAIN = "astro_train"  # Register Metadata
    cfg_loader.DATASETS.TEST = "astro_val"
    cfg_loader.DATALOADER.NUM_WORKERS = 1
    cfg_loader.SOLVER.BASE_LR = 0.001
    cfg_loader.OUTPUT_DIR = output_dir

    cfg_loader.TEST.DETECTIONS_PER_IMAGE = 1000

    cfg_loader.OUTPUT_DIR = output_dir

    cfg.train.init_checkpoint = os.path.join(cfg_loader.OUTPUT_DIR, run_name)

    # predictor = DefaultPredictor(cfg)
    predictor = toolkit.AstroPredictor(cfg_loader, lazy=True, cfglazy=cfg)

    return predictor, cfg


def get_matched_object_classes(dataset_dicts, predictor):

    IOUthresh = 0.5

    # going to assume we only have one test image
    for dn in dataset_names:
        for d in dataset_dicts[dn]:
            filenames = [d["filename_G"], d["filename_R"], d["filename_I"]]
            gt_boxes = np.array([a["bbox"] for a in d["annotations"]])
            # Convert to the mode model expects
            gt_boxes = BoxMode.convert(gt_boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            gt_boxes = structures.Boxes(torch.Tensor(gt_boxes))
            img = toolkit.read_image_hsc(filenames, normalize=args.norm, dtype=dtype, A=1000, do_norm=False)
            outputs = predictor(img)
            pred_boxes = outputs["instances"].pred_boxes
            pred_boxes = pred_boxes.to("cpu")

            IOUs = structures.pairwise_iou(pred_boxes, gt_boxes).numpy()

            # matched_gts holds the indices of the ground truth annotations that correspond to the matched detections
            # matched_dts holds the indices of the detections that corresponds to the ground truth annotations
            matched_gts = []
            matched_dts = []
            for i, dt in enumerate(IOUs):
                IOU = dt[dt.argmax()]
                if IOU >= IOUthresh:
                    matched_gts.append(dt.argmax())
                    matched_dts.append(i)
        true_classes = []
        pred_classes = []
        for gti, dti in zip(matched_gts, matched_dts):
            true_class = d["annotations"][int(gti)]["category_id"]
            pred_class = outputs["instances"].pred_classes.cpu().detach().numpy()[int(dti)]
            true_classes.append(true_class)
            pred_classes.append(pred_class)

    return true_classes, pred_classes


output_dir = args.output_dir
roi_thresh = args.roi_thresh
run_name = args.run_name
dt = args.datatype
if dt == 16:
    dtype = np.int16
elif dt == 8:
    dtype = np.uint8


cfglist = [
    "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml",
    "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml",
    "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
    "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml",
    "Misc/mask_rcnn_R_50_FPN_3x_dconv_c3-c5.yaml",
    "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml",
    "/home/g4merz/deblend/detectron2/projects/ViTDet/configs/COCO/cascade_mask_rcnn_swin_b_in21k_50ep.py",
    "/home/g4merz/deblend/detectron2/projects/ViTDet/configs/COCO/cascade_mask_rcnn_mvitv2_b_in21k_100ep.py",
]


bbs = ["R101C4", "R101dc5", "R101fpn", "X101fpn", "R50def", "R50cas", "Swin", "MViTv2"]


cfgdict = {}
for bb, cfg in zip(bbs, cfglist):
    cfgdict[bb] = cfg


bb = args.run_name.split("_")[0]
cfgfile = cfgdict[bb]
print(bb, cfgfile)
if bb in ["Swin", "MViTv2"]:
    predictor, cfg = return_predictor_transformer(
        cfgfile, run_name, output_dir=output_dir, nc=2, roi_thresh=roi_thresh
    )
else:
    predictor, cfg = return_predictor(cfgfile, run_name, output_dir=output_dir, nc=2, roi_thresh=roi_thresh)

t0 = time.time()


print("Matching objects")
true_classes, pred_classes = get_matched_object_classes(dataset_dicts, predictor)
classes = np.array([true_classes, pred_classes])

savename = f"{bb}_test_matched_classes.npy"
np.save(os.path.join(args.savedir, savename), classes)

print("Took ", time.time() - t0, " seconds")

print(classes)

t0 = time.time()
