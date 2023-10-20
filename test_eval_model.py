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
#from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
# import some common libraries
import numpy as np
import torch
from astropy.visualization import make_lupton_rgb
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (DatasetCatalog, MetadataCatalog,
                             build_detection_train_loader)
from detectron2.data import detection_utils as utils
from detectron2.engine import (DefaultPredictor, DefaultTrainer, HookBase,
                               SimpleTrainer)
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
from deepdisc.inference.match_objects import get_matched_object_classes
from deepdisc.inference.predictors import return_predictor_transformer

'''
This code will read in a trained model and output the classes for predicted objects matched to the ground truth 

'''


parser = argparse.ArgumentParser()


parser.add_argument('--run-name', default='Swin_test.pth', type=str)
parser.add_argument('--output-dir', default='.', type=str)
parser.add_argument('--savedir', default='.', type=str)
parser.add_argument('--norm', default='astrolupton', type=str)
parser.add_argument('--scheme', default=2, type=int)
parser.add_argument('--datatype', default=8, type=int)
parser.add_argument('--roi-thresh', default=0.1, type=float)
parser.add_argument('--testfile', default='/home/shared/hsc/HSC/HSC_DR3/data/single_test.json', type=str)
parser.add_argument('--nc', default=2, type=int)

args = parser.parse_args()





output_dir = args.output_dir
roi_thresh = args.roi_thresh
savedir = args.savedir
print(savedir)
Path(savedir).mkdir(parents=True, exist_ok=True)
run_name = args.run_name


testfile=args.testfile

classes=["star","galaxy"]

dataset_names = ['test'] 
datadir='/home/shared/hsc/HSC/HSC_DR3/data/'
t0 = time.time()
dataset_dicts = {}
for i, d in enumerate(dataset_names):
    dataset_dicts[d] = get_data_from_json(testfile)


    
print('Took ', time.time()-t0, 'seconds to load samples')

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:

def return_predictor(cfgfile, run_name, nc=1, output_dir='/home/shared/hsc/HSC/HSC_DR3/models/noclass/', roi_thresh=0.5):

    '''
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

    '''

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfgfile)) # Get model structure
    cfg.DATASETS.TRAIN = ("astro_train") # Register Metadata
    cfg.DATASETS.TEST = ("astro_test",) # Config calls this TEST, but it should be the val dataset
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.SOLVER.IMS_PER_BATCH = 4  # this is images per iteration. 1 epoch is len(images)/(ims_per_batch iterations*num_gpus)
    cfg.SOLVER.BASE_LR = 0.001    
    cfg.SOLVER.STEPS = []         # do not decay learning rate for retraining
    cfg.SOLVER.MAX_ITER = 100     # for DefaultTrainer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE =512   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = nc
    cfg.OUTPUT_DIR = output_dir
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    cfg.INPUT.MIN_SIZE_TRAIN = 1025
    cfg.INPUT.MAX_SIZE_TRAIN = 1050


    #Defaults
    #PRE_NMS_TOPK_TEST: 6000
    #POST_NMS_TOPK_TEST: 1000
    #PRE_NMS_TOPK_TRAIN: 12000
    #POST_NMS_TOPK_TRAIN: 2000

    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 6000  
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000 

    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, run_name)  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = roi_thresh   # set a custom testing threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
    cfg.MODEL.ROI_BOX_HEAD.GAMMA=1
    cfg.MODEL.ROI_BOX_HEAD.ALPHAS=None

    predictor = toolkit.AstroPredictor(cfg)

    return predictor, cfg


# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
from detectron2.config import LazyConfig, instantiate
from detectron2.engine.defaults import create_ddp_model

bb=args.run_name.split('_')[0]
cfgfile='/home/g4merz/deblend/detectron2/projects/ViTDet/configs/COCO/cascade_mask_rcnn_swin_b_in21k_50ep.py'

cfg = LazyConfig.load(cfgfile)

metadata = MetadataCatalog.get(cfg.dataloader.test.dataset.names) # to get labels from ids
classes = metadata.thing_classes

cfg.model.proposal_generator.anchor_generator.sizes = [[8], [16], [32], [64], [128]]
cfg.dataloader.train.total_batch_size=4
cfg.model.roi_heads.num_classes=args.nc
cfg.model.roi_heads.batch_size_per_image=512

for bp in cfg.model.roi_heads.box_predictors:
    bp.test_score_thresh=roi_thresh
cfg.model.proposal_generator.pre_nms_topk = [6000,6000]
cfg.model.proposal_generator.post_nms_topk = [6000,6000]
cfg.model.proposal_generator.nms_thresh = 0.3


for box_predictor in cfg.model.roi_heads.box_predictors:
    box_predictor.test_topk_per_image = 1000
    box_predictor.test_score_thresh = roi_thresh

cfg_loader = get_cfg()
cfg_loader.SOLVER.IMS_PER_BATCH = 4
cfg_loader.DATASETS.TRAIN = ("astro_train") # Register Metadata
cfg_loader.DATASETS.TEST = ("astro_val")
cfg_loader.DATALOADER.NUM_WORKERS = 1
cfg_loader.SOLVER.BASE_LR = 0.001
cfg_loader.OUTPUT_DIR = output_dir

cfg_loader.TEST.DETECTIONS_PER_IMAGE = 1000

cfg_loader.OUTPUT_DIR = output_dir

cfg.train.init_checkpoint = os.path.join(cfg_loader.OUTPUT_DIR, run_name)

#predictor = return_predictor_transformer(cfg,cfg_loader)

output_dir = args.output_dir
roi_thresh=args.roi_thresh
run_name=args.run_name
dt = args.datatype
if dt==16:
    dtype=np.int16
elif dt==8:
    dtype = np.uint8


    

if bb in ['Swin','MViTv2']:
    predictor= return_predictor_transformer(cfg,cfg_loader)
else:
    predictor,cfg = return_predictor(cfgfile, run_name,output_dir=output_dir,nc=2,roi_thresh=roi_thresh)

t0=time.time()



print('Matching objects')
true_classes, pred_classes = get_matched_object_classes(dataset_dicts['test'],predictor)
classes = np.array([true_classes,pred_classes])

savename =f'{bb}_test_matched_classes.npy'
np.save(os.path.join(args.savedir,savename),classes)

print('Took ', time.time()-t0, ' seconds')

print(classes)

t0 = time.time()
