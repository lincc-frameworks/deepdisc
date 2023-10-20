# Training script for LazyConfig models
try:
    # ignore ShapelyDeprecationWarning from fvcore
    import warnings

    from shapely.errors import ShapelyDeprecationWarning

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

except:
    pass
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

import argparse
import copy
import gc
import json
import logging
import os
import random
import sys
import time
import weakref
from typing import Dict, List, Optional

import cv2
import detectron2.checkpoint as checkpointer
import detectron2.data as data
import detectron2.data.transforms as T
import detectron2.modeling as modeler
import detectron2.solver as solver
import detectron2.utils.comm as comm
import imgaug.augmenters.blur as blur
import imgaug.augmenters.flip as flip

# import some common libraries
import numpy as np
import torch

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_train_loader,
)
from detectron2.data import detection_utils as utils
from detectron2.engine import (
    DefaultPredictor,
    DefaultTrainer,
    HookBase,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
from detectron2.utils.visualizer import Visualizer

from astrodet import astrodet as toolkit
from astrodet import detectron as detectron_addons

# Prettify the plotting
from astrodet.astrodet import set_mpl_style

set_mpl_style()


import glob

from astropy.io import fits
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine.defaults import create_ddp_model
from detectron2.solver import build_lr_scheduler
from detectron2.structures import BoxMode

from astrodet.detectron import _transform_to_aug
from deepdisc.data_format.file_io import get_data_from_json
from deepdisc.model.loaders import return_test_loader, return_train_loader
from deepdisc.model.models import return_lazy_model
from deepdisc.model.trainers import (
    return_evallosshook,
    return_lazy_trainer,
    return_optimizer,
    return_savehook,
    return_schedulerhook,
)


def main(train_head, args):
    # Hack if you get SSL certificate error
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

    output_dir = args.output_dir
    output_name = args.run_name
    dirpath = args.data_dir  # Path to dataset
    scheme = args.scheme
    alphas = args.alphas
    modname = args.modname
    if modname == "swin":
        cfgfile = "/home/shared/hsc/detectron2/projects/ViTDet/configs/COCO/cascade_mask_rcnn_swin_b_in21k_50ep.py"
        initwfile = "/home/shared/hsc/detectron2/projects/ViTDet/model_final_246a82.pkl"
    elif modname == "mvitv2":
        cfgfile = "/home/shared/hsc/detectron2/projects/ViTDet/configs/COCO/cascade_mask_rcnn_mvitv2_b_in21k_100ep.py"
        initwfile = "/home/shared/hsc/detectron2/projects/ViTDet/model_final_8c3da3.pkl"

    elif modname == "vitdet":
        cfgfile = "/home/shared/hsc/detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py"
        # initwfile = '/home/g4merz/deblend/detectron2/projects/ViTDet/model_final_435fa9.pkl'
        initwfile = "/home/shared/hsc/detectron2/projects/ViTDet/model_final_61ccd1.pkl"

    datatype = args.dtype
    if datatype == 8:
        dtype = np.uint8
    elif datatype == 16:
        dtype = np.int16

    trainfile = dirpath + "single_test.json"
    testfile = dirpath + "single_test.json"

    classes = ["star", "galaxy"]
    numclasses = len(classes)

    DatasetCatalog.register("astro_train", lambda: get_data_from_json(trainfile))
    MetadataCatalog.get("astro_train").set(thing_classes=classes)
    astrotrain_metadata = MetadataCatalog.get(
        "astro_train"
    )  # astro_test dataset needs to exist

    DatasetCatalog.register("astro_val", lambda: get_data_from_json(testfile))
    MetadataCatalog.get("astro_val").set(thing_classes=classes)
    astroval_metadata = MetadataCatalog.get(
        "astro_val"
    )  # astro_test dataset needs to exist

    cfg = LazyConfig.load(cfgfile)

    # metadata = MetadataCatalog.get(cfg.dataloader.test.dataset.names) # to get labels from ids

    bs = 1
    cfg.model.proposal_generator.anchor_generator.sizes = [[8], [16], [32], [64], [128]]
    cfg.dataloader.train.total_batch_size = bs
    cfg.model.roi_heads.num_classes = numclasses
    cfg.model.roi_heads.batch_size_per_image = 512

    cfg_loader = get_cfg()
    cfg_loader.SOLVER.IMS_PER_BATCH = bs
    cfg_loader.DATASETS.TRAIN = "astro_train"  # Register Metadata
    cfg_loader.DATASETS.TEST = "astro_val"
    # cfg_loader.DATALOADER.NUM_WORKERS = 0
    cfg_loader.DATALOADER.PREFETCH_FACTOR = 2
    cfg_loader.SOLVER.BASE_LR = 0.001
    cfg_loader.OUTPUT_DIR = output_dir
    os.makedirs(cfg_loader.OUTPUT_DIR, exist_ok=True)

    cfg_loader.SOLVER.CLIP_GRADIENTS.ENABLED = True
    # Type of gradient clipping, currently 2 values are supported:
    # - "value": the absolute values of elements of each gradients are clipped
    # - "norm": the norm of the gradient for each parameter is clipped thus
    #   affecting all elements in the parameter
    cfg_loader.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    # Maximum absolute value used for clipping gradients
    # Floating point number p for L-p norm to be used with the "norm"
    # gradient clipping type; for L-inf, please specify .inf
    cfg_loader.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 5.0

    # iterations for 15,25,35,50 epochs
    epoch = int(args.tl / cfg.dataloader.train.total_batch_size)
    e1 = 20
    e2 = epoch * 10
    e3 = epoch * 20
    efinal = epoch * 35

    val_per = 5

    if train_head:
        # cfg.train.init_checkpoint = initwfile # replace with the path were you have your model
        cfg.train.init_checkpoint = None

        # Step 1)

        model = return_lazy_model(cfg)

        cfg.optimizer.params.model = model
        cfg.optimizer.lr = 0.001

        cfg_loader.SOLVER.STEPS = []  # do not decay learning rate for retraining
        cfg_loader.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
        cfg_loader.SOLVER.WARMUP_ITERS = 0
        cfg_loader.SOLVER.MAX_ITER = e1  # for DefaultTrainer

        # optimizer = instantiate(cfg.optimizer)

        optimizer = return_optimizer(cfg)

        loader = return_train_loader(
            cfg_loader,
            normalize=args.norm,
            ceil_percentile=args.cp,
            dtype=dtype,
            A=args.A,
            stretch=args.stretch,
            Q=args.Q,
            do_norm=args.do_norm,
        )
        test_loader = return_test_loader(
            cfg_loader,
            normalize=args.norm,
            ceil_percentile=args.cp,
            dtype=dtype,
            A=args.A,
            stretch=args.stretch,
            Q=args.Q,
            do_norm=args.do_norm,
        )

        saveHook = return_savehook(output_name)
        lossHook = return_evallosshook(val_per, model, test_loader)
        schedulerHook = return_schedulerhook(optimizer)
        hookList = [lossHook, schedulerHook, saveHook]

        trainer = return_lazy_trainer(
            model, loader, optimizer, cfg, cfg_loader, hookList
        )

        trainer.set_period(5)
        trainer.train(0, 20)
        if comm.is_main_process():
            np.save(output_dir + output_name + "_losses", trainer.lossList)
            np.save(output_dir + output_name + "_val_losses", trainer.vallossList)
        return


def custom_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.
    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.
    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:
Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml
Change some config options:
    $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001
Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="perform evaluation only"
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="number of gpus *per machine*"
    )
    parser.add_argument(
        "--num-machines", type=int, default=1, help="total number of machines"
    )
    parser.add_argument(
        "--run-name", type=str, default="Swin_test", help="output name for run"
    )
    parser.add_argument(
        "--cfgfile",
        type=str,
        default="COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml",
        help="path to model config file",
    )
    parser.add_argument(
        "--norm", type=str, default="astrolupton", help="contrast scaling"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/shared/hsc/HSC/HSC_DR3/data/",
        help="directory with data",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./", help="output directory to save model"
    )
    parser.add_argument(
        "--machine-rank",
        type=int,
        default=0,
        help="the rank of this machine (unique per machine)",
    )
    parser.add_argument(
        "--cp",
        type=float,
        default=99.99,
        help="ceiling percentile for saturation cutoff",
    )
    parser.add_argument("--scheme", type=int, default=1, help="classification scheme")
    parser.add_argument(
        "--alphas", type=float, nargs="*", help="weights for focal loss"
    )
    parser.add_argument("--modname", type=str, default="swin", help="")
    parser.add_argument("--stretch", type=float, default=0.5, help="lupton stretch")
    parser.add_argument("--Q", type=float, default=10, help="lupton Q")
    parser.add_argument("--A", type=float, default=1e3, help="scaling factor for int16")
    parser.add_argument(
        "--do-norm",
        action="store_true",
        help="normalize input image (ignore if lupton)",
    )
    parser.add_argument("--dtype", type=int, default=8, help="data type of array")
    parser.add_argument("--tl", type=int, default=1, help="total size of training set")

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = (
        2**15
        + 2**14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    )
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    args = custom_argument_parser().parse_args()
    print("Command Line Args:", args)

    print("Training head layers")
    train_head = True
    t0 = time.time()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(
            train_head,
            args,
        ),
    )

    torch.cuda.empty_cache()
    gc.collect()

    print(f"Took {time.time()-t0} seconds")
