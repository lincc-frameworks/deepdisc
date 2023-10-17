# Training script for decam data

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import argparse
import logging
import sys

# from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import build_detection_train_loader
from detectron2.engine import default_argument_parser, default_setup, hooks, launch
from typing import Dict, List, Optional
import detectron2.solver as solver
import detectron2.modeling as modeler
import detectron2.data as data
import detectron2.data.transforms as T
import detectron2.checkpoint as checkpointer
from detectron2.data import detection_utils as utils
import detectron2.utils.comm as comm

import weakref
import copy
import torch
import time

import imgaug.augmenters as iaa

from astrodet import astrodet as toolkit
from astrodet.astrodet import read_image
from astrodet import detectron as detectron_addons

# Custom Aug classes have been added to detectron source files
from astrodet.detectron import CustomAug

import imgaug.augmenters.flip as flip
import imgaug.augmenters.blur as blur


# Prettify the plotting
from astrodet.astrodet import set_mpl_style

set_mpl_style()


from detectron2.structures import BoxMode
from astropy.io import fits
import glob


def get_data_from_json(file):
    # Opening JSON file
    with open(file, "r") as f:
        data = json.load(f)
    return data


def main(dataset_names, train_head, args):
    # Hack if you get SSL certificate error
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

    output_dir = args.output_dir
    output_name = args.run_name
    cfgfile = args.cfgfile
    dirpath = args.data_dir  # Path to dataset

    # ### Prepare For Training
    # Training logic:
    # To replicate 2019 methodology, need to
    # 1) run intially with backbone frozen (freeze_at=4) for 15 epochs
    # 2) unfreeze and run for [25,35,50] epochs with lr decaying by 0.1x each time

    for i, d in enumerate(dataset_names):
        filenames_dir = os.path.join(dirpath, d)
        # DatasetCatalog.register("astro_" + d, lambda: get_astro_dicts(filenames_dir))
        # MetadataCatalog.get("astro_" + d).set(thing_classes=["star", "galaxy"], things_colors = ['blue', 'gray'])
        DatasetCatalog.register("astro_" + d, lambda: get_data_from_json(filenames_dir + ".json"))
        MetadataCatalog.get("astro_" + d).set(
            thing_classes=["star", "galaxy"], things_colors=["blue", "gray"]
        )

    astro_metadata = MetadataCatalog.get("astro_train")

    # tl=len(dataset_dicts['train'])
    tl = 1000
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfgfile))  # Get model structure
    cfg.DATASETS.TRAIN = "astro_train"  # Register Metadata
    cfg.DATASETS.TEST = "astro_val"  # Config calls this TEST, but it should be the val dataset
    cfg.TEST.EVAL_PERIOD = 40
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        250  # faster, and good enough for this toy dataset (default: 512)
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 512

    # cfg.MODEL.PIXEL_MEAN = [-200,-200,-200]

    cfg.INPUT.MIN_SIZE_TRAIN = 512
    cfg.INPUT.MAX_SIZE_TRAIN = 512

    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]
    cfg.SOLVER.IMS_PER_BATCH = (
        4  # this is images per iteration. 1 epoch is len(images)/(ims_per_batch iterations*num_gpus)
    )

    cfg.OUTPUT_DIR = output_dir
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000

    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    # Type of gradient clipping, currently 2 values are supported:
    # - "value": the absolute values of elements of each gradients are clipped
    # - "norm": the norm of the gradient for each parameter is clipped thus
    #   affecting all elements in the parameter
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    # Maximum absolute value used for clipping gradients
    # Floating point number p for L-p norm to be used with the "norm"
    # gradient clipping type; for L-inf, please specify .inf
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 5.0

    # itertions for 15,25,35,50 epochs
    epoch = int(tl / cfg.SOLVER.IMS_PER_BATCH)
    e1 = epoch * 15
    e2 = epoch * 10
    e3 = epoch * 20
    efinal = epoch * 35

    val_per = epoch

    # cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE= True

    if train_head:
        # Step 1)
        cfg.MODEL.BACKBONE.FREEZE_AT = 4  # Initial re-training of the head layers (i.e. freeze the backbone)
        cfg.SOLVER.BASE_LR = 0.001
        cfg.SOLVER.STEPS = []  # do not decay learning rate for retraining head layers
        cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
        cfg.SOLVER.WARMUP_ITERS = 0
        cfg.SOLVER.MAX_ITER = e1  # for DefaultTrainer

        init_coco_weights = True  # Start training from MS COCO weights

        if init_coco_weights:
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfgfile)  # Initialize from MS COCO
        else:
            cfg.MODEL.WEIGHTS = os.path.join(output_dir, "model_temp.pth")  # Initialize from a local weights

        print(cfg)

        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        model = modeler.build_model(cfg)
        optimizer = solver.build_optimizer(cfg, model)

        _train_mapper = toolkit.train_mapper_cls(normalize=args.norm, ceil_percentile=99.99)
        _test_mapper = toolkit.test_mapper_cls(normalize=args.norm, ceil_percentile=99.99)

        loader = data.build_detection_train_loader(cfg, mapper=_train_mapper)
        test_loader = data.build_detection_test_loader(cfg, cfg.DATASETS.TEST, mapper=_test_mapper)

        saveHook = detectron_addons.SaveHook()
        saveHook.set_output_name(output_name)
        schedulerHook = detectron_addons.CustomLRScheduler(optimizer=optimizer)
        lossHook = detectron_addons.LossEvalHook(val_per, model, test_loader)
        hookList = [lossHook, schedulerHook, saveHook]

        trainer = toolkit.NewAstroTrainer(model, loader, optimizer, cfg)
        trainer.register_hooks(hookList)
        trainer.set_period(epoch)  # print loss every n iterations
        trainer.train(0, e1)
        if comm.is_main_process():
            np.save(output_dir + output_name + "_losses", trainer.lossList)
            np.save(output_dir + output_name + "_val_losses", trainer.vallossList)

        return
        # return trainer.train(0, e1)

    else:
        # Step 2)

        cfg.MODEL.BACKBONE.FREEZE_AT = 0  # unfreeze all backbone layers
        cfg.SOLVER.BASE_LR = 0.0001
        cfg.SOLVER.STEPS = [e2, e3]  # decay learning rate
        # cfg.SOLVER.STEPS = [50,100]        # decay learning rate
        cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
        cfg.SOLVER.WARMUP_ITERS = 0
        cfg.SOLVER.MAX_ITER = efinal  # for LR scheduling
        cfg.MODEL.WEIGHTS = os.path.join(output_dir, output_name + ".pth")  # Initialize from a local weights

        _train_mapper = toolkit.train_mapper_cls(normalize=args.norm, ceil_percentile=args.cp)
        _test_mapper = toolkit.test_mapper_cls(normalize=args.norm, ceil_percentile=args.cp)

        model = modeler.build_model(cfg)
        optimizer = solver.build_optimizer(cfg, model)
        loader = data.build_detection_train_loader(cfg, mapper=_train_mapper)
        test_loader = data.build_detection_test_loader(cfg, cfg.DATASETS.TEST, mapper=_test_mapper)

        saveHook = detectron_addons.SaveHook()
        saveHook.set_output_name(output_name)
        schedulerHook = detectron_addons.CustomLRScheduler(optimizer=optimizer)
        lossHook = detectron_addons.LossEvalHook(val_per, model, test_loader)
        hookList = [lossHook, schedulerHook, saveHook]

        trainer = toolkit.NewAstroTrainer(model, loader, optimizer, cfg)
        trainer.register_hooks(hookList)
        trainer.set_period(epoch)  # print loss every n iterations
        trainer.train(0, efinal)
        if comm.is_main_process():
            losses = np.load(output_dir + output_name + "_losses.npy")
            losses = np.concatenate((losses, trainer.lossList))
            np.save(output_dir + output_name + "_losses", losses)

            vallosses = np.load(output_dir + output_name + "_val_losses.npy")
            vallosses = np.concatenate((vallosses, trainer.vallossList))
            np.save(output_dir + output_name + "_val_losses", vallosses)
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
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument("--run-name", type=str, default="baseline", help="output name for run")
    parser.add_argument(
        "--cfgfile",
        type=str,
        default="COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml",
        help="path to model config file",
    )
    parser.add_argument("--norm", type=str, default="lupton", help="contrast scaling")
    parser.add_argument(
        "--data-dir", type=str, default="/home/shared/hsc/decam/decam_data/", help="directory with data"
    )
    parser.add_argument("--output-dir", type=str, default="./", help="output directory to save model")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )
    parser.add_argument("--cp", type=float, default=99.99, help="ceiling percentile for saturation cutoff")

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
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

    dataset_names = ["train", "test", "val"]

    print("Training head layers")
    train_head = True
    t0 = time.time()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(dataset_names, train_head, args),
    )

    print("Training full model")
    train_head = False
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(dataset_names, train_head, args),
    )
    print(f"Took {time.time()-t0} seconds")
