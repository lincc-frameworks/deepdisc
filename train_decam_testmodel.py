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
from detectron2.engine import default_argument_parser, default_setup, hooks, launch, SimpleTrainer
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

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine.defaults import create_ddp_model
from detectron2.data import MetadataCatalog
import gc

import warnings

# Get a user warning about some upsampling parameter, just ignoring
warnings.filterwarnings("ignore", category=UserWarning)


def get_data_from_json(file):
    # Opening JSON file
    with open(file, "r") as f:
        data = json.load(f)
    return data


from detectron2.solver import build_lr_scheduler


class LazyAstroTrainer(SimpleTrainer):
    def __init__(self, model, data_loader, optimizer, cfg, cfg_old):
        super().__init__(model, data_loader, optimizer)
        # super().__init__(model, data_loader, optimizer)

        # Borrowed from DefaultTrainer constructor
        # see https://detectron2.readthedocs.io/en/latest/_modules/detectron2/engine/defaults.html#DefaultTrainer
        self.checkpointer = checkpointer.DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg_old.OUTPUT_DIR,
        )
        # load weights
        self.checkpointer.load(cfg.train.init_checkpoint)

        # record loss over iteration
        self.lossList = []
        self.vallossList = []

        self.period = 20
        self.iterCount = 0

        self.scheduler = self.build_lr_scheduler(cfg_old, optimizer)
        # self.scheduler = instantiate(cfg.lr_multiplier)
        self.valloss = 0

    # Note: print out loss over p iterations
    def set_period(self, p):
        self.period = p

    # Copied directly from SimpleTrainer, add in custom manipulation with the loss
    # see https://detectron2.readthedocs.io/en/latest/_modules/detectron2/engine/train_loop.html#SimpleTrainer
    def run_step(self):
        self.iterCount = self.iterCount + 1
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data_time = time.perf_counter() - start
        data = next(self._data_loader_iter)
        # Note: in training mode, model() returns loss
        loss_dict = self.model(data)
        # print('Loss dict',loss_dict)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())
        self.optimizer.zero_grad()
        losses.backward()

        # self._write_metrics(loss_dict,data_time)

        self.optimizer.step()

        self.lossList.append(losses.cpu().detach().numpy())
        if self.iterCount % self.period == 0 and comm.is_main_process():
            print(
                "Iteration: ",
                self.iterCount,
                " time: ",
                data_time,
                " loss: ",
                losses.cpu().detach().numpy(),
                "val loss: ",
                self.valloss,
                "lr: ",
                self.scheduler.get_lr(),
            )
            # print("Iteration: ", self.iterCount, " time: ", data_time," loss: ",losses.cpu().detach().numpy(), "val loss: ",self.valloss, "lr: tbd", )

        del data
        gc.collect()
        torch.cuda.empty_cache()

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    def add_val_loss(self, val_loss):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """

        self.vallossList.append(val_loss)


def main(dataset_names, train_head, args):
    # Hack if you get SSL certificate error
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

    output_dir = args.output_dir
    output_name = args.run_name
    cfgfile = args.cfgfile
    dirpath = args.data_dir  # Path to dataset
    datatype = args.dtype
    if datatype == 8:
        dtype = np.uint8
    elif datatype == 16:
        dtype = np.int16

    modname = args.modname
    if modname == "swin":
        cfgfile = "/home/g4merz/deblend/detectron2/projects/ViTDet/configs/COCO/cascade_mask_rcnn_swin_b_in21k_50ep.py"
        initwfile = "/home/g4merz/deblend/detectron2/projects/ViTDet/model_final_246a82.pkl"
    elif modname == "mvitv2":
        cfgfile = "/home/g4merz/deblend/detectron2/projects/ViTDet/configs/COCO/cascade_mask_rcnn_mvitv2_b_in21k_100ep.py"
        initwfile = "/home/g4merz/deblend/detectron2/projects/ViTDet/model_final_8c3da3.pkl"

    elif modname == "vitdet":
        cfgfile = "/home/g4merz/deblend/detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py"
        # initwfile = '/home/g4merz/deblend/detectron2/projects/ViTDet/model_final_435fa9.pkl'
        initwfile = "/home/g4merz/deblend/detectron2/projects/ViTDet/model_final_61ccd1.pkl"

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

    cfg = LazyConfig.load(cfgfile)

    cfg.train.init_checkpoint = "/home/g4merz/deblend/detectron2/projects/ViTDet/model_final_61ccd1.pkl"  # replace with the path were you have your model
    metadata = MetadataCatalog.get(cfg.dataloader.test.dataset.names)  # to get labels from ids
    classes = metadata.thing_classes

    cfg.model.proposal_generator.anchor_generator.sizes = [[8], [16], [32], [64], [128]]
    cfg.dataloader.train.total_batch_size = 4
    cfg.model.roi_heads.num_classes = 2
    cfg.model.roi_heads.batch_size_per_image = 250

    for box_predictor in cfg.model.roi_heads.box_predictors:
        box_predictor.focal_loss_alphas = None

    cfg_loader = get_cfg()
    cfg_loader.SOLVER.IMS_PER_BATCH = 4
    cfg_loader.DATASETS.TRAIN = "astro_train"  # Register Metadata
    cfg_loader.DATASETS.TEST = "astro_val"
    cfg_loader.DATALOADER.NUM_WORKERS = 1
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

    epoch = int(tl / cfg.dataloader.train.total_batch_size)
    e1 = epoch * 15
    e2 = epoch * 10
    e3 = epoch * 20
    efinal = epoch * 35
    val_per = epoch

    if train_head:
        cfg.train.init_checkpoint = initwfile  # replace with the path were you have your model
        # cfg.train.init_checkpoint =
        # Step 1)
        model = instantiate(cfg.model)

        for param in model.parameters():
            param.requires_grad = False
        # Phase 1: Unfreeze only the roi_heads
        for param in model.roi_heads.parameters():
            param.requires_grad = True
        # Phase 2: Unfreeze region proposal generator with reduced lr
        for param in model.proposal_generator.parameters():
            param.requires_grad = True

        model.to(cfg.train.device)
        model = create_ddp_model(model, **cfg.train.ddp)

        cfg.optimizer.params.model = model
        cfg.optimizer.lr = 0.001
        optimizer = instantiate(cfg.optimizer)

        cfg_loader.SOLVER.STEPS = []  # do not decay learning rate for retraining
        cfg_loader.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
        cfg_loader.SOLVER.WARMUP_ITERS = 0
        cfg_loader.SOLVER.MAX_ITER = e1  # for DefaultTrainer

        _train_mapper = toolkit.train_mapper_cls(
            normalize=args.norm,
            ceil_percentile=args.cp,
            dtype=dtype,
            A=args.A,
            stretch=args.stretch,
            Q=args.Q,
            do_norm=args.do_norm,
            sim=True,
        )
        _test_mapper = toolkit.test_mapper_cls(
            normalize=args.norm,
            ceil_percentile=args.cp,
            dtype=dtype,
            A=args.A,
            stretch=args.stretch,
            Q=args.Q,
            do_norm=args.do_norm,
            sim=True,
        )

        loader = data.build_detection_train_loader(cfg_loader, mapper=_train_mapper)
        test_loader = data.build_detection_test_loader(
            cfg_loader, cfg_loader.DATASETS.TEST, mapper=_test_mapper
        )

        saveHook = detectron_addons.SaveHook()
        saveHook.set_output_name(output_name)
        schedulerHook = detectron_addons.CustomLRScheduler(optimizer=optimizer)
        # schedulerHook = hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier))
        lossHook = detectron_addons.LossEvalHook(val_per, model, test_loader)
        hookList = [lossHook, schedulerHook, saveHook]

        trainer = LazyAstroTrainer(model, loader, optimizer, cfg, cfg_loader)
        trainer.register_hooks(hookList)
        # trainer.set_period(int(epoch/2)) # print loss every n iterations
        trainer.train(0, e1)
        # trainer.set_period(10) # print loss every n iterations
        # trainer.train(0,100)
        if comm.is_main_process():
            np.save(output_dir + output_name + "_losses", trainer.lossList)
            np.save(output_dir + output_name + "_val_losses", trainer.vallossList)

        return
        # return trainer.train(0, e1)

    else:
        cfg.train.init_checkpoint = os.path.join(
            output_dir, output_name + ".pth"
        )  # replace with the path were you have your model

        # Step 2)

        # cfg.MODEL.BACKBONE.FREEZE_AT = 0  # unfreeze all backbone layers
        cfg_loader.SOLVER.BASE_LR = 0.0001
        cfg_loader.SOLVER.STEPS = [e2, e3]  # decay learning rate
        # cfg.SOLVER.STEPS = [50,100]        # decay learning rate
        cfg_loader.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
        cfg_loader.SOLVER.WARMUP_ITERS = 0
        cfg_loader.SOLVER.MAX_ITER = efinal  # for LR scheduling

        _train_mapper = toolkit.train_mapper_cls(
            normalize=args.norm,
            ceil_percentile=args.cp,
            dtype=dtype,
            A=args.A,
            stretch=args.stretch,
            Q=args.Q,
            do_norm=args.do_norm,
            sim=True,
        )
        _test_mapper = toolkit.test_mapper_cls(
            normalize=args.norm,
            ceil_percentile=args.cp,
            dtype=dtype,
            A=args.A,
            stretch=args.stretch,
            Q=args.Q,
            do_norm=args.do_norm,
            sim=True,
        )

        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model, **cfg.train.ddp)
        cfg.optimizer.params.model = model
        optimizer = instantiate(cfg.optimizer)

        loader = data.build_detection_train_loader(cfg_loader, mapper=_train_mapper)
        test_loader = data.build_detection_test_loader(
            cfg_loader, cfg_loader.DATASETS.TEST, mapper=_test_mapper
        )

        saveHook = detectron_addons.SaveHook()
        saveHook.set_output_name(output_name)
        schedulerHook = detectron_addons.CustomLRScheduler(optimizer=optimizer)
        lossHook = detectron_addons.LossEvalHook(val_per, model, test_loader)
        hookList = [lossHook, schedulerHook, saveHook]

        trainer = LazyAstroTrainer(model, loader, optimizer, cfg, cfg_loader)
        trainer.register_hooks(hookList)
        trainer.set_period(epoch)  # print loss every n iterations
        trainer.train(0, efinal)
        # trainer.set_period(10) # print loss every n iterations
        # trainer.train(0,100)
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
    parser.add_argument("--scheme", type=int, default=1, help="classification scheme")
    parser.add_argument("--alphas", type=float, nargs="*", help="weights for focal loss")
    parser.add_argument("--modname", type=str, default="./", help="")
    parser.add_argument("--stretch", type=float, default=0.5, help="lupton stretch")
    parser.add_argument("--Q", type=float, default=10, help="lupton Q")
    parser.add_argument("--A", type=float, default=1e3, help="scaling factor for int16")
    parser.add_argument("--do-norm", action="store_true", help="normalize input image (ignore if lupton)")
    parser.add_argument("--dtype", type=int, default=8, help="data type of array")
    parser.add_argument("--do-fl", action="store_true", help="use focal loss")
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
