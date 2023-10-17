# Training script for decam data
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

except:
    pass
warnings.filterwarnings("ignore", category=RuntimeWarning)


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
import gc

# from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.engine import (
    DefaultTrainer,
    SimpleTrainer,
    HookBase,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
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


import imgaug.augmenters.flip as flip
import imgaug.augmenters.blur as blur


# Prettify the plotting
from astrodet.astrodet import set_mpl_style

set_mpl_style()


from detectron2.structures import BoxMode
from astropy.io import fits
from astropy.visualization import make_lupton_rgb
import glob
from astrodet.detectron import _transform_to_aug

from PIL import Image, ImageEnhance


def get_data_from_json(file):
    # Opening JSON file
    with open(file, "r") as f:
        data = json.load(f)
    return data


# ### Augment Data


def gaussblur(image):
    aug = iaa.GaussianBlur(sigma=(0.0, np.random.random_sample() * 4 + 2))
    return aug.augment_image(image)


def addelementwise16(image):
    aug = iaa.AddElementwise((-3276, 3276))
    return aug.augment_image(image)


def addelementwise8(image):
    aug = iaa.AddElementwise((-25, 25))
    return aug.augment_image(image)


def addelementwise(image):
    aug = iaa.AddElementwise((-image.max() * 0.1, image.max() * 0.1))
    return aug.augment_image(image)


def centercrop(image):
    h, w = image.shape[:2]
    hc = (h - h // 2) // 2
    wc = (w - w // 2) // 2
    image = image[hc : hc + h // 2, wc : wc + w // 2]
    return image


class train_mapper_cls:
    def __init__(self, **read_image_args):
        self.ria = read_image_args

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        filenames = [dataset_dict["filename_G"], dataset_dict["filename_R"], dataset_dict["filename_I"]]

        # image = read_image(dataset_dict["file_name"], normalize=args.norm, ceil_percentile=99.99)
        image = toolkit.read_image_hsc(
            filenames,
            normalize=self.ria["normalize"],
            ceil_percentile=self.ria["ceil_percentile"],
            dtype=self.ria["dtype"],
            A=self.ria["A"],
            stretch=self.ria["stretch"],
            Q=self.ria["Q"],
            do_norm=self.ria["do_norm"],
        )
        """
        augs = T.AugmentationList([
            T.RandomRotation([-90, 90, 180], sample_style='choice'),
            T.RandomFlip(prob=0.5),
            T.RandomFlip(prob=0.5,horizontal=False,vertical=True),
            T.Resize((512,512))
            
        ])
        """

        augs = detectron_addons.KRandomAugmentationList(
            [
                # my custom augs
                T.RandomRotation([-90, 90, 180], sample_style="choice"),
                T.RandomFlip(prob=0.5),
                T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
                # detectron_addons.CustomAug(gaussblur,prob=1.0),
                # detectron_addons.CustomAug(addelementwise,prob=1.0)
                # CustomAug(white),
            ],
            k=-1,
            # cropaug=T.RandomCrop('relative',(0.5,0.5))
            cropaug=_transform_to_aug(
                T.CropTransform(
                    image.shape[1] // 4, image.shape[0] // 4, image.shape[1] // 2, image.shape[0] // 2
                )
            ),
            # cropaug=None
        )

        # Data Augmentation
        auginput = T.AugInput(image)
        # Transformations to model shapes
        transform = augs(auginput)
        image = torch.from_numpy(auginput.image.copy().transpose(2, 0, 1))

        annos = [
            utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
            for annotation in dataset_dict.pop("annotations")
        ]

        instances = utils.annotations_to_instances(annos, image.shape[1:])
        instances = utils.filter_empty_instances(instances)

        return {
            # create the format that the model expects
            "image": image,
            "image_shaped": auginput.image,
            "height": image.shape[1],
            "width": image.shape[2],
            "image_id": dataset_dict["image_id"],
            "instances": instances,
            # "instances": utils.annotations_to_instances(annos, image.shape[1:])
        }


class test_mapper_cls:
    def __init__(self, **read_image_args):
        self.ria = read_image_args

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        filenames = [dataset_dict["filename_G"], dataset_dict["filename_R"], dataset_dict["filename_I"]]

        image = toolkit.read_image_hsc(
            filenames,
            normalize=self.ria["normalize"],
            ceil_percentile=self.ria["ceil_percentile"],
            dtype=self.ria["dtype"],
            A=self.ria["A"],
            stretch=self.ria["stretch"],
            Q=self.ria["Q"],
            do_norm=self.ria["do_norm"],
        )

        augs = T.AugmentationList(
            [
                T.CropTransform(
                    image.shape[1] // 4, image.shape[0] // 4, image.shape[1] // 2, image.shape[0] // 2
                )
            ]
        )

        # Data Augmentation
        auginput = T.AugInput(image)
        # Transformations to model shapes
        transform = augs(auginput)
        image = torch.from_numpy(auginput.image.copy().transpose(2, 0, 1))
        annos = [
            utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
            for annotation in dataset_dict.pop("annotations")
        ]

        instances = utils.annotations_to_instances(annos, image.shape[1:])
        instances = utils.filter_empty_instances(instances)

        return {
            # create the format that the model expects
            "image": image,
            "image_shaped": auginput.image,
            "height": image.shape[1],
            "width": image.shape[2],
            "image_id": dataset_dict["image_id"],
            "instances": instances,
        }


def main(tl, train_head, args):
    """
    This function instantiates the model, dataloaders, and trainer
    and trains the model for 50 epochs following the schedule form Burke et al 2019

    Parameters
    ----------
    tl: int
        Size of training data set
    train_head: bool
        whether or not to train only the head layers
    args: keyword args
        arguments taken from the command line


    """
    # Hack if you get SSL certificate error
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

    output_dir = args.output_dir
    output_name = args.run_name
    cfgfile = args.cfgfile
    dirpath = args.data_dir  # Path to dataset
    scheme = args.scheme
    alphas = args.alphas
    datatype = args.dtype
    if datatype == 8:
        dtype = np.uint8
    elif datatype == 16:
        dtype = np.int16
    # ### Prepare For Training
    # Training logic:
    # To replicate 2019 methodology, need to
    # 1) run intially with backbone frozen (freeze_at=4) for 15 epochs
    # 2) unfreeze and run for [25,35,50] epochs with lr decaying by 0.1x each time
    dataset_names = ["train", "test", "val"]

    trainfile = dirpath + dataset_names[0] + "_sample_new.json"
    testfile = dirpath + dataset_names[1] + "_sample_new.json"
    valfile = dirpath + dataset_names[2] + "_sample_new.json"

    if scheme == 1 or scheme == 3:
        classes = ["star", "galaxy", "bad_fit", "unknown"]
    elif scheme == 2 or scheme == -1:
        # classes =["star", "galaxy","bad_fit"]
        classes = ["star", "galaxy"]
    numclasses = len(classes)

    DatasetCatalog.register("astro_train", lambda: get_data_from_json(trainfile))
    MetadataCatalog.get("astro_train").set(thing_classes=classes)
    astrotrain_metadata = MetadataCatalog.get("astro_train")  # astro_test dataset needs to exist

    # DatasetCatalog.register("astro_val", lambda: get_data_from_json(valfile))
    DatasetCatalog.register("astro_val", lambda: get_data_from_json(valfile))
    MetadataCatalog.get("astro_val").set(thing_classes=classes)
    astroval_metadata = MetadataCatalog.get("astro_val")  # astro_test dataset needs to exist

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfgfile))  # Get model structure
    cfg.DATASETS.TRAIN = "astro_train"  # Register Metadata
    cfg.DATASETS.TEST = "astro_val"  # Config calls this TEST, but it should be the val dataset
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = numclasses
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 512

    if args.norm == "astrolupton":
        cfg.MODEL.PIXEL_MEAN = [13.49794151, 9.11051305, 5.42995532]
    elif args.norm == "astroluptonhc":
        cfg.MODEL.PIXEL_MEAN = [37.92064421, 25.80468069, 14.03756261]
    elif args.norm == "zscore":
        cfg.MODEL.PIXEL_MEAN = [1.02938894, -11.65404583, -26.35697284]

    cfg.INPUT.MIN_SIZE_TRAIN = 500
    cfg.INPUT.MAX_SIZE_TRAIN = 525

    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]
    cfg.SOLVER.IMS_PER_BATCH = (
        8  # this is images per iteration. 1 epoch is len(images)/(ims_per_batch iterations)
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

    # iterations for 15,25,35,50 epochs
    epoch = int(tl / cfg.SOLVER.IMS_PER_BATCH)
    e1 = epoch * 15
    e2 = epoch * 10
    e3 = epoch * 20
    efinal = epoch * 35

    val_per = epoch

    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 6000
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.33

    if train_head:
        # Step 1)

        cfg.MODEL.BACKBONE.FREEZE_AT = 4  # Initial re-training of the head layers (i.e. freeze the backbone)
        if args.from_scratch:
            cfg.MODEL.BACKBONE.FREEZE_AT = 0
        cfg.SOLVER.BASE_LR = 0.001
        cfg.SOLVER.STEPS = []  # do not decay learning rate for retraining
        cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
        cfg.SOLVER.WARMUP_ITERS = 0
        cfg.SOLVER.MAX_ITER = e1  # for DefaultTrainer

        # init_coco_weights = True # Start training from MS COCO weights

        if not args.from_scratch:
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfgfile)  # Initialize from MS COCO
        # else:
        #    cfg.MODEL.WEIGHTS = os.path.join(output_dir, 'model_temp.pth')  # Initialize from a local weights

        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        model = modeler.build_model(cfg)
        optimizer = solver.build_optimizer(cfg, model)

        _train_mapper = train_mapper_cls(
            normalize=args.norm,
            ceil_percentile=args.cp,
            dtype=dtype,
            A=args.A,
            stretch=args.stretch,
            Q=args.Q,
            do_norm=args.do_norm,
        )
        _test_mapper = test_mapper_cls(
            normalize=args.norm,
            ceil_percentile=args.cp,
            dtype=dtype,
            A=args.A,
            stretch=args.stretch,
            Q=args.Q,
            do_norm=args.do_norm,
        )

        loader = data.build_detection_train_loader(cfg, mapper=_train_mapper)
        test_loader = data.build_detection_test_loader(cfg, cfg.DATASETS.TEST, mapper=_test_mapper)

        saveHook = detectron_addons.SaveHook()
        saveHook.set_output_name(output_name)
        schedulerHook = detectron_addons.CustomLRScheduler(optimizer=optimizer)
        lossHook = detectron_addons.LossEvalHook(val_per, model, test_loader)
        hookList = [lossHook, schedulerHook, saveHook]
        # hookList = [schedulerHook,saveHook]

        trainer = toolkit.NewAstroTrainer(model, loader, optimizer, cfg)
        trainer.register_hooks(hookList)
        trainer.set_period(int(epoch / 2))  # print loss every n iterations
        trainer.train(0, e1)
        # trainer.set_period(5)
        # trainer.train(0,20)
        if comm.is_main_process():
            np.save(output_dir + output_name + "_losses", trainer.lossList)
            np.save(output_dir + output_name + "_val_losses", trainer.vallossList)
        return

    else:
        # Step 2)

        cfg.MODEL.BACKBONE.FREEZE_AT = 0  # unfreeze all backbone layers
        cfg.SOLVER.BASE_LR = 0.0001
        cfg.SOLVER.STEPS = [e2, e3]  # decay learning rate
        cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
        cfg.SOLVER.WARMUP_ITERS = 0
        cfg.SOLVER.MAX_ITER = efinal  # for LR scheduling
        cfg.MODEL.WEIGHTS = os.path.join(output_dir, output_name + ".pth")  # Initialize from a local weights

        _train_mapper = train_mapper_cls(
            normalize=args.norm,
            ceil_percentile=args.cp,
            dtype=dtype,
            A=args.A,
            stretch=args.stretch,
            Q=args.Q,
            do_norm=args.do_norm,
        )
        _test_mapper = test_mapper_cls(
            normalize=args.norm,
            ceil_percentile=args.cp,
            dtype=dtype,
            A=args.A,
            stretch=args.stretch,
            Q=args.Q,
            do_norm=args.do_norm,
        )

        model = modeler.build_model(cfg)
        optimizer = solver.build_optimizer(cfg, model)
        loader = data.build_detection_train_loader(cfg, mapper=_train_mapper)
        test_loader = data.build_detection_test_loader(cfg, cfg.DATASETS.TEST, mapper=_test_mapper)

        saveHook = detectron_addons.SaveHook()
        saveHook.set_output_name(output_name)
        schedulerHook = detectron_addons.CustomLRScheduler(optimizer=optimizer)
        lossHook = detectron_addons.LossEvalHook(val_per, model, test_loader)
        hookList = [lossHook, schedulerHook, saveHook]
        # hookList = [schedulerHook,saveHook]

        trainer = toolkit.NewAstroTrainer(model, loader, optimizer, cfg)
        trainer.register_hooks(hookList)
        trainer.set_period(int(epoch / 2))  # print loss every n iterations
        trainer.train(0, efinal)
        # trainer.set_period(5) # print loss every n iterations
        # trainer.train(0,20)

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
        "--data-dir", type=str, default="/home/shared/hsc/HSC/HSC_DR3/data/", help="directory with data"
    )
    parser.add_argument("--output-dir", type=str, default="./", help="output directory to save model")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )
    parser.add_argument("--cp", type=float, default=99.99, help="ceiling percentile for saturation cutoff")
    parser.add_argument("--scheme", type=int, default=1, help="classification scheme")
    parser.add_argument("--stretch", type=float, default=0.5, help="lupton stretch")
    parser.add_argument("--Q", type=float, default=10, help="lupton Q")
    parser.add_argument("--A", type=float, default=1e3, help="scaling factor for int16")
    parser.add_argument("--do-norm", action="store_true", help="normalize input image (ignore if lupton)")
    parser.add_argument("--dtype", type=int, default=8, help="data type of array")
    parser.add_argument("--do-fl", action="store_true", help="use focal loss")
    parser.add_argument("--alphas", type=float, nargs="*", help="weights for focal loss")
    parser.add_argument(
        "--from-scratch", action="store_true", help="use this if you don't want to use pretrained weights"
    )

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
    """
    Runs the training of the head layers for 15 epochs
    then runs the training of the full model for an additional 35 epochs
    """

    args = custom_argument_parser().parse_args()
    print("Command Line Args:", args)

    dirpath = "/home/shared/hsc/HSC/HSC_DR3/data/"  # Path to dataset

    dataset_names = ["train", "test", "val"]

    traind = get_data_from_json(os.path.join(dirpath, dataset_names[0]) + "_sample.json")
    testd = get_data_from_json(os.path.join(dirpath, dataset_names[2]) + "_sample.json")

    # number of total samples
    print("# of train sample: ", len(traind))
    print("# of val sample: ", len(testd))

    tl = len(traind)
    del traind, testd
    gc.collect()

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
            tl,
            dataset_names,
            train_head,
            args,
        ),
    )

    print("Training full model")
    train_head = False
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(
            tl,
            dataset_names,
            train_head,
            args,
        ),
    )
    print(f"Took {time.time()-t0} seconds")
