try:
    # ignore ShapelyDeprecationWarning from fvcore
    import warnings
    from shapely.errors import ShapelyDeprecationWarning
    warnings.filterwarnings("ignore", category=sShapelyDeprecationWarning)
except:
    pass
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

import gc
import os
import time

import detectron2.utils.comm as comm

# import some common libraries
import numpy as np
import torch

# import some common detectron2 utilities
from detectron2.config import LazyConfig, get_cfg
from detectron2.engine import launch

from deepdisc.data_format.augment_image import hsc_test_augs, train_augs
from deepdisc.data_format.image_readers import DC2ImageReader, HSCImageReader
from deepdisc.data_format.register_data import register_data_set
from deepdisc.model.loaders import DictMapper, RedshiftDictMapper, return_test_loader, return_train_loader
from deepdisc.model.models import RedshiftPDFCasROIHeads, return_lazy_model
from deepdisc.training.trainers import (
    return_evallosshook,
    return_lazy_trainer,
    return_optimizer,
    return_savehook,
    return_schedulerhook,
)
from deepdisc.utils.parse_arguments import dtype_from_args, make_training_arg_parser


def main(train_head, args):
    # Hack if you get SSL certificate error
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    # Handle args
    output_dir = args.output_dir
    output_name = args.run_name
    dirpath = args.data_dir  # Path to dataset
    scheme = args.scheme
    alphas = args.alphas
    modname = args.modname
    datatype = args.dtype
    dtype = dtype_from_args(args.dtype)

    # Get file locations
    trainfile = dirpath + "train_scarlet_public.json"
    testfile = dirpath + "test_scarlet_public.json"
    if args.use_dc2:
        if modname == "swin":
            cfgfile = "./tests/deepdisc/test_data/configs/solo/solo_cascade_mask_rcnn_swin_b_in21k_50ep_DC2.py"
        elif modname == "mvitv2":
            cfgfile = "./tests/deepdisc/test_data/configs/solo/solo_cascade_mask_rcnn_mvitv2_b_in21k_100ep_DC2.py"
    else:
        if modname == "swin":
            cfgfile = "./tests/deepdisc/test_data/configs/solo/solo_cascade_mask_rcnn_swin_b_in21k_50ep.py"
        elif modname == "mvitv2":
            cfgfile = "./tests/deepdisc/test_data/configs/solo/solo_cascade_mask_rcnn_mvitv2_b_in21k_100ep.py"

    # Load the config
    cfg = LazyConfig.load(cfgfile)
    for key in cfg.get("MISC", dict()).keys():
        cfg[key] = cfg.MISC[key]

    # Register the data sets
    astrotrain_metadata = register_data_set(
        cfg.DATASETS.TRAIN, trainfile, thing_classes=cfg.metadata.classes
    )
    astroval_metadata = register_data_set(
        cfg.DATASETS.TEST, testfile, thing_classes=cfg.metadata.classes
    )
    
    # Set the output directory
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Iterations for 15, 25, 35, 50 epochs
    #! could this stuff be moved to a config too?
    epoch = int(args.tl / cfg.dataloader.train.total_batch_size)
    e1 = 20
    e2 = epoch * 10
    e3 = epoch * 20
    efinal = epoch * 35

    val_per = 5

    if train_head:
        cfg.train.init_checkpoint = None # or initwfile, the path to your model
    
        model = return_lazy_model(cfg)

        cfg.SOLVER.STEPS = []  # do not decay learning rate for retraining 
        cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
        cfg.SOLVER.WARMUP_ITERS = 0
        cfg.SOLVER.MAX_ITER = e1  # for DefaultTrainer
        
        cfg.optimizer.params.model = model
        cfg.optimizer.lr = 0.001
        optimizer = return_optimizer(cfg)

        if args.use_dc2:
            # key_mapper function should take a dataset_dict as input and output a key used by the image_reader function
            def dc2_key_mapper(dataset_dict):
                filename = dataset_dict["filename"]
                base = filename.split(".")[0].split("/")[-1]
                dirpath = "/home/g4merz/DC2/nersc_data/scarlet_data"
                fn = os.path.join(dirpath, base) + ".npy"
                return fn
            if args.use_redshift:
                IR = DC2ImageReader()
                mapper = RedshiftDictMapper(IR, dc2_key_mapper, train_augs).map_data
                loader = return_train_loader(cfg, mapper)
                test_mapper = RedshiftDictMapper(IR, dc2_key_mapper).map_data
                test_loader = return_test_loader(cfg, test_mapper)
            else:
                IR = DC2ImageReader(norm=args.norm)
                mapper = DictMapper(IR, dc2_key_mapper, train_augs).map_data
                loader = return_train_loader(cfg, mapper)
                test_mapper = DictMapper(IR, dc2_key_mapper).map_data
                test_loader = return_test_loader(cfg, test_mapper)
        else:
            def hsc_key_mapper(dataset_dict):
                filenames = [
                    dataset_dict["filename_G"],
                    dataset_dict["filename_R"],
                    dataset_dict["filename_I"],
                ]
                return filenames
            IR = HSCImageReader(norm=args.norm)
            mapper = DictMapper(IR, hsc_key_mapper, train_augs).map_data
            loader = return_train_loader(cfg, mapper)
            test_mapper = DictMapper(IR, hsc_key_mapper, hsc_test_augs).map_data
            test_loader = return_test_loader(cfg, test_mapper)

        saveHook = return_savehook(output_name)
        lossHook = return_evallosshook(val_per, model, test_loader)
        schedulerHook = return_schedulerhook(optimizer)
        hookList = [lossHook, schedulerHook, saveHook]

        trainer = return_lazy_trainer(model, loader, optimizer, cfg, hookList)
        trainer.set_period(5)
        trainer.train(0, 20)
        if comm.is_main_process():
            np.save(output_dir + output_name + "_losses", trainer.lossList)
            np.save(output_dir + output_name + "_val_losses", trainer.vallossList)
        return


if __name__ == "__main__":
    args = make_training_arg_parser().parse_args()
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
    