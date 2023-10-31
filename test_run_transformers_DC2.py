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
from detectron2.engine import (
    launch,
)

from deepdisc.data_format.image_readers import DC2ImageReader
from deepdisc.data_format.register_data import register_data_set
from deepdisc.model.loaders import (
    redshift_test_mapper_cls,
    redshift_train_mapper_cls,
    return_test_loader,
    return_train_loader,
)
from deepdisc.model.models import return_lazy_model
from deepdisc.training.trainers import (
    return_evallosshook,
    return_lazy_trainer,
    return_optimizer,
    return_savehook,
    return_schedulerhook,
)
from deepdisc.utils.parse_arguments import make_training_arg_parser


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
        cfgfile = "./tests/deepdisc/test_data/configs/COCO/cascade_mask_rcnn_swin_b_in21k_50ep.py"
        # initwfile = "/home/shared/hsc/detectron2/projects/ViTDet/model_final_246a82.pkl"
    elif modname == "mvitv2":
        cfgfile = "/home/shared/hsc/detectron2/projects/ViTDet/configs/COCO/cascade_mask_rcnn_mvitv2_b_in21k_100ep.py"
        # initwfile = "/home/shared/hsc/detectron2/projects/ViTDet/model_final_8c3da3.pkl"

    elif modname == "vitdet":
        cfgfile = "/home/shared/hsc/detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py"
        # initwfile = '/home/g4merz/deblend/detectron2/projects/ViTDet/model_final_435fa9.pkl'
        # initwfile = "/home/shared/hsc/detectron2/projects/ViTDet/model_final_61ccd1.pkl"

    datatype = args.dtype
    if datatype == 8:
        dtype = np.uint8
    elif datatype == 16:
        dtype = np.int16

    trainfile = dirpath + "single_test.json"
    testfile = dirpath + "single_test.json"

    classes = ["object"]
    numclasses = len(classes)

    # Register the data sets and get the metadata.
    astrotrain_metadata = register_data_set("astro_train", trainfile, thing_classes=classes)
    astroval_metadata = register_data_set("astro_val", testfile, thing_classes=classes)

    cfg = LazyConfig.load(cfgfile)

    # metadata = MetadataCatalog.get(cfg.dataloader.test.dataset.names) # to get labels from ids

    bs = 1
    cfg.model.proposal_generator.anchor_generator.sizes = [[8], [16], [32], [64], [128]]
    cfg.dataloader.train.total_batch_size = bs
    cfg.model.roi_heads.num_classes = numclasses
    cfg.model.roi_heads.batch_size_per_image = 512
    cfg.model.backbone.bottom_up.in_chans = 6
    cfg.model.pixel_mean = [0.05381286, 0.04986344, 0.07526361, 0.10420945, 0.14229655, 0.21245764]
    cfg.model.pixel_std = [2.9318833, 1.8443471, 2.581817, 3.5950038, 4.5809164, 7.302009]
    # cfg.model.roi_heads.num_components=5
    # cfg.model.roi_heads._target_ = RedshiftPDFCasROIHeads

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

        IR = DC2ImageReader(norm=args.norm)
        mapper = redshift_train_mapper_cls(IR)
        loader = return_train_loader(cfg_loader, mapper)
        test_mapper = redshift_test_mapper_cls(IR)
        test_loader = return_test_loader(cfg_loader, test_mapper)

        saveHook = return_savehook(output_name)
        lossHook = return_evallosshook(val_per, model, test_loader)
        schedulerHook = return_schedulerhook(optimizer)
        hookList = [lossHook, schedulerHook, saveHook]

        trainer = return_lazy_trainer(model, loader, optimizer, cfg, cfg_loader, hookList)

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
