# Training script for decam data
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

import copy
import gc

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:150"
import os
import time

import detectron2.checkpoint as checkpointer
import detectron2.data as data
import detectron2.data.transforms as T
import detectron2.utils.comm as comm
import imgaug.augmenters as iaa

# from google.colab.patches import cv2_imshow

# import some common libraries
import numpy as np
import torch

# import some common detectron2 utilities
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.engine import (
    SimpleTrainer,
    launch,
)

from astrodet import astrodet as toolkit
from astrodet import detectron as detectron_addons

# Prettify the plotting
from astrodet.astrodet import set_mpl_style

set_mpl_style()



from detectron2.config import LazyConfig, instantiate
from detectron2.engine.defaults import create_ddp_model
from detectron2.solver import build_lr_scheduler


from deepdisc.data_format.file_io import get_data_from_json
from deepdisc.data_format.register_data import register_data_set
from deepdisc.utils.parse_arguments import dtype_from_args, make_training_arg_parser

from deepdisc.astrodet.astrodet import LazyAstroTrainer

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


def main(tl, dataset_names, train_head, args):
    # Hack if you get SSL certificate error
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

    output_dir = args.output_dir
    output_name = args.run_name
    # cfgfile=args.cfgfile
    dirpath = args.data_dir  # Path to dataset
    scheme = args.scheme
    alphas = args.alphas
    modname = args.modname
    if modname == "swin":
        cfgfile = (
            "/home/g4merz/detectron2/projects/ViTDet/configs/COCO/cascade_mask_rcnn_swin_b_in21k_50ep.py"
        )
        initwfile = "/home/g4merz/detectron2/projects/ViTDet/model_final_246a82.pkl"
    elif modname == "mvitv2":
        cfgfile = (
            "/home/g4merz/detectron2/projects/ViTDet/configs/COCO/cascade_mask_rcnn_mvitv2_b_in21k_100ep.py"
        )
        initwfile = "/home/g4merz/detectron2/projects/ViTDet/model_final_8c3da3.pkl"

    elif modname == "vitdet":
        cfgfile = "/home/g4merz/detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py"
        # initwfile = '/home/g4merz/deblend/detectron2/projects/ViTDet/model_final_435fa9.pkl'
        initwfile = "/home/g4merz/detectron2/projects/ViTDet/model_final_61ccd1.pkl"

    dtype = dtype_from_args(args.dtype)

    # ### Prepare For Training
    # Training logic:
    # To replicate 2019 methodology, need to
    # 1) run intially with backbone frozen (freeze_at=4) for 15 epochs
    # 2) unfreeze and run for [25,35,50] epochs with lr decaying by 0.1x each time

    trainfile = dirpath + dataset_names[0] + "_sample_new.json"
    testfile = dirpath + dataset_names[1] + "_sample_new.json"
    valfile = dirpath + dataset_names[2] + "_sample_new.json"

    if scheme == 1 or scheme == 3:
        classes = ["star", "galaxy", "bad_fit", "unknown"]
    elif scheme == 2 or scheme == -1:
        # classes =["star", "galaxy","bad_fit"]
        classes = ["star", "galaxy"]
    numclasses = len(classes)

    # Register the data sets and get the metadata.
    astrotrain_metadata = register_data_set("astro_train", trainfile, thing_classes=classes)
    astroval_metadata = register_data_set("astro_val", testfile, thing_classes=classes)

    # cfg = LazyConfig.load("/home/g4merz/deblend/detectron2/projects/ViTDet/configs/COCO/cascade_mask_rcnn_swin_b_in21k_50ep.py")
    cfg = LazyConfig.load(cfgfile)

    metadata = MetadataCatalog.get(cfg.dataloader.test.dataset.names)  # to get labels from ids
    classes = metadata.thing_classes
    bs = 2

    cfg.model.proposal_generator.anchor_generator.sizes = [[8], [16], [32], [64], [128]]
    cfg.dataloader.train.total_batch_size = bs
    cfg.model.roi_heads.num_classes = 2
    cfg.model.roi_heads.batch_size_per_image = 512

    # cfg.model.rpn.post_nms_topk_train=6000
    # cfg.model.roi_heads.positive_fraction = 0.33

    # cfg.model.roi_heads.box_predictor.focal_loss_alphas=None
    # print(cfg.model.roi_heads.box_head)
    # cfg.box_predictor.focal_loss_alphas=None
    # for box_predictor in cfg.model.roi_heads.box_predictors:
    #    box_predictor.focal_loss_alphas=None

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
    epoch = int(tl / cfg.dataloader.train.total_batch_size)
    e1 = epoch * 15
    e2 = epoch * 10
    e3 = epoch * 20
    efinal = epoch * 35

    val_per = epoch
    # val_per = 100

    if train_head:
        # cfg.train.init_checkpoint = initwfile # replace with the path were you have your model
        cfg.train.init_checkpoint = None

        # Step 1)
        model = instantiate(cfg.model)
        """
        for param in model.parameters():
            param.requires_grad = False
        # Phase 1: Unfreeze only the roi_heads
        for param in model.roi_heads.parameters():
            param.requires_grad = True
        # Phase 2: Unfreeze region proposal generator with reduced lr
        for param in model.proposal_generator.parameters():
            param.requires_grad = True
        """

        # print(cfg.train.ddp)
        model.to(cfg.train.device)
        model = create_ddp_model(model, **cfg.train.ddp)

        cfg.optimizer.params.model = model
        cfg.optimizer.lr = 0.001
        optimizer = instantiate(cfg.optimizer)

        cfg_loader.SOLVER.STEPS = []  # do not decay learning rate for retraining
        cfg_loader.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
        cfg_loader.SOLVER.WARMUP_ITERS = 0
        cfg_loader.SOLVER.MAX_ITER = e1  # for DefaultTrainer

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

        loader = data.build_detection_train_loader(cfg_loader, mapper=_train_mapper)
        test_loader = data.build_detection_test_loader(
            cfg_loader, cfg_loader.DATASETS.TEST, mapper=_test_mapper
        )

        saveHook = detectron_addons.SaveHook()
        saveHook.set_output_name(output_name)
        schedulerHook = detectron_addons.CustomLRScheduler(optimizer=optimizer)
        lossHook = detectron_addons.LossEvalHook(val_per, model, test_loader)
        hookList = [lossHook, schedulerHook, saveHook]
        # hookList = [schedulerHook,saveHook]

        # print(cfg.train.init_checkpoint)
        trainer = LazyAstroTrainer(model, loader, optimizer, cfg)
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
        cfg.train.init_checkpoint = os.path.join(
            output_dir, output_name + ".pth"
        )  # replace with the path were you have your model

        # Step 2)

        # cfg.MODEL.BACKBONE.FREEZE_AT = 0  # unfreeze all backbone layers
        # cfg_loader.SOLVER.BASE_LR = 0.0001
        cfg_loader.SOLVER.BASE_LR = 0.000001
        cfg_loader.SOLVER.STEPS = []  # [e2,e3]        # decay learning rate
        # cfg.SOLVER.STEPS = [50,100]        # decay learning rate
        cfg_loader.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
        cfg_loader.SOLVER.WARMUP_ITERS = 0
        cfg_loader.SOLVER.MAX_ITER = efinal  # for LR scheduling

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

        model = instantiate(cfg.model)

        # VitDet model doesn't fit fully into memory if training the whole thing
        if modname == "vitdet":
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
        cfg.optimizer.lr = 0.0001
        optimizer = instantiate(cfg.optimizer)

        # optimizer = solver.build_optimizer(cfg, model)
        loader = data.build_detection_train_loader(cfg_loader, mapper=_train_mapper)
        test_loader = data.build_detection_test_loader(
            cfg_loader, cfg_loader.DATASETS.TEST, mapper=_test_mapper
        )

        saveHook = detectron_addons.SaveHook()
        saveHook.set_output_name(output_name)
        schedulerHook = detectron_addons.CustomLRScheduler(optimizer=optimizer)
        lossHook = detectron_addons.LossEvalHook(val_per, model, test_loader)
        hookList = [lossHook, schedulerHook, saveHook]
        # hookList = [schedulerHook,saveHook]

        trainer = LazyAstroTrainer(model, loader, optimizer, cfg)
        trainer.register_hooks(hookList)
        # trainer.set_period(int(epoch/2)) # print loss every n iterations
        # trainer.train(0,efinal)
        trainer.set_period(5)  # print loss every n iterations
        trainer.train(0, 20)

        if comm.is_main_process():
            losses = np.load(output_dir + output_name + "_losses.npy")
            losses = np.concatenate((losses, trainer.lossList))
            np.save(output_dir + output_name + "_losses", losses)

            vallosses = np.load(output_dir + output_name + "_val_losses.npy")
            vallosses = np.concatenate((vallosses, trainer.vallossList))
            np.save(output_dir + output_name + "_val_losses", vallosses)

        return


if __name__ == "__main__":
    args = make_training_arg_parser().parse_args()
    print("Command Line Args:", args)

    dirpath = "/home/shared/hsc/HSC/HSC_DR3/data/"  # Path to dataset
    # dirpath = '/home/shared/hsc/decam/decam_data/' # Path to dataset

    dataset_names = ["train", "test", "val"]

    # filenames_dict_list = get_dict_lists(dataset_names,dirpath,args.sample_number)
    traind = get_data_from_json(os.path.join(dirpath, dataset_names[0]) + "_sample_new.json")
    testd = get_data_from_json(os.path.join(dirpath, dataset_names[2]) + "_sample_new.json")

    # number of total samples
    print("# of train sample: ", len(traind))
    print("# of val sample: ", len(testd))

    tl = len(traind)
    del traind, testd
    gc.collect()

    # dataset_dicts = data_register_and_load(dataset_names,filenames_dict_list)

    # for i, d in enumerate(dataset_names):
    #    filenames_dir = os.path.join(dirpath,d)
    #    DatasetCatalog.register("astro_" + d, lambda: get_astro_dicts(filenames_dir))
    #    MetadataCatalog.get("astro_" + d).set(thing_classes=["star", "galaxy"], things_colors = ['blue', 'gray'])

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

    torch.cuda.empty_cache()
    gc.collect()

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
