# Training script for decam data

# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger

setup_logger()

import os
import time

import detectron2.data as data
import detectron2.modeling as modeler
import detectron2.solver as solver
import detectron2.utils.comm as comm

# from google.colab.patches import cv2_imshow

# import some common libraries
import numpy as np

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import launch

import deepdisc.astrodet.astrodet as toolkit
from deepdisc.astrodet import detectron as detectron_addons

# Prettify the plotting
from astrodet.astrodet import set_mpl_style

# Custom Aug classes have been added to detectron source files

set_mpl_style()




from deepdisc.data_format.register_data import register_data_set
from deepdisc.utils.parse_arguments import make_training_arg_parser


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
        _ = register_data_set(
            "astro_" + d,
            filenames_dir + ".json",
            thing_classes=["star", "galaxy"],
            things_colors=["blue", "gray"],
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


if __name__ == "__main__":
    args = make_training_arg_parser().parse_args()
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
