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
from detectron2.data import MetadataCatalog, DatasetCatalog

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


def main(args, freeze):
    # Hack if you get SSL certificate error
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    # Handle args
    output_dir = args.output_dir
    run_name = args.run_name    

    # Get file locations
    trainfile = args.train_metadata
    evalfile = args.eval_metadata
   

    cfgfile = args.cfgfile
    
    # Load the config
    cfg = LazyConfig.load(cfgfile)
    for key in cfg.get("MISC", dict()).keys():
        cfg[key] = cfg.MISC[key]

    
    if args.num_gpus==1 and not freeze:
        DatasetCatalog.remove(cfg.DATASETS.TRAIN)
        MetadataCatalog.remove(cfg.DATASETS.TRAIN)
        DatasetCatalog.remove(cfg.DATASETS.TEST)
        MetadataCatalog.remove(cfg.DATASETS.TEST)
        
    # Register the data sets
    astrotrain_metadata = register_data_set(
        cfg.DATASETS.TRAIN, trainfile, thing_classes=cfg.metadata.classes
    )
    astroval_metadata = register_data_set(
        cfg.DATASETS.TEST, evalfile, thing_classes=cfg.metadata.classes
    )

    # Set the output directory
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Iterations for 15, 25, 35, 50 epochs
    epoch = cfg.dataloader.epoch
    e1 = epoch * 15
    e2 = epoch * 25
    e3 = epoch * 30
    efinal = epoch * 50
    

    val_per = epoch
    #val_per=5
    
    model = return_lazy_model(cfg,freeze)

    mapper = cfg.dataloader.train.mapper(
            cfg.dataloader.imagereader, cfg.dataloader.key_mapper, cfg.dataloader.augs
        ).map_data


    loader = return_train_loader(cfg, mapper)
    eval_loader = return_test_loader(cfg, mapper)    
    
    cfg.optimizer.params.model = model

        
    if freeze:

        cfg.optimizer.lr = 0.001
        optimizer = return_optimizer(cfg)


        saveHook = return_savehook(run_name)
        lossHook = return_evallosshook(val_per, model, eval_loader)
        schedulerHook = return_schedulerhook(optimizer)
        hookList = [lossHook, schedulerHook, saveHook]

        trainer = return_lazy_trainer(model, loader, optimizer, cfg, hookList)
        trainer.set_period(epoch//2)
        trainer.train(0, e1)
        #trainer.train(0, 10)
        if comm.is_main_process():
            np.save(output_dir + run_name + "_losses", trainer.lossList)
            np.save(output_dir + run_name + "_val_losses", trainer.vallossList)
            
        return
            
    else:
        
        cfg.train.init_checkpoint = os.path.join(output_dir, run_name + ".pth")
        cfg.SOLVER.BASE_LR = 0.0001
        cfg.SOLVER.MAX_ITER = efinal  # for DefaultTrainer
        cfg.SOLVER.STEPS=[e2,e3]
        
        cfg.optimizer.lr = 0.0001
        
        optimizer = return_optimizer(cfg)
        schedulerHook = return_schedulerhook(optimizer)
        
        saveHook = return_savehook(run_name)
        lossHook = return_evallosshook(val_per, model, eval_loader)
        schedulerHook = return_schedulerhook(optimizer)
        hookList = [lossHook, schedulerHook, saveHook]

        trainer = return_lazy_trainer(model, loader, optimizer, cfg, hookList)
        trainer.set_period(epoch//2)
        trainer.train(e1, efinal)
        #trainer.train(10, 20)
        if comm.is_main_process():
            losses = np.load(output_dir + run_name + "_losses.npy")
            losses = np.concatenate((losses, trainer.lossList))
            np.save(output_dir + run_name + "_losses", losses)
        return
            
    

if __name__ == "__main__":
    args = make_training_arg_parser().parse_args()
    print("Command Line Args:", args)

    print("Training head layers")
    freeze = True
    t0 = time.time()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(
            args,
            freeze
        ),
    )

    torch.cuda.empty_cache()
    gc.collect()
    
    
    ######
    # After finetuning the head layers, train the whole model
    ######
    
    print("Training all layers")
    freeze = False
    t0 = time.time()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(
            args,
            freeze
        ),
    )

    torch.cuda.empty_cache()
    gc.collect()


    
    print(f"Took {time.time()-t0} seconds")
    