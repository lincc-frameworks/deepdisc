"""
This code will read in a trained model and output the classes for predicted objects matched to the ground truth 

"""
import logging
import os
import time

import numpy as np
import deepdisc.astrodet.astrodet as toolkit

from deepdisc.data_format.file_io import get_data_from_json
from deepdisc.data_format.image_readers import HSCImageReader, DC2ImageReader
from deepdisc.inference.match_objects import get_matched_object_classes, get_matched_z_pdfs
from deepdisc.inference.predictors import return_predictor_transformer
from deepdisc.model.models import RedshiftPDFCasROIHeads #! is this necessary if it's now used only in the config?
from deepdisc.utils.parse_arguments import dtype_from_args, make_inference_arg_parser

from detectron2 import model_zoo
from detectron2.config import LazyConfig
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger

from pathlib import Path

setup_logger()
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # --------- Handle args
    args = make_inference_arg_parser().parse_args()
    print("Command Line Args:", args)
    
    run_name = args.run_name
    testfile = args.testfile
    savedir = args.savedir
    Path(savedir).mkdir(parents=True, exist_ok=True)
    output_dir = args.output_dir
    dtype=dtype_from_args(args.datatype)

    # --------- Load data
    dataset_names = ["test"]
    
    if args.use_dc2:
        datadir = "./tests/deepdisc/test_data/dc2/"
    else:
        datadir = "/home/shared/hsc/HSC/HSC_DR3/data/"
        
    t0 = time.time()
    dataset_dicts = {}
    for i, d in enumerate(dataset_names):
        dataset_dicts[d] = get_data_from_json(testfile)
    print("Took ", time.time() - t0, "seconds to load samples")
    
    # Local vars/metadata
    bb = args.run_name.split("_")[0] # backbone
    
    # --------- Get config
    cfg_dir = "./tests/deepdisc/test_data/configs/solo"
    if args.use_dc2:
        if args.use_redshift:
            file_name = "solo_cascade_mask_rcnn_swin_b_in21k_50ep_test_eval_DC2_redshift.py"
        file_name = "solo_cascade_mask_rcnn_swin_b_in21k_50ep_test_eval_DC2.py"
    else:
        file_name = "solo_cascade_mask_rcnn_swin_b_in21k_50ep_test_eval.py"
    cfg_file = f"{cfg_dir}/{file_name}")

    cfg = LazyConfig.load(cfg_file)
    for key in cfg.get("MISC", dict()).keys():
        cfg[key] = cfg.MISC[key]
    
    # --------- Setting config values that depend on command line inputs
    cfg.OUTPUT_DIR = output_dir
    cfg.train.init_checkpoint = os.path.join(output_dir, run_name)

    # --------- Get predictor
    predictor = return_predictor_transformer(cfg)

    # --------- Key mappers
    if args.use_dc2:
        def dc2_key_mapper(dataset_dict):
            filename = dataset_dict["filename"]
            return filename
        IR = DC2ImageReader(norm=args.norm)
    else:
        def hsc_key_mapper(dataset_dict):
            filenames = [
                dataset_dict["filename_G"],
                dataset_dict["filename_R"],
                dataset_dict["filename_I"],
            ]
            return filenames
        IR = HSCImageReader(norm=args.norm)
    
    # --------- Do the thing
    t0 = time.time()
    print("Matching objects")
    
    if args.use_dc2:
        true_classes, pred_classes = get_matched_object_classes(dataset_dicts["test"], IR, dc2_key_mapper, predictor)
        if args.use_redshift:
            true_zs, pred_pdfs, matched_ids = get_matched_z_pdfs(dataset_dicts["test"], IR, dc2_key_mapper, predictor)
            print(true_zs)
            print(f"{str(pred_pdfs)[:1000]}...")
    else:
        true_classes, pred_classes = get_matched_object_classes(dataset_dicts["test"], IR, hsc_key_mapper, predictor)

    classes = np.array([true_classes, pred_classes])
    savename = f"{bb}_test_matched_classes.npy"
    np.save(os.path.join(args.savedir, savename), classes)

    print("Took ", time.time() - t0, " seconds")
    print(classes)
