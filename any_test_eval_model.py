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
from deepdisc.inference.match_objects import get_matched_object_classes, get_matched_z_pdfs, get_matched_z_pdfs_new #! here
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


def load_data(testfile):
    """Load the data into dataset_dicts and output how long loading took.
    """
    dataset_names = ["test"]
    t0 = time.time()
    dataset_dicts = {}
    for i, d in enumerate(dataset_names):
        dataset_dicts[d] = get_data_from_json(testfile)
    print("Took ", time.time() - t0, "seconds to load samples")
    
    return dataset_dicts


def get_config(use_dc2, use_redshift, output_dir, run_name):
    """Get the relevant config based on if using dc2/redshifts.
    
    Adds the MISC keys into the the top level of the config
    (these would otherwise be ignored as config file local vars when importing)
    """
    cfg_dir = "./tests/deepdisc/test_data/configs/solo"
    if use_dc2:
        if use_redshift:
            file_name = "solo_cascade_mask_rcnn_swin_b_in21k_50ep_test_eval_DC2_redshift.py"
        file_name = "solo_cascade_mask_rcnn_swin_b_in21k_50ep_test_eval_DC2.py"
    else:
        file_name = "solo_cascade_mask_rcnn_swin_b_in21k_50ep_test_eval.py"
    cfg_file = f"{cfg_dir}/{file_name}"
    cfg = LazyConfig.load(cfg_file)
    
    # Set misc vals as top level vals
    for key in cfg.get("MISC", dict()).keys():
        cfg[key] = cfg.MISC[key]
    
    # Set command line args as config vals
    cfg.OUTPUT_DIR = output_dir
    
    init_checkpoint = os.path.join(output_dir, run_name)
    if init_checkpoint.split(".")[-1] != "pth":
        init_checkpoint = f"{init_checkpoint}.pth"
    cfg.train.init_checkpoint = init_checkpoint
    
    return cfg


def get_IR_and_keymapper(use_dc2, norm):
    """Returns the appropriate image reader and key mapper for DC2 or HSC data.
    """
    if use_dc2:
        def dc2_key_mapper(dataset_dict):
            filename = dataset_dict["filename"]
            return filename
        IR = DC2ImageReader(norm=norm)
        return IR, dc2_key_mapper
    else:
        def hsc_key_mapper(dataset_dict):
            filenames = [
                dataset_dict["filename_G"],
                dataset_dict["filename_R"],
                dataset_dict["filename_I"],
            ]
            return filenames
        IR = HSCImageReader(norm=norm)
        return IR, hsc_key_mapper
    

def get_classes(dataset_dicts, IR, keymapper, predictor, use_dc2, use_redshift):
    """Get true and predicted classes.
    """
    if use_dc2:
        if use_redshift:
            #! Note: this currently results in a:
            # AttributeError: Cannot find field 'pred_redshift_pdf' in the given Instances!
            true_zs, pred_pdfs, matched_ids = get_matched_z_pdfs(
                dataset_dicts["test"], IR, keymapper, predictor
            )
        else:
            true_classes, pred_classes = get_matched_object_classes(
                dataset_dicts["test"], IR, keymapper, predictor
            )
    else:
        true_classes, pred_classes = get_matched_object_classes(
            dataset_dicts["test"], IR, keymapper, predictor
        )
    return np.array([true_classes, pred_classes])


if __name__ == "__main__":
    args = make_inference_arg_parser().parse_args()
    print("Command Line Args:", args)
    bb = args.run_name.split("_")[0] # backbone
    output_dir = args.output_dir
    run_name = args.run_name
    savedir = args.savedir
    testfile = args.testfile
    Path(savedir).mkdir(parents=True, exist_ok=True)

    dataset_dicts = load_data(testfile)
    
    cfg = get_config(args.use_dc2, args.use_redshift, output_dir, run_name)

    predictor = return_predictor_transformer(cfg)

    IR, keymapper = get_IR_and_keymapper(args.use_dc2, args.norm)
    
    t0 = time.time()
    print("Matching objects")
    classes = get_classes(
        dataset_dicts, 
        IR, keymapper, 
        predictor, 
        args.use_dc2, 
        args.use_redshift
    )
    savename = f"{bb}_test_matched_classes.npy"
    np.save(os.path.join(args.savedir, savename), classes)
    print("Took ", time.time() - t0, " seconds")
    print(classes)
