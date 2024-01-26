"""
This code will read in a trained model and output the classes for predicted objects matched to the ground truth 

"""
import logging
import os
import time

import numpy as np
import deepdisc.astrodet.astrodet as toolkit

from deepdisc.data_format.file_io import get_data_from_json
from deepdisc.data_format.image_readers import DC2ImageReader
from deepdisc.inference.match_objects import get_matched_object_classes, get_matched_z_pdfs, run_batched_match_class, run_batched_match_redshift
from deepdisc.inference.predictors import return_predictor_transformer
from deepdisc.utils.parse_arguments import dtype_from_args, make_inference_arg_parser
from deepdisc.model.loaders import RedshiftDictMapperEval, return_test_loader, return_train_loader


from detectron2 import model_zoo
from detectron2.config import LazyConfig
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger
import detectron2.data as d2data

from pathlib import Path
from detectron2.engine import launch

setup_logger()
logger = logging.getLogger(__name__)
import torch.distributed as dist
import torch.multiprocessing as mp

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:


import torch.distributed as dist


def gather_predictions(array, array_list=None, root=0, group=None):
    """
        Sends tensor to root process, which store it in tensor_list.
    """
  
    rank = dist.get_rank()
    if group is None:
        group = dist.group.WORLD
    if rank == root:
        assert(array_list is not None)
        dist.gather_object(array, object_gather_list=array_list, group=group)
    else:
        dist.gather_object(array, dst=root, group=group)



def main(q,args):
    size = args.num_gpus * args.num_machines
    
    # --------- Handle args
    roi_thresh = args.roi_thresh
    run_name = args.run_name
    testfile = args.testfile
    savedir = args.savedir
    Path(savedir).mkdir(parents=True, exist_ok=True)
    output_dir = args.output_dir
    dtype=dtype_from_args(args.datatype)
        
    # --------- Load data
    dataset_names = ["test"]
    t0 = time.time()
    dataset_dicts = {}
    for i, d in enumerate(dataset_names):
        dataset_dicts[d] = get_data_from_json(testfile)
    print("Took ", time.time() - t0, "seconds to load samples")
    
    # Local vars/metadata
    #classes = ["star", "galaxy"]
    bb = args.run_name.split("_")[0] # backbone
    
    # --------- Start config stuff
    
    cfgfile = "./tests/deepdisc/test_data/configs/solo/solo_cascade_mask_rcnn_swin_b_in21k_50ep_DC2_redshift.py"
    cfg = LazyConfig.load(cfgfile)
    
    # --------- Setting a bunch of config stuff

    cfg.train.init_checkpoint = os.path.join(output_dir, run_name)
    
    # --------- Now we case predictor on model type (the second case has way different config vals it appears)
    
    cfg.OUTPUT_DIR = output_dir
    if bb in ['Swin','MViTv2']:
        predictor= return_predictor_transformer(cfg)
    else:
        cfgfile = "./tests/deepdisc/test_data/configs/solo/solo_cascade_mask_rcnn_swin_b_in21k_50ep_DC2.py"
        predictor, cfg = return_predictor(cfgfile, run_name, output_dir=output_dir, nc=2, roi_thresh=roi_thresh)

    # --------- 
    #def dc2_key_mapper(dataset_dict):
    #    filename = dataset_dict["filename"]
    #    return filename

    
    def dc2_key_mapper(dataset_dict):
        filename = dataset_dict["filename"]
        base = filename.split(".")[0].split("/")[-1]
        dirpath = "/home/g4merz/DC2/nersc_data/scarlet_data"
        fn = os.path.join(dirpath, base) + ".npy"
        return fn
    
    IR = DC2ImageReader()
    
    
    mapper = RedshiftDictMapperEval(IR, dc2_key_mapper).map_data

    #loader = return_test_loader(cfg, mapper)
    loader = d2data.build_detection_test_loader(
        dataset_dicts['test'], mapper=mapper, batch_size=1
    )
    
    
    # --------- Do the thing
    t0 = time.time()
    print("Matching objects")
    #true_classes, pred_classes = get_matched_object_classes(dataset_dicts["test"], IR, dc2_key_mapper, predictor)
    #true_zs, pred_pdfs = get_matched_z_pdfs(dataset_dicts["test"], IR, dc2_key_mapper, predictor)

    true_classes, pred_classes = run_batched_match_class(loader, predictor)
    
    classes = np.array([true_classes, pred_classes])
    
    
    true_zs, pred_pdfs, ids = run_batched_match_redshift(loader, predictor, ids=True)
    
    print(len(true_zs))
        
    if size==1:
        np.save(os.path.join(args.savedir,'predicted_pdfs.npy'),pred_pdfs)
        np.save(os.path.join(args.savedir,'true_zs.npy'),true_zs)
        np.save(os.path.join(args.savedir,'ids.npy'),ids)

        return


    else:
        #size is the world size
        true_zlist = [None for _ in range(size)]
        pred_zlist = [None for _ in range(size)]
        id_list = [None for _ in range(size)]

        if dist.get_rank() == 0:
            gather_predictions(true_zs, true_zlist)
            gather_predictions(pred_pdfs, pred_zlist)
            gather_predictions(ids, id_list)

        else:
            gather_predictions(true_zs)
            gather_predictions(pred_pdfs)
            gather_predictions(ids)

    
        if dist.get_rank() == 0:
            #pred_zlist = np.concatenate([pred_list for pred_list in pred_zlist])
            #np.save(os.path.join(args.savedir,'predicted_pdfs.npy'),pred_zlist)
            #np.save(os.path.join(args.savedir,'true_zs.npy'),true_zlist)
            #np.save(os.path.join(args.savedir,'ids.npy'),id_list)
            q.put(pred_zlist)

    

if __name__ == "__main__":
    args = make_inference_arg_parser().parse_args()
    
    print('Inference')
    train_head = True
    
    q = mp.Queue()
    
    t0 = time.time()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(
            q,
            args,
        ),
    )
    
    print(q.get())


    print(f"Took {time.time()-t0} seconds")
    