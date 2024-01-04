import copy
import os
import sys
import time
import weakref
from typing import Dict, List, Mapping, Optional

# Some basic setup:
# Setup detectron2 logger
import detectron2
import detectron2.checkpoint as checkpointer
import detectron2.data as data
import detectron2.data.transforms as T
import detectron2.modeling as modeler
import detectron2.solver as solver
import numpy as np
import torch
from detectron2.data import detection_utils as utils
from detectron2.data.transforms import Augmentation, Transform
from detectron2.engine import DefaultPredictor, DefaultTrainer, SimpleTrainer
from detectron2.engine.defaults import create_ddp_model
from detectron2.utils.logger import setup_logger

setup_logger()
import argparse
import contextlib
import copy
import datetime
import gc
import glob
import io
import itertools
import json
import logging
import os
import pickle
import random
import shutil
import weakref
from collections import OrderedDict
from typing import Optional

import cv2
import detectron2.data.transforms as T
import detectron2.utils.comm as comm
import imgaug.augmenters as iaa
import imgaug.augmenters.flip as flip

# from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt

# import some common libraries
import numpy as np
import pycocotools.mask as mask_util
import torch
from astropy.io import fits
from astropy.visualization import make_lupton_rgb

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, LazyConfig, get_cfg, instantiate
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.engine.hooks import LRScheduler
from detectron2.evaluation import DatasetEvaluator, inference_on_dataset, print_csv_format, verify_results

# yufeng 6/11 import cocoevaluator
from detectron2.evaluation.coco_evaluation import COCOEvaluator
from detectron2.evaluation.fast_eval_api import COCOeval_opt
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes, pairwise_iou
from detectron2.utils import comm
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.env import seed_all_rng
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
    get_event_storage,
)
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table, log_every_n_seconds, setup_logger
from detectron2.utils.visualizer import Visualizer
from fvcore.common.param_scheduler import ParamScheduler
from fvcore.nn.precise_bn import get_bn_modules
from iopath.common.file_io import file_lock
from omegaconf import OmegaConf
from PIL import Image, ImageEnhance
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate
from torch.nn.parallel import DistributedDataParallel


def set_mpl_style():
    """Function to set MPL style"""

    fsize = 15
    tsize = 18
    tdir = "in"
    major = 5.0
    minor = 3.0
    lwidth = 1.8
    lhandle = 2.0
    plt.style.use("default")
    plt.rcParams["text.usetex"] = False
    plt.rcParams["font.size"] = fsize
    plt.rcParams["legend.fontsize"] = tsize
    plt.rcParams["xtick.direction"] = tdir
    plt.rcParams["ytick.direction"] = tdir
    plt.rcParams["xtick.major.size"] = major
    plt.rcParams["xtick.minor.size"] = minor
    plt.rcParams["ytick.major.size"] = 5.0
    plt.rcParams["ytick.minor.size"] = 3.0
    plt.rcParams["axes.linewidth"] = lwidth
    plt.rcParams["legend.handlelength"] = lhandle

    return


class NewAstroTrainer(SimpleTrainer):
    """
    Use this for models that use yacs cfg files

    Parameters
    ----------
    model: torch nn.module
        The model being trained
    data_loader: detectron2 DataLoader
        The data loader that loads the training set
    optimizer:
        The learning optimizer
    cfg: config file
        The model config


    """

    def __init__(self, model, data_loader, optimizer, cfg):
        super().__init__(model, data_loader, optimizer)
        # super().__init__(model, data_loader, optimizer)

        # Borrowed from DefaultTrainer constructor
        # see https://detectron2.readthedocs.io/en/latest/_modules/detectron2/engine/defaults.html#DefaultTrainer
        self.checkpointer = checkpointer.DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
        )
        # load weights
        self.checkpointer.load(cfg.MODEL.WEIGHTS)

        # record loss over iteration
        self.lossList = []
        self.vallossList = []

        self.period = 20
        self.iterCount = 0

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
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
        # print('Loss dict',loss_dict.values())
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())
            all_losses = [l.cpu().detach().item() for l in loss_dict.values()]
        self.optimizer.zero_grad()
        losses.backward()

        # self._write_metrics(loss_dict,data_time)

        self.optimizer.step()

        self.lossList.append(losses.cpu().detach().numpy())
        if self.iterCount % self.period == 0 and comm.is_main_process():
            # print("Iteration: ", self.iterCount, " time: ", data_time," loss: ",losses.cpu().detach().numpy(), "val loss: ",self.valloss, "lr: ", self.scheduler.get_lr())
            print(
                "Iteration: ",
                self.iterCount,
                " time: ",
                data_time,
                loss_dict.keys(),
                all_losses,
                "val loss: ",
                self.valloss,
                "lr: ",
                self.scheduler.get_lr(),
            )

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


class AstroPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.
    Compared to using the model directly, this class does the following additions:
    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.
    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more complicated, please refer to its source code as
    examples to build and use the model manually.
    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.
    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """
    def __init__(self, cfg, lazy=False, cfglazy=None, checkpoint=None):
        self.cfg = copy.deepcopy(cfg) # cfg can be modified by model
        
        if "model" in self.cfg: # This is when were using a LazyConfig-style model in the solo config
            self.model = instantiate(self.cfg.model)
            self.model.to(self.cfg.train.device)
            self.model = create_ddp_model(self.model)
            
        else: # This is when we're using a yacs-style model in the solo config (will be specified as "MODEL")
            self.model = build_model(self.cfg)

        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)

        # If we provide AstroPredictor with a checkpoint already loaded in memory
        # just simply load the weights into the model.
        if checkpoint:
            checkpointer._load_model(checkpoint)
        else:
            checkpointer.load(cfg.train.init_checkpoint)
        
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format
      
    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            # image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions


class COCOeval_opt_custom(COCOeval_opt):
    """ """

    def evaluate_custom(self):
        """
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        """
        tic = time.time()
        print("Running per image evaluation...")
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = "segm" if p.useSegm == 1 else "bbox"
            print("useSegm (deprecated) is not None. Running {} evaluation".format(p.iouType))
        print("Evaluate annotation type *{}*".format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p
        print(p.areaRng)
        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == "segm" or p.iouType == "bbox":
            computeIoU = self.computeIoU
        elif p.iouType == "keypoints":
            computeIoU = self.computeOks
        self.ious = {(imgId, catId): computeIoU(imgId, catId) for imgId in p.imgIds for catId in catIds}
        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [
            evaluateImg(imgId, catId, areaRng, maxDet)
            for catId in catIds
            for areaRng in p.areaRng
            for imgId in p.imgIds
        ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print("DONE (t={:0.2f}s).".format(toc - tic))

    def accumulate_custom(self, p=None):
        """
        YL: Override in order to put in some output commands

        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        """
        print("Accumulating evaluation results...")
        tic = time.time()
        if not self.evalImgs:
            print("Please run evaluate() first")
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = len(p.catIds) if p.useCats else 1
        A = len(p.areaRng)
        M = len(p.maxDets)
        precision = -np.ones((T, R, K, A, M))  # -1 for the precision of absent categories
        recall = -np.ones((T, K, A, M))
        scores = -np.ones((T, R, K, A, M))

        precision_raw = -np.ones((T, R, K, A, M))
        recall_raw = -np.ones((T, R, K, A, M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e["dtScores"][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind="mergesort")
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate([e["dtMatches"][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    dtIg = np.concatenate([e["dtIgnore"][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    gtIg = np.concatenate([e["gtIgnore"] for e in E])

                    npig = np.count_nonzero(gtIg == 0)
                    # print('npig', npig)
                    if npig == 0:
                        continue
                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))
                    # print('tps cumsum', np.cumsum(tps))
                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    # print('TP and FP sums', tp_sum.shape, fp_sum.shape)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))
                        q = np.zeros((R,))
                        ss = np.zeros((R,))
                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist()
                        q = q.tolist()

                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side="left")
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t, :, k, a, m] = np.array(q)
                        scores[t, :, k, a, m] = np.array(ss)
        self.eval = {
            "params": p,
            "counts": [T, R, K, A, M],
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": precision,
            "recall": recall,
            "scores": scores,
        }
        toc = time.time()
        print("DONE (t={:0.2f}s).".format(toc - tic))

    def summarize_custom(self):
        """
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        """

        def _summarize(ap=1, iouThr=None, areaRng="all", maxDets=100):
            p = self.params
            iStr = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
            titleStr = "Average Precision" if ap == 1 else "Average Recall"
            typeStr = "(AP)" if ap == 1 else "(AR)"
            iouStr = (
                "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1])
                if iouThr is None
                else "{:0.2f}".format(iouThr)
            )

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval["precision"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval["recall"]
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=0.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=0.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng="small", iouThr=0.5, maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng="medium", iouThr=0.5, maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng="large", iouThr=0.5, maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng="small", maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng="medium", maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng="large", maxDets=self.params.maxDets[2])
            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=0.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=0.75)
            stats[3] = _summarize(1, maxDets=20, areaRng="medium")
            stats[4] = _summarize(1, maxDets=20, areaRng="large")
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=0.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=0.75)
            stats[8] = _summarize(0, maxDets=20, areaRng="medium")
            stats[9] = _summarize(0, maxDets=20, areaRng="large")
            return stats

        if not self.eval:
            raise Exception("Please run accumulate() first")
        iouType = self.params.iouType
        if iouType == "segm" or iouType == "bbox":
            summarize = _summarizeDets
        elif iouType == "keypoints":
            summarize = _summarizeKps
        self.stats = summarize()


def _evaluate_predictions_on_coco(
    coco_gt,
    coco_results,
    iou_type,
    kpt_oks_sigmas=None,
    use_fast_impl=True,
    img_ids=None,
    max_dets_per_image=None,
    areaRng=None,
):
    # Evaluate the coco results using COCOEval API.
    assert len(coco_results) > 0
    print("_evaluate_predictions_on_coco")
    # 6/27 this override function is not called
    if iou_type == "segm":
        coco_results = copy.deepcopy(coco_results)
        # When evaluating mask AP, if the results contain bbox, cocoapi will
        # use the box area as the area of the instance, instead of the mask area.
        # This leads to a different definition of small/medium/large.
        # We remove the bbox field to let mask AP use mask area.
        for c in coco_results:
            c.pop("bbox", None)

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = (COCOeval_opt_custom if use_fast_impl else COCOeval)(coco_gt, coco_dt, iou_type)
    # change COCOeval_opt_custom to COCO_eval_opt to call the default function
    print("++++++++++", type(coco_eval))
    if img_ids is not None:
        coco_eval.params.imgIds = img_ids

    if iou_type == "keypoints":
        # Use the COCO default keypoint OKS sigmas unless overrides are specified
        if kpt_oks_sigmas:
            assert hasattr(coco_eval.params, "kpt_oks_sigmas"), "pycocotools is too old!"
            coco_eval.params.kpt_oks_sigmas = np.array(kpt_oks_sigmas)
        # COCOAPI requires every detection and every gt to have keypoints, so
        # we just take the first entry from both
        num_keypoints_dt = len(coco_results[0]["keypoints"]) // 3
        num_keypoints_gt = len(next(iter(coco_gt.anns.values()))["keypoints"]) // 3
        num_keypoints_oks = len(coco_eval.params.kpt_oks_sigmas)
        assert num_keypoints_oks == num_keypoints_dt == num_keypoints_gt, (
            f"[COCOEvaluator] Prediction contain {num_keypoints_dt} keypoints. "
            f"Ground truth contains {num_keypoints_gt} keypoints. "
            f"The length of cfg.TEST.KEYPOINT_OKS_SIGMAS is {num_keypoints_oks}. "
            "They have to agree with each other. For meaning of OKS, please refer to "
            "http://cocodataset.org/#keypoints-eval."
        )

    coco_eval.params.maxDets = (
        max_dets_per_image  # by default it is [1,10,100], our datasets have more than 100 instances
    )
    coco_eval.params.areaRng = areaRng
    coco_eval.evaluate_custom()
    coco_eval.accumulate_custom()
    coco_eval.summarize_custom()
    # coco_eval.summarize()

    return coco_eval


def convert_to_coco_dict(dataset_name, mbins, mind, logger):
    """
    Convert an instance detection/segmentation or keypoint detection dataset
    in detectron2's standard format into COCO json format.

    Generic dataset description can be found here:
    https://detectron2.readthedocs.io/tutorials/datasets.html#register-a-dataset

    COCO data format description can be found here:
    http://cocodataset.org/#format-data

    Args:
        dataset_name (str):
            name of the source dataset
            Must be registered in DatastCatalog and in detectron2's standard format.
            Must have corresponding metadata "thing_classes"
    Returns:
        coco_dict: serializable dict in COCO json format
    """
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    # unmap the category mapping ids for COCO
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
        reverse_id_mapping = {v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()}
        reverse_id_mapper = lambda contiguous_id: reverse_id_mapping[contiguous_id]  # noqa
    else:
        reverse_id_mapper = lambda contiguous_id: contiguous_id  # noqa

    categories = [
        {"id": reverse_id_mapper(id), "name": name} for id, name in enumerate(metadata.thing_classes)
    ]

    logger.info("Converting dataset dicts into COCO format")
    coco_images = []
    coco_annotations = []

    for image_id, image_dict in enumerate(dataset_dicts):
        coco_image = {
            "id": image_dict.get("image_id", image_id),
            "width": int(image_dict["width"]),
            "height": int(image_dict["height"]),
            "file_name": str(image_dict["file_name"]),
        }
        coco_images.append(coco_image)

        anns_per_image = image_dict.get("annotations", [])
        for annotation in anns_per_image:
            # create a new dict with only COCO fields
            coco_annotation = {}

            # COCO requirement: XYWH box format for axis-align and XYWHA for rotated
            bbox = annotation["bbox"]
            if isinstance(bbox, np.ndarray):
                if bbox.ndim != 1:
                    raise ValueError(f"bbox has to be 1-dimensional. Got shape={bbox.shape}.")
                bbox = bbox.tolist()
            if len(bbox) not in [4, 5]:
                raise ValueError(f"bbox has to has length 4 or 5. Got {bbox}.")
            from_bbox_mode = annotation["bbox_mode"]
            to_bbox_mode = BoxMode.XYWH_ABS if len(bbox) == 4 else BoxMode.XYWHA_ABS
            bbox = BoxMode.convert(bbox, from_bbox_mode, to_bbox_mode)

            # COCO requirement: instance area
            if "segmentation" in annotation:
                # Computing areas for instances by counting the pixels
                segmentation = annotation["segmentation"]
                # TODO: check segmentation type: RLE, BinaryMask or Polygon
                if isinstance(segmentation, list):
                    polygons = PolygonMasks([segmentation])
                    area = polygons.area()[0].item()
                elif isinstance(segmentation, dict):  # RLE
                    area = mask_util.area(segmentation).item()
                else:
                    raise TypeError(f"Unknown segmentation type {type(segmentation)}!")
            else:
                # Computing areas using bounding boxes
                if to_bbox_mode == BoxMode.XYWH_ABS:
                    bbox_xy = BoxMode.convert(bbox, to_bbox_mode, BoxMode.XYXY_ABS)
                    area = Boxes([bbox_xy]).area()[0].item()
                else:
                    area = RotatedBoxes([bbox]).area()[0].item()

            if "keypoints" in annotation:
                keypoints = annotation["keypoints"]  # list[int]
                for idx, v in enumerate(keypoints):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # For COCO format consistency we substract 0.5
                        # https://github.com/facebookresearch/detectron2/pull/175#issuecomment-551202163
                        keypoints[idx] = v - 0.5
                if "num_keypoints" in annotation:
                    num_keypoints = annotation["num_keypoints"]
                else:
                    num_keypoints = sum(kp > 0 for kp in keypoints[2::3])

            # COCO requirement:
            #   linking annotations to images
            #   "id" field must start with 1
            coco_annotation["id"] = len(coco_annotations) + 1
            coco_annotation["image_id"] = coco_image["id"]
            coco_annotation["bbox"] = [round(float(x), 3) for x in bbox]
            coco_annotation["area"] = float(area)
            coco_annotation["iscrowd"] = int(annotation.get("iscrowd", 0))
            # coco_annotation["iscrowd"] = 1 if annotation.get('imag')>24 else 0
            # coco_annotation["ignore"] = 1 if annotation.get('imag')>24 else 0
            if mind != len(mbins) - 1 and mind != -1:
                coco_annotation["ignore"] = (
                    0
                    if annotation.get("imag") > mbins[mind] and annotation.get("imag") <= mbins[mind + 1]
                    else 1
                )
            elif mind == len(mbins) - 1:
                coco_annotation["ignore"] = 0 if annotation.get("imag") > mbins[mind] else 1
            else:
                coco_annotation["ignore"] = int(annotation.get("ignore", 0))

            coco_annotation["category_id"] = int(reverse_id_mapper(annotation["category_id"]))

            # Add optional fields
            if "keypoints" in annotation:
                coco_annotation["keypoints"] = keypoints
                coco_annotation["num_keypoints"] = num_keypoints

            if "segmentation" in annotation:
                seg = coco_annotation["segmentation"] = annotation["segmentation"]
                if isinstance(seg, dict):  # RLE
                    counts = seg["counts"]
                    if not isinstance(counts, str):
                        # make it json-serializable
                        seg["counts"] = counts.decode("ascii")

            coco_annotations.append(coco_annotation)

    logger.info("Conversion finished, " f"#images: {len(coco_images)}, #annotations: {len(coco_annotations)}")

    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated COCO json file for Detectron2.",
    }
    coco_dict = {
        "info": info,
        "images": coco_images,
        "categories": categories,
        "licenses": None,
    }
    if len(coco_annotations) > 0:
        coco_dict["annotations"] = coco_annotations
    return coco_dict


def convert_to_coco_json(dataset_name, output_file, mbins=[0, 1], mind=-1, allow_cached=True):
    """
    Converts dataset into COCO format and saves it to a json file.
    dataset_name must be registered in DatasetCatalog and in detectron2's standard format.

    Args:
        dataset_name:
            reference from the config file to the catalogs
            must be registered in DatasetCatalog and in detectron2's standard format
        output_file: path of json file that will be saved to
        allow_cached: if json file is already present then skip conversion
    """

    logger = logging.getLogger(__name__)
    PathManager.mkdirs(os.path.dirname(output_file))
    with file_lock(output_file):
        if PathManager.exists(output_file) and allow_cached:
            logger.warning(
                f"Using previously cached COCO format annotations at '{output_file}'. "
                "You need to clear the cache file if your dataset has been modified."
            )
        else:
            logger.info(f"Converting annotations of dataset '{dataset_name}' to COCO format ...)")
            coco_dict = convert_to_coco_dict(dataset_name, mbins, mind, logger)

            logger.info(f"Caching COCO format annotations at '{output_file}' ...")
            tmp_file = output_file + ".tmp"
            with PathManager.open(tmp_file, "w") as f:
                json.dump(coco_dict, f)
            shutil.move(tmp_file, output_file)


class COCOEvaluatorRecall(COCOEvaluator):

    """
    Override this class in order to call the custom function above

    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.
    The metrics range from 0 to 100 (instead of 0 to 1), where a -1 or NaN means
    the metric cannot be computed (e.g. due to no predictions made).

    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def __init__(
        self,
        dataset_name,
        tasks=None,
        distributed=True,
        output_dir=None,
        *,
        max_dets_per_image=None,
        areaRng=None,
        use_fast_impl=True,
        kpt_oks_sigmas=(),
        allow_cached_coco=True,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:
                    "json_file": the path to the COCO format annotation
                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. A task is one of "bbox", "segm", "keypoints".
                By default, will infer this automatically from predictions.
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:
                1. "instances_predictions.pth" a file that can be loaded with `torch.load` and
                   contains all the results in the format they are produced by the model.
                2. "coco_instances_results.json" a json file in COCO's result format.
            max_dets_per_image (int): limit on the maximum number of detections per image.
                By default in COCO, this limit is to 100, but this can be customized
                to be greater, as is needed in evaluation metrics AP fixed and AP pool
                (see https://arxiv.org/pdf/2102.01066.pdf)
                This doesn't affect keypoint evaluation.
            use_fast_impl (bool): use a fast but **unofficial** implementation to compute AP.
                Although the results should be very close to the official implementation in COCO
                API, it is still recommended to compute results with the official API for use in
                papers. The faster implementation also uses more RAM.
            kpt_oks_sigmas (list[float]): The sigmas used to calculate keypoint OKS.
                See http://cocodataset.org/#keypoints-eval
                When empty, it will use the defaults in COCO.
                Otherwise it should be the same length as ROI_KEYPOINT_HEAD.NUM_KEYPOINTS.
            allow_cached_coco (bool): Whether to use cached coco json from previous validation
                runs. You should set this to False if you need to use different validation data.
                Defaults to True.
        """

        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._output_dir = output_dir

        if use_fast_impl and (COCOeval_opt is COCOeval):
            self._logger.info("Fast COCO eval is not built. Falling back to official COCO eval.")
            use_fast_impl = False
        self._use_fast_impl = use_fast_impl

        # COCOeval requires the limit on the number of detections per image (maxDets) to be a list
        # with at least 3 elements. The default maxDets in COCOeval is [1, 10, 100], in which the
        # 3rd element (100) is used as the limit on the number of detections per image when
        # evaluating AP. COCOEvaluator expects an integer for max_dets_per_image, so for COCOeval,
        # we reformat max_dets_per_image into [1, 10, max_dets_per_image], based on the defaults.
        if max_dets_per_image is None:
            max_dets_per_image = [1, 10, 100]
        else:
            max_dets_per_image = [1, 10, max_dets_per_image]
        self._max_dets_per_image = max_dets_per_image

        if areaRng is None:
            areaRng = [
                [0, 10000000000.0],
                [0, 1024],
                [1024, 9216],
                [9216, 10000000000.0],
            ]
        self._areaRng = areaRng

        if tasks is not None and isinstance(tasks, CfgNode):
            kpt_oks_sigmas = tasks.TEST.KEYPOINT_OKS_SIGMAS if not kpt_oks_sigmas else kpt_oks_sigmas
            self._logger.warn(
                "COCO Evaluator instantiated using config, this is deprecated behavior."
                " Please pass in explicit arguments instead."
            )
            self._tasks = None  # Infering it from predictions should be better
        else:
            self._tasks = tasks

        self._cpu_device = torch.device("cpu")

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            if output_dir is None:
                raise ValueError(
                    "output_dir must be provided to COCOEvaluator " "for datasets not in COCO format."
                )
            self._logger.info(f"Trying to convert '{dataset_name}' to COCO format ...")

            cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
            self._metadata.json_file = cache_path
            convert_to_coco_json(dataset_name, cache_path, allow_cached=allow_cached_coco)
        json_file = PathManager.get_local_path(self._metadata.json_file)
        print("Loading ", json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset
        if self._do_evaluation:
            self._kpt_oks_sigmas = kpt_oks_sigmas

    def _eval_predictions(self, predictions, img_ids=None):
        # Evaluate predictions. Fill self._results with the metrics of the tasks.

        self._logger.info("Preparing results for COCO format ...")
        # for splitting by magnitude, take the instances that are matched to objects with that mag thresh
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(coco_results)

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
            all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            num_classes = len(all_contiguous_ids)
            assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            for result in coco_results:
                category_id = result["category_id"]
                assert category_id < num_classes, (
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has {num_classes} classes and "
                    f"predicted class id should be in [0, {num_classes - 1}]."
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )

        self.coco_eval_list = []

        for task in sorted(tasks):
            assert task in {"bbox", "segm", "keypoints"}, f"Got unknown task: {task}!"
            print(self._kpt_oks_sigmas)
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api,
                    coco_results,
                    task,
                    kpt_oks_sigmas=None,
                    use_fast_impl=self._use_fast_impl,
                    img_ids=img_ids,
                    max_dets_per_image=self._max_dets_per_image,
                    areaRng=self._areaRng,
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            self.coco_eval_list.append(coco_eval)

            res = self._derive_coco_results(coco_eval, task, class_names=self._metadata.get("thing_classes"))
            self._results[task] = res

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        # Derive the desired score numbers from summarized COCOeval.

        """Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}"""

        print("++++++++derive_coco_results")
        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}
        # the standard metrics
        print(type(coco_eval))
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info("Evaluation results for {}: \n".format(iou_type) + create_small_table(results))
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        # if class_names is None or len(class_names) <= 1:
        #    return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        precision_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            precision_per_category.append(precisions)
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        # Save the precision-recall per category
        results["results_per_category"] = precision_per_category

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results


def read_image_hsc(
    filenames,
    normalize="lupton",
    stretch=0.5,
    Q=10,
    m=0,
    ceil_percentile=99.995,
    dtype=np.uint8,
    A=1e4,
    do_norm=False,
):
    """
    Read in a formatted HSC image

    Parameters
    ----------
    filenames: list
        The list of g,r,i band files
    normalize: str
        The key word for the normalization scheme
    stretch, Q, m: float, int, float
        Parameters for lupton normalization
    ceil_percentile:
        If do_norm is true, cuts data off at this percentile
    dtype: numpy datatype
        data type of the output array
    A: float
        scaling factor for zscoring
    do_norm: boolean
        For normalizing top fit dtype range

    Returns
    -------
    Scaled image

    """

    def norm(z, r, g):
        max_RGB = np.nanpercentile([z, r, g], ceil_percentile)
        print(max_RGB)

        max_z = np.nanpercentile([z], ceil_percentile)
        max_r = np.nanpercentile([r], ceil_percentile)
        max_g = np.nanpercentile([g], ceil_percentile)

        # z = np.clip(z,None,max_RGB)
        # r = np.clip(r,None,max_RGB)
        # g = np.clip(g,None,max_RGB)

        # avoid saturation
        r = r / max_RGB
        g = g / max_RGB
        z = z / max_RGB
        # r = r/max_r; g = g/max_g; z = z/max_z

        # Rescale to 0-255 for dtype=np.uint8
        max_dtype = np.iinfo(dtype).max
        r = r * max_dtype
        g = g * max_dtype
        z = z * max_dtype

        # 0-255 RGB image
        image[:, :, 0] = z  # R
        image[:, :, 1] = r  # G
        image[:, :, 2] = g  # B

        return image

    # Read image
    g = fits.getdata(os.path.join(filenames[0]), memmap=False)
    r = fits.getdata(os.path.join(filenames[1]), memmap=False)
    z = fits.getdata(os.path.join(filenames[2]), memmap=False)

    # Contrast scaling / normalization
    I = (z + r + g) / 3.0

    length, width = g.shape
    image = np.empty([length, width, 3], dtype=dtype)

    # asinh(Q (I - minimum)/stretch)/Q

    # Options for contrast scaling
    if normalize.lower() == "lupton" or normalize.lower() == "luptonhc":
        z = z * np.arcsinh(stretch * Q * (I - m)) / (Q * I)
        r = r * np.arcsinh(stretch * Q * (I - m)) / (Q * I)
        g = g * np.arcsinh(stretch * Q * (I - m)) / (Q * I)

        # z = z*np.arcsinh(Q*(I - m)/stretch)/(Q)
        # r = r*np.arcsinh(Q*(I - m)/stretch)/(Q)
        # g = g*np.arcsinh(Q*(I - m)/stretch)/(Q)
        image[:, :, 0] = z  # R
        image[:, :, 1] = r  # G
        image[:, :, 2] = g  # B
        if do_norm:
            return norm(z, r, g)
        else:
            return image

    elif normalize.lower() == "astrolupton":
        image = make_lupton_rgb(z, r, g, minimum=m, stretch=stretch, Q=Q)
        return image

    elif normalize.lower() == "zscore":
        Imean = np.nanmean(I)
        Isigma = np.nanstd(I)

        z = A * (z - Imean - m) / Isigma
        r = A * (r - Imean - m) / Isigma
        g = A * (g - Imean - m) / Isigma

        image[:, :, 0] = z  # R
        image[:, :, 1] = r  # G
        image[:, :, 2] = g  # B
        if do_norm:
            return norm(z, r, g)
        else:
            return image

    elif normalize.lower() == "zscore_orig":
        zsigma = np.nanstd(z)
        rsigma = np.nanstd(r)
        gsigma = np.nanstd(g)

        z = A * (z - np.nanmean(z) - m) / zsigma
        r = A * (r - np.nanmean(r) - m) / rsigma
        g = A * (g - np.nanmean(g) - m) / gsigma

        image[:, :, 0] = z  # R
        image[:, :, 1] = r  # G
        image[:, :, 2] = g  # B

        return image

    elif normalize.lower() == "sinh":
        z = np.sinh((z - m))
        r = np.sinh((r - m))
        g = np.sinh((g - m))

    # sqrt(Q (I - minimum)/stretch)/Q
    elif normalize.lower() == "sqrt":
        z = z * np.sqrt((I - m) * Q / stretch) / I / stretch
        r = r * np.sqrt((I - m) * Q / stretch) / I / stretch
        g = g * np.sqrt((I - m) * Q / stretch) / I / stretch
        image[:, :, 0] = z  # R
        image[:, :, 1] = r  # G
        image[:, :, 2] = g  # B
        if do_norm:
            return norm(z, r, g)
        else:
            return image

    elif normalize.lower() == "sqrt-old":
        z = np.sqrt(z)
        r = np.sqrt(r)
        g = np.sqrt(g)
        image[:, :, 0] = z  # R
        image[:, :, 1] = r  # G
        image[:, :, 2] = g  # B
        if do_norm:
            return norm(z, r, g)
        else:
            return image

    elif normalize.lower() == "linear":
        z = A * (z - m)
        r = A * (r - m)
        g = A * (g - m)
        # z = (z - m)
        # r = (r - m)
        # g = (g - m)

        image[:, :, 0] = z  # R
        image[:, :, 1] = r  # G
        image[:, :, 2] = g  # B
        return image

    elif normalize.lower() == "normlinear":
        # image = np.empty([length, width, 3], dtype=dtype)

        z = A * (z - m)
        r = A * (r - m)
        g = A * (g - m)
        # z = (z - m)
        # r = (r - m)
        # g = (g - m)

        # image[:,:,0] = z # R
        # image[:,:,1] = r # G
        # image[:,:,2] = g # B
        # return image

    elif normalize.lower() == "astroluptonhc":
        image = make_lupton_rgb(z, r, g, minimum=m, stretch=stretch, Q=Q)
        factor = 2  # gives original image
        cenhancer = ImageEnhance.Contrast(Image.fromarray(image))
        im_output = cenhancer.enhance(factor)
        benhancer = ImageEnhance.Brightness(im_output)
        image = benhancer.enhance(factor)
        image = np.asarray(image)
        return image

    else:
        print("Normalize keyword not recognized.")


def read_image_decam(
    filename,
    normalize="lupton",
    stretch=0.5,
    Q=10,
    m=0,
    ceil_percentile=99.995,
    dtype=np.uint8,
    A=1e4,
    do_norm=False,
):
    """
    Read in a formatted simulated DECam image

    Parameters
    ----------
    filenames: list
        The list of g,r,i band files
    normalize: str
        The key word for the normalization scheme
    stretch, Q, m: float, int, float
        Parameters for lupton normalization
    ceil_percentile:
        If do_norm is true, cuts data off at this percentile
    dtype: numpy datatype
        data type of the output array
    A: float
        scaling factor for zscoring
    do_norm: boolean
        For normalizing top fit dtype range

    Returns
    -------
    Scaled image

    """

    def norm(z, r, g):
        max_RGB = np.nanpercentile([z, r, g], ceil_percentile)
        print(max_RGB)

        max_z = np.nanpercentile([z], ceil_percentile)
        max_r = np.nanpercentile([r], ceil_percentile)
        max_g = np.nanpercentile([g], ceil_percentile)

        # z = np.clip(z,None,max_RGB)
        # r = np.clip(r,None,max_RGB)
        # g = np.clip(g,None,max_RGB)

        # avoid saturation
        r = r / max_RGB
        g = g / max_RGB
        z = z / max_RGB
        # r = r/max_r; g = g/max_g; z = z/max_z

        # Rescale to 0-255 for dtype=np.uint8
        max_dtype = np.iinfo(dtype).max
        r = r * max_dtype
        g = g * max_dtype
        z = z * max_dtype

        # 0-255 RGB image
        image[:, :, 0] = z  # R
        image[:, :, 1] = r  # G
        image[:, :, 2] = g  # B

        return image

    # Read image
    g = fits.getdata(os.path.join(filename + "_g.fits"), memmap=False)
    r = fits.getdata(os.path.join(filename + "_r.fits"), memmap=False)
    z = fits.getdata(os.path.join(filename + "_z.fits"), memmap=False)

    # Contrast scaling / normalization
    I = (z + r + g) / 3.0

    length, width = g.shape
    image = np.empty([length, width, 3], dtype=dtype)

    # asinh(Q (I - minimum)/stretch)/Q

    # Options for contrast scaling
    if normalize.lower() == "lupton" or normalize.lower() == "luptonhc":
        z = z * np.arcsinh(stretch * Q * (I - m)) / (Q * I)
        r = r * np.arcsinh(stretch * Q * (I - m)) / (Q * I)
        g = g * np.arcsinh(stretch * Q * (I - m)) / (Q * I)

        # z = z*np.arcsinh(Q*(I - m)/stretch)/(Q)
        # r = r*np.arcsinh(Q*(I - m)/stretch)/(Q)
        # g = g*np.arcsinh(Q*(I - m)/stretch)/(Q)
        image[:, :, 0] = z  # R
        image[:, :, 1] = r  # G
        image[:, :, 2] = g  # B
        if do_norm:
            return norm(z, r, g)
        else:
            return image

    elif normalize.lower() == "astrolupton":
        image = make_lupton_rgb(z, r, g, minimum=m, stretch=stretch, Q=Q)
        return image

    elif normalize.lower() == "zscore":
        Imean = np.nanmean(I)
        Isigma = np.nanstd(I)

        z = A * (z - Imean - m) / Isigma
        r = A * (r - Imean - m) / Isigma
        g = A * (g - Imean - m) / Isigma

        image[:, :, 0] = z  # R
        image[:, :, 1] = r  # G
        image[:, :, 2] = g  # B
        if do_norm:
            return norm(z, r, g)
        else:
            return image

    elif normalize.lower() == "zscore_orig":
        zsigma = np.nanstd(z)
        rsigma = np.nanstd(r)
        gsigma = np.nanstd(g)

        z = A * (z - np.nanmean(z) - m) / zsigma
        r = A * (r - np.nanmean(r) - m) / rsigma
        g = A * (g - np.nanmean(g) - m) / gsigma

        image[:, :, 0] = z  # R
        image[:, :, 1] = r  # G
        image[:, :, 2] = g  # B

        return image

    elif normalize.lower() == "sinh":
        z = np.sinh((z - m))
        r = np.sinh((r - m))
        g = np.sinh((g - m))

    # sqrt(Q (I - minimum)/stretch)/Q
    elif normalize.lower() == "sqrt":
        z = z * np.sqrt((I - m) * Q / stretch) / I / stretch
        r = r * np.sqrt((I - m) * Q / stretch) / I / stretch
        g = g * np.sqrt((I - m) * Q / stretch) / I / stretch
        image[:, :, 0] = z  # R
        image[:, :, 1] = r  # G
        image[:, :, 2] = g  # B
        if do_norm:
            return norm(z, r, g)
        else:
            return image

    elif normalize.lower() == "sqrt-old":
        z = np.sqrt(z)
        r = np.sqrt(r)
        g = np.sqrt(g)
        image[:, :, 0] = z  # R
        image[:, :, 1] = r  # G
        image[:, :, 2] = g  # B
        if do_norm:
            return norm(z, r, g)
        else:
            return image

    elif normalize.lower() == "linear":
        z = A * (z - m)
        r = A * (r - m)
        g = A * (g - m)
        # z = (z - m)
        # r = (r - m)
        # g = (g - m)

        image[:, :, 0] = z  # R
        image[:, :, 1] = r  # G
        image[:, :, 2] = g  # B
        return image

    elif normalize.lower() == "normlinear":
        # image = np.empty([length, width, 3], dtype=dtype)

        z = A * (z - m)
        r = A * (r - m)
        g = A * (g - m)
        # z = (z - m)
        # r = (r - m)
        # g = (g - m)

        # image[:,:,0] = z # R
        # image[:,:,1] = r # G
        # image[:,:,2] = g # B
        # return image

    elif normalize.lower() == "astroluptonhc":
        image = make_lupton_rgb(z, r, g, minimum=m, stretch=stretch, Q=Q)
        factor = 2  # gives original image
        cenhancer = ImageEnhance.Contrast(Image.fromarray(image))
        im_output = cenhancer.enhance(factor)
        benhancer = ImageEnhance.Brightness(im_output)
        image = benhancer.enhance(factor)
        image = np.asarray(image)
        return image

    else:
        print("Normalize keyword not recognized.")


# ### Augment Data
def gaussblur(image):
    """
    Parameters
    ----------
    image: ndarray

    Returns
    -------
    augmented image

    """
    aug = iaa.GaussianBlur(sigma=(0.0, np.random.random_sample() * 4 + 2))
    return aug.augment_image(image)


def addelementwise16(image):
    """
    Parameters
    ----------
    image: ndarray

    Returns
    -------
    augmented image

    """
    aug = iaa.AddElementwise((-3276, 3276))
    return aug.augment_image(image)


def addelementwise8(image):
    """
    Parameters
    ----------
    image: ndarray

    Returns
    -------
    augmented image

    """
    aug = iaa.AddElementwise((-25, 25))
    return aug.augment_image(image)


def addelementwise(image):
    """
    Parameters
    ----------
    image: ndarray

    Returns
    -------
    augmented image

    """
    aug = iaa.AddElementwise((-image.max() * 0.1, image.max() * 0.1))
    return aug.augment_image(image)


# ### Format Astro R-CNN dataset for detectron instance segmentation models
def get_astro_dicts(img_dir):
    """
    This function reads in the scarlet model files and formats annotations for detectron2

    Parameters
    ----------
    img_dir: str
        Directory where the scarlet outputs are stored

    Returns
    -------
    dataset_dicts: list
        A list of dictionaries that contain annotations for train, test, val sets


    """

    # It's weird to call this img_dir
    set_dirs = sorted(glob.glob("%s/set_*" % img_dir))

    dataset_dicts = []

    # Loop through each set
    for idx, set_dir in enumerate(set_dirs[0:10]):
        record = {}

        mask_dir = os.path.join(img_dir, set_dir, "masks.fits")
        filename = os.path.join(img_dir, set_dir, "img")

        # Open each FITS image
        with fits.open(mask_dir, memmap=False, lazy_load_hdus=False) as hdul:
            sources = len(hdul)
            height, width = hdul[0].data.shape
            data = [hdu.data / np.max(hdu.data) for hdu in hdul]
            category_ids = [hdu.header["CLASS_ID"] for hdu in hdul]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        objs = []

        # Mask value thresholds per category_id
        thresh = [0.005 if i == 1 else 0.08 for i in category_ids]

        # Generate segmentation masks
        for i in range(sources):
            image = data[i]
            mask = np.zeros([height, width], dtype=np.uint8)
            # Create mask from threshold
            mask[:, :][image > thresh[i]] = 1
            # Smooth mask
            mask[:, :] = cv2.GaussianBlur(mask[:, :], (9, 9), 2)

            # https://github.com/facebookresearch/Detectron/issues/100
            contours, hierarchy = cv2.findContours(
                (mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            segmentation = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                contour = contour.flatten().tolist()
                # segmentation.append(contour)
                if len(contour) > 4:
                    segmentation.append(contour)
            # No valid countors
            if len(segmentation) == 0:
                continue

            # Add to dict
            obj = {
                "bbox": [x, y, w, h],
                "area": w * h,
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": segmentation,
                "category_id": category_ids[i] - 1,
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts
