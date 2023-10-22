import numpy as np
import torch
from detectron2 import structures
from detectron2.structures import BoxMode
import deepdisc.astrodet.astrodet as toolkit
from deepdisc.inference.predictors import get_predictions

def get_matched_object_inds(dataset_dict,outputs):
    """Returns indices for matched pairs of ground truth and detected objects in an image

    Parameters
    ----------
    dataset_dict : dictionary
        the dictionary metadata for a single image

    Returns
    -------
        matched_gts: list(int)
            The indices of matched objects in the ground truth list
        matched_dts: list(int)
            The indices of matched objects in the detections list
        outputs: list(Intances)
            The list of detected object Instances
    """

    IOUthresh=0.5
    
    gt_boxes = np.array([a['bbox'] for a in dataset_dict['annotations']])
    # Convert to the mode model expects
    gt_boxes = BoxMode.convert(gt_boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
    gt_boxes = structures.Boxes(torch.Tensor(gt_boxes))
    pred_boxes = outputs["instances"].pred_boxes
    pred_boxes = pred_boxes.to('cpu')

    IOUs = structures.pairwise_iou(pred_boxes, gt_boxes).numpy()
    #matched_gts holds the indices of the ground truth annotations that correspond to the matched detections
    #matched_dts holds the indices of the detections that corresponds to the ground truth annotations
    matched_gts = []
    matched_dts =[]
    for i,dt in enumerate(IOUs):
        IOU = dt[dt.argmax()]  
        if IOU>=IOUthresh:
            matched_gts.append(dt.argmax())
            matched_dts.append(i) 

    return matched_gts, matched_dts

def get_matched_object_classes(dataset_dicts,predictor, **kwargs):

    IOUthresh=0.5
    
    #going to assume we only have one test image

    for d in dataset_dicts:
        outputs = get_predictions(d, predictor, **kwargs)
        matched_gts, matched_dts = get_matched_object_inds(d,outputs)
    true_classes =[]
    pred_classes =[]
    for gti,dti in zip(matched_gts,matched_dts):
        true_class = d['annotations'][int(gti)]['category_id']
        pred_class = outputs['instances'].pred_classes.cpu().detach().numpy()[int(dti)]
        true_classes.append(true_class)
        pred_classes.append(pred_class)
    
    return true_classes, pred_classes

