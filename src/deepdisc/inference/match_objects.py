import numpy as np
import torch
from detectron2 import structures
from detectron2.structures import BoxMode
from astropy.wcs import WCS

import deepdisc.astrodet.astrodet as toolkit
from deepdisc.inference.predictors import get_predictions, get_predictions_new


def get_matched_object_inds(dataset_dict, outputs):
    """Returns indices for matched pairs of ground truth and detected objects in an image

    Parameters
    ----------
    dataset_dict : dictionary
        The dictionary metadata for a single image

    Returns
    -------
        matched_gts: list(int)
            The indices of matched objects in the ground truth list
        matched_dts: list(int)
            The indices of matched objects in the detections list
        outputs: list(Intances)
            The list of detected object Instances
    """

    IOUthresh = 0.5

    gt_boxes = np.array([a["bbox"] for a in dataset_dict["annotations"]])
    # Convert to the mode model expects
    # Make sure the input bboxes are in XYWH mode so they can be converted here
    gt_boxes = BoxMode.convert(gt_boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
    gt_boxes = structures.Boxes(torch.Tensor(gt_boxes))
    pred_boxes = outputs["instances"].pred_boxes
    pred_boxes = pred_boxes.to("cpu")

    IOUs = structures.pairwise_iou(pred_boxes, gt_boxes).numpy()
    # matched_gts holds the indices of the ground truth annotations that correspond to the matched detections
    # matched_dts holds the indices of the detections that corresponds to the ground truth annotations
    matched_gts = []
    matched_dts = []
    for i, dt in enumerate(IOUs):
        IOU = dt[dt.argmax()]
        if IOU >= IOUthresh:
            matched_gts.append(dt.argmax())
            matched_dts.append(i)

    return matched_gts, matched_dts



def get_object_coords(dataset_dict, outputs):
    """Returns indices for matched pairs of ground truth and detected objects in an image

    Parameters
    ----------
    dataset_dict : dictionary
        The dictionary metadata containing the wcs for a single image

    Returns
    -------
        matched_gts: list(int)
            The indices of matched objects in the ground truth list
        matched_dts: list(int)
            The indices of matched objects in the detections list
        outputs: list(Intances)
            The list of detected object Instances
    """

    wcs = WCS(dataset_dict['wcs'])
    pred_boxes = outputs["instances"].pred_boxes
    pred_boxes = pred_boxes.to("cpu")
    
    centers = outputs['instances'].pred_boxes.get_centers().cpu().numpy()
    coords = wcs.pixel_to_world(centers[:,0],centers[:,1])
    
    ras = coords.ra.degree
    decs = coords.dec.degree
    return ras, decs


def get_matched_object_classes(dataset_dicts, imreader, key_mapper, predictor):
    """Returns object classes for matched pairs of ground truth and detected objects in an image

    Parameters
    ----------
    dataset_dicts : list[dict]
        The dictionary metadata for a test images
    imreader: ImageReader object
        An object derived from ImageReader base class to read the images.
    key_mapper: function
        The key_mapper should take a dataset_dict as input and return the key used by imreader
    predictor: AstroPredictor
        The predictor object used to make predictions on the test set

    Returns
    -------
        true_classes: list(int)
            The classes of matched objects in the ground truth list
        pred_classes: list(int)
            The classes of matched objects in the detections list
    """
    true_classes = []
    pred_classes = []
    for d in dataset_dicts:
        outputs = get_predictions(d, imreader, key_mapper, predictor)
        matched_gts, matched_dts = get_matched_object_inds(d, outputs)

        for gti, dti in zip(matched_gts, matched_dts):
            true_class = d["annotations"][int(gti)]["category_id"]
            pred_class = outputs["instances"].pred_classes.cpu().detach().numpy()[int(dti)]
            true_classes.append(true_class)
            pred_classes.append(pred_class)

    return true_classes, pred_classes


def get_matched_object_classes_new(dataset_dicts, predictor):
    """Returns object classes for matched pairs of ground truth and detected objects test images
    assuming the dataset_dicts have the image HxWxC in the 'image_shaped' field

    Parameters
    ----------
    dataset_dicts : list[dict]
        The dictionary metadata for a test images
    predictor: AstroPredictor
        The predictor object used to make predictions on the test set

    Returns
    -------
        true_classes: list(int)
            The classes of matched objects in the ground truth list
        pred_classes: list(int)
            The classes of matched objects in the detections list
    """
    true_classes = []
    pred_classes = []
    for d in dataset_dicts:
        outputs = get_predictions_new(d, predictor)
        matched_gts, matched_dts = get_matched_object_inds(d, outputs)

        for gti, dti in zip(matched_gts, matched_dts):
            true_class = d["annotations"][int(gti)]["category_id"]
            pred_class = outputs["instances"].pred_classes.cpu().detach().numpy()[int(dti)]
            true_classes.append(true_class)
            pred_classes.append(pred_class)

    return true_classes, pred_classes


def get_matched_z_pdfs(dataset_dicts, imreader, key_mapper, predictor, ids=False, blendedness=False):
    """Returns redshift pdfs for matched pairs of ground truth and detected objects test images

    Parameters
    ----------
    dataset_dicts : list[dict]
        The dictionary metadata for a test images
    imreader: ImageReader object
        An object derived from ImageReader base class to read the images.
    key_mapper: function
        The key_mapper should take a dataset_dict as input and return the key used by imreader
    predictor: AstroPredictor
        The predictor object used to make predictions on the test set

    Returns
    -------
        z_trues: list(float)
            The redshifts of matched objects in the ground truth list
        z_preds: list(array(float))
            The redshift pdfs of matched objects in the detections list
    """

    IOUthresh = 0.5
    zs = np.linspace(-1, 5.0, 200)

    ztrues = []
    zpreds = []
    matched_ids=[]
    matched_bnds=[]

    for d in dataset_dicts:
        outputs = get_predictions(d, imreader, key_mapper, predictor)
        matched_gts, matched_dts = get_matched_object_inds(d, outputs)

        for gti, dti in zip(matched_gts, matched_dts):
            ztrue = d["annotations"][int(gti)]["redshift"]
            pdf = np.exp(outputs["instances"].pred_redshift_pdf[int(dti)].cpu().numpy())
            if ids:
                oid = d["annotations"][int(gti)]["obj_id"]
                matched_ids.append(oid)
            if blendedness:
                bnd = d["annotations"][int(gti)]["blendedness"]
                matched_bnds.append(bnd)

            ztrues.append(ztrue)
            zpreds.append(pdf)

    return ztrues, zpreds, matched_ids, matched_bnds


def get_matched_z_pdfs_new(dataset_dicts, predictor, ids=False, blendedness=False):
    """Returns redshift pdfs for matched pairs of ground truth and detected objects test images
    assuming the dataset_dicts have the image HxWxC in the 'image_shaped' field

    Parameters
    ----------
    dataset_dicts : list[dict]
        The dictionary metadata for a test images
    predictor: AstroPredictor
        The predictor object used to make predictions on the test set

    Returns
    -------
        z_trues: list(float)
            The redshifts of matched objects in the ground truth list
        z_preds: list(array(float))
            The redshift pdfs of matched objects in the detections list
    """
    IOUthresh = 0.5
    zs = np.linspace(-1, 5.0, 200)

    ztrues = []
    zpreds = []
    matched_ids=[]
    matched_bnds=[]

    for d in dataset_dicts:
        outputs = get_predictions_new(d, predictor)
        matched_gts, matched_dts = get_matched_object_inds(d, outputs)

        for gti, dti in zip(matched_gts, matched_dts):
            ztrue = d["annotations"][int(gti)]["redshift"]
            pdf = np.exp(outputs["instances"].pred_redshift_pdf[int(dti)].cpu().numpy())
            if ids:
                oid = d["annotations"][int(gti)]["obj_id"]
                matched_ids.append(oid)
            if blendedness:
                bnd = d["annotations"][int(gti)]["blendedness"]
                matched_bnds.append(bnd)


            ztrues.append(ztrue)
            zpreds.append(pdf)

    return ztrues, zpreds, matched_ids, matched_bnds


def get_matched_z_points_new(dataset_dicts, predictor):
    """Returns redshift point estimates for matched pairs of ground truth and detected objects test images
    assuming the dataset_dicts have the image in the 'image_shaped' field

    Parameters
    ----------
    dataset_dicts : list[dict]
        The dictionary metadata for a test images
    predictor: AstroPredictor
        The predictor object used to make predictions on the test set

    Returns
    -------
        z_trues: list(float)
            The redshifts of matched objects in the ground truth list
        z_preds: list(array(float))
            The redshift pdfs of matched objects in the detections list
    """
    IOUthresh = 0.5
    zs = np.linspace(-1, 5.0, 200)

    ztrues = []
    zpreds = []

    for d in dataset_dicts:
        outputs = get_predictions_new(d, predictor)
        matched_gts, matched_dts = get_matched_object_inds(d, outputs)

        for gti, dti in zip(matched_gts, matched_dts):
            ztrue = d["annotations"][int(gti)]["redshift"]
            zpred = outputs["instances"].pred_redshift[int(dti)].cpu().numpy()

            ztrues.append(ztrue)
            zpreds.append(zpred)

    return ztrues, zpreds



def run_batched_match_class(dataloader, predictor):
    """
    Test function not yet implemented for batch prediction

    """
    true_classes = []
    pred_classes = []
    with torch.no_grad():
        for i, dataset_dicts in enumerate(dataloader):
            batched_outputs = predictor.model(dataset_dicts)
            for outputs,d in zip(batched_outputs, dataset_dicts):
                matched_gts, matched_dts = get_matched_object_inds(d, outputs)
                for gti, dti in zip(matched_gts, matched_dts):
                    true_class = d["annotations"][int(gti)]["category_id"]
                    pred_class = outputs["instances"].pred_classes.cpu().detach().numpy()[int(dti)]
                    true_classes.append(true_class)
                    pred_classes.append(pred_class)
    return true_classes, pred_classes


def run_batched_match_redshift(dataloader, predictor, ids=False, blendedness=False):
    """
    Test function not yet implemented for batch prediction

    """
    ztrues = []
    zpreds = []
    matched_ids = []
    matched_bnds = []

    with torch.no_grad():
        for i, dataset_dicts in enumerate(dataloader):
            batched_outputs = predictor.model(dataset_dicts)
            for outputs,d in zip(batched_outputs, dataset_dicts):
                matched_gts, matched_dts = get_matched_object_inds(d, outputs)
                for gti, dti in zip(matched_gts, matched_dts):
                    ztrue = d["annotations"][int(gti)]["redshift"]
                    pdf = np.exp(outputs["instances"].pred_redshift_pdf[int(dti)].cpu().numpy())
                    ztrues.append(ztrue)
                    zpreds.append(pdf)
                    if ids:
                        oid = d["annotations"][int(gti)]["obj_id"]
                        matched_ids.append(oid)
                    if blendedness:
                        bnd = d["annotations"][int(gti)]["blendedness"]
                        matched_bnds.append(bnd)


    return ztrues, zpreds, matched_ids, matched_bnds



def run_batched_get_object_coords(dataloader, predictor, oclass=True, gmm=False):
    """
    Test function not yet implemented for batch prediction

    """
    zpreds = []
    all_decs = []
    all_ras = []
    scores = []
    
    if gmm:
        gmms =[]
    if oclass:
        oclasses=[]

    with torch.no_grad():
        for i, dataset_dicts in enumerate(dataloader):
            batched_outputs = predictor.model(dataset_dicts)
            for outputs,d in zip(batched_outputs, dataset_dicts):
                ras,decs = get_object_coords(d, outputs)
                #all_ras.append(*ras)
                #all_decs.append(*decs)
                list(map(all_ras.append, ras))
                list(map(all_decs.append, decs))

               
                for dti in range(len(outputs['instances'])):
                    #ztrue = d["annotations"][int(gti)]["redshift"]
                    pdf = np.exp(outputs["instances"].pred_redshift_pdf[int(dti)].cpu().numpy())
                    zpreds.append(pdf)
                    
                    gmms.append(outputs['instances'].pred_gmm.cpu()[int(dti)].cpu().numpy())
                    oclasses.append(outputs['instances'].pred_classes.cpu()[int(dti)].numpy())
                    scores.append(outputs['instances'].scores.cpu()[int(dti)].numpy())

                    
    if gmm:
        return zpreds, all_ras, all_decs, oclasses, gmms, scores
    
    else:
        return zpreds, all_ras, all_decs, oclasses, scores
