import cv2
import numpy as np
from astropy.io import fits
from detectron2.structures import BoxMode

FILT_INX = 0


def annotate_dc2(images, mask, idx, filters):
    """
    This can needs to be customized to your training data format

    """

    record = {}

    # Open FITS image of first filter (each should have same shape)
    with fits.open(images[FILT_INX], memmap=False, lazy_load_hdus=False) as hdul:
        height, width = hdul[0].data.shape

    # Open each FITS mask image
    with fits.open(mask, memmap=False, lazy_load_hdus=False) as hdul:
        hdul = hdul[1:]
        sources = len(hdul)
        # Normalize data
        data = [hdu.data for hdu in hdul]
        category_ids = [0 for hdu in hdul]

        # ellipse_pars = [hdu.header["ELL_PARM"] for hdu in hdul]
        bbox = [list(map(int, hdu.header["BBOX"].split(","))) for hdu in hdul]
        area = [hdu.header["AREA"] for hdu in hdul]
        # imags = [hdu.header["IMAG"] for hdu in hdul]
        # oids = [hdu.header["hsc_oid"] for hdu in hdul]
        redshifts = [hdu.header["redshift"] for hdu in hdul]
        obj_ids = [hdu.header["objid"] for hdu in hdul]
        mag_is = [hdu.header["mag_i"] for hdu in hdul]

    tract = int(images[FILT_INX].split("_")[1])
    patch = (
        int(images[FILT_INX].split("_")[2].split("_")[0][0]),
        int(images[FILT_INX].split("_")[2].split("_")[0][-1]),
    )
    sp = int(images[FILT_INX].split("_")[3])
    record[f"filename"] = f"/home/g4merz/DC2/nersc_data/data/{tract}_{patch[0]},{patch[1]}_{sp}_images.npy"
    record["image_id"] = idx
    record["height"] = height
    record["width"] = width
    objs = []

    # Generate segmentation masks from model
    for i in range(sources):
        image = data[i]
        # Why do we need this?
        if len(image.shape) != 2:
            continue
        height_mask, width_mask = image.shape
        # Create mask from threshold
        mask = data[i]
        # Smooth mask
        # mask = cv2.GaussianBlur(mask, (9,9), 2)
        x, y, w, h = bbox[i]  # (x0, y0, w, h)

        # https://github.com/facebookresearch/Detectron/issues/100
        contours, hierarchy = cv2.findContours(
            (mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        segmentation = []
        for contour in contours:
            # contour = [x1, y1, ..., xn, yn]
            contour = contour.flatten()
            if len(contour) > 4:
                contour[::2] += x - w // 2
                contour[1::2] += y - h // 2
                segmentation.append(contour.tolist())
        # No valid countors
        if len(segmentation) == 0:
            print(i)
            continue

        # Add to dict
        obj = {
            "bbox": [x - w // 2, y - h // 2, w, h],
            "area": w * h,
            "bbox_mode": BoxMode.XYWH_ABS,
            "segmentation": segmentation,
            "category_id": category_ids[i],
            # "ellipse_pars": ellipse_pars[i],
            "redshift": redshifts[i],
            "obj_id": obj_ids[i],
            "mag_i": mag_is[i],
        }

        objs.append(obj)

    record["annotations"] = objs

    return record
