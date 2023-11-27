
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random
import deepdisc
from deepdisc.data_format.image_readers import DC2ImageReader


def flatten_dc2(ddicts):
    """Reads in large cutouts and creates postage stamp images centered on individual objects
    Flattens these images+metadata into one tabular dataset. Ignores segmentation maps.

    Parameters
    ----------
    ddicts : list[dicts]
        The metadata dictionaries for large cutouts with multiple objects.
    
    Returns
    -------
    flattened_data : np array
        The images + metadata that have now been flattened into a tabular array.  
        Each row has 98316 columns (6x128x128 + 12 metadata values)
    """
    
    i=0
    images=[]
    metadatas = []
    image_reader = DC2ImageReader(norm="raw")

    for d in ddicts:
        filename= d[f"filename"]
        for a in d['annotations']:
            new_dict = {}
            new_dict["image_id"] = 1
            new_dict["height"] = 128
            new_dict["width"] = 128

            x = a['bbox'][0]
            y = a['bbox'][1]
            w = a['bbox'][2]
            h = a['bbox'][3]

            xnew = x+w//2-64
            ynew = y+h//2-64

            if xnew<0 or ynew <0 or xnew+128>d['height'] or ynew+128>d['height'] or a['mag_i']>25.3:
                continue

            bxnew = x-(x+w//2 - 64)
            bynew = y-(y+h//2 - 64)
            
            image = image_reader(filename)
            image = np.transpose(image, axes=(2, 0, 1))


            imagecut = image[:,ynew:ynew+128,xnew:xnew+128]

            images.append(imagecut.flatten())

            metadata =[128,128,i,bxnew,bynew,w,h,1,a['category_id'],a['redshift'],a['obj_id'],a['mag_i']]
            metadatas.append(metadata)
            i+=1
            
    images = np.array(images)
    metadatas = np.array(metadatas)
    
    flattened_data = []
    for image,metadata in zip(images,metadatas):
        flatdat = np.concatenate((image,metadata))
        flattened_data.append(flatdat)

            
    return flattened_data
                    
    




