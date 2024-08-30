import copy

import detectron2.data as data
import detectron2.data.transforms as T
import numpy as np
import torch
from detectron2.data import detection_utils as utils

import deepdisc.astrodet.astrodet as toolkit
import deepdisc.astrodet.detectron as detectron_addons
from astropy.wcs import WCS


class DataMapper:
    """Base class that will map data to the format necessary for the model

    To implement a data mapper for a new class, the derived class needs to have an
    __init__() function that calls super().__init__(*args, **kwargs)
    and a custom version of map_data().
    """

    def __init__(self, imreader=None, key_mapper=None, augmentations=None):
        """
        Parameters
        ----------
        imreader : ImageReader
            The class that will load and contrast scale the images.
            They can be stored separately from the dataset or with it.
        key_mapper : function
            The function that takes the data set and returns the key that will be used to load the image.
            If the image is stored with the dataset, this is not needed
            Default = None
        augmentations : detectron2 AugmentationList or a detectron_addons.KRandomAugmentationList
            The list of augmentations to apply to the image
            Default = None
        """
        self.IR = imreader
        self.km = key_mapper
        self.augmentations = augmentations

    def map_data(self, data):
        return data


class DictMapper(DataMapper):
    """Class that will map COCO dictionary data to the format necessary for the model"""

    def __init__(self, *args, **kwargs):
        # Pass arguments to the parent function.
        super().__init__(*args, **kwargs)

    def map_data(self, dataset_dict):
        """Map COCO dict data to the correct format

        Parameters
        ----------
        dataset_dict: dict
            a dictionary of COCO formatted metadata

        Returns
        -------
        reformatted dictionary including image and instances
        """

        dataset_dict = copy.deepcopy(dataset_dict)
        key = self.km(dataset_dict)
        image = self.IR(key)

        # Data Augmentation
        auginput = T.AugInput(image)
        # Transformations to model shapes
        if self.augmentations is not None:
            augs = self.augmentations(image)
        else:
            augs = T.AugmentationList([])
        transform = augs(auginput)
        image = torch.from_numpy(auginput.image.copy().transpose(2, 0, 1))
        annos = [
            utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
            for annotation in dataset_dict.pop("annotations")
        ]

        instances = utils.annotations_to_instances(annos, image.shape[1:])
        instances = utils.filter_empty_instances(instances)

        return {
            # create the format that the model expects
            "image": image,
            "image_shaped": auginput.image,
            "height": image.shape[1],
            "width": image.shape[2],
            "image_id": dataset_dict["image_id"],
            "instances": instances,
        }

    
class MagRedshiftDictMapper(DataMapper):
    def __init__(self, *args, **kwargs):
        # Pass arguments to the parent function.
        super().__init__(*args, **kwargs)

    def map_data(self, dataset_dict):
        """Map COCO dict data to the correct format, add ground truth redhshift

        Parameters
        ----------
        dataset_dict: dict
            a dictionary of COCO formatted metadata

        Returns
        -------
        reformatted dictionary including image and instances+redshift
        """

        dataset_dict = copy.deepcopy(dataset_dict)
        key = self.km(dataset_dict)
        image = self.IR(key)

        # Data Augmentation
        auginput = T.AugInput(image)
        # Transformations to model shapes
        if self.augmentations is not None:
            augs = self.augmentations(image)
        else:
            augs = T.AugmentationList([])

        transform = augs(auginput)
        image = torch.from_numpy(auginput.image.copy().transpose(2, 0, 1))

        annos = [
            utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
            for annotation in dataset_dict.pop("annotations")
            if annotation["redshift"] != 0.0 ]# and annotation["mag_i"] < 25.3]

        instances = utils.annotations_to_instances(annos, image.shape[1:])

        instances.gt_magi = torch.tensor([a["mag_i"] for a in annos])
        instances.gt_redshift = torch.tensor([a["redshift"] for a in annos])

        instances = utils.filter_empty_instances(instances)

        return {
            # create the format that the model expects
            "image": image,
            "image_shaped": auginput.image,
            "height": image.shape[1],
            "width": image.shape[2],
            "image_id": dataset_dict["image_id"],
            "instances": instances,
            "annotations": annos
        }


    
    
class MagRedshiftDictMapperNlimWithStarsSpecSelect(DataMapper):
    def __init__(self, *args, **kwargs):
        # Pass arguments to the parent function.
        super().__init__(*args, **kwargs)

    def map_data(self, dataset_dict):
        """Map COCO dict data to the correct format, add ground truth redhshift

        Parameters
        ----------
        dataset_dict: dict
            a dictionary of COCO formatted metadata

        Returns
        -------
        reformatted dictionary including image and instances+redshift
        """

        dataset_dict = copy.deepcopy(dataset_dict)
        key = self.km(dataset_dict)
        image = self.IR(key)

        # Data Augmentation
        auginput = T.AugInput(image)
        # Transformations to model shapes
        if self.augmentations is not None:
            augs = self.augmentations(image)
        else:
            augs = T.AugmentationList([])

        transform = augs(auginput)
        image = torch.from_numpy(auginput.image.copy().transpose(2, 0, 1))

        goldinds =[]

        goldannos = []
        fillannos = []
        for i,annotation in enumerate(dataset_dict['annotations']):
            if annotation['redshift']==0:
                annotation['category_id'] = 1
                
            if annotation["mag_i"] <= 25.3 :
                goldannos.append(utils.transform_instance_annotations(annotation, [transform], image.shape[1:]))
                goldinds.append(i)
            elif annotation["mag_i"] > 25.3:
                fillannos.append(utils.transform_instance_annotations(annotation, [transform], image.shape[1:]))

        Nfill=500-len(goldinds)
        fillannos = list(np.random.choice(fillannos,Nfill))

        annos = goldannos+fillannos
        
        instances = utils.annotations_to_instances(annos, image.shape[1:])

        instances.gt_magi = torch.tensor([a["mag_i"] for a in annos])
        instances.gt_redshift = torch.tensor([a["redshift"] for a in annos])
        instances.gt_spec_select = torch.tensor([a["spec_selection"] for a in annos])

        instances = utils.filter_empty_instances(instances)
        
        if 'wcs' in dataset_dict.keys():
            wcs = dataset_dict["wcs"]
        else:
            wcs=None
        
        return {
            # create the format that the model expects
            "image": image,
            "image_shaped": auginput.image,
            "height": image.shape[1],
            "width": image.shape[2],
            "image_id": dataset_dict["image_id"],
            "instances": instances,
            "annotations": annos,
            "wcs": wcs
        }
    
    
class MagRedshiftDictMapperNlim(DataMapper):
    def __init__(self, *args, **kwargs):
        # Pass arguments to the parent function.
        super().__init__(*args, **kwargs)

    def map_data(self, dataset_dict):
        """Map COCO dict data to the correct format, add ground truth redhshift

        Parameters
        ----------
        dataset_dict: dict
            a dictionary of COCO formatted metadata

        Returns
        -------
        reformatted dictionary including image and instances+redshift
        """

        dataset_dict = copy.deepcopy(dataset_dict)
        key = self.km(dataset_dict)
        image = self.IR(key)

        # Data Augmentation
        auginput = T.AugInput(image)
        # Transformations to model shapes
        if self.augmentations is not None:
            augs = self.augmentations(image)
        else:
            augs = T.AugmentationList([])

        transform = augs(auginput)
        image = torch.from_numpy(auginput.image.copy().transpose(2, 0, 1))

        goldinds =[]

        goldannos = []
        fillannos = []
        for i,annotation in enumerate(dataset_dict['annotations']):
            if annotation["redshift"] != 0.0  and annotation["mag_i"] <= 25.3 :
                goldannos.append(utils.transform_instance_annotations(annotation, [transform], image.shape[1:]))
                goldinds.append(i)
            elif annotation["redshift"] != 0.0  and annotation["mag_i"] > 25.3:
                fillannos.append(utils.transform_instance_annotations(annotation, [transform], image.shape[1:]))

        Nfill=500-len(goldinds)
        fillannos = list(np.random.choice(fillannos,Nfill))

        annos = goldannos+fillannos
        
        instances = utils.annotations_to_instances(annos, image.shape[1:])

        instances.gt_magi = torch.tensor([a["mag_i"] for a in annos])
        instances.gt_redshift = torch.tensor([a["redshift"] for a in annos])

        instances = utils.filter_empty_instances(instances)

        return {
            # create the format that the model expects
            "image": image,
            "image_shaped": auginput.image,
            "height": image.shape[1],
            "width": image.shape[2],
            "image_id": dataset_dict["image_id"],
            "instances": instances,
            "annotations": annos
        }
    
    
    
class MagRedshiftDictMapperNlimWithStars(DataMapper):
    def __init__(self, *args, **kwargs):
        # Pass arguments to the parent function.
        super().__init__(*args, **kwargs)

    def map_data(self, dataset_dict):
        """Map COCO dict data to the correct format, add ground truth redhshift

        Parameters
        ----------
        dataset_dict: dict
            a dictionary of COCO formatted metadata

        Returns
        -------
        reformatted dictionary including image and instances+redshift
        """

        dataset_dict = copy.deepcopy(dataset_dict)
        key = self.km(dataset_dict)
        image = self.IR(key)

        # Data Augmentation
        auginput = T.AugInput(image)
        # Transformations to model shapes
        if self.augmentations is not None:
            augs = self.augmentations(image)
        else:
            augs = T.AugmentationList([])

        transform = augs(auginput)
        image = torch.from_numpy(auginput.image.copy().transpose(2, 0, 1))

        goldinds =[]

        goldannos = []
        fillannos = []
        for i,annotation in enumerate(dataset_dict['annotations']):
            if annotation['redshift']==0:
                annotation['category_id'] = 1
                
            if annotation["mag_i"] <= 25.3 :
                goldannos.append(utils.transform_instance_annotations(annotation, [transform], image.shape[1:]))
                goldinds.append(i)
            elif annotation["mag_i"] > 25.3:
                fillannos.append(utils.transform_instance_annotations(annotation, [transform], image.shape[1:]))

        Nfill=500-len(goldinds)
        fillannos = list(np.random.choice(fillannos,Nfill))

        annos = goldannos+fillannos
        
        instances = utils.annotations_to_instances(annos, image.shape[1:])

        instances.gt_magi = torch.tensor([a["mag_i"] for a in annos])
        instances.gt_redshift = torch.tensor([a["redshift"] for a in annos])

        instances = utils.filter_empty_instances(instances)
        
        if 'wcs' in dataset_dict.keys():
            wcs = dataset_dict["wcs"]
        else:
            wcs=None
        
        return {
            # create the format that the model expects
            "image": image,
            "image_shaped": auginput.image,
            "height": image.shape[1],
            "width": image.shape[2],
            "image_id": dataset_dict["image_id"],
            "instances": instances,
            "annotations": annos,
            "wcs": wcs
        }
    
class MagRedshiftDictMapperWithStars(DataMapper):
    def __init__(self, *args, **kwargs):
        # Pass arguments to the parent function.
        super().__init__(*args, **kwargs)

    def map_data(self, dataset_dict):
        """Map COCO dict data to the correct format, add ground truth redhshift

        Parameters
        ----------
        dataset_dict: dict
            a dictionary of COCO formatted metadata

        Returns
        -------
        reformatted dictionary including image and instances+redshift
        """

        dataset_dict = copy.deepcopy(dataset_dict)
        key = self.km(dataset_dict)
        image = self.IR(key)

        # Data Augmentation
        auginput = T.AugInput(image)
        # Transformations to model shapes
        if self.augmentations is not None:
            augs = self.augmentations(image)
        else:
            augs = T.AugmentationList([])

        transform = augs(auginput)
        image = torch.from_numpy(auginput.image.copy().transpose(2, 0, 1))
        
        annos = dataset_dict['annotations']
        
        instances = utils.annotations_to_instances(annos, image.shape[1:])

        instances.gt_magi = torch.tensor([a["mag_i"] for a in annos])
        instances.gt_redshift = torch.tensor([a["redshift"] for a in annos])

        instances = utils.filter_empty_instances(instances)
        
        if 'wcs' in dataset_dict.keys():
            wcs = dataset_dict["wcs"]
        else:
            wcs=None
        
        return {
            # create the format that the model expects
            "image": image,
            "image_shaped": auginput.image,
            "height": image.shape[1],
            "width": image.shape[2],
            "image_id": dataset_dict["image_id"],
            "instances": instances,
            "annotations": annos,
            "wcs": wcs
        }

class RedshiftDictMapper(DataMapper):
    def __init__(self, *args, **kwargs):
        # Pass arguments to the parent function.
        super().__init__(*args, **kwargs)

    def map_data(self, dataset_dict):
        """Map COCO dict data to the correct format, add ground truth redhshift

        Parameters
        ----------
        dataset_dict: dict
            a dictionary of COCO formatted metadata

        Returns
        -------
        reformatted dictionary including image and instances+redshift
        """

        dataset_dict = copy.deepcopy(dataset_dict)
        key = self.km(dataset_dict)
        image = self.IR(key)

        # Data Augmentation
        auginput = T.AugInput(image)
        # Transformations to model shapes
        if self.augmentations is not None:
            augs = self.augmentations(image)
        else:
            augs = T.AugmentationList([])

        transform = augs(auginput)
        image = torch.from_numpy(auginput.image.copy().transpose(2, 0, 1))

        annos = [
            utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
            for annotation in dataset_dict.pop("annotations")
            #if annotation["redshift"] != 0.0
        ]

        instances = utils.annotations_to_instances(annos, image.shape[1:])

        instances.gt_redshift = torch.tensor([a["redshift"] for a in annos])
        
        instances.gt_imageid = torch.tensor([dataset_dict["image_id"] for a in annos])

        instances = utils.filter_empty_instances(instances)
        
        
        if 'wcs' in dataset_dict.keys():
            wcs = dataset_dict["wcs"]
        else:
            wcs=None
        
        return {
            # create the format that the model expects
            "image": image,
            "image_shaped": auginput.image,
            "height": image.shape[1],
            "width": image.shape[2],
            "image_id": dataset_dict["image_id"],
            "instances": instances,
            #"annotations": annos
            "wcs": wcs
        }
    
    
class GoldRedshiftDictMapper(DataMapper):
    def __init__(self, *args, **kwargs):
        # Pass arguments to the parent function.
        super().__init__(*args, **kwargs)

    def map_data(self, dataset_dict):
        """Map COCO dict data to the correct format, add ground truth redhshift

        Parameters
        ----------
        dataset_dict: dict
            a dictionary of COCO formatted metadata

        Returns
        -------
        reformatted dictionary including image and instances+redshift
        """

        dataset_dict = copy.deepcopy(dataset_dict)
        key = self.km(dataset_dict)
        image = self.IR(key)

        # Data Augmentation
        auginput = T.AugInput(image)
        # Transformations to model shapes
        if self.augmentations is not None:
            augs = self.augmentations(image)
        else:
            augs = T.AugmentationList([])

        transform = augs(auginput)
        image = torch.from_numpy(auginput.image.copy().transpose(2, 0, 1))

        annos = [
            utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
            for annotation in dataset_dict.pop("annotations")
            if annotation["redshift"] != 0.0 and annotation["mag_i"] < 25.3
        ]

        instances = utils.annotations_to_instances(annos, image.shape[1:])

        instances.gt_redshift = torch.tensor([a["redshift"] for a in annos])

        instances = utils.filter_empty_instances(instances)

        return {
            # create the format that the model expects
            "image": image,
            "image_shaped": auginput.image,
            "height": image.shape[1],
            "width": image.shape[2],
            "image_id": dataset_dict["image_id"],
            "instances": instances,
            "annotations": annos
        }

    
class RedshiftEBVDictMapper(DataMapper):
    def __init__(self, *args, **kwargs):
        # Pass arguments to the parent function.
        super().__init__(*args, **kwargs)

    def map_data(self, dataset_dict):
        """Map COCO dict data to the correct format, add ground truth redhshift

        Parameters
        ----------
        dataset_dict: dict
            a dictionary of COCO formatted metadata

        Returns
        -------
        reformatted dictionary including image and instances+redshift
        """

        dataset_dict = copy.deepcopy(dataset_dict)
        key = self.km(dataset_dict)
        image = self.IR(key)

        # Data Augmentation
        auginput = T.AugInput(image)
        # Transformations to model shapes
        if self.augmentations is not None:
            augs = self.augmentations(image)
        else:
            augs = T.AugmentationList([])

        transform = augs(auginput)
        image = torch.from_numpy(auginput.image.copy().transpose(2, 0, 1))

        annos = [
            utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
            for annotation in dataset_dict.pop("annotations")
            if annotation["redshift"] != 0.0
        ]

        instances = utils.annotations_to_instances(annos, image.shape[1:])

        instances.gt_redshift = torch.tensor([a["redshift"] for a in annos])
        instances.gt_ebv = torch.tensor([a["EBV"] for a in annos])

        instances = utils.filter_empty_instances(instances)

        return {
            # create the format that the model expects
            "image": image,
            "image_shaped": auginput.image,
            "height": image.shape[1],
            "width": image.shape[2],
            "image_id": dataset_dict["image_id"],
            "instances": instances,
            #"annotations": annos
        }
    
    
class GoldRedshiftDictMapperEval(DataMapper):
    def __init__(self, *args, **kwargs):
        # Pass arguments to the parent function.
        super().__init__(*args, **kwargs)

    def map_data(self, dataset_dict):
        """Map COCO dict data to the correct format, add ground truth redhshift

        Parameters
        ----------
        dataset_dict: dict
            a dictionary of COCO formatted metadata

        Returns
        -------
        reformatted dictionary including image and instances+redshift
        """

        dataset_dict = copy.deepcopy(dataset_dict)
        key = self.km(dataset_dict)
        image = self.IR(key)

        # Data Augmentation
        auginput = T.AugInput(image)
        # Transformations to model shapes
        if self.augmentations is not None:
            augs = self.augmentations(image)
        else:
            augs = T.AugmentationList([])

        transform = augs(auginput)
        image = torch.from_numpy(auginput.image.copy().transpose(2, 0, 1))

        annotations = [annotation for annotation in dataset_dict["annotations"]
                       if annotation["redshift"] != 0.0 and annotation["mag_i"] < 25.3]
        
        #annos = [
        #    utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
        #    for annotation in dataset_dict.pop("annotations")
        #    if annotation["redshift"] != 0.0
        #]

        #instances = utils.annotations_to_instances(annos, image.shape[1:])

        #instances.gt_redshift = torch.tensor([a["redshift"] for a in annos])

        #instances = utils.filter_empty_instances(instances)

        return {
            # create the format that the model expects
            "image": image,
            "image_shaped": auginput.image,
            "height": image.shape[1],
            "width": image.shape[2],
            "image_id": dataset_dict["image_id"],
            #"instances": instances,
            "annotations": annotations
        }



    
class RedshiftDictMapperEval(DataMapper):
    def __init__(self, *args, **kwargs):
        # Pass arguments to the parent function.
        super().__init__(*args, **kwargs)

    def map_data(self, dataset_dict):
        """Map COCO dict data to the correct format, add ground truth redhshift

        Parameters
        ----------
        dataset_dict: dict
            a dictionary of COCO formatted metadata

        Returns
        -------
        reformatted dictionary including image and instances+redshift
        """

        dataset_dict = copy.deepcopy(dataset_dict)
        key = self.km(dataset_dict)
        image = self.IR(key)

        # Data Augmentation
        auginput = T.AugInput(image)
        # Transformations to model shapes
        if self.augmentations is not None:
            augs = self.augmentations(image)
        else:
            augs = T.AugmentationList([])

        transform = augs(auginput)
        image = torch.from_numpy(auginput.image.copy().transpose(2, 0, 1))

        annotations = [annotation for annotation in dataset_dict["annotations"]
                       if annotation["redshift"] != 0.0]
        
        #annos = [
        #    utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
        #    for annotation in dataset_dict.pop("annotations")
        #    if annotation["redshift"] != 0.0
        #]

        #instances = utils.annotations_to_instances(annos, image.shape[1:])

        #instances.gt_redshift = torch.tensor([a["redshift"] for a in annos])

        #instances = utils.filter_empty_instances(instances)

        return {
            # create the format that the model expects
            "image": image,
            "image_shaped": auginput.image,
            "height": image.shape[1],
            "width": image.shape[2],
            "image_id": dataset_dict["image_id"],
            #"instances": instances,
            "annotations": annotations
        }


class WCSDictmapper(DataMapper):
    def __init__(self, *args, **kwargs):
        # Pass arguments to the parent function.
        super().__init__(*args, **kwargs)

    def map_data(self, dataset_dict):
        """Map COCO dict data to the correct format, add ground truth redhshift

        Parameters
        ----------
        dataset_dict: dict
            a dictionary of COCO formatted metadata

        Returns
        -------
        reformatted dictionary including image and instances+redshift
        """

        dataset_dict = copy.deepcopy(dataset_dict)
        key = self.km(dataset_dict)
        image = self.IR(key)

        # Data Augmentation
        auginput = T.AugInput(image)
        # Transformations to model shapes
        if self.augmentations is not None:
            augs = self.augmentations(image)
        else:
            augs = T.AugmentationList([])

        transform = augs(auginput)
        image = torch.from_numpy(auginput.image.copy().transpose(2, 0, 1))


        return {
            # create the format that the model expects
            "image": image,
            "image_shaped": auginput.image,
            "height": image.shape[1],
            "width": image.shape[2],
            #"image_id": dataset_dict["image_id"],
            #"instances": instances,
            "wcs": dataset_dict['wcs']
        }

    
class RedshiftDictMapperRemove(DataMapper):
    def __init__(self, *args, **kwargs):
        # Pass arguments to the parent function.
        super().__init__(*args, **kwargs)

    def map_data(self, dataset_dict):
        """Map COCO dict data to the correct format, add ground truth redhshift

        Parameters
        ----------
        dataset_dict: dict
            a dictionary of COCO formatted metadata

        Returns
        -------
        reformatted dictionary including image and instances+redshift
        """

        dataset_dict = copy.deepcopy(dataset_dict)
        key = self.km(dataset_dict)
        image = self.IR(key)
        

        # Data Augmentation
        auginput = T.AugInput(image)
        # Transformations to model shapes
        if self.augmentations is not None:
            augs = self.augmentations(image)
        else:
            augs = T.AugmentationList([])

        transform = augs(auginput)
        image = torch.from_numpy(auginput.image.copy().transpose(2, 0, 1))
        
        image[1]=torch.zeros(image[1].shape)

        annos = [
            utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
            for annotation in dataset_dict.pop("annotations")
            #if annotation["redshift"] != 0.0
        ]

        instances = utils.annotations_to_instances(annos, image.shape[1:])

        instances.gt_redshift = torch.tensor([a["redshift"] for a in annos])
        
        instances.gt_imageid = torch.tensor([dataset_dict["image_id"] for a in annos])

        instances = utils.filter_empty_instances(instances)
        
        
        if 'wcs' in dataset_dict.keys():
            wcs = dataset_dict["wcs"]
        else:
            wcs=None
        
        return {
            # create the format that the model expects
            "image": image,
            "image_shaped": auginput.image,
            "height": image.shape[1],
            "width": image.shape[2],
            "image_id": dataset_dict["image_id"],
            "instances": instances,
            #"annotations": annos
            "wcs": wcs
        }

    

def return_train_loader(cfg, mapper):
    """Returns a train loader

    Parameters
    ----------
    cfg : LazyConfig
        The lazy config, which contains data loader config values

    **kwargs for the read_image functionality

    Returns
    -------
        a train loader
    """
    loader = data.build_detection_train_loader(cfg, mapper=mapper)
    return loader


def return_test_loader(cfg, mapper):
    """Returns a test loader

    Parameters
    ----------
    cfg : LazyConfig
        The lazy config, which contains data loader config values

    **kwargs for the read_image functionality

    Returns
    -------
        a test loader
    """
    loader = data.build_detection_test_loader(cfg, cfg.DATASETS.TEST, mapper=mapper)
    return loader
