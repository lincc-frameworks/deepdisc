import copy

import detectron2.data as data
import detectron2.data.transforms as T
import numpy as np
import torch
from detectron2.data import detection_utils as utils

import deepdisc.astrodet.astrodet as toolkit
import deepdisc.astrodet.detectron as detectron_addons


class train_mapper_cls:
    def __init__(self, imreader):
        self.IR = imreader

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = self.IR.read(dataset_dict)

        augs = detectron_addons.KRandomAugmentationList(
            [
                # my custom augs
                T.RandomRotation([-90, 90, 180], sample_style="choice"),
                T.RandomFlip(prob=0.5),
                T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            ],
            k=-1,
            cropaug=T.RandomCrop("relative", (0.5, 0.5)),
        )
        # Data Augmentation
        auginput = T.AugInput(image)
        # Transformations to model shapes
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


class redshift_train_mapper_cls:
    def __init__(self, imreader):
        self.IR = imreader

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = self.IR.read(dataset_dict)

        augs = detectron_addons.KRandomAugmentationList(
            [
                # my custom augs
                T.RandomRotation([-90, 90, 180], sample_style="choice"),
                T.RandomFlip(prob=0.5),
                T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            ],
            k=-1,
            cropaug=T.RandomCrop("relative", (0.5, 0.5)),
        )
        # Data Augmentation
        auginput = T.AugInput(image)
        # Transformations to model shapes
        transform = augs(auginput)
        image = torch.from_numpy(auginput.image.copy().transpose(2, 0, 1))

        annos = [
            utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
            for annotation in dataset_dict.pop("annotations")
            if annotation["redshift"] != 0.0
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
        }


def return_train_loader(cfg_loader, mapper):
    """Returns a train loader

    Parameters
    ----------
    cfg_loader: .yml file
        The config used for the data loaders

    **kwargs for the read_image functionality

    Returns
    -------
        a train loader
    """
    loader = data.build_detection_train_loader(cfg_loader, mapper=mapper)
    return loader


class test_mapper_cls:
    def __init__(self, imreader):
        self.IR = imreader

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = self.IR.read(dataset_dict)

        augs = T.AugmentationList(
            [
                T.CropTransform(
                    image.shape[1] // 4,
                    image.shape[0] // 4,
                    image.shape[1] // 2,
                    image.shape[0] // 2,
                )
            ]
        )
        # Data Augmentation
        auginput = T.AugInput(image)
        # Transformations to model shapes
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


class redshift_test_mapper_cls:
    def __init__(self, imreader):
        self.IR = imreader

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = self.IR.read(dataset_dict)

        augs = T.AugmentationList([])
        # Data Augmentation
        auginput = T.AugInput(image)
        # Transformations to model shapes
        transform = augs(auginput)
        image = torch.from_numpy(auginput.image.copy().transpose(2, 0, 1))
        annos = [
            utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
            for annotation in dataset_dict.pop("annotations")
            if annotation["redshift"] != 0.0
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
        }


def return_test_loader(cfg_loader, mapper):
    """Returns a train loader

    Parameters
    ----------
    cfg_loader: .yml file
        The config used for the data loaders

    **kwargs for the read_image functionality

    Returns
    -------
        a train loader
    """
    # _train_mapper = mapper
    loader = data.build_detection_test_loader(cfg_loader, cfg_loader.DATASETS.TEST, mapper=mapper)
    return loader
