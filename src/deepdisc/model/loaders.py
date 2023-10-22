import detectron2.data as data
import copy
import numpy as np
import deepdisc.astrodet.astrodet as toolkit
import deepdisc.astrodet.detectron as detectron_addons
import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils
import torch
import detectron2.data as data


class train_mapper_cls:
    def __init__(self,**read_image_args):
        self.ria = read_image_args

    def __call__(self,dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        filenames=[dataset_dict['filename_G'],dataset_dict['filename_R'],dataset_dict['filename_I']]

        #image = read_image(dataset_dict["file_name"], normalize=args.norm, ceil_percentile=99.99)
        if self.ria['normalize'] != 'random':
            image = toolkit.read_image_hsc(filenames, normalize = self.ria['normalize'],
            ceil_percentile = self.ria['ceil_percentile'], dtype=self.ria['dtype'],
            A=self.ria['A'],stretch=self.ria['stretch'],Q=self.ria['Q'],do_norm=self.ria['do_norm'])
        elif self.ria['normalize'] =='random':
            scalings = np.array(['astrolupton','astroluptonhc','zscore'])
            norm = np.random.choice(scalings,1)[0]
            if norm=='zscore':
                dtype = np.int16
            else:
                dtype = np.uint8
            image = toolkit.read_image_hsc(filenames, normalize = norm,
            ceil_percentile = self.ria['ceil_percentile'], dtype=dtype,
            A=self.ria['A'],stretch=self.ria['stretch'],Q=self.ria['Q'],do_norm=self.ria['do_norm'])

        
        augs = detectron_addons.KRandomAugmentationList([
            # my custom augs
            T.RandomRotation([-90, 90, 180], sample_style='choice'),
            T.RandomFlip(prob=0.5),
            T.RandomFlip(prob=0.5,horizontal=False,vertical=True),
            ],
            k=-1,
            cropaug=T.RandomCrop('relative',(0.5,0.5))
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



class test_mapper_cls:
    def __init__(self,**read_image_args):
        self.ria = read_image_args

    def __call__(self,dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        filenames=[dataset_dict['filename_G'],dataset_dict['filename_R'],dataset_dict['filename_I']]

        if self.ria['normalize'] != 'random':
            image = toolkit.read_image_hsc(filenames, normalize = self.ria['normalize'],
            ceil_percentile = self.ria['ceil_percentile'], dtype=self.ria['dtype'],
            A=self.ria['A'],stretch=self.ria['stretch'],Q=self.ria['Q'],do_norm=self.ria['do_norm'])
        elif self.ria['normalize'] =='random':
            scalings = np.array(['astrolupton','astroluptonhc','zscore'])
            norm = np.random.choice(scalings,1)[0]
            if norm=='zscore':
                dtype = np.int16
            else:
                dtype = np.uint8
            image = toolkit.read_image_hsc(filenames, normalize = norm,
            ceil_percentile = self.ria['ceil_percentile'], dtype=dtype,
            A=self.ria['A'],stretch=self.ria['stretch'],Q=self.ria['Q'],do_norm=self.ria['do_norm'])


        augs = T.AugmentationList([
        T.CropTransform(image.shape[1]//4,image.shape[0]//4,image.shape[1]//2,image.shape[0]//2)
        ])


        
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



def return_train_loader(cfg_loader,**kwargs):
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
    _train_mapper = train_mapper_cls(**kwargs)
    loader = data.build_detection_train_loader(cfg_loader, mapper=_train_mapper)
    return loader
    


def return_test_loader(cfg_loader,**kwargs):
    """Returns a test loader

    Parameters
    ----------
    cfg_loader: .yml file
        The config used for the data loaders

    **kwargs for the read_image functionality

    Returns
    -------
        a test loader
    """
    _test_mapper = test_mapper_cls(**kwargs)
    test_loader = data.build_detection_test_loader(cfg_loader,cfg_loader.DATASETS.TEST,mapper=_test_mapper)
    return test_loader
