import deepdisc.astrodet.astrodet as toolkit


def return_predictor_transformer(cfg, checkpoint=None):
    """This function returns a trained model and its config file.
    
    Used for models with lazy config files. Also assumes a cascade roi head structure.

    Parameters
    ----------
    cfg : .py file
        a LazyConfig
    Returns
    -------
        torch model

    """
    predictor = toolkit.AstroPredictor(cfg, checkpoint=checkpoint)
    return predictor


def get_predictions(dataset_dict, imreader, key_mapper, predictor):
    """Returns indices for matched pairs of ground truth and detected objects in an image

    Parameters
    ----------
    dataset_dict : dictionary
        The dictionary metadata for a single image
    imreader: ImageReader object
        An object derived from ImageReader base class to read the images.
    key_mapper: function
        The key_mapper should take a dataset_dict as input and return the key used by imreader
    predictor: AstroPredictor
        The predictor object used to make predictions on the test set

    Returns
    -------
        matched_gts: list(int)
            The indices of matched objects in the ground truth list
        matched_dts: list(int)
            The indices of matched objects in the detections list
        outputs: list(Intances)
            The list of detected object Instances
    """

    key = key_mapper(dataset_dict)
    img = imreader(key)
    outputs = predictor(img)
    return outputs


def get_predictions_new(dataset_dict, predictor):
    """Returns indices for matched pairs of ground truth and detected objects in an image

    Parameters
    ----------
    dataset_dict : dictionary
        The dictionary metadata for a single image
    predictor: AstroPredictor
        The predictor object used to make predictions on the test set

    Returns
    -------
        outputs: list(Intances)
            The list of detected object Instances
    """

    img = dataset_dict["image_shaped"]
    outputs = predictor(img)
    return outputs
