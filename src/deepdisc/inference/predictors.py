from astrodet import astrodet as toolkit

def return_predictor_transformer(cfg, cfg_loader):

    '''
    This function returns a trained model and its config file.
    Used for models with lazy config files.  Also assumes a cascade roi head structure

    Parameters
    ----------
    cfg : .py file
        a LazyConfig
    cfg_loader : .py file
        a LazyConfig
    
    Returns
    -------
        torch model

    '''

    predictor = toolkit.AstroPredictor(cfg_loader, lazy=True, cfglazy=cfg)

    return predictor


def get_predictions(dataset_dict,predictor, **kwargs):
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
    
    #We will want to make this handle other image readers eventually 
    filenames=[dataset_dict['filename_G'],dataset_dict['filename_R'],dataset_dict['filename_I']]
    img = toolkit.read_image_hsc(filenames,**kwargs)
    outputs = predictor(img)
    return outputs