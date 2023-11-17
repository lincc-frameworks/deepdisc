from pathlib import Path

from detectron2.data import DatasetCatalog, MetadataCatalog

from deepdisc.data_format.file_io import get_data_from_json


def register_data_set(data_set_name, filename, load_func=get_data_from_json, **kwargs):
    """Register the data set and get the MetadataCatalog.

    Parameters
    ----------
    data_set_name: str
        The name of the data set.
    filename: str
        The name of the file.
    load_func: function
        The function to use to load the data set. Defaults to get_data_from_json().
    kwargs:
        Additional parameters to pass into the metadata
        set function. Example:
        register_data_set(name1, name2, thing_classes=['star', 'galaxy'])

    Returns
    -------
    meta: Metadata
        The metadata for this data set.

    Raises
    ------
    FileNotFoundError if the data set file cannot be found.
    """
    if not Path(filename).exists():
        raise FileNotFoundError(f"Unable to load data set file {filename}")

    DatasetCatalog.register(data_set_name, lambda: load_func(filename))
    MetadataCatalog.get(data_set_name).set(**kwargs)
    meta = MetadataCatalog.get(data_set_name)

    return meta


def register_loaded_data_set(data_set_name, data, load_func=get_data_from_json, **kwargs):
    """Register the data set and get the MetadataCatalog.

    Parameters
    ----------
    data_set_name: str
        The name of the data set.
    data: any
        The dataset.
    load_func: function
        The function to use to load the data set. Defaults to get_data_from_json().
    kwargs:
        Additional parameters to pass into the metadata
        set function. Example:
        register_data_set(name1, name2, thing_classes=['star', 'galaxy'])

    Returns
    -------
    meta: Metadata
        The metadata for this data set.

    Raises
    ------
    FileNotFoundError if the data set file cannot be found.
    """

    DatasetCatalog.register(data_set_name, lambda: load_func(data))
    MetadataCatalog.get(data_set_name).set(**kwargs)
    meta = MetadataCatalog.get(data_set_name)

    return meta