"""Utilities for augmenting image data."""

import imgaug.augmenters as iaa
import numpy as np


def gaussblur(image):
    """
    Parameters
    ----------
    image: ndarray

    Returns
    -------
    augmented image

    """
    aug = iaa.GaussianBlur(sigma=(0.0, np.random.random_sample() * 4 + 2))
    return aug.augment_image(image)


def addelementwise16(image):
    """
    Parameters
    ----------
    image: ndarray

    Returns
    -------
    augmented image

    """
    aug = iaa.AddElementwise((-3276, 3276))
    return aug.augment_image(image)


def addelementwise8(image):
    """
    Parameters
    ----------
    image: ndarray

    Returns
    -------
    augmented image

    """
    aug = iaa.AddElementwise((-25, 25))
    return aug.augment_image(image)


def addelementwise(image):
    """
    Parameters
    ----------
    image: ndarray

    Returns
    -------
    augmented image

    """
    aug = iaa.AddElementwise((-image.max() * 0.1, image.max() * 0.1))
    return aug.augment_image(image)


def centercrop(image):
    """Crop an image to just the center portion

    Parameters
    ----------
    image: ndarray

    Returns
    -------
    cropped image
    """
    h, w = image.shape[:2]
    hc = (h - h // 2) // 2
    wc = (w - w // 2) // 2
    image = image[hc : hc + h // 2, wc : wc + w // 2]
    return image
