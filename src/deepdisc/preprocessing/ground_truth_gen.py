import scarlet
import scarlet.wavelet
from astropy.stats import median_absolute_deviation as astromad


def mad_wavelet_own(image):
    """image: Median absolute deviation of the first wavelet scale.
    (WARNING: sorry to disapoint, this is not a wavelet for mad scientists)

    Have to use astropy mad as scipy mad does not like ignoring NaN and computing over multiple axes

    Parameters
    ----------
    image: array
        An image or cube of images
    Returns
    -------
    mad: array
        median absolute deviation for each image in the cube
    """

    # Scale =1/1.4826 to replicate older scipy MAD behavior
    scale = 1 / 1.4826
    sigma = astromad(
        scarlet.Starlet.from_image(image, scales=2).coefficients[0, ...],
        axis=(-2, -1),
        ignore_nan=True,
    )
    return sigma / scale
