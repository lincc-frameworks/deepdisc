import numpy as np
import pytest

from deepdisc.preprocessing.ground_truth_gen import mad_wavelet_own


@pytest.fixture
def one_channel_image():
    input_image = np.arange(0, 100, dtype=np.int16)
    input_image = input_image.reshape(10, 10)
    return input_image


def test_mad_wavelet_own(one_channel_image):
    mad_own = mad_wavelet_own(one_channel_image)
    assert mad_own > 0
