"""Utilities for augmenting image data."""

import numpy as np
import pytest

from deepdisc.data_format.augment_image import (addelementwise,
                                                addelementwise8,
                                                addelementwise16, centercrop,
                                                gaussblur)


@pytest.fixture
def simple_image():
    input_image = np.arange(0, 100, dtype=np.int16)
    input_image = input_image.reshape(10, 10)
    return input_image


def test_gaussblur(simple_image):
    output = gaussblur(simple_image)
    assert len(output) == len(simple_image)


def test_addelementwise16(simple_image):
    output = addelementwise16(simple_image)
    assert len(output) == len(simple_image)


def test_addelementwise8(simple_image):
    output = addelementwise8(simple_image)
    assert len(output) == len(simple_image)


def test_addelementwise(simple_image):
    output = addelementwise(simple_image)
    assert len(output) == len(simple_image)


def test_centercrop(simple_image):
    output = centercrop(simple_image)
    assert len(output) == len(simple_image) / 2
