import os
import numpy as np
import pytest

from deepdisc.data_format.image_readers import DC2ImageReader, HSCImageReader

def test_read_hsc_data(hsc_triple_test_file):
    """Test that we can read the test DC2 data."""
    ir = HSCImageReader(norm="raw")
    img = ir.read(hsc_triple_test_file)
    assert img.shape[0] == 1050
    assert img.shape[1] == 1025
    assert img.shape[2] == 3

    # Check we can load from a dictionary as well.
    config_dict = {
        "filename_G": hsc_triple_test_file[0],
        "filename_R": hsc_triple_test_file[1],
        "filename_I": hsc_triple_test_file[2],    
    }
    img2 = ir.read(config_dict)
    assert img2.shape[0] == 1050
    assert img2.shape[1] == 1025
    assert img2.shape[2] == 3

    # Unknown input type.
    with pytest.raises(TypeError):
        img3 = ir.read(10.1)

def test_read_dc2_data(dc2_single_test_file):
    """Test that we can read the test DC2 data."""
    ir = DC2ImageReader(norm="raw")
    img = ir.read(dc2_single_test_file)
    assert img.shape[0] == 525
    assert img.shape[1] == 525
    assert img.shape[2] == 6

    # Check we can load from a dictionary as well.
    config_dict = { "filename": dc2_single_test_file }
    img2 = ir.read(config_dict)
    assert img2.shape[0] == 525
    assert img2.shape[1] == 525
    assert img2.shape[2] == 6

    # Unknown input type.
    with pytest.raises(TypeError):
        img3 = ir.read(10.1)

def test_read_numpy_data():
    """Test that we take data from a numpy array."""
    arr = np.zeros((10, 15, 3))

    # DC2 reader
    ir = DC2ImageReader(norm="raw")
    img = ir.read(arr)
    assert img.shape[0] == 10
    assert img.shape[1] == 15
    assert img.shape[2] == 3

    # HSC Reader
    ir2 = HSCImageReader(norm="raw")
    img2 = ir2.read(arr)
    assert img2.shape[0] == 10
    assert img2.shape[1] == 15
    assert img2.shape[2] == 3
