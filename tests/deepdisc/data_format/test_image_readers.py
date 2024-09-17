import os

import pytest

from deepdisc.data_format.image_readers import DC2ImageReader, HSCImageReader


def test_add_user_scaling_function():
    base_reader = HSCImageReader()
    initial_num_functions = len(base_reader.norm_dict)
    def silly_function():
        pass
    base_reader.add_scaling("silly_func", silly_function)
    after_num_functions = len(base_reader.norm_dict)

    assert after_num_functions > initial_num_functions
    assert "silly_func" in base_reader.norm_dict
    assert base_reader.norm_dict["silly_func"] == silly_function


def test_read_hsc_data(hsc_triple_test_file):
    """Test that we can read the test DC2 data."""
    ir = HSCImageReader(norm="raw")
    img = ir(hsc_triple_test_file)
    assert img.shape[0] == 1050
    assert img.shape[1] == 1025
    assert img.shape[2] == 3


def test_hsc_reader_raises_with_incorrect_input(hsc_triple_test_file):
    """Test that HSC reader raises error with incorrect number of files."""
    ir = HSCImageReader(norm="raw")
    with pytest.raises(ValueError) as excinfo:
        _ = ir(hsc_triple_test_file[0:1])
        assert "Incorrect number" in excinfo.value


def test_read_dc2_data(dc2_single_test_file):
    """Test that we can read the test DC2 data."""
    ir = DC2ImageReader(norm="raw")
    img = ir(dc2_single_test_file)
    assert img.shape[0] == 525
    assert img.shape[1] == 525
    assert img.shape[2] == 6

def test_read_roman_data(roman_single_test_file):
    """Test that we can read the test Roman data."""
    ir = RomanImageReader(norm="raw")
    img = ir(dc2_single_test_file)
    assert img.shape[0] == 525
    assert img.shape[1] == 525
    assert img.shape[2] == 4

def test_lupton_base_case(dc2_single_test_file):
    """Test that we can call lupton scaling without crashing"""
    ir = DC2ImageReader(norm="lupton")
    original_image = ir._read_image(dc2_single_test_file)
    scaled_img = ir(dc2_single_test_file)

    # pixel dimensions should be equal (number of bands may or may not be equal)
    assert original_image.shape[0:2] == scaled_img.shape[0:2]


def test_zscore_base_case(hsc_triple_test_file):
    """Test that we can call zscore scaling without crashing"""
    ir = HSCImageReader(norm="zscore")
    original_image = ir._read_image(hsc_triple_test_file)
    scaled_img = ir(hsc_triple_test_file)

    # pixel dimensions should be equal (number of bands may or may not be equal)
    assert original_image.shape[0:2] == scaled_img.shape[0:2]
