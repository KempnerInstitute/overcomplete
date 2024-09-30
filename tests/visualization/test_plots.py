import pytest
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from PIL import Image

from overcomplete.data import to_npf32
from overcomplete.visualization.plot_utils import (np_channel_last, normalize, clip_percentile, get_image_dimensions,
                                                   interpolate_cv2, interpolate_torch)
from overcomplete.visualization import show

from ..utils import epsilon_equal


def test_to_numpy():
    tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
    np_array = to_npf32(tensor)
    assert isinstance(np_array, np.ndarray), "Output is not a NumPy array"
    assert np_array.tolist() == [1.0, 2.0, 3.0], f"Expected [1.0, 2.0, 3.0], but got {np_array.tolist()}"

    pil = Image.new('RGB', (2, 2), (255, 0, 0))
    np_array = to_npf32(pil)
    assert isinstance(np_array, np.ndarray), "Output is not a numpy array"
    assert np_array.shape == (2, 2, 3), f"Expected shape (2, 2, 3), but got {np_array.shape}"
    assert epsilon_equal(np_array[0, 0], [255, 0, 0]), f"Expected [255, 0, 0], but got {np_array[0, 0]}"


def test_np_channel_last():
    tensor = torch.rand(3, 224, 224)
    formatted = np_channel_last(tensor)
    assert formatted.shape == (224, 224, 3)

    tensor = torch.rand(1, 224, 224)
    formatted = np_channel_last(tensor)
    assert formatted.shape == (224, 224, 1)

    tensor = torch.rand(1, 3, 224, 224)
    formatted = np_channel_last(tensor)
    assert formatted.shape == (224, 224, 3)

    tensor = torch.rand(1, 224, 224, 3)
    formatted = np_channel_last(tensor)
    assert formatted.shape == (224, 224, 3)

    tensor = torch.rand(224, 224)
    formatted = np_channel_last(tensor)
    assert formatted.shape == (224, 224, 1)


def test_normalize():
    image = np.array([[1, 2], [3, 4]], dtype=np.float32)
    norm_image = normalize(image)
    expected = (image - 1) / 3
    assert epsilon_equal(norm_image, expected), f"Expected {expected}, but got {norm_image}"

    image = np.array([[0, 0], [0, 0]], dtype=np.float32)
    norm_image = normalize(image)
    expected = np.zeros_like(image)
    assert epsilon_equal(norm_image, expected), f"Expected {expected}, but got {norm_image}"


def test_clip_percentile():
    image = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    clipped = clip_percentile(image, percentile=10)
    expected = [1, 1, 2, 3, 4, 5, 6, 7, 8, 8]
    assert epsilon_equal(clipped, expected), f"Expected {expected}, but got {clipped}"


def test_get_image_dimensions():

    img = Image.new('RGB', (100, 200))
    width, height = get_image_dimensions(img)
    assert width == 100
    assert height == 200

    img = Image.new('L', (100, 200))
    width, height = get_image_dimensions(img)
    assert width == 100
    assert height == 200

    img = np.random.rand(200, 100)
    width, height = get_image_dimensions(img)
    assert width == 100
    assert height == 200

    img = np.random.rand(3, 200, 100)
    width, height = get_image_dimensions(img)
    assert width == 100
    assert height == 200

    img = np.random.rand(200, 100, 3)
    width, height = get_image_dimensions(img)
    assert width == 100
    assert height == 200

    img = np.random.rand(200, 100, 3).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    width, height = get_image_dimensions(img)
    assert width == 100
    assert height == 200


def test_get_image_dimensions_invalid_type():
    with pytest.raises(TypeError):
        get_image_dimensions("invalid_type")


def test_show():
    imgs = [
        np.random.rand(224, 224, 3),
        np.random.rand(224, 224),
        np.random.rand(224, 224, 1),
        np.random.rand(1, 224, 224),
        np.random.rand(1, 3, 224, 224),
        np.random.rand(1, 224, 224, 3),
        torch.rand(3, 224, 224),
        torch.rand(1, 224, 224),
        torch.rand(1, 3, 224, 224),
        torch.rand(1, 224, 224, 3),
        Image.new('RGB', (224, 224)),
        Image.new('L', (224, 224)),
    ]
    for img in imgs:
        try:
            show(img)
        except Exception as e:
            pytest.fail(f"show raised an exception: {e}")


def test_interpolate_torch():
    img = torch.rand(3, 16, 16)
    target_size = 80

    # ensure interpolation work with and without channel dimension
    result = interpolate_torch(img, target=(target_size, target_size))
    assert result.shape == (3, target_size, target_size)

    result_2d = interpolate_torch(img[0], target=target_size)
    assert result_2d.shape == (target_size, target_size)


def test_interpolate_cv2():
    # this interpolate should be more robust and handle torch, pil, numpy format
    # and channel first or last
    target_size = 80

    img = torch.rand(3, 16, 16)
    result = interpolate_cv2(img, target=(target_size, target_size))
    assert result.shape == (target_size, target_size, 3)

    img = torch.rand(16, 16, 3)
    result = interpolate_cv2(img, target=(target_size, target_size))
    assert result.shape == (target_size, target_size, 3)

    img = torch.rand(16, 16)
    result = interpolate_cv2(img, target=(target_size, target_size))
    assert result.shape == (target_size, target_size)

    img = Image.new('RGB', (16, 16))
    result = interpolate_cv2(img, target=(target_size, target_size))
    assert result.shape == (target_size, target_size, 3)

    img = np.random.rand(16, 16, 3)
    result = interpolate_cv2(img, target=(target_size, target_size))
    assert result.shape == (target_size, target_size, 3)

    img = np.random.rand(3, 16, 16)
    result = interpolate_cv2(img, target=(target_size, target_size))
    assert result.shape == (target_size, target_size, 3)

    img = np.random.rand(16, 16)
    result = interpolate_cv2(img, target=(target_size, target_size))
    assert result.shape == (target_size, target_size)
