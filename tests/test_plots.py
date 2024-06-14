import pytest
import numpy as np
import matplotlib.pyplot as plt
import torch

from overcomplete.plots import to_numpy, check_format, normalize, clip_percentile, show


def test_to_numpy():
    tensor = torch.tensor([1, 2, 3])
    np_array = to_numpy(tensor)
    assert isinstance(np_array, np.ndarray), "Output is not a NumPy array"
    assert np_array.tolist() == [1, 2, 3], f"Expected [1, 2, 3], but got {np_array.tolist()}"


def test_check_format():
    tensor = torch.rand(3, 224, 224)
    formatted = check_format(tensor)
    assert formatted.shape == (224, 224, 3), f"Expected shape (224, 224, 3), but got {formatted.shape}"


def test_normalize():
    image = np.array([[1, 2], [3, 4]], dtype=np.float32)
    norm_image = normalize(image)
    expected = (image - 1) / 3
    assert np.allclose(norm_image, expected), f"Expected {expected}, but got {norm_image}"


def test_clip_percentile():
    image = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    clipped = clip_percentile(image, percentile=10)
    expected = np.clip(image, np.percentile(image, 10), np.percentile(image, 90))
    assert np.allclose(clipped, expected), f"Expected {expected}, but got {clipped}"
