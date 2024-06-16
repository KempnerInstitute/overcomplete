import pytest
import torch
import numpy as np
from unittest.mock import patch
import matplotlib.pyplot as plt

from overcomplete.visualization import (interpolate_torch, overlay_top_heatmaps,
                                        zoom_top_images, contour_top_image)


@pytest.fixture
def sample_images():
    return torch.randn(20, 3, 64, 64)


@pytest.fixture
def sample_heatmaps():
    return torch.randn(20, 56, 56, 10)


@pytest.fixture
def concept_id():
    return 3


def test_interpolate(sample_images):
    img = sample_images[0]
    target_size = (80, 80)

    # ensure interpolation work with and without channel dimension
    result = interpolate_torch(img, target=target_size)
    assert result.shape[-2:] == target_size

    img_2d = sample_images[0, 0]
    result_2d = interpolate_torch(img_2d, target=target_size)
    assert result_2d.shape == target_size


def test_overlay_top_heatmaps(sample_images, sample_heatmaps, concept_id):
    overlay_top_heatmaps(sample_images, sample_heatmaps, concept_id)
    fig = plt.gcf()
    assert fig is not None
    assert len(fig.axes) == 10


def test_zoom_top_images(sample_images, sample_heatmaps, concept_id):
    zoom_top_images(sample_images, sample_heatmaps, concept_id)
    fig = plt.gcf()
    assert fig is not None
    assert len(fig.axes) == 10


def test_contour_top_image(sample_images, sample_heatmaps, concept_id):
    contour_top_image(sample_images, sample_heatmaps, concept_id)
    fig = plt.gcf()
    assert fig is not None
    assert len(fig.axes) == 10
