import pytest
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from overcomplete.visualization import (overlay_top_heatmaps, zoom_top_images, contour_top_image, evidence_top_images)
from overcomplete.visualization.top_concepts import _get_representative_ids


@pytest.fixture
def sample_images():
    return torch.randn(20, 3, 64, 64)


@pytest.fixture
def sample_heatmaps():
    return torch.randn(20, 56, 56, 10)


@pytest.fixture
def concept_id():
    return 3


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


def test_evidence_top_images(sample_images, sample_heatmaps, concept_id):
    evidence_top_images(sample_images, sample_heatmaps, concept_id)
    fig = plt.gcf()
    assert fig is not None
    assert len(fig.axes) == 10


@pytest.mark.parametrize("heatmaps", [
    torch.randn(20, 56, 56, 10),
    np.random.randn(20, 56, 56, 10)
])
def test_get_representative_ids(heatmaps):
    concept_id = 3
    ids = _get_representative_ids(heatmaps, concept_id)
    assert ids.shape == (10,)
    assert ids.dtype == torch.int64 if isinstance(heatmaps, torch.Tensor) else np.int64


IMG_SIZE = 64
BATCH_SIZE = 10


def generate_sample_image(image_type='torch', channels_first=True):
    if image_type == 'torch':
        if channels_first:
            return torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
        else:
            return torch.randn(BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3)
    elif image_type == 'numpy':
        if channels_first:
            return np.random.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
        else:
            return np.random.randn(BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3)
    elif image_type == 'pil':
        return Image.fromarray(np.uint8(np.random.rand(BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3) * 255))
    else:
        raise ValueError("Unsupported image type")


def generate_sample_heatmap(image_type='torch'):
    if image_type == 'torch':
        return torch.randn(BATCH_SIZE, IMG_SIZE//2, IMG_SIZE//2, 10)
    elif image_type == 'numpy':
        return np.random.randn(BATCH_SIZE, IMG_SIZE//2, IMG_SIZE//2, 10)
    else:
        raise ValueError("Unsupported heatmap type")


@pytest.mark.parametrize("img_type,channels_first", [
    ('torch', True),
    ('torch', False),
    ('numpy', True),
    ('numpy', False)
])
def test_overlay_advanced_types(img_type, channels_first):
    sample_image = generate_sample_image(img_type, channels_first)
    sample_heatmap = generate_sample_heatmap(img_type)
    concept_id = 3
    overlay_top_heatmaps(sample_image, sample_heatmap, concept_id)
    fig = plt.gcf()
    assert fig is not None
    assert len(fig.axes) == 10
    imgs = [ax.images[0].get_array().data for ax in fig.axes if ax.images]
    assert len(imgs) == 10
    assert imgs[0].shape[0] == imgs[0].shape[1]


@pytest.mark.parametrize("img_type,channels_first", [
    ('torch', True),
    ('torch', False),
    ('numpy', True),
    ('numpy', False)
])
def test_zoom_advanced_types(img_type, channels_first):
    sample_image = generate_sample_image(img_type, channels_first)
    sample_heatmap = generate_sample_heatmap(img_type)
    concept_id = 3
    zoom_top_images(sample_image, sample_heatmap, concept_id, zoom_size=16)
    fig = plt.gcf()
    assert fig is not None
    assert len(fig.axes) == 10
    imgs = [ax.images[0].get_array().data for ax in fig.axes if ax.images]
    assert len(imgs) == 10
    assert imgs[0].shape[0] == imgs[0].shape[1]


@pytest.mark.parametrize("img_type,channels_first", [
    ('torch', True),
    ('torch', False),
    ('numpy', True),
    ('numpy', False)
])
def test_contour_advanced_types(img_type, channels_first):
    sample_image = generate_sample_image(img_type, channels_first)
    sample_heatmap = generate_sample_heatmap(img_type)
    concept_id = 3
    contour_top_image(sample_image, sample_heatmap, concept_id)
    fig = plt.gcf()
    assert fig is not None
    assert len(fig.axes) == 10
    imgs = [ax.images[0].get_array().data for ax in fig.axes if ax.images]
    assert len(imgs) == 10
    assert imgs[0].shape[0] == imgs[0].shape[1]


@pytest.mark.parametrize("img_type,channels_first", [
    ('torch', True),
    ('torch', False),
    ('numpy', True),
    ('numpy', False)
])
def test_evidence_advanced_types(img_type, channels_first):
    sample_image = generate_sample_image(img_type, channels_first)
    sample_heatmap = generate_sample_heatmap(img_type)
    concept_id = 3
    evidence_top_images(sample_image, sample_heatmap, concept_id)
    fig = plt.gcf()
    assert fig is not None
    assert len(fig.axes) == 10
    imgs = [ax.images[0].get_array().data for ax in fig.axes if ax.images]
    assert len(imgs) == 10
    assert imgs[0].shape[0] == imgs[0].shape[1]
