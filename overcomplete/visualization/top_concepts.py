"""
Module dedicated for visualizing top concepts in a batch of images.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import torch

from .plot_utils import show, interpolate_cv2, get_image_dimensions, np_channel_last
from .cmaps import VIRIDIS_ALPHA, TAB10_ALPHA


def _get_representative_ids(heatmaps, concept_id):
    """
    Get the top 10 images based on the mean value of the heatmaps for a given concept.

    Parameters
    ----------
    heatmaps : torch.Tensor or np.ndarray
        Batch of heatmaps corresponding to the input images of shape (batch_size, height, width, num_concepts).
    concept_id : int
        Index of the concept to visualize.

    Returns
    -------
    torch.Tensor or np.ndarray
        Indices of the top 10 images based on the mean value of the heatmaps for a given concept.
    """
    if isinstance(heatmaps, torch.Tensor):
        return torch.mean(heatmaps[:, :, :, concept_id], dim=(1, 2)).argsort()[-10:]
    return np.mean(heatmaps[:, :, :, concept_id], axis=(1, 2)).argsort()[-10:]


def overlay_top_heatmaps(images, heatmaps, concept_id, cmap=None, alpha=0.35):
    """
    Visualize the top activating image for a concepts and overlay the associated heatmap.

    This function sorts images based on the mean value of the heatmaps for a given concept and
    visualizes the top 10 images with their corresponding heatmaps.

    Parameters
    ----------
    images : torch.Tensor or PIL.Image or np.ndarray
        Batch of input images of shape (batch_size, channels, height, width).
    z_heatmaps : torch.Tensor or np.ndarray
        Batch of heatmaps corresponding to the input images of
        shape (batch_size, height, width, num_concepts).
    concept_id : int
        Index of the concept to visualize.
    cmap : str, optional
        Colormap for the heatmap, by default 'jet'.
    alpha : float, optional
        Transparency of the heatmap overlay, by default 0.35.
    """
    assert len(images) == len(heatmaps)
    assert heatmaps.shape[-1] > concept_id
    assert heatmaps.ndim == 4

    # if we handle the cmap, choose tab10 if number of concepts is less than 10
    # else choose a normal one
    if cmap is None:
        cmap = TAB10_ALPHA[concept_id] if heatmaps.shape[-1] < 10 else VIRIDIS_ALPHA
        # and enforce the alpha value to one, as the alpha is already handled by the colormap
        alpha = 1.0

    best_ids = _get_representative_ids(heatmaps, concept_id)

    for i, idx in enumerate(best_ids):
        image = images[idx]
        width, height = get_image_dimensions(image)

        heatmap = interpolate_cv2(heatmaps[idx, :, :, concept_id], (width, height))

        plt.subplot(2, 5, i + 1)
        show(image)
        show(heatmap, cmap=cmap, alpha=alpha)


def evidence_top_images(images, heatmaps, concept_id, percentiles=None):
    """
    Visualize the top activating image for a concept and highlight the top activating pixels.

    This function identifies the top 10 images based on the mean value of the heatmaps for a given concept,
    then use the heatmap to highlights the top activating area depending on their percentile value.

    Parameters
    ----------
    images : torch.Tensor or PIL.Image or np.ndarray
        Batch of input images of shape (batch_size, channels, height, width).
    heatmaps : torch.Tensor or np.ndarray
        Batch of heatmaps corresponding to the input images of
        shape (batch_size, height, width, num_concepts).
    concept_id : int
        Index of the concept to visualize.
    percentiles : list of int, optional
        List of percentiles to highlight, by default None.
    """
    assert len(images) == len(heatmaps)
    assert heatmaps.shape[-1] > concept_id
    assert heatmaps.ndim == 4

    if percentiles is None:
        # gradation from 50% top activating pixels to 95% top activating pixels
        # with alpha 0 at start and 1 at the end
        percentiles = np.linspace(50, 95, 10)

    best_ids = _get_representative_ids(heatmaps, concept_id)

    for i, idx in enumerate(best_ids):
        image = images[idx]
        image = np_channel_last(image)
        width, height = get_image_dimensions(image)

        heatmap = interpolate_cv2(heatmaps[idx, :, :, concept_id], (width, height))

        # use the heatmap to apply alpha depending on the percentile
        mask = np.zeros_like(heatmap)
        for percentile in percentiles:
            mask[heatmap > np.percentile(heatmap, percentile)] += 1.0
        mask = mask / len(percentiles)

        plt.subplot(2, 5, i + 1)
        show(image*mask[:, :, None])


def zoom_top_images(images, heatmaps, concept_id, zoom_size=100):
    """
    Zoom into the hottest point in the heatmaps for a specific concept.

    This function identifies the top 10 images based on the mean value of the heatmaps for a given concept,
    then zooms into the hottest point of the heatmap for each of these images.

    Parameters
    ----------
    images : torch.Tensor or PIL.Image or np.ndarray
        Batch of input images of shape (batch_size, channels, height, width).
    heatmaps : torch.Tensor or np.ndarray
        Batch of heatmaps corresponding to the input images of
        shape (batch_size, height, width, num_concepts).
    concept_id : int
        Index of the concept to visualize.
    zoom_size : int, optional
        Size of the zoomed area around the hottest point, by default 100.
    """
    assert len(images) == len(heatmaps)
    assert heatmaps.shape[-1] > concept_id
    assert heatmaps.ndim == 4

    best_ids = _get_representative_ids(heatmaps, concept_id)

    for i, idx in enumerate(best_ids):
        image = np_channel_last(images[idx])
        width, height = get_image_dimensions(image)

        heatmap = interpolate_cv2(heatmaps[idx, :, :, concept_id], (width, height))
        hottest_point = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)

        x_min = max(hottest_point[0] - zoom_size // 2, 0)
        x_max = min(hottest_point[0] + zoom_size // 2, image.shape[0])
        y_min = max(hottest_point[1] - zoom_size // 2, 0)
        y_max = min(hottest_point[1] + zoom_size // 2, image.shape[1])

        zoomed_image = image[x_min:x_max, y_min:y_max]

        plt.subplot(2, 5, i + 1)
        show(zoomed_image)


def contour_top_image(images, heatmaps, concept_id, percentiles=None, cmap="viridis", linewidth=1.0):
    """
    Contour the best images for a specific concept using heatmap percentiles.

    This function identifies the top 10 images based on the mean value of the heatmaps for a given concept,
    then draws contours at specified percentiles on the heatmap overlaid on the original image.

    Parameters
    ----------
    images : torch.Tensor or PIL.Image or np.ndarray
        Batch of input images of shape (batch_size, channels, height, width).
    heatmaps : torch.Tensor or np.ndarray
        Batch of heatmaps corresponding to the input images of shape (batch_size, height, width, num_concepts).
    concept_id : int
        Index of the concept to visualize.
    percentiles : list of int, optional
        List of percentiles to contour, by default [70].
    cmap : str, optional
        Colormap for the contours, by default "viridis".
    linewidth : float, optional
        Width of the contour lines, by default 1.0.
    """
    assert len(images) == len(heatmaps)
    assert heatmaps.shape[-1] > concept_id
    assert heatmaps.ndim == 4

    if percentiles is None:
        percentiles = [70]

    cmap = plt.get_cmap(cmap)
    best_ids = _get_representative_ids(heatmaps, concept_id)

    for i, idx in enumerate(best_ids):
        image = images[idx]
        width, height = get_image_dimensions(image)
        plt.subplot(2, 5, i + 1)
        show(image)

        heatmap = heatmaps[idx, :, :, concept_id]
        heatmap = interpolate_cv2(heatmap, (width, height))

        for percentile in percentiles:
            if len(percentiles) == 1:
                color_value = cmap(0.0)
            else:
                # color value is a remap of percentile between [0, 1] depending on value of percentiles
                color_value = (percentile - percentiles[-1]) / (percentiles[0] - percentiles[-1])
                color_value = cmap(color_value)

            cut_value = np.percentile(heatmap, percentile)
            contours = measure.find_contours(heatmap, cut_value)
            for contour in contours:
                plt.plot(contour[:, 1], contour[:, 0], linewidth=linewidth, color=color_value)
