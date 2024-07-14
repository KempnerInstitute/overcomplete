import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import torch

from .plot_utils import show, interpolate_cv2, get_image_dimensions, np_channel_last


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


def overlay_top_heatmaps(images, heatmaps, concept_id, cmap="jet", alpha=0.35):
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

    Returns
    -------
    None
    """
    assert len(images) == len(heatmaps)
    assert heatmaps.shape[-1] > concept_id
    assert heatmaps.ndim == 4

    best_ids = _get_representative_ids(heatmaps, concept_id)

    for i, idx in enumerate(best_ids):
        image = images[idx]
        width, height = get_image_dimensions(image)

        heatmap = interpolate_cv2(heatmaps[idx, :, :, concept_id], (width, height))

        plt.subplot(2, 5, i + 1)
        show(image)
        show(heatmap, cmap=cmap, alpha=alpha)


def zoom_top_images(images, heatmaps, concept_id, zoom_size=100):
    """
    Zoom into the hottest point in the heatmaps for a specific concept.

    This function identifies the top 10 images based on the mean value of the heatmaps for a given concept,
    then zooms into the hottest point of the heatmap for each of these images.

    Parameters
    ----------
    images : torch.Tensor
        Batch of input images of shape (batch_size, channels, height, width).
    heatmaps : torch.Tensor
        Batch of heatmaps corresponding to the input images of
        shape (batch_size, height, width, num_concepts).
    concept_id : int
        Index of the concept to visualize.
    zoom_size : int, optional
        Size of the zoomed area around the hottest point, by default 100.

    Returns
    -------
    None
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
    images : torch.Tensor
        Batch of input images of shape (batch_size, channels, height, width).
    heatmaps : torch.Tensor
        Batch of heatmaps corresponding to the input images of shape (batch_size, height, width, num_concepts).
    concept_id : int
        Index of the concept to visualize.
    percentiles : list of int, optional
        List of percentiles to contour, by default [70].
    cmap : str, optional
        Colormap for the contours, by default "viridis".
    linewidth : float, optional
        Width of the contour lines, by default 1.0.

    Returns
    -------
    None
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
