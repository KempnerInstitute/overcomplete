"""
Module dedicated to utilities for image visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from PIL import Image
import cv2

from ..data import to_npf32


def interpolate_torch(img, target=(224, 224), mode='bicubic'):
    """
    Interpolate a tensor to a target size using bicubic interpolation.

    Parameters
    ----------
    img : torch.Tensor
        Input image tensor. Can be 2D (single channel) or 3D (multiple channels).
    target : tuple of int, optional
        Target size (height, width), by default (224, 224).

    Returns
    -------
    torch.Tensor
        Interpolated image tensor.
    """
    if img.ndim == 2:
        return F.interpolate(img[None, None, ...], target, mode=mode, antialias=True)[0, 0]
    if img.ndim == 3:
        return F.interpolate(img[None, ...], target, mode=mode, antialias=True)[0]
    return F.interpolate(img, target, mode=mode, antialias=True)


def interpolate_cv2(img, target=(224, 224), interpolation=cv2.INTER_CUBIC):
    """
    Robust interpolate an image to a target size using OpenCV.
    Handle numpy, torch, PIL, channels first or last.

    Parameters
    ----------
    img : numpy.ndarray, torch.Tensor, or PIL.Image
        Input image array. Can be 2D (single channel) or 3D (multiple channels),
        channels can be in the first or last dimension.
    target : tuple of int, optional
        Target size (height, width), by default (224, 224).
    interpolation : int, optional
        Interpolation method, see OpenCV documentation for details, by default cv2.INTER_CUBIC.

    Returns
    -------
    numpy.ndarray
        Interpolated image array.
    """
    img = to_npf32(img)
    img = np_channel_last(img)
    assert img.ndim in (2, 3), f"Expected 2 or 3 dimensions, but got {img.shape}"
    return cv2.resize(img, target, interpolation=interpolation)


def get_image_dimensions(img):
    """
    Get the width and height of an image, which can be a PIL image,
    a NumPy array, or an OpenCV (cv2) image.

    Parameters
    ----------
    img : PIL.Image.Image, numpy.ndarray, or cv2.Mat
        Input image to get the dimensions from

    Returns
    -------
    tuple of int
        A tuple (width, height) representing the dimensions of the image.

    Raises
    ------
    TypeError
        If the image type is unsupported.
    """
    # PIL case
    if isinstance(img, Image.Image):
        return img.size

    # numpy / torch case
    if isinstance(img, (np.ndarray, torch.Tensor)):
        shape = img.shape
        if len(shape) == 2:  # grayscale
            return shape[1], shape[0]  # inversed width, height
        if len(shape) == 3:
            if shape[0] == 3 or shape[0] == 1:  # it's channel first
                return shape[2], shape[1]
            # then it's channel last
            return shape[1], shape[0]

    # cv2 case
    if isinstance(img, cv2.Mat):
        return img.shape[1], img.shape[0]

    raise TypeError("Unsupported image type")


def np_channel_last(arr):
    """
    Ensure and convert if necessary to a numpy array and move channels to the last
    dimension if they are in the first dimension.

    Parameters
    ----------
    arr : numpy.ndarray, torch.Tensor, or PIL.Image
        Input array, expected to have channels in the first dimension if it
        has 3 channels.

    Returns
    -------
    numpy.ndarray
        The input array with 3 dimensions and channels moved to the last
        dimension if necessary.
    """
    arr = to_npf32(arr)

    # if batch dimension, take the first element
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    # if grayscale, return as is with extra dimension
    if arr.ndim == 2:
        return arr[:, :, None]
    # assume channels dimension are 3 or 1
    if arr.shape[0] == 3 or arr.shape[0] == 1:
        return np.moveaxis(arr, 0, -1)

    assert arr.ndim == 3, f"Expected 3 dimensions, but got {arr.shape}"

    return arr


def normalize(image, eps=1e-6):
    """
    Normalize the image to the 0-1 range.

    Parameters
    ----------
    image : numpy.ndarray
        Input image array.
    eps : float, optional
        Epsilon value to avoid division by zero (default is 1e-6).

    Returns
    -------
    numpy.ndarray
        Normalized image array.
    """
    image = np.array(image, dtype=np.float32)
    image -= image.min()
    image /= (image.max() + eps)
    return image


def clip_percentile(img, percentile=0.1, clip_method='nearest'):
    """
    Clip pixel values to a specified percentile range.

    Parameters
    ----------
    img : numpy.ndarray
        Input image array.
    percentile : float, optional
        Percentile for clipping (default is 0.1).
    clip_method : str, optional
        Method for clipping, see numpy documentation for details.

    Returns
    -------
    numpy.ndarray
        Image array with pixel values clipped to the specified percentile range.
    """
    return np.clip(img, np.percentile(img, percentile, method=clip_method),
                   np.percentile(img, 100 - percentile, method=clip_method),)


def show(img, **kwargs):
    """
    Display an image with normalization and channels in the last dimension.

    Parameters
    ----------
    img : numpy.ndarray
        Input image array.
    kwargs : dict
        Additional keyword arguments for plt.imshow.

    Returns
    -------
    None
    """
    img = np_channel_last(img)
    img = normalize(img)
    plt.imshow(img, **kwargs)
    plt.axis('off')
