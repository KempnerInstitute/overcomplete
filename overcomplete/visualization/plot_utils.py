import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

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


def check_format(arr):
    """
    Ensure the input is a NumPy array and move channels to the last dimension if they are in the first dimension.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array, expected to have channels in the first dimension if it has 3 channels.

    Returns
    -------
    numpy.ndarray
        The input array with channels moved to the last dimension if necessary.
    """
    arr = to_npf32(arr)
    if arr.shape[0] == 3:
        return np.moveaxis(arr, 0, -1)
    return arr


def normalize(image):
    """
    Normalize the image to the 0-1 range.

    Parameters
    ----------
    image : numpy.ndarray
        Input image array.

    Returns
    -------
    numpy.ndarray
        Normalized image array.
    """
    image = np.array(image, dtype=np.float32)
    image -= image.min()
    image /= image.max()
    return image


def clip_percentile(img, percentile=0.1):
    """
    Clip pixel values to a specified percentile range.

    Parameters
    ----------
    img : numpy.ndarray
        Input image array.
    percentile : float, optional
        Percentile for clipping (default is 0.1).

    Returns
    -------
    numpy.ndarray
        Image array with pixel values clipped to the specified percentile range.
    """
    return np.clip(img, np.percentile(img, percentile), np.percentile(img, 100 - percentile))


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
    img = check_format(img)
    img = normalize(img)
    plt.imshow(img, **kwargs)
    plt.axis('off')
