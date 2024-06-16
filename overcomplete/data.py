import os

from PIL import Image
import numpy as np
import torch


def load_directory(directory):
    """
    Load all images from a directory.

    Parameters
    ----------
    dir : str
        Directory path.

    Returns
    -------
    list
        List of PIL images.
    """
    paths = os.listdir(directory)
    images = []
    for path in paths:
        img = Image.open(os.path.join(directory, path)).convert('RGB')
        images.append(img)
    return images


def to_npf32(tensor):
    """
    Check if tensor is torch, ensure it is on CPU and convert to NumPy.

    Parameters
    ----------
    tensor : torch.Tensor or np.ndarray
        Input tensor.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy().astype(np.float32)
    return np.array(tensor).astype(np.float32)


def unwrap_dataloader(dataloader):
    """
    Unwrap a DataLoader into a single tensor.

    Parameters
    ----------
    dataloader : DataLoader
        DataLoader object.
    """
    return torch.cat([batch[0] if isinstance(batch, (tuple, list))
                      else batch for batch in dataloader], dim=0)
