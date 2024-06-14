import os
from PIL import Image


def load_directory(dir):
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
    paths = os.listdir(dir)
    images = []
    for path in paths:
        img = Image.open(os.path.join(dir, path)).convert('RGB')
        images.append(img)
    return images
