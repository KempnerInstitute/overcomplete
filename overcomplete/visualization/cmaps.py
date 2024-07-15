"""
Module for creating and storing custom colormaps.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def create_alpha_cmap(color_input, name=None):
    """
    Create an alpha colormap from an input colormap or RGB tuple.
    The alpha values will range from 0 (fully transparent) to 1 (fully opaque).

    Parameters
    ----------
    color_input : str or tuple
        A Matplotlib colormap name or an RGB tuple.
    name : str, optional
        The name of the colormap, by default None.

    Returns
    -------
    mcolors.LinearSegmentedColormap
        A Matplotlib LinearSegmentedColormap with alpha values ranging from 0 to 1.

    Raises
    ------
    ValueError
        If color_input is neither a valid colormap name nor an RGB tuple.
    """
    if isinstance(color_input, str):
        # If the input is a colormap name
        base_cmap = plt.cm.get_cmap(color_input)
    elif isinstance(color_input, tuple) and len(color_input) == 3:
        if np.max(color_input) > 1:
            color_input = (
                color_input[0] / 255,
                color_input[1] / 255,
                color_input[2] / 255,
            )
        if name is None:
            name = f'RGB{color_input}'
        base_cmap = mcolors.LinearSegmentedColormap.from_list(name, [color_input, color_input])
    else:
        raise ValueError("Invalid color_input. Must be a colormap name or an RGB tuple.")

    # alpha values ranging from 0 to 1
    # tfel: useful to start above zero?
    alpha = np.linspace(0.0, 1, base_cmap.N)

    colors = base_cmap(np.arange(base_cmap.N))
    colors[:, -1] = alpha

    alpha_cmap = mcolors.LinearSegmentedColormap.from_list('alpha_cmap', colors)

    return alpha_cmap


# predefine some alpha cmaps
JET_ALPHA = create_alpha_cmap('jet')
VIRIDIS_ALPHA = create_alpha_cmap('viridis')

_tab_10_colors = [
    (31, 119, 180),   # tab:blue
    (255, 127, 14),   # tab:orange
    (44, 160, 44),    # tab:green
    (214, 39, 40),    # tab:red
    (148, 103, 189),  # tab:purple
    (140, 86, 75),    # tab:brown
    (227, 119, 194),  # tab:pink
    (127, 127, 127),  # tab:gray
    (188, 189, 34),   # tab:olive
    (23, 190, 207)    # tab:cyan
]

TAB10_ALPHA = [create_alpha_cmap(color) for color in _tab_10_colors]
