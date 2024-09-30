import numpy as np
from overcomplete.visualization.cmaps import create_alpha_cmap


def test_alpha_values_rgb_tuple():
    rgb_color = (66, 133, 244)
    cmap = create_alpha_cmap(rgb_color)

    alpha_values = cmap(np.linspace(0, 1, cmap.N))[:, -1]

    assert alpha_values[0] == 0.0, "Alpha value at the beginning should be 0"
    assert alpha_values[-1] == 1.0, "Alpha value at the end should be 1"


def test_alpha_values_colormap_name():
    cmap_name = 'viridis'
    cmap = create_alpha_cmap(cmap_name)

    alpha_values = cmap(np.linspace(0, 1, cmap.N))[:, -1]

    assert alpha_values[0] == 0.0, "Alpha value at the beginning should be 0"
    assert alpha_values[-1] == 1.0, "Alpha value at the end should be 1"


def test_color_range_rgb_tuple():
    rgb_color = (66, 133, 244)
    cmap = create_alpha_cmap(rgb_color)

    colors = cmap(np.linspace(0, 1, cmap.N))

    assert np.all(colors[:, :3] >= 0) and np.all(colors[:, :3] <= 1), "Color values should be in the range [0, 1]"


def test_color_range_colormap_name():
    cmap_name = 'viridis'
    cmap = create_alpha_cmap(cmap_name)

    colors = cmap(np.linspace(0, 1, cmap.N))

    assert np.all(colors[:, :3] >= 0) and np.all(colors[:, :3] <= 1), "Color values should be in the range [0, 1]"
