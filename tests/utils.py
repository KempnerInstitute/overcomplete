"""
Utility functions for testing.

"""

import torch
import numpy as np


def _to_npf64(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float64)
    if isinstance(x, torch.Tensor):
        return x.numpy().astype(np.float64)
    return np.array(x).astype(np.float64)


def epsilon_equal(x, y, epsilon=1e-6):
    x = _to_npf64(x)
    y = _to_npf64(y)

    assert x.shape == y.shape, "Tensors must have the same shape"

    return np.allclose(x, y, atol=epsilon)
