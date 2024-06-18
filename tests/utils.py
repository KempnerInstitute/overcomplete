"""
Utility functions for testing.

"""

import torch


def epsilon_equal(x, y, epsilon=1e-6):
    if isinstance(x, float):
        x = torch.tensor(x)
    if isinstance(y, float):
        y = torch.tensor(y)

    assert x.shape == y.shape, "Tensors must have the same shape"

    return torch.allclose(x, y, atol=epsilon)
