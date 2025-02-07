import os
import pytest
import torch

from overcomplete.optimization import SemiNMF, ConvexNMF, NMF

from ..utils import epsilon_equal

data_shape = (50, 10)
nb_concepts = 5
A = torch.rand(data_shape, dtype=torch.float32)

methods = [SemiNMF, ConvexNMF, NMF]


@pytest.mark.parametrize("methods", methods)
def test_methods_save_and_load(methods):
    """Test that the methods can be saved and loaded."""

    model = methods(nb_concepts=nb_concepts, max_iter=2)
    model.fit(A)

    D = model.D

    torch.save(model, 'test_optimization_model.pth')
    model = torch.load('test_optimization_model.pth')

    assert epsilon_equal(model.D, D), "Loaded model does not produce the same results."

    os.remove('test_optimization_model.pth')
