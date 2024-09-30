import pytest
import torch

from overcomplete.sae.kernels import (
    rectangle_kernel, gaussian_kernel, triangular_kernel,
    cosine_kernel, epanechnikov_kernel, quartic_kernel,
    silverman_kernel, cauchy_kernel
)


@pytest.fixture
def sample_tensor():
    return torch.tensor([-1.5, -0.5, 0.0, 0.5, 1.5], dtype=torch.float32)


@pytest.fixture
def bandwith():
    return 1.0


all_kernels = [
    rectangle_kernel, gaussian_kernel, triangular_kernel,
    cosine_kernel, epanechnikov_kernel, quartic_kernel,
    silverman_kernel, cauchy_kernel
]


@pytest.mark.parametrize("kernel_fn", all_kernels)
def test_kernel_shape(kernel_fn, sample_tensor, bandwith):
    output = kernel_fn(sample_tensor, bandwith)
    assert output.shape == sample_tensor.shape, f"{kernel_fn.__name__} output shape mismatch."


@pytest.mark.parametrize("kernel_fn", all_kernels)
def test_kernel_non_nan(kernel_fn, sample_tensor, bandwith):
    output = kernel_fn(sample_tensor, bandwith)
    assert not torch.isnan(output).any(), f"{kernel_fn.__name__} contains NaN values."


@pytest.mark.parametrize("kernel_fn", all_kernels)
def test_kernel_non_negative(kernel_fn, sample_tensor, bandwith):
    output = kernel_fn(sample_tensor, bandwith)
    assert (output >= 0).all(), f"{kernel_fn.__name__} has negative values."


# additional specific tests for rectangle kernel
def test_rectangle_kernel_values(sample_tensor, bandwith):
    expected_values = torch.tensor([0, 1, 1, 1, 0], dtype=torch.float32) / bandwith
    output = rectangle_kernel(sample_tensor, bandwith)
    assert torch.allclose(output, expected_values), f"Expected {expected_values}, but got {output}"


@pytest.mark.parametrize("test_input, expected", [
    (torch.tensor([-0.6, -0.5, 0.0, 0.5, 0.6]), torch.tensor([0, 1, 1, 1, 0])),  # within bandwith
    (torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]), torch.tensor([0, 0, 1, 0, 0])),  # eEdges just outside
    (torch.tensor([0.25, -0.25]), torch.tensor([1, 1])),  # values within bandwith boundaries
])
def test_rectangle_kernel_edge_cases(test_input, expected, bandwith):
    output = rectangle_kernel(test_input, bandwith)
    expected_output = expected.float() / bandwith
    assert torch.allclose(output, expected_output), f"Expected {expected_output}, but got {output}"
