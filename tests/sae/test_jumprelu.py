import pytest
import torch

from overcomplete.sae.jump_sae import jump_relu, heaviside
from overcomplete.sae.kernels import (
    rectangle_kernel, gaussian_kernel, triangular_kernel,
    cosine_kernel, epanechnikov_kernel, quartic_kernel,
    silverman_kernel, cauchy_kernel
)

all_kernels = [
    rectangle_kernel, gaussian_kernel, triangular_kernel,
    cosine_kernel, epanechnikov_kernel, quartic_kernel,
    silverman_kernel, cauchy_kernel
]


@pytest.fixture
def test_tensors():
    # random tensor to test:
    # for input 1 : first one pass, secondone fails and third one passes
    # for input 2 : just the third one pass
    # however for the gradient:
    # for input 1: all gradients are active
    # for input 2: only the first one is active
    x = torch.tensor([[1.5, -0.5, 2.0], [0.2, -1.2, 3.0]], dtype=torch.float32, requires_grad=True)
    threshold = torch.tensor([1.0, 1.0, 1.5], dtype=torch.float32, requires_grad=True)
    bandwith = 1.0
    return x, threshold, rectangle_kernel, bandwith


@pytest.fixture
def grad_output():
    # custom gradient 1.0 gradient output
    return torch.ones((2, 3), dtype=torch.float32)


def test_jumprelu_shape(test_tensors):
    x, threshold, kernel_fn, bandwith = test_tensors
    output = jump_relu(x, threshold, kernel_fn, bandwith)

    assert output.shape == x.shape


def test_jumprelu_non_nan(test_tensors):
    x, threshold, kernel_fn, bandwith = test_tensors
    output = jump_relu(x, threshold, kernel_fn, bandwith)

    assert not torch.isnan(output).any()


def test_jumprelu_value(test_tensors):
    # recall, for the forward
    # for input 1 : first one pass, secondone fails and third one passes
    # for input 2 : just the third one pass
    x, threshold, kernel_fn, bandwith = test_tensors
    output = jump_relu(x, threshold, kernel_fn, bandwith)

    expected_output = torch.tensor([[1.5, 0.0, 2.0], [0.0, 0.0, 3.0]], dtype=torch.float32)
    assert torch.allclose(output, expected_output), f"Expected {expected_output}, but got {output}"


def test_jumprelu_backward(test_tensors, grad_output):
    x, threshold, kernel_fn, bandwith = test_tensors

    output = jump_relu(x, threshold, kernel_fn, bandwith)
    output.backward(grad_output)

    assert x.grad is not None, "Gradient for x is None"
    assert threshold.grad is not None, "Gradient for threshold is None"


@pytest.mark.parametrize("kernel_fn", all_kernels)
def test_jumprelu_kernels(kernel_fn, test_tensors):
    x, threshold, _, bandwith = test_tensors
    output = jump_relu(x, threshold, kernel_fn, bandwith)

    assert not torch.isnan(output).any(), f"{kernel_fn.__name__} contains NaN values."
    assert (output >= 0).all(), f"{kernel_fn.__name__} has negative values."


def test_heaviside_shape(test_tensors):
    x, threshold, kernel_fn, bandwith = test_tensors
    output = heaviside(x, threshold, kernel_fn, bandwith)

    assert output.shape == x.shape


def test_heaviside_non_nan(test_tensors):
    x, threshold, kernel_fn, bandwith = test_tensors
    output = heaviside(x, threshold, kernel_fn, bandwith)

    assert not torch.isnan(output).any()


def test_heaviside_value(test_tensors):
    x, threshold, kernel_fn, bandwith = test_tensors
    output = heaviside(x, threshold, kernel_fn, bandwith)

    expected_output = (x > threshold).float()
    assert torch.allclose(output, expected_output), f"Expected {expected_output}, but got {output}"


def test_heaviside_backward(test_tensors, grad_output):
    x, threshold, kernel_fn, bandwith = test_tensors

    output = heaviside(x, threshold, kernel_fn, bandwith)

    output.backward(grad_output)

    assert x.grad is not None, "Gradient for x is None"
    assert threshold.grad is not None, "Gradient for threshold is None"
    assert torch.allclose(x.grad, torch.zeros_like(x)), f"Expected grad_x 0, but got {x.grad}"


@pytest.mark.parametrize("kernel_fn", all_kernels)
def test_heaviside_kernels(kernel_fn, test_tensors):
    x, threshold, _, bandwith = test_tensors
    output = heaviside(x, threshold, kernel_fn, bandwith)

    assert not torch.isnan(output).any(), f"{kernel_fn.__name__} contains NaN values."
    assert (output >= 0).all(), f"{kernel_fn.__name__} has negative values."


def test_jumprelu_backward_specific_values():
    # random tensor to test:
    # for input 1 : first and second one fail, third one passes
    # for input 2 : all of them pass
    # however for the gradient wrt threshold:
    # for input 1: second is active
    # for input 2: first is active
    x = torch.tensor([[0.1, 0.6, 2.0], [1.4, 2.0, 3.0]], dtype=torch.float32, requires_grad=True)
    threshold = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, requires_grad=True)
    bandwith = 1.0

    output = jump_relu(x, threshold, rectangle_kernel, bandwith)

    grad_output = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
    output.backward(grad_output)

    expected_grad_x = torch.tensor([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
    assert torch.allclose(x.grad, expected_grad_x), f"Expected grad_x {expected_grad_x}, but got {x.grad}"

    # see the comment, only the first doors are active
    # minus sign come from the fact that:
    # 'increasing the threshold will decrease the output'
    expected_grad_threshold = torch.tensor([-1.0, -1.0, 0.0], dtype=torch.float32)
    assert torch.allclose(threshold.grad, expected_grad_threshold), "Grad Threshold error"
