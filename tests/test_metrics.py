import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

from overcomplete.metrics import (
    reconstruction_loss,
    relative_reconstruction_loss,
    sparsity,
    sparsity_eps,
    dead_codes,
    hungarian_loss,
    _max_non_diagonal,
    _cosine_distance_matrix,
    cosine_hungarian_loss,
    dictionary_collinearity,
    wasserstein_1d,
    codes_correlation_matrix,
    energy_of_codes
)

from .utils import epsilon_equal


def test_reconstruction_loss():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x_hat = torch.tensor([[1.0, 2.0], [2.0, 4.0]])

    # Avg(sqrt(x^2).sum(-1))
    expected_loss = 1.0 / 2.0

    assert epsilon_equal(reconstruction_loss(x, x_hat), expected_loss)


def test_relative_reconstruction_loss():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x_hat = torch.tensor([[1.0, 2.0], [2.0, 4.0]])

    l2_err_per_sample = torch.tensor([0.0, 1.0]).sqrt()
    l2_x = torch.tensor([5.0, 25.0]).sqrt()

    expected_loss = torch.mean(l2_err_per_sample / l2_x).item()

    assert epsilon_equal(relative_reconstruction_loss(x, x_hat), expected_loss)


def test_sparsity():
    x = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    expected_sparsity = 3 / 4

    assert epsilon_equal(sparsity(x), expected_sparsity)
    assert epsilon_equal(sparsity(x, dims=0), torch.tensor([1.0, 0.5]))


def test_sparsity_eps():
    x = torch.tensor([[0.0, 1.0], [0.0, 1e-7]])
    expected_sparsity = 3 / 4

    assert epsilon_equal(sparsity_eps(x, threshold=1e-6), expected_sparsity)
    assert epsilon_equal(sparsity_eps(x, dims=0, threshold=1e-6), torch.tensor([1.0, 0.5]))


def test_dead_codes():
    z = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    expected_dead_codes = torch.tensor([1.0, 0.0])
    assert torch.equal(dead_codes(z), expected_dead_codes)


def test_hungarian_loss():
    dict1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    dict2 = torch.tensor([[1.0, 2.0], [2.0, 4.0]])
    expected_loss = 1.0

    assert epsilon_equal(hungarian_loss(dict1, dict2), expected_loss)
    assert epsilon_equal(hungarian_loss(dict1, dict1), 0.0)


def test_max_non_diagonal():
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expected_max = 3.0

    assert _max_non_diagonal(matrix) == expected_max


def test_cosine_distance_matrix():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    y = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    expected_matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    assert epsilon_equal(_cosine_distance_matrix(x, y), expected_matrix)


def test_cosine_hungarian_loss():
    dict1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    dict2 = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    # just a permutation to go from one to the other
    assert cosine_hungarian_loss(dict1, dict1) == 0.0
    assert cosine_hungarian_loss(dict1, dict2) == 0.0


def test_dictionary_collinearity():
    dict1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    dict2 = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    max_col, _ = dictionary_collinearity(dict1)
    assert epsilon_equal(max_col, 0.0, 1e-3)

    max_col, _ = dictionary_collinearity(dict2)
    assert epsilon_equal(max_col, 1.0, 1e-3)


def test_wasserstein_1d():
    x1 = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
    x2 = torch.tensor([[2.0, 3.0], [2.0, 3.0]])

    assert epsilon_equal(wasserstein_1d(x1, x2), 1.0)


def test_codes_correlation_matrix():
    codes = torch.tensor([[0.0, 1.0],
                          [1.0, 0.0]])
    expected_max_corr = 1.0  # abs(-1) correlation
    max_corr, m = codes_correlation_matrix(codes)

    assert epsilon_equal(max_corr, expected_max_corr, 1e-4)


def test_energy_of_codes():
    codes = torch.tensor([[1.0, 0.0],
                          [0.0, 1.0]])
    dictionary = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    expected_energy = torch.tensor([((0.5 * 1.0)**2 + (0.5 * 2.0)**2)**0.5,
                                    ((0.5 * 3.0)**2 + (0.5 * 4.0)**2)**0.5])

    assert epsilon_equal(energy_of_codes(codes, dictionary), expected_energy)
