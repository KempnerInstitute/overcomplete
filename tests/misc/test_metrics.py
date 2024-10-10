import numpy as np
import pytest
import torch
import scipy

from overcomplete.metrics import (
    avg_l2_loss,
    avg_l1_loss,
    relative_avg_l1_loss,
    relative_avg_l2_loss,
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
    energy_of_codes,
    frechet_distance,
    l0,
    l1,
    l2,
    lp,
    l1_l2_ratio,
    hoyer,
    kappa_4,
    r2_score
)

from ..utils import epsilon_equal


def test_l2_loss():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x_hat = torch.tensor([[1.0, 2.0], [2.0, 4.0]])

    # Avg(sqrt(x^2).sum(-1))
    expected_loss = 1.0 / 2.0

    assert epsilon_equal(avg_l2_loss(x, x_hat), expected_loss)


def test_relative_l2_loss():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x_hat = torch.tensor([[1.0, 2.0], [2.0, 4.0]])

    l2_err_per_sample = torch.tensor([0.0, 1.0]).sqrt()
    l2_x = torch.tensor([5.0, 25.0]).sqrt()

    expected_loss = torch.mean(l2_err_per_sample / l2_x).item()

    assert epsilon_equal(relative_avg_l2_loss(x, x_hat), expected_loss)


def test_l1_loss():
    x = torch.tensor([[1.0, 2.0],
                      [3.0, 4.0]])
    x_hat = torch.tensor([[1.0, 2.0],
                          [2.0, 4.0]])

    # Avg(abs(x).sum(-1))
    expected_loss = 1.0 / 2.0

    assert epsilon_equal(avg_l1_loss(x, x_hat), expected_loss)


def test_relative_l1_loss():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x_hat = torch.tensor([[1.0, 2.0], [2.0, 4.0]])

    l1_err_per_sample = torch.tensor([0.0, 1.0])
    l1_x = torch.tensor([3.0, 7.0])

    expected_loss = torch.mean(l1_err_per_sample / l1_x).item()

    assert epsilon_equal(relative_avg_l1_loss(x, x_hat), expected_loss)


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


def test_frechet_distance():
    mean1 = torch.tensor([0.0, 0.0])
    cov1 = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    mean2 = torch.tensor([1.0, 1.0])
    cov2 = torch.tensor([[1.5, 0.3], [0.3, 1.5]])

    size = 1_000_000

    # Generate random samples
    x1 = torch.distributions.MultivariateNormal(mean1, cov1).sample((size,))
    x2 = torch.distributions.MultivariateNormal(mean2, cov2).sample((size,))

    # Calculate the expected Frechet distance
    mean_diff = mean1 - mean2
    mean_diff_squared = torch.sum(mean_diff ** 2)

    cov_prod = torch.matmul(cov1, cov2)

    # pytorch don't have sqrtm, we use sum of sqrt of eigvals
    cov_prod_sqrt = scipy.linalg.sqrtm(cov_prod.cpu().numpy())
    cov_prod_sqrt = torch.tensor(cov_prod_sqrt)

    expected_distance = mean_diff_squared + torch.trace(cov1 + cov2 - 2 * cov_prod_sqrt)

    computed_distance = frechet_distance(x1, x2)

    assert epsilon_equal(computed_distance, expected_distance, epsilon=1e-1)

    x1 = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    x2 = torch.tensor([[0.0, 0.0], [0.0, 0.0]])

    expected_distance = 0.0
    assert epsilon_equal(frechet_distance(x1, x2), expected_distance, 1e-1)

    x1 = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
    x2 = torch.tensor([[2.0, 3.0], [2.0, 3.0]])

    assert frechet_distance(x1, x2) > 0.0


def test_l0():
    x = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    expected_sparsity = 0.5  # half of the elements are zero
    assert epsilon_equal(l0(x), torch.tensor(expected_sparsity))


def test_l2():
    v = torch.tensor([[3.0, 4.0], [0.0, 5.0]])
    expected_norm = torch.tensor([5.0, 5.0])
    assert epsilon_equal(l2(v, dims=1), expected_norm)
    assert epsilon_equal(l2(v), torch.tensor(torch.sqrt(torch.tensor(50.0))))


def test_l1():
    v = torch.tensor([[3.0, 4.0], [0.0, 5.0]])
    expected_norm = torch.tensor([7.0, 5.0])
    assert epsilon_equal(l1(v, dims=1), expected_norm)
    assert epsilon_equal(l1(v), torch.tensor(12.0))


def test_lp():
    v = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert epsilon_equal(lp(v, p=2, dims=1), l2(v, dims=1))
    assert epsilon_equal(lp(v, p=2), l2(v))

    for p in [0.1, 0.5, 0.8]:
        val = lp(v, p=p)
        assert torch.isnan(val).sum() == 0


def test_l1_l2_ratio():
    x = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    expected_ratio = torch.tensor([1.0, 2 / torch.sqrt(torch.tensor(2.0))])
    assert epsilon_equal(l1_l2_ratio(x, dims=1), expected_ratio)


def test_hoyer():
    x = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    expected_hoyer = torch.tensor([1.0, 0.0])
    assert epsilon_equal(hoyer(x), expected_hoyer, epsilon=1e-4)


def test_kappa_4():
    x = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    expected_kappa = torch.tensor([0.5, 1.0])
    assert epsilon_equal(kappa_4(x), expected_kappa)


def test_r2_score_perfect_reconstruction():
    # Perfect reconstruction
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x_hat = x.clone()
    assert epsilon_equal(r2_score(x, x_hat), 1.0)


def test_r2_score_zero_reconstruction():
    # Mean reconstruction
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x_hat = x.mean() * torch.ones_like(x)
    assert epsilon_equal(r2_score(x, x_hat), 0.0)


def test_r2_worst_than_mean_reconstruction():
    # Worst than mean reconstruction
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x_hat = torch.zeros_like(x)
    r2 = r2_score(x, x_hat)
    assert r2 <= 0


def test_r2_score_partial_reconstruction():
    # Partial reconstruction
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x_hat = torch.tensor([[1.0, 1.5], [3.0, 3.5]])
    r2 = r2_score(x, x_hat)
    assert 0 < r2 < 1
