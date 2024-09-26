import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import NMF as SkNMF

from overcomplete.metrics import relative_avg_l2_loss
from overcomplete.optimization import NMF


data_shape = (50, 10)
n_components = 5

A = torch.rand(data_shape, dtype=torch.float32)
optimizers = ['hals', 'mu', 'pgd', 'anls']

sk_model = SkNMF(n_components=n_components, max_iter=1000, solver='mu')
Z_sk = sk_model.fit_transform(A.numpy())
D_sk = sk_model.components_
sk_error = relative_avg_l2_loss(A, Z_sk @ D_sk)


@pytest.mark.parametrize("optimizer", optimizers)
def test_nmf_initialization(optimizer):
    """Test that the NMF class initializes properly with valid parameters."""
    model = NMF(n_components=n_components, optimizer=optimizer)
    assert model.n_components == n_components, "Number of components not set correctly"
    assert model.optimizer == optimizer, "Optimizer not set correctly"


def test_nmf_invalid_optimizer():
    """Test that initializing NMF with an invalid optimizer raises an error."""
    with pytest.raises(AssertionError):
        NMF(n_components=n_components, optimizer='invalid_optimizer')


@pytest.mark.parametrize("optimizer", optimizers)
def test_nmf_fit(optimizer):
    """Test that the NMF model can fit to the data."""
    model = NMF(n_components=n_components, optimizer=optimizer, max_iter=2)
    Z, D = model.fit(A)
    assert D.shape == (n_components, data_shape[1]), "Dictionary D has incorrect shape after fitting"
    assert Z.shape == (data_shape[0], n_components), "Codes Z have incorrect shape after fitting"


def test_nmf_encode_decode():
    """Test the encode and decode methods of the NMF model."""
    model = NMF(n_components=n_components, max_iter=2)
    model.fit(A)

    Z = model.encode(A)
    assert Z.shape == (data_shape[0], n_components), "Encoded data has incorrect shape"
    assert (Z >= 0).all(), "Encoded data contains negative values"

    A_hat = model.decode(Z)
    assert A_hat.shape == (data_shape[0], data_shape[1]), "Decoded data has incorrect shape"
    assert (A_hat >= 0).all(), "Decoded data contains negative values"


@pytest.mark.parametrize("optimizer", optimizers)
def test_nmf_reconstruction_error(optimizer):
    """Test that the reconstruction error decreases after fitting."""
    model = NMF(n_components=n_components, optimizer=optimizer, max_iter=100)
    initial_error = torch.norm(A - model.init_random_z(A) @ model.init_random_d(A), 'fro')
    model.fit(A)
    Z = model.encode(A)
    A_hat = model.decode(Z)
    final_error = torch.norm(A - A_hat, 'fro')
    assert final_error < initial_error, "Reconstruction error did not decrease after fitting"


def test_nmf_zero_data():
    """Test how the model handles data with zeros."""
    zero_data = torch.zeros(data_shape)
    model = NMF(n_components=n_components)
    Z, D = model.fit(zero_data)
    assert torch.norm(Z) == 0, "Codes Z should be zero for zero input data"
    assert torch.norm(D) == 0, "Dictionary D should be zero for zero input data"


@pytest.mark.parametrize("optimizer", optimizers)
def test_nmf_large_number_of_components(optimizer):
    """Test the NMF model with a number of components equal to the number of features."""
    little_nmf = NMF(n_components=1, optimizer=optimizer)
    little_nmf.fit(A)
    error_little = torch.square(A - little_nmf.decode(little_nmf.encode(A))).sum()

    big_nmf = NMF(n_components=100, optimizer=optimizer)
    big_nmf.fit(A)
    error_big = torch.square(A - big_nmf.decode(big_nmf.encode(A))).sum()

    assert error_big <= error_little, "Reconstruction error is higher for maximal components"


@pytest.mark.parametrize("optimizer", optimizers)
def test_compare_to_sklearn(optimizer, repetitions=10):
    """Test that each NMF method can at least achieve one time better or similar perf to sklearn model."""
    results = []
    for _ in range(repetitions):
        our_nmf = NMF(n_components=n_components, optimizer=optimizer, max_iter=10_000)
        Z, D = our_nmf.fit(A)
        our_error = relative_avg_l2_loss(A, Z @ D)
        results.append(our_error <= 5.0 * sk_error)

    assert any(results), f"The {optimizer} runs can't achieved similar performance to sklearn NMF"
