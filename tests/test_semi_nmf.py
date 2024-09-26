import pytest
import torch
from sklearn.decomposition import NMF as SkNMF

from overcomplete.optimization import SemiNMF
from overcomplete.metrics import relative_avg_l2_loss

data_shape = (50, 10)
n_components = 5

A = torch.rand(data_shape, dtype=torch.float32)

# Sklearn NMF for benchmarking
sk_model = SkNMF(n_components=n_components, init='random', solver='mu', max_iter=1000)
Z_sk = sk_model.fit_transform(A.numpy())
D_sk = sk_model.components_
sk_error = relative_avg_l2_loss(A, Z_sk @ D_sk)


def test_semi_nmf_initialization():
    """Test that the SemiNMF class initializes properly."""
    model = SemiNMF(n_components=n_components)
    assert model.n_components == n_components, "Incorrect number of components."


def test_semi_nmf_fit():
    """Test that the SemiNMF model can fit to the data."""
    model = SemiNMF(n_components=n_components, max_iter=2)
    Z, D = model.fit(A)
    assert D.shape == (n_components, data_shape[1]), "Incorrect shape for D."
    assert Z.shape == (data_shape[0], n_components), "Incorrect shape for Z."


def test_semi_nmf_encode_decode():
    """Test the encode and decode methods of the SemiNMF model."""
    model = SemiNMF(n_components=n_components, max_iter=2)
    model.fit(A)

    Z = model.encode(A)
    assert Z.shape == (data_shape[0], n_components), "Incorrect shape for encoded data."

    A_hat = model.decode(Z)
    assert A_hat.shape == (data_shape[0], data_shape[1]), "Incorrect shape for decoded data."


def test_semi_nmf_reconstruction_error():
    """Test that the reconstruction error decreases after fitting."""
    model = SemiNMF(n_components=n_components, max_iter=100)
    initial_error = torch.norm(A - model.init_random_z(A) @ model.init_random_d(A), 'fro')
    model.fit(A)
    Z = model.encode(A)
    A_hat = model.decode(Z)
    final_error = torch.norm(A - A_hat, 'fro')
    assert final_error < initial_error, "Reconstruction error did not decrease."


def test_semi_nmf_zero_data():
    """Test how the model handles data with zeros."""
    zero_data = torch.zeros(data_shape)
    model = SemiNMF(n_components=n_components)
    Z, D = model.fit(zero_data)
    reconstruction_error = torch.norm(zero_data - Z @ D, 'fro')
    assert reconstruction_error < 1e-5, "Model did not reconstruct zero data correctly."


def test_semi_nmf_large_number_of_components():
    """Test the SemiNMF model with varying number of components."""
    small_model = SemiNMF(n_components=1)
    small_model.fit(A)
    error_small = torch.square(A - small_model.decode(small_model.encode(A))).sum()

    large_model = SemiNMF(n_components=100)
    large_model.fit(A)
    error_large = torch.square(A - large_model.decode(large_model.encode(A))).sum()

    assert error_large <= error_small, "Higher error with more components."


def test_compare_to_sklearn(repetitions=10):
    """
    Test that SemiNMF achieves similar performance to sklearn NMF.
    """
    results = []
    for _ in range(repetitions):
        our_model = SemiNMF(n_components=n_components, max_iter=1000)
        Z, D = our_model.fit(A)
        our_error = relative_avg_l2_loss(A, Z @ D)
        results.append(our_error <= 5.0 * sk_error)

    assert any(results), "SemiNMF did not match sklearn NMF performance."
