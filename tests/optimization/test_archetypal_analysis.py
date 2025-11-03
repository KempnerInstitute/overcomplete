import pytest
import torch
import numpy as np

from overcomplete.metrics import relative_avg_l2_loss
from overcomplete.optimization.archetypal_analysis import ArchetypalAnalysis, project_simplex

data_shape = (50, 10)
nb_concepts = 5
A = torch.rand(data_shape, dtype=torch.float32)


def test_archetypal_initialization():
    model = ArchetypalAnalysis(nb_concepts=nb_concepts)
    assert model.nb_concepts == nb_concepts


def test_archetypal_fit_shapes():
    model = ArchetypalAnalysis(nb_concepts=nb_concepts)
    Z, D = model.fit(A)
    assert Z.shape == (data_shape[0], nb_concepts)
    assert D.shape == (nb_concepts, data_shape[1])


def test_archetypal_encoding():
    model = ArchetypalAnalysis(nb_concepts=nb_concepts)
    model.fit(A)
    Z = model.encode(A)
    assert Z.shape == (data_shape[0], nb_concepts)
    row_sums = torch.sum(Z, dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)


def test_archetypal_decoding():
    model = ArchetypalAnalysis(nb_concepts=nb_concepts)
    model.fit(A)
    Z = model.encode(A)
    A_hat = model.decode(Z)
    assert A_hat.shape == A.shape


def test_archetypal_simplex_W():
    model = ArchetypalAnalysis(nb_concepts=nb_concepts)
    model.fit(A)
    W = model.W
    row_sums = torch.sum(W, dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)
    assert torch.all(W >= 0)


@pytest.mark.flaky(reruns=9, reruns_delay=0)
def test_archetypal_loss_decrease():
    model = ArchetypalAnalysis(nb_concepts=nb_concepts)
    Z0 = model.init_random_z(A)
    W0 = model.init_random_w(A)
    D0 = W0 @ A
    initial_loss = torch.mean((A - Z0 @ D0).pow(2)).item()
    Z, D = model.fit(A)
    final_loss = torch.mean((A - Z @ D).pow(2)).item()
    assert final_loss < initial_loss


def test_project_simplex_behavior():
    W = torch.randn(20, 10)
    P = project_simplex(W)
    assert torch.allclose(torch.sum(P, dim=1), torch.ones(P.size(0)), atol=1e-3)
    assert torch.all(P >= 0)


def test_archetypal_zero_input():
    A_zero = torch.zeros_like(A)
    model = ArchetypalAnalysis(nb_concepts=nb_concepts)
    Z, D = model.fit(A_zero)
    assert torch.allclose(Z@D, A_zero, atol=1e-4)


def test_archetypal_doubly_stoch():
    model = ArchetypalAnalysis(nb_concepts=nb_concepts)
    Z, D = model.fit(A)
    W = model.W

    assert Z.shape == (data_shape[0], nb_concepts)
    assert D.shape == (nb_concepts, data_shape[1])
    assert W.shape == (nb_concepts, data_shape[0])

    assert torch.all(Z >= 0)
    assert torch.all(W >= 0)

    assert torch.allclose(torch.sum(Z, dim=1), torch.ones(data_shape[0]), atol=1e-3)
    assert torch.allclose(torch.sum(W, dim=1), torch.ones(nb_concepts), atol=1e-3)
