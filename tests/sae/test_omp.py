import pytest
import torch

from overcomplete.sae.omp_sae import OMPSAE


@pytest.fixture
def dummy_input():
    torch.manual_seed(42)
    x = torch.randn(4, 10)
    return x


@pytest.fixture
def model():
    return OMPSAE(input_shape=10, nb_concepts=6, k=3, dropout=None, device='cpu', max_iter=15)


def test_ompsae_output_shapes(model, dummy_input):
    residual, codes = model.encode(dummy_input)
    assert residual.shape == dummy_input.shape
    assert codes.shape == (dummy_input.shape[0], model.nb_concepts)


def test_ompsae_no_nan(model, dummy_input):
    residual, codes = model.encode(dummy_input)
    assert not torch.isnan(residual).any()
    assert not torch.isnan(codes).any()


def test_ompsae_overridden_k(dummy_input):
    model = OMPSAE(input_shape=10, nb_concepts=8, k=2)
    _, codes1 = model.encode(dummy_input)
    _, codes2 = model.encode(dummy_input, k=5)
    # more iterations should generally give denser codes (more nonzeros)
    assert codes2.abs().sum() >= codes1.abs().sum()


def test_ompsae_overridden_max_iter(dummy_input):
    model = OMPSAE(input_shape=10, nb_concepts=8, k=3, max_iter=1)
    _, codes_low_iter = model.encode(dummy_input)
    _, codes_high_iter = model.encode(dummy_input, max_iter=20)

    # not a strict check, but high max_iter should give smaller residual
    residual_low = dummy_input - codes_low_iter @ model.get_dictionary()
    residual_high = dummy_input - codes_high_iter @ model.get_dictionary()
    assert residual_high.norm() <= residual_low.norm() + 1e-4


def test_ompsae_dropout_effect(dummy_input):
    model = OMPSAE(input_shape=10, nb_concepts=10, k=10, dropout=0.9)
    model.eval()
    residual_1, codes_1 = model.encode(dummy_input)
    residual_2, codes_2 = model.encode(dummy_input)

    assert torch.sum(torch.square(codes_1 - codes_2)) < 1e-5, "Dropout in eval did affect output"
