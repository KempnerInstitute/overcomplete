import os
import pytest

import torch
from overcomplete.sae import SAE, DictionaryLayer, JumpSAE, TopKSAE, QSAE, BatchTopKSAE, MpSAE

from ..utils import epsilon_equal

all_sae = [SAE, JumpSAE, TopKSAE, QSAE, BatchTopKSAE, MpSAE]


def test_save_and_load_dictionary_layer():
    nb_concepts = 5
    dimensions = 10
    layer = DictionaryLayer(dimensions, nb_concepts)
    z = torch.randn(3, nb_concepts)
    x_hat = layer(z)
    assert x_hat.shape == (3, dimensions)

    torch.save(layer, 'test_dictionary_layer.pth')
    layer = torch.load('test_dictionary_layer.pth')
    assert isinstance(layer, DictionaryLayer)

    x_hat_loaded = layer(z)
    assert epsilon_equal(x_hat, x_hat_loaded)

    os.remove('test_dictionary_layer.pth')


@pytest.mark.parametrize("sae_class", all_sae)
def test_save_and_load_sae(sae_class):
    input_size = 10
    nb_concepts = 5
    model = sae_class(input_size, nb_concepts)

    x = torch.randn(3, input_size)
    output = model(x)
    z_pre, z, x_hat = output

    torch.save(model, 'test_sae.pth')
    model_loaded = torch.load('test_sae.pth')
    assert isinstance(model_loaded, sae_class)

    output_loaded = model_loaded(x)
    z_pre_loaded, z_loaded, x_hat_loaded = output_loaded

    assert epsilon_equal(z, z_loaded)
    assert epsilon_equal(x_hat, x_hat_loaded)
    assert epsilon_equal(z_pre, z_pre_loaded)

    os.remove('test_sae.pth')


@pytest.mark.parametrize("sae_class", all_sae)
def test_eval_and_save_sae(sae_class):
    input_size = 10
    nb_concepts = 5
    model = sae_class(input_size, nb_concepts)

    x = torch.randn(3, input_size)
    output = model(x)
    z_pre, z, x_hat = output

    torch.save(model, 'test_sae.pth')
    model_loaded = torch.load('test_sae.pth').eval().cpu()
    assert isinstance(model_loaded, sae_class)

    output_loaded = model_loaded(x)
    z_pre_loaded, z_loaded, x_hat_loaded = output_loaded

    assert epsilon_equal(z, z_loaded)
    assert epsilon_equal(x_hat, x_hat_loaded)
    assert epsilon_equal(z_pre, z_pre_loaded)

    os.remove('test_sae.pth')
