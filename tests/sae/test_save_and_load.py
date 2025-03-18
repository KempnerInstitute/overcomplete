import os
import pytest

import torch
from overcomplete.sae import SAE, DictionaryLayer, JumpSAE, TopKSAE, QSAE, BatchTopKSAE, MpSAE, OMPSAE

from ..utils import epsilon_equal

all_sae = [SAE, JumpSAE, TopKSAE, QSAE, BatchTopKSAE, MpSAE, OMPSAE]


@pytest.mark.parametrize("nb_concepts, dimensions", [(5, 10)])
def test_save_and_load_dictionary_layer(nb_concepts, dimensions, tmp_path):
    # Initialize and run layer
    layer = DictionaryLayer(dimensions, nb_concepts)
    z = torch.randn(3, nb_concepts)
    x_hat = layer(z)

    # Validate output shape
    assert x_hat.shape == (3, dimensions)

    # Save to temporary file
    model_path = tmp_path / "test_dictionary_layer.pth"
    torch.save(layer, model_path)

    # Reload and validate
    layer = torch.load(model_path, map_location="cpu")
    assert isinstance(layer, DictionaryLayer)

    # Check consistency after loading
    x_hat_loaded = layer(z)
    assert epsilon_equal(x_hat, x_hat_loaded)


@pytest.mark.parametrize("sae_class", all_sae)
def test_save_and_load_sae(sae_class, tmp_path):
    input_size = 10
    nb_concepts = 5
    model = sae_class(input_size, nb_concepts)

    x = torch.randn(3, input_size)
    output = model(x)
    z_pre, z, x_hat = output

    # Save using tmp_path
    model_path = tmp_path / "test_sae.pth"
    torch.save(model, model_path)

    # Load and validate
    model_loaded = torch.load(model_path, map_location="cpu")
    assert isinstance(model_loaded, sae_class)

    # Run inference again
    output_loaded = model_loaded(x)
    z_pre_loaded, z_loaded, x_hat_loaded = output_loaded

    # Validate numerical consistency
    assert epsilon_equal(z, z_loaded)
    assert epsilon_equal(x_hat, x_hat_loaded)
    assert epsilon_equal(z_pre, z_pre_loaded)


@pytest.mark.parametrize("sae_class", all_sae)
def test_eval_and_save_sae(sae_class, tmp_path):
    input_size = 10
    nb_concepts = 5
    model = sae_class(input_size, nb_concepts)

    x = torch.randn(3, input_size)
    output = model(x)
    z_pre, z, x_hat = output

    # Save using tmp_path
    model_path = tmp_path / "test_sae.pth"
    torch.save(model, model_path)

    # Load, set to eval mode, and validate
    model_loaded = torch.load(model_path, map_location="cpu").eval()
    assert isinstance(model_loaded, sae_class)

    # Run inference again
    output_loaded = model_loaded(x)
    z_pre_loaded, z_loaded, x_hat_loaded = output_loaded

    # Validate numerical consistency
    assert epsilon_equal(z, z_loaded)
    assert epsilon_equal(x_hat, x_hat_loaded)
    assert epsilon_equal(z_pre, z_pre_loaded)
