import pytest

import torch
from overcomplete.sae import SAE, DictionaryLayer, JumpSAE, TopKSAE, QSAE, BatchTopKSAE, MpSAE

all_sae = [SAE, JumpSAE, TopKSAE, QSAE, BatchTopKSAE, MpSAE]


def test_dictionary_layer():
    nb_concepts = 5
    in_dimensions = 10
    layer = DictionaryLayer(in_dimensions, nb_concepts)
    z = torch.randn(3, nb_concepts)
    x_hat = layer(z)
    assert x_hat.shape == (3, in_dimensions)
    assert layer.get_dictionary().shape == (nb_concepts, in_dimensions)


@pytest.mark.parametrize("sae_class", all_sae)
def test_sae(sae_class):
    input_size = 10
    nb_concepts = 5
    model = sae_class(input_size, nb_concepts)

    x = torch.randn(3, input_size)
    output = model(x)

    z_pre, z, x_hat = output

    assert z.shape == (3, nb_concepts)
    assert x_hat.shape == (3, input_size)
    assert z_pre.shape == (3, nb_concepts)

    dictionary = model.get_dictionary()
    assert dictionary.shape == (nb_concepts, input_size)


@pytest.mark.parametrize("sae_class", all_sae)
def test_sae_device(sae_class):
    " Use meta device to test moving the model to a different device"
    input_size = 10
    nb_components = 5

    model = sae_class(input_size, nb_components, device='meta')

    # ensure dictionary is on the meta device
    dictionary = model.get_dictionary()
    assert dictionary.device.type == 'meta'

    model = sae_class(input_size, nb_components, device='cpu')

    # ensure dictionary is on the cpu device
    dictionary = model.get_dictionary()
    assert dictionary.device.type == 'cpu'

    model.to('meta')

    # ensure dictionary is on the meta device
    dictionary = model.get_dictionary()
    assert dictionary.device.type == 'meta'
