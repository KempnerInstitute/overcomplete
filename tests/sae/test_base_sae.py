import pytest

import torch
from overcomplete.sae import SAE, DictionaryLayer, JumpSAE
from overcomplete.sae.base import SAEOutput

all_sae = [SAE, JumpSAE]


def test_dictionary_layer():
    nb_components = 5
    dimensions = 10
    layer = DictionaryLayer(nb_components, dimensions)
    z = torch.randn(3, nb_components)
    x_hat = layer(z)
    assert x_hat.shape == (3, dimensions)

    x = torch.randn(10, dimensions)
    layer.initialize_dictionary(x, method='svd')
    assert layer.get_dictionary().shape == (nb_components, dimensions)


@pytest.mark.parametrize("sae_class", all_sae)
def test_sae(sae_class):
    input_size = 10
    nb_components = 5
    model = sae_class(input_size, nb_components)

    x = torch.randn(3, input_size)
    output = model(x)

    assert isinstance(output, SAEOutput)

    z_pre, z, x_hat = output

    assert z.shape == (3, nb_components)
    assert x_hat.shape == (3, input_size)
    assert z_pre.shape == (3, nb_components)

    dictionary = model.get_dictionary()
    assert dictionary.shape == (nb_components, input_size)
