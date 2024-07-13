import torch
from overcomplete.sae import SAE, DictionaryLayer


def test_dictionary_layer():
    nb_components = 5
    dimensions = 10
    layer = DictionaryLayer(nb_components, dimensions)
    z = torch.randn(3, nb_components)
    x_hat = layer(z)
    assert x_hat.shape == (3, dimensions)

    x = torch.randn(10, dimensions)
    layer.initialize_dictionary(x, method='svd')
    assert layer.dictionary.shape == (nb_components, dimensions)


def test_sae():
    input_size = 10
    nb_components = 5
    model = SAE(input_size, nb_components)

    x = torch.randn(3, input_size)
    z, x_hat = model(x)
    assert z.shape == (3, nb_components)
    assert x_hat.shape == (3, input_size)

    dictionary = model.get_dictionary()
    assert dictionary.shape == (nb_components, input_size)
