import torch
import pytest

from overcomplete.sae import DictionaryLayer

from .utils import epsilon_equal


def test_dictionary_layer_initialization():
    nb_components = 10
    dimensions = 20
    device = 'cpu'
    normalize = 'l2'
    layer = DictionaryLayer(nb_components, dimensions, normalize, device)

    assert layer.nb_components == nb_components
    assert layer.dimensions == dimensions
    assert layer.device == device
    assert callable(layer.normalize)
    assert layer.get_dictionary().shape == (nb_components, dimensions)


def test_dictionary_layer_custom_normalization():
    def custom_normalize(x):
        return x / torch.max(torch.norm(x, p=2, dim=1, keepdim=True), torch.tensor(1.0))

    layer = DictionaryLayer(10, 20, normalize=custom_normalize)
    assert layer.normalize == custom_normalize


def test_dictionary_layer_forward():
    nb_components = 5
    dimensions = 10
    batch_size = 3

    layer = DictionaryLayer(nb_components, dimensions)
    z = torch.randn(batch_size, nb_components)
    x_hat = layer.forward(z)

    assert x_hat.shape == (batch_size, dimensions)


def test_dictionary_layer_get_dictionary():
    nb_components = 5
    dimensions = 10

    layer = DictionaryLayer(nb_components, dimensions, normalize='l2')
    dictionary = layer.get_dictionary()
    norms = torch.norm(dictionary, p=2, dim=1)

    expected_norms = torch.ones(nb_components)
    assert epsilon_equal(norms, expected_norms)


def test_dictionary_layer_normalizations():
    nb_components = 5
    dimensions = 10

    # Test 'l2' normalization
    layer_l2 = DictionaryLayer(nb_components, dimensions, normalize='l2')
    dictionary_l2 = layer_l2.get_dictionary()
    norms_l2 = torch.norm(dictionary_l2, p=2, dim=1)
    expected_norms_l2 = torch.ones(nb_components)
    assert epsilon_equal(norms_l2, expected_norms_l2)

    # Test 'max_l2' normalization
    layer_max_l2 = DictionaryLayer(nb_components, dimensions, normalize='max_l2')
    layer_max_l2._weights.data *= 2  # Set norms greater than 1
    dictionary_max_l2 = layer_max_l2.get_dictionary()
    norms_max_l2 = torch.norm(dictionary_max_l2, p=2, dim=1)
    print(norms_max_l2)
    assert torch.all(norms_max_l2 <= 1.0 + 1e-4)

    # Test 'l1' normalization
    layer_l1 = DictionaryLayer(nb_components, dimensions, normalize='l1')
    dictionary_l1 = layer_l1.get_dictionary()
    norms_l1 = torch.norm(dictionary_l1, p=1, dim=1)
    expected_norms_l1 = torch.ones(nb_components)
    assert epsilon_equal(norms_l1, expected_norms_l1)

    # Test 'max_l1' normalization
    layer_max_l1 = DictionaryLayer(nb_components, dimensions, normalize='max_l1')
    layer_max_l1._weights.data *= 2  # Set norms greater than 1
    dictionary_max_l1 = layer_max_l1.get_dictionary()
    norms_max_l1 = torch.norm(dictionary_max_l1, p=1, dim=1)
    assert torch.all(norms_max_l1 <= 1.0 + 1e-4)

    # Test 'identity' normalization
    layer_identity = DictionaryLayer(nb_components, dimensions, normalize='identity')
    dictionary_identity = layer_identity.get_dictionary()
    assert torch.equal(dictionary_identity, layer_identity._weights)


def test_dictionary_layer_initialize_dictionary_svd():
    nb_components = 2
    dimensions = 3
    layer = DictionaryLayer(nb_components, dimensions)
    x = torch.randn(20, dimensions)

    layer.initialize_dictionary(x, method='svd')
    dictionary = layer.get_dictionary()
    assert dictionary.shape == (nb_components, dimensions)


def test_dictionary_layer_initialize_dictionary_custom_method():
    class DummyDictionaryLearning:
        def __init__(self, nb_components):
            self.nb_components = nb_components

        def fit(self, x):
            self.dictionary = torch.randn(self.nb_components, x.shape[1])

        def get_dictionary(self):
            return self.dictionary

    nb_components = 5
    dimensions = 10
    layer = DictionaryLayer(nb_components, dimensions)
    x = torch.randn(100, dimensions)
    custom_method = DummyDictionaryLearning(nb_components)

    layer.initialize_dictionary(x, method=custom_method)
    dictionary = layer.get_dictionary()
    assert dictionary.shape == (nb_components, dimensions)


def test_dictionary_layer_get_dictionary_normalization():
    nb_components = 5
    dimensions = 10

    # Manually set weights
    layer = DictionaryLayer(nb_components, dimensions, normalize='l2')
    layer._weights.data = torch.randn(nb_components, dimensions)
    dictionary = layer.get_dictionary()
    norms = torch.norm(dictionary, p=2, dim=1)
    expected_norms = torch.ones(nb_components)
    assert epsilon_equal(norms, expected_norms)
