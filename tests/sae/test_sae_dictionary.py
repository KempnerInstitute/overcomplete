import torch
import pytest

from overcomplete.sae import DictionaryLayer, SAE, QSAE, TopKSAE, JumpSAE, BatchTopKSAE, MpSAE

from ..utils import epsilon_equal

all_saes = [SAE, QSAE, TopKSAE, JumpSAE, BatchTopKSAE, MpSAE]


def test_dictionary_layer_initialization():
    nb_concepts = 10
    dimensions = 20
    device = 'cpu'
    normalization = 'l2'
    layer = DictionaryLayer(dimensions, nb_concepts, normalization=normalization, device=device)

    assert layer.nb_concepts == nb_concepts
    assert layer.in_dimensions == dimensions
    assert layer.device == device
    assert callable(layer.normalization)
    assert layer.get_dictionary().shape == (nb_concepts, dimensions)


def test_dictionary_layer_custom_normalization():
    def custom_normalization(x):
        return x / torch.max(torch.norm(x, p=2, dim=1, keepdim=True), torch.tensor(1.0))

    layer = DictionaryLayer(10, 20, normalization=custom_normalization)
    assert layer.normalization == custom_normalization


def test_dictionary_layer_forward():
    nb_concepts = 5
    dimensions = 10
    batch_size = 3

    layer = DictionaryLayer(dimensions, nb_concepts)
    z = torch.randn(batch_size, nb_concepts)
    x_hat = layer.forward(z)

    assert x_hat.shape == (batch_size, dimensions)


def test_dictionary_layer_get_dictionary():
    nb_concepts = 5
    dimensions = 10

    layer = DictionaryLayer(dimensions, nb_concepts, normalization='l2')
    dictionary = layer.get_dictionary()
    norms = torch.norm(dictionary, p=2, dim=1)

    expected_norms = torch.ones(nb_concepts)
    assert epsilon_equal(norms, expected_norms)


def test_dictionary_layer_normalizations():
    nb_concepts = 5
    dimensions = 10

    # Test 'l2' normalization
    layer_l2 = DictionaryLayer(dimensions, nb_concepts, normalization='l2')
    dictionary_l2 = layer_l2.get_dictionary()
    norms_l2 = torch.norm(dictionary_l2, p=2, dim=1)
    expected_norms_l2 = torch.ones(nb_concepts)
    assert epsilon_equal(norms_l2, expected_norms_l2)

    # Test 'max_l2' normalization
    layer_max_l2 = DictionaryLayer(dimensions, nb_concepts, normalization='max_l2')
    layer_max_l2._weights.data *= 2  # Set norms greater than 1
    dictionary_max_l2 = layer_max_l2.get_dictionary()
    norms_max_l2 = torch.norm(dictionary_max_l2, p=2, dim=1)
    assert torch.all(norms_max_l2 <= 1.0 + 1e-4)

    # Test 'l1' normalization
    layer_l1 = DictionaryLayer(dimensions, nb_concepts, normalization='l1')
    dictionary_l1 = layer_l1.get_dictionary()
    norms_l1 = torch.norm(dictionary_l1, p=1, dim=1)
    expected_norms_l1 = torch.ones(nb_concepts)
    assert epsilon_equal(norms_l1, expected_norms_l1)

    # Test 'max_l1' normalization
    layer_max_l1 = DictionaryLayer(dimensions, nb_concepts, normalization='max_l1')
    layer_max_l1._weights.data *= 2  # Set norms greater than 1
    dictionary_max_l1 = layer_max_l1.get_dictionary()
    norms_max_l1 = torch.norm(dictionary_max_l1, p=1, dim=1)
    assert torch.all(norms_max_l1 <= 1.0 + 1e-4)

    # Test 'identity' normalization
    layer_identity = DictionaryLayer(dimensions, nb_concepts, normalization='identity')
    dictionary_identity = layer_identity.get_dictionary()
    assert torch.equal(dictionary_identity, layer_identity._weights)


def test_dictionary_layer_get_dictionary_normalization():
    nb_concepts = 5
    dimensions = 10

    # Manually set weights
    layer = DictionaryLayer(dimensions, nb_concepts, normalization='l2')
    layer._weights.data = torch.randn(nb_concepts, dimensions)
    dictionary = layer.get_dictionary()
    norms = torch.norm(dictionary, p=2, dim=1)
    expected_norms = torch.ones(nb_concepts)
    assert epsilon_equal(norms, expected_norms)


@pytest.mark.parametrize("sae_class", all_saes)
def test_sae_init_dictionary_layer_normalizations(sae_class):
    nb_concepts = 5
    dimensions = 10

    # Test 'l2' normalization
    sae_l2 = sae_class(input_shape=dimensions, nb_concepts=nb_concepts,
                       dictionary_params={'normalization': 'l2'})

    dictionary_l2 = sae_l2.get_dictionary()
    norms_l2 = torch.norm(dictionary_l2, p=2, dim=1)
    expected_norms_l2 = torch.ones(nb_concepts)
    assert epsilon_equal(norms_l2, expected_norms_l2)

    # Test 'max_l2' normalization
    sae_max_l2 = sae_class(input_shape=dimensions, nb_concepts=nb_concepts,
                           dictionary_params={'normalization': 'max_l2'})
    dictionary_max_l2 = sae_max_l2.get_dictionary()
    norms_max_l2 = torch.norm(dictionary_max_l2, p=2, dim=1)
    assert torch.all(norms_max_l2 <= 1.0 + 1e-4)

    # Test 'l1' normalization
    sae_l1 = sae_class(input_shape=dimensions, nb_concepts=nb_concepts,
                       dictionary_params={'normalization': 'l1'})
    dictionary_l1 = sae_l1.get_dictionary()
    norms_l1 = torch.norm(dictionary_l1, p=1, dim=1)
    expected_norms_l1 = torch.ones(nb_concepts)
    assert epsilon_equal(norms_l1, expected_norms_l1)

    # Test 'max_l1' normalization
    sae_max_l1 = sae_class(input_shape=dimensions, nb_concepts=nb_concepts,
                           dictionary_params={'normalization': 'max_l1'})
    dictionary_max_l1 = sae_max_l1.get_dictionary()
    norms_max_l1 = torch.norm(dictionary_max_l1, p=1, dim=1)
    assert torch.all(norms_max_l1 <= 1.0 + 1e-4)

    # Test 'identity' normalization
    sae = sae_class(input_shape=dimensions, nb_concepts=nb_concepts,
                    dictionary_params={'normalization': 'identity'})
    dictionary_identity = sae.get_dictionary()
    assert torch.equal(dictionary_identity, sae.dictionary._weights)


@pytest.mark.parametrize("sae_class", all_saes)
def test_class_sae_dictionary_init(sae_class):
    # ensure every sae class can pass an initializer for the dictionary, and check if
    # the dictionary is correctly initialized

    nb_concepts = 10
    dimensions = 20
    seed = torch.randn(nb_concepts, dimensions)

    sae = sae_class(input_shape=dimensions, nb_concepts=nb_concepts,
                    dictionary_params={'initializer': seed, 'normalization': 'identity'})
    assert torch.equal(sae.get_dictionary(), seed)


@pytest.mark.parametrize("sae_class", [SAE, QSAE, TopKSAE, JumpSAE, BatchTopKSAE])
def test_class_sae_dictionary_multiplier(sae_class):
    # ensure every sae class can pass arg to make multiplier of dictionary trainable

    nb_concepts = 10
    dimensions = 20

    sae = sae_class(input_shape=dimensions, nb_concepts=nb_concepts)
    # default, multiplier is a buffer
    assert not sae.dictionary.multiplier.requires_grad

    sae = sae_class(input_shape=dimensions, nb_concepts=nb_concepts,
                    dictionary_params={'use_multiplier': True})
    # multiplier is a parameter
    assert sae.dictionary.multiplier.requires_grad


def test_dictionary_initialization():
    nb_concepts = 10
    dimensions = 20
    seed = torch.randn(nb_concepts, dimensions)

    dictionary = DictionaryLayer(dimensions, nb_concepts, initializer=seed, normalization='identity')
    assert torch.equal(dictionary.get_dictionary(), seed)


def test_multiplier_initial_value():
    """
    Test that when the multiplier is used, its initial value (0) results in an effective scaling of 1.
    """
    nb_concepts = 5
    dimensions = 10
    layer = DictionaryLayer(dimensions, nb_concepts, normalization='l2', use_multiplier=True)
    dictionary = layer.get_dictionary()
    expected = layer._weights
    assert epsilon_equal(dictionary, expected)


def test_multiplier_effect_on_dictionary():
    """
    Test that manually updating the multiplier changes the output dictionary accordingly.
    """
    nb_concepts = 5
    dimensions = 10
    layer = DictionaryLayer(dimensions, nb_concepts, normalization='l2', use_multiplier=True)
    new_value = 0.7  # arbitrary new multiplier value
    with torch.no_grad():
        layer.multiplier.copy_(torch.tensor(new_value, device=layer.device))
    dictionary = layer.get_dictionary()
    expected = layer._weights * torch.exp(torch.tensor(new_value, device=layer.device))
    assert epsilon_equal(dictionary, expected)


def test_multiplier_not_trainable_when_disabled():
    """
    Test that when use_multiplier is False, the multiplier is not a trainable parameter.
    """
    nb_concepts = 5
    dimensions = 10
    layer = DictionaryLayer(dimensions, nb_concepts, normalization='l2', use_multiplier=False)
    # The multiplier should be registered as a buffer, not a Parameter.
    assert not isinstance(layer.multiplier, torch.nn.Parameter)
    assert not layer.multiplier.requires_grad


def test_multiplier_gradient_computation():
    """
    Test that when use_multiplier is True, the multiplier receives gradients.
    """
    nb_concepts = 5
    dimensions = 10
    batch_size = 3
    layer = DictionaryLayer(dimensions, nb_concepts, normalization='l2', use_multiplier=True)
    # Ensure multiplier is trainable.
    assert layer.multiplier.requires_grad

    # Create a dummy latent code and compute a simple loss.
    z = torch.randn(batch_size, nb_concepts)
    x_hat = layer.forward(z)
    target = torch.zeros_like(x_hat)
    loss = ((x_hat - target) ** 2).mean()
    loss.backward()
    assert layer.multiplier.grad is not None


def test_multiplier_optimizer_step():
    """
    Test that an optimizer updates the multiplier during training.
    """
    nb_concepts = 5
    dimensions = 10
    batch_size = 3
    layer = DictionaryLayer(dimensions, nb_concepts, normalization='l2', use_multiplier=True)
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)

    # Record the initial multiplier value.
    init_multiplier = layer.multiplier.clone().detach()

    # Compute a dummy forward/backward pass.
    z = torch.randn(batch_size, nb_concepts)
    x_hat = layer.forward(z)
    target = torch.zeros_like(x_hat)
    loss = ((x_hat - target) ** 2).mean()
    loss.backward()
    optimizer.step()

    # Check that the multiplier has been updated.
    assert not torch.allclose(init_multiplier, layer.multiplier.detach(), atol=1e-6)
