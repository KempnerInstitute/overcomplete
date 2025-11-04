import pytest

import torch
from overcomplete.sae import SAE, DictionaryLayer, JumpSAE, TopKSAE, QSAE, BatchTopKSAE, MpSAE, OMPSAE
from overcomplete.sae.modules import TieableEncoder

from ..utils import epsilon_equal

all_sae = [SAE, JumpSAE, TopKSAE, QSAE, BatchTopKSAE, MpSAE, OMPSAE]


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


def test_tieable_encoder_basic():
    """Test TieableEncoder can be created in both tied and untied modes."""
    input_size = 10
    nb_concepts = 5

    # Create a dummy dictionary layer
    dictionary = DictionaryLayer(input_size, nb_concepts)

    # Test untied mode
    encoder_untied = TieableEncoder(input_size, nb_concepts, tied_to=None)
    assert encoder_untied.weight is not None
    assert encoder_untied.tied_to is None

    # Test tied mode
    encoder_tied = TieableEncoder(input_size, nb_concepts, tied_to=dictionary)
    assert encoder_tied.weight is None
    assert encoder_tied.tied_to is dictionary


def test_tieable_encoder_forward():
    """Test TieableEncoder forward pass in both modes."""
    input_size = 10
    nb_concepts = 5
    batch_size = 3

    dictionary = DictionaryLayer(input_size, nb_concepts)
    x = torch.randn(batch_size, input_size)

    # Test untied forward
    encoder_untied = TieableEncoder(input_size, nb_concepts, tied_to=None)
    z_pre, z = encoder_untied(x)
    assert z_pre.shape == (batch_size, nb_concepts)
    assert z.shape == (batch_size, nb_concepts)
    assert (z >= 0).all()  # ReLU activation

    # Test tied forward
    encoder_tied = TieableEncoder(input_size, nb_concepts, tied_to=dictionary)
    z_pre, z = encoder_tied(x)
    assert z_pre.shape == (batch_size, nb_concepts)
    assert z.shape == (batch_size, nb_concepts)
    assert (z >= 0).all()


@pytest.mark.parametrize("sae_class", all_sae)
def test_sae_tied_untied(sae_class):
    """Test that SAE can switch between tied and untied modes."""
    input_size = 10
    nb_concepts = 5

    model = sae_class(input_size, nb_concepts)

    # Tie weights
    model.tied()
    assert isinstance(model.encoder, TieableEncoder)
    assert model.encoder.tied_to is not None

    # Untie weights
    model.untied()
    assert isinstance(model.encoder, TieableEncoder)
    assert model.encoder.tied_to is None


@pytest.mark.parametrize("sae_class", all_sae)
def test_sae_tied_forward(sae_class):
    """Test that tied SAE produces valid outputs."""
    input_size = 10
    nb_concepts = 5

    model = sae_class(input_size, nb_concepts)
    model.tied()

    x = torch.randn(3, input_size)
    z_pre, z, x_hat = model(x)

    assert z.shape == (3, nb_concepts)
    assert x_hat.shape == (3, input_size)


@pytest.mark.parametrize("sae_class", all_sae)
def test_sae_untied_copy_weights(sae_class):
    """Test that untied with copy_from_dictionary copies weights correctly."""
    input_size = 10
    nb_concepts = 5

    model = sae_class(input_size, nb_concepts)
    model.tied()

    # Get dictionary before untying
    dict_before = model.get_dictionary().clone()

    # Untie and copy
    model.untied(copy_from_dictionary=True)

    # Check that encoder weights match dictionary
    assert epsilon_equal(model.encoder.weight, dict_before)
