import pytest
from torch import nn
from overcomplete.sae.factory import SAEFactory
from overcomplete.sae.modules import MLPEncoder, ResNetEncoder, AttentionEncoder


def contains_batchnorm(module):
    return any(isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)) for layer in module.modules())


def contains_layernorm(module):
    return any(isinstance(layer, nn.LayerNorm) for layer in module.modules())


@pytest.mark.parametrize("model_name", [
    "mlp_bn_1", "mlp_bn_3", "mlp_bn_3_gelu", "resnet_3b", "mlp_bn_3_gelu_no_res"
])
def test_contains_batchnorm(model_name):
    input_shape = (3, 32, 32) if "resnet" in model_name else 10
    model = SAEFactory.create_module(model_name, input_shape=input_shape, n_components=2)
    assert contains_batchnorm(model), f"Model {model_name} does not contain a BatchNorm layer"


@pytest.mark.parametrize("model_name", [
    "mlp_ln_1", "mlp_ln_3", "attention_1b", "attention_3b", "mlp_ln_3_gelu_no_res"
])
def test_contains_layer_norm(model_name):
    input_shape = (10, 16) if "attention" in model_name else 10
    model = SAEFactory.create_module(model_name, input_shape=input_shape, n_components=2)
    assert contains_layernorm(model), f"Model {model_name} does not contain a LayerNorm layer"


@pytest.mark.parametrize("model_name, expected_blocks", [
    ("mlp_ln_1", 1),
    ("mlp_bn_3", 3),
    ("resnet_3b", 3),
    ("attention_1b", 1),
    ("mlp_bn_3_gelu", 3),
])
def test_number_of_blocks(model_name, expected_blocks):
    if "resnet" in model_name:
        input_shape = (3, 32, 32)
    elif "attention" in model_name:
        input_shape = (10, 16)
    else:
        input_shape = 10

    model = SAEFactory.create_module(model_name, input_shape=input_shape, n_components=2)

    # Count the number of MLP blocks, ResNet blocks, or Attention blocks
    if isinstance(model, MLPEncoder):
        num_blocks = len(model.mlp_blocks)
    elif isinstance(model, ResNetEncoder):
        num_blocks = len(model.resnet_blocks)
    elif isinstance(model, AttentionEncoder):
        num_blocks = len(model.attention_blocks)
    else:
        num_blocks = 0

    if "mlp" in model_name:
        num_blocks = num_blocks + 1

    assert num_blocks == expected_blocks, f"Model {model_name} expected {expected_blocks} blocks, \
                                                found {num_blocks}"


@pytest.mark.parametrize("model_name", SAEFactory.list_modules())
def test_model_creation(model_name):
    if "resnet" in model_name:
        input_shape = (3, 32, 32)
    elif "attention" in model_name:
        input_shape = (10, 16)
    else:
        input_shape = 10

    model = SAEFactory.create_module(model_name, input_shape=input_shape, n_components=2)
    assert isinstance(model, nn.Module), f"Model {model_name} creation failed or is not an nn.Module"
