import pytest
import torch
from torch import nn

from overcomplete.sae.modules import (MLPEncoder, AttentionEncoder, ResNetEncoder,
                                      ResNetBlock, AttentionBlock)
from overcomplete.sae.factory import EncoderFactory

BATCH_SIZE = 4
SEQ_LENGTH = 16
INPUT_SIZE = 32
INPUT_CHANNELS = 3
HEIGHT = 28
WIDTH = 28
N_COMPONENTS = 10


@pytest.mark.parametrize("input_size, n_components, hidden_dim, nb_blocks", [
    (INPUT_SIZE, N_COMPONENTS, None, 1),
    (INPUT_SIZE, N_COMPONENTS, 64, 3),
    (INPUT_SIZE, N_COMPONENTS, 128, 5)
])
def test_mlp_encoder(input_size, n_components, hidden_dim, nb_blocks):
    x = torch.randn(BATCH_SIZE, input_size)
    model = MLPEncoder(input_size, n_components, hidden_dim, nb_blocks)
    pre_codes, codes = model(x)
    assert pre_codes.shape == codes.shape == (BATCH_SIZE, n_components)


@pytest.mark.parametrize("dims, num_heads, mlp_ratio", [
    ((SEQ_LENGTH, INPUT_SIZE), 4, 4.0),
    ((SEQ_LENGTH, INPUT_SIZE), 8, 2.0)
])
def test_attention_block(dims, num_heads, mlp_ratio):
    x = torch.randn(BATCH_SIZE, *dims)
    model = AttentionBlock(dims, num_heads, mlp_ratio)
    activations = model(x)
    assert activations.shape == (BATCH_SIZE, dims[0], dims[1])
    assert isinstance(activations, torch.Tensor)


@pytest.mark.parametrize("dim, num_heads, mlp_ratio, drop, attn_drop, act_layer", [
    (INPUT_SIZE, 4, 4.0, 0.0, 0.0, nn.GELU),
    (INPUT_SIZE, 8, 2.0, 0.1, 0.1, nn.GELU),
    (INPUT_SIZE, 4, 4.0, 0.2, 0.2, nn.ReLU),
    (INPUT_SIZE, 8, 2.0, 0.3, 0.3, nn.ReLU),
])
def test_attention_block_configurations(dim, num_heads, mlp_ratio, drop, attn_drop, act_layer):
    x = torch.randn(BATCH_SIZE, SEQ_LENGTH, dim)
    model = AttentionBlock((SEQ_LENGTH, dim), num_heads, mlp_ratio, drop, attn_drop, act_layer)
    activations = model(x)
    assert activations.shape == (BATCH_SIZE, SEQ_LENGTH, dim)
    assert isinstance(activations, torch.Tensor)


@pytest.mark.parametrize("batch_size, seq_length, dim", [
    (8, 32, 64),
    (16, 64, 128),
])
def test_attention_block_input_shapes(batch_size, seq_length, dim):
    x = torch.randn(batch_size, seq_length, dim)
    model = AttentionBlock((seq_length, dim), num_heads=4, mlp_ratio=4.0)
    output = model(x)
    assert output.shape == (batch_size, seq_length, dim)
    assert isinstance(output, torch.Tensor)


@pytest.mark.parametrize("input_size, n_components, hidden_dim, nb_blocks, attention_heads, mlp_ratio", [
    (INPUT_SIZE, N_COMPONENTS, None, 1, 4, 4.0),
    (INPUT_SIZE, N_COMPONENTS, 64, 3, 8, 2.0)
])
def test_attention_encoder(input_size, n_components, hidden_dim, nb_blocks, attention_heads, mlp_ratio):
    x = torch.randn(BATCH_SIZE, SEQ_LENGTH, input_size)
    model = AttentionEncoder((SEQ_LENGTH, input_size), n_components, hidden_dim, nb_blocks,
                             attention_heads=attention_heads, mlp_ratio=mlp_ratio)
    pre_codes, codes = model(x)
    assert pre_codes.shape == codes.shape == (BATCH_SIZE * SEQ_LENGTH, n_components)
    assert isinstance(pre_codes, torch.Tensor)
    assert isinstance(codes, torch.Tensor)


@pytest.mark.parametrize("input_shape, n_components, hidden_dim, nb_blocks", [
    ((INPUT_CHANNELS, HEIGHT, WIDTH), N_COMPONENTS, None, 1),
    ((INPUT_CHANNELS, HEIGHT, WIDTH), N_COMPONENTS, 64, 3),
    ((INPUT_CHANNELS, HEIGHT, WIDTH), N_COMPONENTS, 128, 5)
])
def test_resnet_encoder(input_shape, n_components, hidden_dim, nb_blocks):
    x = torch.randn(BATCH_SIZE, *input_shape)
    model = ResNetEncoder(input_shape, n_components, hidden_dim, nb_blocks)
    pre_codes, codes = model(x)
    assert pre_codes.shape == codes.shape == (BATCH_SIZE * HEIGHT * WIDTH, n_components)
    assert isinstance(pre_codes, torch.Tensor)
    assert isinstance(codes, torch.Tensor)


@pytest.mark.parametrize("input_channels, out_channels, stride, activation", [
    (INPUT_CHANNELS, 64, 1, nn.ReLU),
    (INPUT_CHANNELS, 128, 2,  nn.ReLU),
    (INPUT_CHANNELS, 64, 1,  nn.GELU),
    (INPUT_CHANNELS, 128, 2, nn.GELU),
])
def test_resnet_block_configurations(input_channels, out_channels, stride, activation):
    x = torch.randn(BATCH_SIZE, input_channels, HEIGHT, WIDTH)
    model = ResNetBlock(input_channels, out_channels, stride, activation)
    output = model(x)
    assert output.shape == (BATCH_SIZE, out_channels, HEIGHT // stride, WIDTH // stride)
    assert isinstance(output, torch.Tensor)


@pytest.mark.parametrize("batch_size, height, width, input_channels, out_channels", [
    (8, 56, 56, 3, 64),
    (16, 112, 112, 3, 128),
])
def test_resnet_block_input_shapes(batch_size, height, width, input_channels, out_channels):
    x = torch.randn(batch_size, input_channels, height, width)
    model = ResNetBlock(input_channels, out_channels, stride=2)
    output = model(x)
    assert output.shape == (batch_size, out_channels, height // 2, width // 2)
    assert isinstance(output, torch.Tensor)


@pytest.mark.parametrize("input_channels, out_channels, stride", [
    (INPUT_CHANNELS, 64, 1),
    (INPUT_CHANNELS, 128, 2)
])
def test_resnet_block(input_channels, out_channels, stride):
    x = torch.randn(BATCH_SIZE, input_channels, HEIGHT, WIDTH)
    model = ResNetBlock(input_channels, out_channels, stride)
    output = model(x)
    assert output.shape == (BATCH_SIZE, out_channels, HEIGHT // stride, WIDTH // stride)
    assert isinstance(output, torch.Tensor)


def test_resnet_block_downsampling():
    x = torch.randn(BATCH_SIZE, INPUT_CHANNELS, HEIGHT, WIDTH)
    model = ResNetBlock(INPUT_CHANNELS, 64, stride=2)
    output = model(x)
    assert output.shape == (BATCH_SIZE, 64, HEIGHT // 2, WIDTH // 2)

    model_no_downsample = ResNetBlock(INPUT_CHANNELS, 64, stride=1)
    output_no_downsample = model_no_downsample(x)
    assert output_no_downsample.shape == (BATCH_SIZE, 64, HEIGHT, WIDTH)


@pytest.mark.parametrize("module_name, args, kwargs", [
    ("linear", (INPUT_SIZE, N_COMPONENTS), {}),
    ("mlp_bn_1", (INPUT_SIZE, N_COMPONENTS), {}),
    ("mlp_ln_1", (INPUT_SIZE, N_COMPONENTS), {}),
    ("mlp_bn_3", (INPUT_SIZE, N_COMPONENTS), {"hidden_dim": 64}),
    ("mlp_ln_3", (INPUT_SIZE, N_COMPONENTS), {"hidden_dim": 64}),
    ("resnet_1b", ((INPUT_CHANNELS, HEIGHT, WIDTH), N_COMPONENTS), {"hidden_dim": 64}),
    ("resnet_3b", ((INPUT_CHANNELS, HEIGHT, WIDTH), N_COMPONENTS), {"hidden_dim": 128}),
    ("attention_1b", ((SEQ_LENGTH, INPUT_SIZE), N_COMPONENTS), {"hidden_dim": 64}),
    ("attention_3b", ((SEQ_LENGTH, INPUT_SIZE), N_COMPONENTS), {"hidden_dim": 64})
])
def test_module_factory(module_name, args, kwargs):

    model = EncoderFactory.create_module(module_name, *args, **kwargs)

    if module_name.startswith("mlp"):
        x = torch.randn(BATCH_SIZE, INPUT_SIZE)
        pre_codes, codes = model(x)
        assert pre_codes.shape == codes.shape == (BATCH_SIZE, N_COMPONENTS)
    elif module_name.startswith("resnet"):
        x = torch.randn(BATCH_SIZE, INPUT_CHANNELS, HEIGHT, WIDTH)
        pre_codes, codes = model(x)
        assert pre_codes.shape == codes.shape == (BATCH_SIZE * HEIGHT * WIDTH, N_COMPONENTS)
    elif module_name.startswith("attention"):
        x = torch.randn(BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
        pre_codes, codes = model(x)
        assert pre_codes.shape == codes.shape == (BATCH_SIZE * SEQ_LENGTH, N_COMPONENTS)


def test_invalid_module():
    with pytest.raises(ValueError):
        EncoderFactory.create_module("invalid_module_name", INPUT_SIZE, N_COMPONENTS)
