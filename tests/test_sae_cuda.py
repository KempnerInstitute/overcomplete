import pytest

from overcomplete.sae import MLPEncoder, AttentionEncoder, ResNetEncoder, SAEFactory, SAE


INPUT_SIZE = 32
N_COMPONENTS = 16
INPUT_CHANNELS = 3
SEQ_LENGTH = 4
HEIGHT = 7
WIDTH = 7


@pytest.mark.parametrize("device", ['cpu', 'meta'])
def test_mlp_encoder_device_propagation(device):
    encoder = MLPEncoder(INPUT_SIZE, N_COMPONENTS, device=device)

    for param in encoder.parameters():
        assert param.device.type == device


@ pytest.mark.parametrize("device", ['cpu', 'meta'])
def test_attention_encoder_device_propagation(device):
    encoder = AttentionEncoder((SEQ_LENGTH, INPUT_SIZE), N_COMPONENTS, device=device)

    for param in encoder.parameters():
        assert param.device.type == device


@ pytest.mark.parametrize("device", ['cpu', 'meta'])
def test_resnet_encoder_device_propagation(device):
    encoder = ResNetEncoder((INPUT_CHANNELS, HEIGHT, WIDTH), N_COMPONENTS, device=device)

    for param in encoder.parameters():
        assert param.device.type == device


@ pytest.mark.parametrize("device", ['cpu', 'meta'])
def test_default_sae_device_propagation(device):
    model = SAE(32, 5, encoder_module=None, device=device)

    for param in model.encoder.parameters():
        assert param.device.type == device
    for param in model.dictionary.parameters():
        assert param.device.type == device
    for param in model.parameters():
        assert param.device.type == device


@ pytest.mark.parametrize(
    "device, module_name, args, kwargs",
    [
        ('cpu', 'mlp_bn_1', (INPUT_SIZE, N_COMPONENTS), {}),
        ('cpu', 'mlp_ln_1', (INPUT_SIZE, N_COMPONENTS), {}),
        ('cpu', 'mlp_bn_3', (INPUT_SIZE, N_COMPONENTS), {"hidden_dim": 64}),
        ('cpu', 'mlp_ln_3', (INPUT_SIZE, N_COMPONENTS), {"hidden_dim": 64}),
        ('cpu', 'mlp_bn_3_no_res', (INPUT_SIZE, N_COMPONENTS), {"hidden_dim": 64}),
        ('cpu', 'mlp_ln_3_no_res', (INPUT_SIZE, N_COMPONENTS), {"hidden_dim": 64}),
        ('cpu', 'mlp_bn_3_gelu', (INPUT_SIZE, N_COMPONENTS), {"hidden_dim": 64}),
        ('cpu', 'mlp_ln_3_gelu', (INPUT_SIZE, N_COMPONENTS), {"hidden_dim": 64}),
        ('cpu', 'resnet_1b', ((INPUT_CHANNELS, HEIGHT, WIDTH), N_COMPONENTS), {"hidden_dim": 64}),
        ('cpu', 'resnet_3b', ((INPUT_CHANNELS, HEIGHT, WIDTH), N_COMPONENTS), {"hidden_dim": 128}),
        ('cpu', 'attention_1b', ((SEQ_LENGTH, INPUT_SIZE), N_COMPONENTS), {"hidden_dim": 64}),
        ('cpu', 'attention_3b', ((SEQ_LENGTH, INPUT_SIZE), N_COMPONENTS), {"hidden_dim": 64}),
        ('meta', 'mlp_bn_1', (INPUT_SIZE, N_COMPONENTS), {}),
        ('meta', 'mlp_ln_1', (INPUT_SIZE, N_COMPONENTS), {}),
        ('meta', 'mlp_bn_3', (INPUT_SIZE, N_COMPONENTS), {"hidden_dim": 64}),
        ('meta', 'mlp_ln_3', (INPUT_SIZE, N_COMPONENTS), {"hidden_dim": 64}),
        ('meta', 'mlp_bn_3_no_res', (INPUT_SIZE, N_COMPONENTS), {"hidden_dim": 64}),
        ('meta', 'mlp_ln_3_no_res', (INPUT_SIZE, N_COMPONENTS), {"hidden_dim": 64}),
        ('meta', 'mlp_bn_3_gelu', (INPUT_SIZE, N_COMPONENTS), {"hidden_dim": 64}),
        ('meta', 'mlp_ln_3_gelu', (INPUT_SIZE, N_COMPONENTS), {"hidden_dim": 64}),
    ]
)
def test_module_factory(device, module_name, args, kwargs):
    model = SAEFactory.create_module(module_name, *args, **kwargs, device=device)

    for param in model.parameters():
        assert param.device.type == device
