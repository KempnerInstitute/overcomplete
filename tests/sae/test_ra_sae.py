import pytest
import torch

from overcomplete.sae import RATopKSAE, RAJumpSAE

INPUT_SIZE = 128
NB_CONCEPTS = 20
POINTS_COUNT = 100
BATCH_SIZE = 32
TOP_K = 5
BANDWIDTH = 0.001


@pytest.mark.parametrize("device", ['cpu'])
@pytest.mark.parametrize("ra_class", [RATopKSAE, RAJumpSAE])
def test_ra_sae_device_propagation(device, ra_class):
    # points need to be on the same device as the module for initialization
    points = torch.randn(POINTS_COUNT, INPUT_SIZE, device=device)

    if ra_class == RATopKSAE:
        model = ra_class(INPUT_SIZE, NB_CONCEPTS, points=points, top_k=TOP_K, device=device)
    else:
        model = ra_class(INPUT_SIZE, NB_CONCEPTS, points=points, bandwidth=BANDWIDTH, device=device)

    # check encoder parameters
    for param in model.encoder.parameters():
        assert param.device.type == device

    # check dictionary parameters
    for param in model.dictionary.parameters():
        assert param.device.type == device

    # check all parameters
    for param in model.parameters():
        assert param.device.type == device


@pytest.mark.parametrize("ra_class", [RATopKSAE, RAJumpSAE])
def test_ra_sae_forward_shape(ra_class):
    # run forward pass on cpu to ensure shape correctness
    device = 'cpu'
    points = torch.randn(POINTS_COUNT, INPUT_SIZE, device=device)
    input_data = torch.randn(BATCH_SIZE, INPUT_SIZE, device=device)

    if ra_class == RATopKSAE:
        model = ra_class(INPUT_SIZE, NB_CONCEPTS, points=points, top_k=TOP_K, device=device)
    else:
        model = ra_class(INPUT_SIZE, NB_CONCEPTS, points=points, bandwidth=BANDWIDTH, device=device)

    z_pre, z, x_hat = model(input_data)

    # check output shapes
    assert x_hat.shape == input_data.shape
    assert z.shape == (BATCH_SIZE, NB_CONCEPTS)
    assert z_pre.shape == (BATCH_SIZE, NB_CONCEPTS)
