import pytest
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
from einops import rearrange

from overcomplete.sae.train import train_sae
from overcomplete.sae.losses import mse_l1
from overcomplete.sae import SAE


@pytest.mark.parametrize(
    "module_name",
    [
        'mlp_ln_1',
        'mlp_ln_1_no_res',
        'mlp_ln_1_gelu',
        'mlp_ln_1_gelu_no_res',
        'mlp_ln_3',
        'mlp_ln_3_no_res',
        'mlp_ln_3_gelu',
        'mlp_ln_3_gelu_no_res',
        'mlp_bn_1',
        'mlp_bn_1_no_res',
        'mlp_bn_1_gelu',
        'mlp_bn_1_gelu_no_res',
        'mlp_bn_3',
        'mlp_bn_3_no_res',
        'mlp_bn_3_gelu',
        'mlp_bn_3_gelu_no_res',
    ]
)
def test_train_mlp_sae(module_name):
    """Ensure we can train MLP SAE using common configurations."""
    torch.autograd.set_detect_anomaly(True)

    data = torch.randn(10, 10)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=10)
    criterion = mse_l1
    n_components = 2

    model = SAE(data.shape[1], n_components, encoder_module=module_name)

    optimizer = optim.SGD(model.parameters(), lr=0.001)
    scheduler = None

    # first training pass without monitoring
    logs = train_sae(
        model,
        dataloader,
        criterion,
        optimizer,
        scheduler,
        nb_epochs=1,
        monitoring=False,
        device="cpu",
    )

    assert isinstance(logs, defaultdict)
    assert len(logs) == 0

    # second training pass with monitoring enabled
    logs = train_sae(
        model,
        dataloader,
        criterion,
        optimizer,
        scheduler,
        nb_epochs=2,
        monitoring=True,
        device="cpu",
    )

    assert isinstance(logs, defaultdict)
    assert "z" in logs
    assert "z_l2" in logs
    assert "z_sparsity" in logs
    assert "time_epoch" in logs


def test_train_resnet_sae():
    """Ensure we can train resnet sae"""
    torch.autograd.set_detect_anomaly(True)

    def criterion(x, x_hat, z, dictionary):
        x = rearrange(x, 'n c w h -> (n w h) c')
        return mse_l1(x, x_hat, z, dictionary)

    data = torch.randn(10, 10, 5, 5)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=10)
    n_components = 2

    model = SAE(data.shape[1:], n_components, encoder_module="resnet_3b")

    optimizer = optim.SGD(model.parameters(), lr=0.001)
    scheduler = None

    logs = train_sae(model, dataloader, criterion, optimizer, scheduler, nb_epochs=2, monitoring=False, device="cpu")

    assert isinstance(logs, defaultdict)
    assert len(logs) == 0

    logs = train_sae(model, dataloader, criterion, optimizer, scheduler, nb_epochs=2, monitoring=True, device="cpu")
    assert isinstance(logs, defaultdict)
    assert "z" in logs
    assert "z_l2" in logs
    assert "z_sparsity" in logs
    assert "time_epoch" in logs


def test_train_attention_sae():
    """Ensure we can train attention sae"""
    torch.autograd.set_detect_anomaly(True)

    def criterion(x, x_hat, z, dictionary):
        x = rearrange(x, 'n t c -> (n t) c')
        return mse_l1(x, x_hat, z, dictionary)

    data = torch.randn(10, 10, 64)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=10)
    n_components = 2

    model = SAE(data.shape[1:], n_components, encoder_module="attention_3b")

    optimizer = optim.SGD(model.parameters(), lr=0.001)
    scheduler = None

    logs = train_sae(model, dataloader, criterion, optimizer, scheduler, nb_epochs=2, monitoring=False, device="cpu")

    assert isinstance(logs, defaultdict)
    assert len(logs) == 0

    logs = train_sae(model, dataloader, criterion, optimizer, scheduler, nb_epochs=2, monitoring=True, device="cpu")
    assert isinstance(logs, defaultdict)
    assert "z" in logs
    assert "z_l2" in logs
    assert "z_sparsity" in logs
    assert "time_epoch" in logs


@pytest.mark.parametrize(
    "module_name",
    [
        'mlp_ln_1',
        'mlp_ln_1_no_res',
        'mlp_ln_1_gelu',
        'mlp_ln_1_gelu_no_res',
        'mlp_ln_3',
        'mlp_ln_3_no_res',
        'mlp_ln_3_gelu',
        'mlp_ln_3_gelu_no_res',
        'mlp_bn_1',
        'mlp_bn_1_no_res',
        'mlp_bn_1_gelu',
        'mlp_bn_1_gelu_no_res',
        'mlp_bn_3',
        'mlp_bn_3_no_res',
        'mlp_bn_3_gelu',
        'mlp_bn_3_gelu_no_res',
    ]
)
def test_train_without_amp(module_name):
    """Ensure we can train SAE without AMP."""
    data = torch.randn(10, 10)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=10)
    criterion = mse_l1
    n_components = 2

    model = SAE(data.shape[1], n_components, encoder_module=module_name)

    optimizer = optim.SGD(model.parameters(), lr=0.001)
    scheduler = None

    logs = train_sae(
        model,
        dataloader,
        criterion,
        optimizer,
        scheduler,
        nb_epochs=2,
        monitoring=False,
        device="cpu",
        use_amp=False
    )

    assert isinstance(logs, defaultdict)
    assert len(logs) == 0
