import pytest
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
from einops import rearrange

from overcomplete.sae.train import train_sae, train_sae_amp
from overcomplete.sae.losses import mse_l1
from overcomplete.sae import SAE, JumpSAE, TopKSAE, QSAE

from ..utils import epsilon_equal

all_sae = [SAE, JumpSAE, TopKSAE, QSAE]


@pytest.mark.parametrize(
    "module_name",
    [
        'linear',
        'mlp_ln_1',
        'mlp_ln_3',
        'mlp_bn_1',
        'mlp_bn_3',
    ]
)
@pytest.mark.parametrize("sae_class", all_sae)
def test_train_mlp_sae(module_name, sae_class):
    """Ensure we can train MLP SAE using common configurations."""
    torch.autograd.set_detect_anomaly(True)

    data = torch.randn(10, 10)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=10)
    criterion = mse_l1
    n_components = 2

    model = sae_class(data.shape[1], n_components, encoder_module=module_name)

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
    logs = train_sae_amp(
        model,
        dataloader,
        criterion,
        optimizer,
        scheduler,
        nb_epochs=2,
        monitoring=2,
        device="cpu",
    )

    assert isinstance(logs, defaultdict)
    assert "z" in logs
    assert "z_l2" in logs
    assert "z_sparsity" in logs
    assert "time_epoch" in logs


@pytest.mark.parametrize("sae_class", all_sae)
def test_train_resnet_sae(sae_class):
    """Ensure we can train resnet sae"""
    torch.autograd.set_detect_anomaly(True)

    def criterion(x, x_hat, z_pre, z, dictionary):
        x = rearrange(x, 'n c w h -> (n w h) c')
        return mse_l1(x, x_hat, z_pre, z, dictionary)

    data = torch.randn(10, 10, 5, 5)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=10)
    n_components = 2

    model = sae_class(data.shape[1:], n_components, encoder_module="resnet_3b")

    optimizer = optim.SGD(model.parameters(), lr=0.001)
    scheduler = None

    logs = train_sae(model, dataloader, criterion, optimizer, scheduler, nb_epochs=2, monitoring=False, device="cpu")

    assert isinstance(logs, defaultdict)
    assert len(logs) == 0

    logs = train_sae_amp(model, dataloader, criterion, optimizer, scheduler, nb_epochs=2, monitoring=2, device="cpu")
    assert isinstance(logs, defaultdict)
    assert "z" in logs
    assert "z_l2" in logs
    assert "z_sparsity" in logs
    assert "time_epoch" in logs


@pytest.mark.parametrize("sae_class", all_sae)
def test_train_attention_sae(sae_class):
    """Ensure we can train attention sae"""
    torch.autograd.set_detect_anomaly(True)

    def criterion(x, x_hat, z_pre, z, dictionary):
        x = rearrange(x, 'n t c -> (n t) c')
        return mse_l1(x, x_hat, z_pre, z, dictionary)

    data = torch.randn(10, 10, 64)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=10)
    n_components = 2

    model = sae_class(data.shape[1:], n_components, encoder_module="attention_3b")

    optimizer = optim.SGD(model.parameters(), lr=0.001)
    scheduler = None

    logs = train_sae_amp(model, dataloader, criterion, optimizer, scheduler,
                         nb_epochs=2, monitoring=False, device="cpu")

    assert isinstance(logs, defaultdict)
    assert len(logs) == 0

    logs = train_sae(model, dataloader, criterion, optimizer, scheduler, nb_epochs=2, monitoring=2, device="cpu")
    assert isinstance(logs, defaultdict)
    assert "z" in logs
    assert "z_l2" in logs
    assert "z_sparsity" in logs
    assert "time_epoch" in logs
    assert "dead_features" in logs


@pytest.mark.parametrize(
    "module_name",
    [
        'linear',
        'mlp_ln_1',
        'mlp_ln_3',
        'mlp_bn_1',
        'mlp_bn_3',
    ]
)
@pytest.mark.parametrize("sae_class", all_sae)
def test_train_without_amp(module_name, sae_class):
    """Ensure we can train SAE without AMP."""
    data = torch.randn(10, 10)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=10)
    criterion = mse_l1
    n_components = 2

    model = sae_class(data.shape[1], n_components, encoder_module=module_name)

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
    )

    assert isinstance(logs, defaultdict)
    assert len(logs) == 0


def test_monitoring():
    """Ensure monitoring granularity is working."""
    data = torch.randn(10, 10)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=10)
    criterion = mse_l1
    n_components = 2

    model = SAE(data.shape[1], n_components, encoder_module="linear")

    optimizer = optim.SGD(model.parameters(), lr=0.001)
    scheduler = None

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

    logs = train_sae(
        model,
        dataloader,
        criterion,
        optimizer,
        scheduler,
        nb_epochs=1,
        monitoring=1,
        device="cpu",
    )

    assert isinstance(logs, defaultdict)
    assert "lr" in logs
    assert "z" not in logs

    logs = train_sae(
        model,
        dataloader,
        criterion,
        optimizer,
        scheduler,
        nb_epochs=1,
        monitoring=2,
        device="cpu",
    )

    assert isinstance(logs, defaultdict)
    assert "z" in logs


def test_top_k_constraint():
    # test that top-k sae are returning only k non-zero codes before and after training
    N = 10
    top_k = 2

    data = torch.randn(N, 10)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=10)
    criterion = mse_l1
    n_components = 4

    model = TopKSAE(data.shape[1], n_components, encoder_module="linear", top_k=top_k)

    _, code_at_init = model.encode(data)
    # top-k is an upper bound, so we can have less than k non-zero codes
    assert (code_at_init > 0).float().sum() <= N * top_k

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
    )

    _, code_after_training = model.encode(data)
    # top-k is an upper bound, so we can have less than k non-zero codes
    assert (code_after_training > 0).float().sum() <= N * top_k

    assert isinstance(logs, defaultdict)
    assert len(logs) == 0


def test_q_sae_quantization_levels():
    # ensure the quantization levels of q-sae are respected
    # and are also trainable
    N = 10
    quantization_levels = 2

    data = torch.randn(N, 10)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=10)
    criterion = mse_l1

    model = QSAE(data.shape[1], 4, encoder_module="linear", q=quantization_levels,
                 hard=True)

    _, code_at_init = model.encode(data)
    unique_values = np.unique(code_at_init.detach().numpy().astype(np.float16))

    sorted_unique_values = np.sort(unique_values)
    sorted_quantization_state = np.array(model.Q.data.sort().values.detach().numpy())

    assert len(unique_values) == quantization_levels
    assert epsilon_equal(sorted_unique_values, np.clip(sorted_quantization_state, 0, None))

    optimizer = optim.SGD(model.parameters(), lr=0.1)
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
    )

    _, code_after_training = model.encode(data)

    unique_values_after_training = np.unique(code_after_training.detach().numpy().astype(np.float16))
    quantization_state_after_training = np.array(model.Q.data.sort().values.detach().numpy().astype(np.float16))

    assert len(unique_values_after_training) == quantization_levels
    assert epsilon_equal(np.clip(quantization_state_after_training, 0, None), unique_values_after_training, 1e-4)

    # ensure the quantization levels are trainable / not the same
    assert np.square(sorted_unique_values - quantization_state_after_training).sum() > 1e-4

    assert isinstance(logs, defaultdict)
    assert len(logs) == 0
