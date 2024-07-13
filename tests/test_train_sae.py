import pytest
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict

from overcomplete.sae.train import l2, train_sae

from .utils import epsilon_equal


def test_l2():
    x = torch.tensor([3.0, 4.0])
    expected = 5.0
    assert epsilon_equal(l2(x), expected)


def test_train_sae():
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.encoder = nn.Linear(10, 5)
            self.decoder = nn.Linear(5, 10)

        def forward(self, x):
            z = self.encoder(x)
            x_hat = self.decoder(z)
            return z, x_hat

        def get_dictionary(self):
            return torch.eye(5, 10)

    model = SimpleModel()
    data = torch.randn(10, 10)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=10)

    def criterion(x, x_hat, z, dictionary):
        return ((x - x_hat).pow(2).mean() + z.abs().mean())

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
