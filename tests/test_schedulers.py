import pytest
import torch
from torch import nn
from torch.optim import SGD
from overcomplete.methods.sae import CosineScheduler, mse_l1_loss

from .utils import epsilon_equal


def test_cosine_scheduler():
    model = nn.Linear(10, 1)
    optimizer = SGD(model.parameters(), lr=0.1)
    scheduler = CosineScheduler(optimizer, base_value=0.1, final_value=0.01, total_iters=10, warmup_iters=2)

    expected_schedule = [0.0, 0.05, 0.1, 0.087, 0.065, 0.035, 0.01, 0.01, 0.01, 0.01]
    for i, expected_lr in enumerate(expected_schedule):
        lr = scheduler[i]
        assert epsilon_equal(lr, expected_lr)
        scheduler.step()


def test_mse_l1_loss():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x_hat = torch.tensor([[1.1, 1.9], [2.9, 4.1]])
    codes = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    dictionary = torch.eye(2)
    penalty = 0.5

    loss = mse_l1_loss(x, x_hat, codes, dictionary, penalty)
    expected_loss = ((x - x_hat).pow(2).mean() + penalty * codes.abs().mean()).item()

    assert epsilon_equal(loss, expected_loss)
