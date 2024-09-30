import pytest
import torch
from torch import nn
from torch.optim import SGD
from overcomplete.sae import CosineScheduler, mse_l1

from ..utils import epsilon_equal


def test_cosine_scheduler():
    model = nn.Linear(10, 1)
    optimizer = SGD(model.parameters(), lr=0.1)
    scheduler = CosineScheduler(optimizer, base_value=0.1, final_value=0.01, total_iters=10, warmup_iters=2)

    expected_schedule = [0.0, 0.1, 0.1, 0.0966, 0.0868, 0.0722, 0.0550, 0.0378, 0.0232, 0.0134]
    for i, expected_lr in enumerate(expected_schedule):
        lr = scheduler[i]
        assert epsilon_equal(lr, expected_lr, 1e-3)
        scheduler.step()
