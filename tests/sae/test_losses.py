import torch

from overcomplete.sae.losses import mse_l1
from ..utils import epsilon_equal


def test_mse_l1_criterion():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x_hat = torch.tensor([[1.1, 1.9], [2.9, 4.1]])
    codes = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    dictionary = torch.eye(2)
    penalty = 0.5

    loss = mse_l1(x, x_hat, codes, codes, dictionary, penalty)
    expected_loss = ((x - x_hat).pow(2).mean() + penalty * codes.abs().mean()).item()

    assert epsilon_equal(loss, expected_loss)
