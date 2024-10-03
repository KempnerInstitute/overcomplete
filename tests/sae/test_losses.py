import torch

from overcomplete.sae.losses import mse_l1, top_k_auxiliary_loss
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


def test_top_k_auxiliary_loss():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x_hat = torch.tensor([[1.1, 1.9], [2.9, 4.1]])

    # everything pass, so no auxilary loss
    codes = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    pre_codes = codes

    dictionary = torch.eye(2)

    loss = top_k_auxiliary_loss(x, x_hat, pre_codes, codes, dictionary)
    expected_loss = (x - x_hat).square().mean()

    assert epsilon_equal(loss, expected_loss)

    # now remove the code, so the aux loss will be non zero and the overall
    # loss will be higher
    codes = codes * 0.0

    loss = top_k_auxiliary_loss(x, x_hat, pre_codes, codes, dictionary)

    assert loss > expected_loss
