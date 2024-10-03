import torch

from overcomplete.sae.losses import (mse_l1, mse_hoyer, mse_kappa_4, top_k_auxiliary_loss, reanimation_regularizer)
from overcomplete.metrics import hoyer, kappa_4, dead_codes, l1

from ..utils import epsilon_equal


def test_mse_l1_criterion():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x_hat = torch.tensor([[1.1, 1.9], [2.9, 4.1]])
    codes = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    dictionary = torch.eye(2)
    penalty = 0.5

    loss = mse_l1(x, x_hat, codes, codes, dictionary, penalty)
    expected_loss = ((x - x_hat).pow(2).mean() + penalty * l1(codes).mean()).item()

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


def test_mse_hoyer_criterion():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x_hat = torch.tensor([[1.1, 1.9], [2.9, 4.1]])
    codes = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    dictionary = torch.eye(2)
    penalty = 0.5

    hoyer_loss = hoyer(codes).mean()

    loss = mse_hoyer(x, x_hat, codes, codes, dictionary, penalty)
    expected_loss = ((x - x_hat).pow(2).mean() + penalty * hoyer_loss).item()

    assert epsilon_equal(loss, expected_loss)


def test_mse_kappa_4_criterion():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x_hat = torch.tensor([[1.1, 1.9], [2.9, 4.1]])
    codes = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    dictionary = torch.eye(2)
    penalty = 0.5

    kappa_4_loss = kappa_4(codes).mean()

    loss = mse_kappa_4(x, x_hat, codes, codes, dictionary, penalty)
    expected_loss = ((x - x_hat).pow(2).mean() + penalty * kappa_4_loss).item()

    assert epsilon_equal(loss, expected_loss)


def test_reanimation_regularizer_gradients():
    # test for random inputs that gradient for dead codes are non zero
    for _ in range(10):

        batch_size, code_size = 10, 5
        x = torch.randn(batch_size, code_size, requires_grad=False)
        x_hat = torch.randn(batch_size, code_size, requires_grad=False)
        dictionary = torch.randn(code_size, code_size, requires_grad=False)

        pre_codes = torch.randn(batch_size, code_size)
        pre_codes[:, 0] = (pre_codes[:, 0] * 0.0).detach() - 1.0  # introduce dead code
        pre_codes.requires_grad = True

        optimizer = torch.optim.Adam([pre_codes], lr=1e-3)
        previous_loss = torch.inf
        previous_pre_codes = pre_codes.clone()

        for _ in range(5):
            optimizer.zero_grad()
            codes = torch.relu(pre_codes)

            reg_loss = reanimation_regularizer(x, x_hat, pre_codes, codes, dictionary)
            reg_loss.backward()
            optimizer.step()

            dead_mask = dead_codes(codes)
            pre_codes_grad = pre_codes.grad

            assert torch.any((torch.abs(pre_codes_grad) > 1e-6) & (dead_mask > 1e-6)
                             ), "Gradients for dead neurons are zero."
            assert reg_loss.item() > 0, "Regularization loss should always be positive with Relu."
            assert reg_loss.item() < previous_loss, "Regularization loss should decrease."
            assert torch.sum(previous_pre_codes) < torch.sum(pre_codes), "Pre codes could only increase."

            previous_loss = reg_loss.item()
            previous_pre_codes = pre_codes.clone()

    # pre-defined examples
    pre_codes = torch.tensor([[-1.0, 2.0], [-1.0, 0.0]], requires_grad=True)
    codes = torch.relu(pre_codes)

    reg_loss = reanimation_regularizer(x, x_hat, pre_codes, codes, dictionary, penalty=1.0)
    reg_loss.backward()

    assert epsilon_equal(reg_loss.item(), 2), "Regularization loss should be 2."
    assert epsilon_equal(pre_codes.grad, torch.tensor([[-1.0, 0.0], [-1.0, 0.0]])), "Gradients are not correct."
