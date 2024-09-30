"""
Module containing loss functions for the Sparse Autoencoder (SAE) model.

In the Overcomplete library, the loss functions are defined as standalone functions.
They all share the same signature:
    - x: torch.Tensor
        Input tensor.
    - x_hat: torch.Tensor
        Reconstructed tensor.
    - pre_codes: torch.Tensor
        Encoded tensor before activation function.
    - codes: torch.Tensor
        Encoded tensor.
    - dictionary: torch.Tensor
        Dictionary tensor.

Additional arguments can be passed as keyword arguments.
"""

import torch

# disable W0613 (unused-argument) to keep the same signature for all loss functions
# pylint: disable=W0613


def mse_l1(x, x_hat, pre_codes, codes, dictionary, penalty=1.0):
    """
    Compute the Mean Squared Error (MSE) loss with L1 penalty on the codes.

    Loss = ||x - x_hat||^2 + penalty * ||z||_1

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    x_hat : torch.Tensor
        Reconstructed tensor.
    pre_codes : torch.Tensor
        Encoded tensor before activation function.
    codes : torch.Tensor
        Encoded tensor.
    dictionary : torch.Tensor
        Dictionary tensor.
    penalty : float, optional
        L1 penalty coefficient, by default 1.0.

    Returns
    -------
    torch.Tensor
        Loss value.
    """
    mse = (x - x_hat).square().mean()
    l1 = codes.abs().mean()
    return mse + penalty * l1


def mse_elastic(x, x_hat, pre_codes, codes, dictionary, alpha=0.5):
    """
    Compute the Mean Squared Error (MSE) loss with L1 penalty on the codes.

    Loss = ||x - x_hat||^2 + (1 - alpha) * ||z||_1 + alpha * ||D||^2

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    x_hat : torch.Tensor
        Reconstructed tensor.
    pre_codes : torch.Tensor
        Encoded tensor before activation function.
    codes : torch.Tensor
        Encoded tensor.
    dictionary : torch.Tensor
        Dictionary tensor.
    alpha : float, optional
        Alpha coefficient in the Elastic-net loss, control the ratio of l1 vs l2.
        alpha=0 means l1 only, alpha=1 means l2 only.

    Returns
    -------
    torch.Tensor
        Loss value.
    """
    assert 0.0 <= alpha <= 1.0

    mse = (x - x_hat).square().mean()

    l1_loss = codes.abs().mean()
    l2_loss = dictionary.square().mean()

    loss = mse + (1.0 - alpha) * l1_loss + alpha * l2_loss

    return loss


def mse_l1_double(x, x_hat, pre_codes, codes, dictionary, penalty_codes=0.5, penalty_dictionary=0.5):
    """
    Compute the Mean Squared Error (MSE) loss with L1 penalty on the codes and dictionary.

    Loss = ||x - x_hat||^2 + penalty_codes * ||z||_1 + penalty_dictionary * ||D||_1

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    x_hat : torch.Tensor
        Reconstructed tensor.
    pre_codes : torch.Tensor
        Encoded tensor before activation function.
    codes : torch.Tensor
        Encoded tensor.
    dictionary : torch.Tensor
        Dictionary tensor.
    penalty_codes : float, optional
        L1 penalty for concepts coefficient / codes, by default 1/2.
    penalty_dictionary : float, optional
        L1 penalty for dictionary / codebook, by default 1/2.

    Returns
    -------
    torch.Tensor
        Loss value.
    """
    mse = (x - x_hat).square().mean()

    l1_codes = codes.abs().mean()
    l1_dict = dictionary.abs().mean()

    loss = mse + penalty_codes * l1_codes + penalty_dictionary * l1_dict

    return loss


def top_k_auxiliary_loss(x, x_hat, pre_codes, codes, dictionary):
    """
    The Top-K Auxiliary loss (AuxK).

    The loss is defined in the original Top-K SAE paper:
        "Scaling and evaluating sparse autoencoders"
        by Gao et al. (2024).

    Similar to Ghost-grads, it consist in trying to "revive" the dead codes
    by trying the predict the residual using the 50% of the top non choosen codes.

    Loss = ||x - x_hat||^2 + ||x - (x_hat D * top_half(z_pre - z)||^2

    @tfel the order actually matter here! residual is x - x_hat and
    should be in this specific order.
    """
    # select the 50% of non choosen codes and predict the residual
    # using those non choosen codes
    # the code choosen are the non-zero element of codes

    residual = x - x_hat
    mse = residual.square().mean()

    pre_codes = torch.relu(pre_codes)
    pre_codes = pre_codes - codes  # removing the choosen codes

    auxiliary_topk = torch.topk(pre_codes, k=pre_codes.shape[1] // 2, dim=1)
    pre_codes = torch.zeros_like(codes).scatter(-1, auxiliary_topk.indices,
                                                auxiliary_topk.values)

    residual_hat = pre_codes @ dictionary
    auxilary_mse = (residual - residual_hat).square().mean()

    loss = mse + auxilary_mse

    return loss
