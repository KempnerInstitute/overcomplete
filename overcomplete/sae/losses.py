"""
Module containing loss functions for the Sparse Autoencoder (SAE) model.
Every loss function should take the following arguments:
- x: torch.Tensor
    Input tensor.
- x_hat: torch.Tensor
    Reconstructed tensor.
- codes: torch.Tensor
    Encoded tensor.
- dictionary: torch.Tensor
    Dictionary tensor.
Additional arguments can be passed as keyword arguments.
"""

# disable W0613 (unused-argument) to keep the same signature for all loss functions
# pylint: disable=W0613


def mse_l1(x, x_hat, codes, dictionary, penalty=1.0):
    """
    Compute the Mean Squared Error (MSE) loss with L1 penalty on the codes.

    Loss = ||x - x_hat||^2 + penalty * ||z||_1

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    x_hat : torch.Tensor
        Reconstructed tensor.
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


def mse_elastic(x, x_hat, codes, dictionary, alpha=0.5):
    """
    Compute the Mean Squared Error (MSE) loss with L1 penalty on the codes.

    Loss = ||x - x_hat||^2 + (1 - alpha) * ||z||_1 + alpha * ||D||^2

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    x_hat : torch.Tensor
        Reconstructed tensor.
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


def mse_l1_double(x, x_hat, codes, dictionary, penalty_codes=0.5, penalty_dictionary=0.5):
    """
    Compute the Mean Squared Error (MSE) loss with L1 penalty on the codes and dictionary.

    Loss = ||x - x_hat||^2 + penalty_codes * ||z||_1 + penalty_dictionary * ||D||_1

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    x_hat : torch.Tensor
        Reconstructed tensor.
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
