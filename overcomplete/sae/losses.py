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


def mse_l1_loss(x, x_hat, codes, dictionary, penalty=1.0):
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
