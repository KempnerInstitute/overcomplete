"""
Semi-NMF module for PyTorch.

For sake of simplicity, we will use the following notation:
- A: pattern of activations of a neural net, tensor of shape (batch_size, n_features)
- Z: codes in the concepts (overcomplete) basis, tensor of shape (batch_size, n_components)
- D: dictionary of concepts, tensor of shape (n_components, n_features)
"""

import torch
from .base import BaseOptimDictionaryLearning
from .utils import matrix_nnls, stopping_criterion, _assert_shapes


def _one_step_semi_nmf(A, Z, D, update_Z=True, update_D=True):
    """
    One step of the Semi-NMF update rules.

    The Semi-NMF algorithm updates Z and D alternately:

    1. Update Z by solving Z = A D^T (D D^T)^{-1}
    2. Update D by solving D = argmin_{D >= 0} ||A - Z D||_F^2

    Parameters
    ----------
    A : torch.Tensor
        Activation tensor, should be (batch_size, n_features).
    Z : torch.Tensor
        Codes tensor, should be (batch_size, n_components).
    D : torch.Tensor
        Dictionary tensor, should be (n_components, n_features).
    update_Z : bool, optional
        Whether to update Z, by default True.
    update_D : bool, optional
        Whether to update D, by default True.

    Returns
    -------
    Z : torch.Tensor
        Updated codes tensor.
    D : torch.Tensor
        Updated dictionary tensor.
    """
    _assert_shapes(A, Z, D)

    if update_Z:
        DDT = D @ D.T
        Z = A @ D.T @ torch.linalg.pinv(DDT + 1e-8 * torch.eye(DDT.shape[0]))

    if update_D:
        D = matrix_nnls(Z, A)

    return Z, D


def semi_nmf_solver(A, Z, D, update_Z=True, update_D=True, max_iter=500, tol=1e-5):
    """
    Semi-NMF optimizer.

    Alternately updates Z and D using the Semi-NMF update rules.
    See: "Convex and Semi-Nonnegative Matrix Factorizations"
    Chris Ding, Tao Li, and Michael I. Jordan (2008).

    Parameters
    ----------
    A : torch.Tensor
        Activation tensor, should be (batch_size, n_features).
    Z : torch.Tensor
        Codes tensor, should (batch_size, n_components).
    D : torch.Tensor
        Dictionary tensor, should be (n_components, n_features).
    update_Z : bool, optional
        Whether to update Z, by default True.
    update_D : bool, optional
        Whether to update D, by default True.
    max_iter : int, optional
        Maximum number of iterations, by default 500.
    tol : float, optional
        Tolerance value for the stopping criterion, by default 1e-5.

    Returns
    -------
    Z : torch.Tensor
        Updated codes tensor.
    D : torch.Tensor
        Updated dictionary tensor.
    """
    _assert_shapes(A, Z, D)

    for _ in range(max_iter):
        Z_old = Z.clone()
        Z, D = _one_step_semi_nmf(A, Z, D, update_Z, update_D)

        if update_Z:
            should_stop = stopping_criterion(Z, Z_old, tol)
            if should_stop:
                break

    return Z, D


class SemiNMF(BaseOptimDictionaryLearning):
    """
    Torch Semi-NMF-based Dictionary Learning model.

    Solve the following optimization problem:
    min ||A - ZD||_F^2, s.t. D >= 0.

    Parameters
    ----------
    n_components: int
        Number of components to learn.
    device: str, optional
        Device to use for tensor computations, by default 'cpu'
    tol: float, optional
        Tolerance value for the stopping criterion, by default 1e-5.
    """

    def __init__(self, n_components, device='cpu', tol=1e-5, **kwargs):
        super().__init__(n_components, device)
        self.tol = tol
        self.D = None

    def encode(self, A, max_iter=500, tol=1e-5):
        """
        Encode the input tensor (the activations) using Semi-NMF.

        Parameters
        ----------
        A : torch.Tensor or Iterable
            Input tensor of shape (batch_size, n_features).
        max_iter : int, optional
            Maximum number of iterations, by default 500.
        tol : float, optional
            Tolerance value for the stopping criterion, by default 1e-5.

        Returns
        -------
        torch.Tensor
            Encoded features (the codes).
        """
        self._assert_fitted()

        Z = self.init_random_z(A)

        Z, _ = semi_nmf_solver(A, Z, self.D, update_Z=True, update_D=False, max_iter=max_iter, tol=tol)

        return Z

    def decode(self, Z):
        """
        Decode the input tensor (the codes) using Semi-NMF.

        Parameters
        ----------
        Z : torch.Tensor
            Encoded tensor (the codes) of shape (batch_size, n_components).

        Returns
        -------
        torch.Tensor
            Decoded output (the activations).
        """
        self._assert_fitted()

        A_hat = Z @ self.D

        return A_hat

    def fit(self, A, max_iter=500, tol=1e-5):
        """
        Fit the Semi-NMF model to the input data.

        Parameters
        ----------
        A : torch.Tensor or Iterable
            Input tensor of shape (batch_size, n_features).
        max_iter : int, optional
            Maximum number of iterations, by default 500.
        tol : float, optional
            Tolerance value, by default 1e-5.
        """
        Z = self.init_random_z(A)
        D = self.init_random_d(A)

        Z, D = semi_nmf_solver(A, Z, D, max_iter=max_iter, tol=tol)

        self.D = D
        self._set_fitted()

        return Z, D

    def get_dictionary(self):
        """
        Return the learned dictionary components from Semi-NMF.

        Returns
        -------
        torch.Tensor
            Dictionary components.
        """
        self._assert_fitted()
        return self.D

    def init_random_d(self, A):
        """
        Initialize the dictionary D using non-negative random values.

        Parameters
        ----------
        A : torch.Tensor
            Input tensor of shape (batch_size, n_features).

        Returns
        -------
        D : torch.Tensor
            Initialized dictionary tensor.
        """
        mu = torch.sqrt(torch.mean(A) / self.n_components)

        D = torch.randn(self.n_components, A.shape[1], device=self.device) * mu
        D = torch.abs(D)

        return D

    def init_random_z(self, A):
        """
        Initialize the codes Z using random values (can be negative).

        Parameters
        ----------
        A : torch.Tensor
            Input tensor of shape (batch_size, n_features).

        Returns
        -------
        Z : torch.Tensor
            Initialized codes tensor.
        """
        mu = torch.sqrt(torch.mean(A) / self.n_components)

        Z = torch.randn(A.shape[0], self.n_components, device=self.device) * mu

        return Z
