"""
Convex NMF module for PyTorch.

We use the following notation:
- A: data matrix, tensor of shape (n_samples, n_features)
- Z: codes matrix, tensor of shape (n_samples, n_components)
- W: coefficient matrix, tensor of shape (n_components, n_samples)
- D: dictionary matrix, computed as D = W A

In Convex NMF, we aim to factorize A ≈ Z D = Z A W.
"""

import torch

from .base import BaseOptimDictionaryLearning
from .semi_nmf import SemiNMF
from .utils import stopping_criterion, pos_part, neg_part


def _one_step_multiplicative_update(A, Z, W, update_Z=True, update_W=True,
                                    strict_convex=False, epsilon=1e-10):

    AAt = A @ A.T
    Wt = W.T

    if update_Z:

        ZW = Z @ W

        numerator = (pos_part(AAt) @ Wt) + (ZW @ neg_part(AAt) @ Wt)
        denominator = (neg_part(AAt) @ Wt) + (ZW @ pos_part(AAt) @ Wt)

        Z = Z * torch.sqrt((numerator / (denominator + epsilon)) + epsilon)

    if update_W:
        WZtZ = W @ Z @ Z.T

        numerator = (Z.T @ pos_part(AAt)) + (WZtZ @ neg_part(AAt))
        denominator = (Z.T @ neg_part(AAt)) + (WZtZ @ pos_part(AAt))

        W = W * torch.sqrt((numerator / (denominator + epsilon)) + epsilon)

        # @tfel: note that the original article don't enforce this constraint
        # however it seems to stabilize the optimization
        if strict_convex:
            W = W / (torch.sum(W, dim=1, keepdim=True) + epsilon)

    return Z, W


def cnmf_multiplicative_update_solver(
        A, Z, W, update_Z=True, update_W=True, strict_convex=False, max_iter=500, tol=1e-5):
    """
    Convex NMF Multiplicative update optimizer.

    Alternately updates Z and W using the original Convex NMF update rules.

    Parameters
    ----------
    A : torch.Tensor
        Data tensor, shape (n_samples, n_features).
    Z : torch.Tensor
        Codes tensor, shape (n_samples, n_components).
    W : torch.Tensor
        Coefficient tensor, shape (n_features, n_components).
    update_Z : bool, optional
        Whether to update Z, by default True.
    update_W : bool, optional
        Whether to update W, by default True.
    strict_convex : bool, optional
        Whether to enforce that the columns of W sum (convex combination), by default False.
    max_iter : int, optional
        Maximum number of iterations, by default 500.
    tol : float, optional
        Tolerance value for the stopping criterion, by default 1e-5.

    Returns
    -------
    Z : torch.Tensor
        Updated codes tensor.
    W : torch.Tensor
        Updated coefficient tensor.
    """
    for _ in range(max_iter):
        Z_old = Z.clone()
        Z, W = _one_step_multiplicative_update(A, Z, W, update_Z=update_Z, update_W=update_W,
                                               strict_convex=strict_convex)

        if update_Z and tol > 0 and stopping_criterion(Z, Z_old, tol):
            break

    return Z, W


def cnmf_pgd_solver(
        A, Z, W, lr=1e-2, update_Z=True, update_W=True, strict_convex=False, max_iter=500, tol=1e-5):
    """
    Convex NMF PGD optimizer.

    Alternately updates Z and W using the Convex NMF update rules.

    Parameters
    ----------
    A : torch.Tensor
        Data tensor, shape (n_samples, n_features).
    Z : torch.Tensor
        Codes tensor, shape (n_samples, n_components).
    W : torch.Tensor
        Coefficient tensor, shape (n_features, n_components).
    lr : float, optional
        Learning rate, by default 1e-1.
    update_Z : bool, optional
        Whether to update Z, by default True.
    update_W : bool, optional
        Whether to update W, by default True.
    strict_convex : bool, optional
        Whether to enforce that the columns of W sum (convex combination), by default False.
    max_iter : int, optional
        Maximum number of iterations, by default 500.
    tol : float, optional
        Tolerance value for the stopping criterion, by default 1e-5.

    Returns
    -------
    Z : torch.Tensor
        Updated codes tensor.
    W : torch.Tensor
        Updated coefficient tensor.
    """
    if update_Z:
        Z = torch.nn.Parameter(Z)
    if update_W:
        W = torch.nn.Parameter(W)

    to_optimize = []
    if update_Z:
        to_optimize.append(Z)
    if update_W:
        to_optimize.append(W)

    optimizer = torch.optim.Adam(to_optimize, lr=lr)

    for iter_i in range(max_iter):
        optimizer.zero_grad()
        # @tfel: add possibility to pass custom loss here
        D = W @ A
        loss = torch.mean(torch.square(A - (Z @ D)))

        if update_Z:
            Z_old = Z.data.clone()

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if update_Z:
                Z.clamp_(min=0)
            if update_W:
                W.clamp_(min=0)
            if strict_convex:
                W /= (torch.sum(W, dim=1, keepdim=True) + 1e-10)

        if update_Z:
            should_stop = stopping_criterion(Z, Z_old, tol)
            if should_stop:
                break

    # return pure torch tensor (not a parameter)
    Z = Z.detach() if update_Z else Z
    W = W.detach() if update_W else W
    return Z, W


class ConvexNMF(BaseOptimDictionaryLearning):
    """
    PyTorch Convex NMF-based Dictionary Learning model.

    Solves the optimization problem:
        min_{Z,W} ||A - Z D||_F^2, s.t. D = W A, Z >= 0, W >= 0,
    columns of W sum to 1 if strict_convex is True.

    Parameters
    ----------
    n_components: int
        Number of components to learn.
    device: str, optional
        Device to use for tensor computations, by default 'cpu'.
    tol: float, optional
        Tolerance value for the stopping criterion, by default 1e-4.
    strict_convex: bool, optional
        Whether to enforce the convexity constraint, by default False.
    solver: str, optional
        Optimization solver to use, either 'mu' (Multiplicative Update) or 'pgd' like method,
        by default 'mu'.
    """

    _SOLVERS = {
        'mu': cnmf_multiplicative_update_solver,
        'pgd': cnmf_pgd_solver,
    }

    def __init__(self, n_components, device='cpu', tol=1e-4, strict_convex=False, solver='mu', **kwargs):
        assert solver in self._SOLVERS, f"Unknown optimizer: {
            solver}, should be one of {list(self._SOLVERS.keys())}"

        super().__init__(n_components, device)

        self.tol = tol
        self.strict_convex = strict_convex

        self.solver = solver
        self.solver_fn = self._SOLVERS[solver]
        self.tol = tol

    def encode(self, A, max_iter=300, tol=None):
        """
        Encode the input tensor (the data) using Convex NMF.

        Parameters
        ----------
        A : torch.Tensor
            Input tensor of shape (n_samples, n_features).
        max_iter : int, optional
            Maximum number of iterations, by default 300.
        tol : float, optional
            Tolerance value for the stopping criterion, by default the value set at initialization.

        Returns
        -------
        torch.Tensor
            Encoded features (the codes Z).
        """
        self._assert_fitted()
        if tol is None:
            tol = self.tol

        Z = self.init_random_z(A)
        Z, _ = self.solver_fn(A, Z, self.W, update_Z=True, update_W=False,
                              strict_convex=self.strict_convex, max_iter=max_iter, tol=tol)

        return Z

    def decode(self, Z):
        """
        Decode the input tensor (the codes) using Convex NMF.

        Parameters
        ----------
        Z : torch.Tensor
            Codes tensor of shape (n_samples, n_components).

        Returns
        -------
        torch.Tensor
            Decoded output (the approximation of A).
        """
        self._assert_fitted()

        D = self.get_dictionary()
        A_hat = Z @ D

        return A_hat

    def fit(self, A, max_iter=500):
        """
        Fit the Convex NMF model to the input data.

        Parameters
        ----------
        A : torch.Tensor
            Input tensor of shape (n_samples, n_features).
        max_iter : int, optional
            Maximum number of iterations, by default 500.
        """
        # @tfel could random init, after experiment it seems that
        # cnmf is really sensitive to the initialization
        Z, W = self.init_semi_nmf(A)
        # Z, W = self.init_random_z(A), self.init_random_w(A)

        Z, W = self.solver_fn(A, Z, W, update_Z=True, update_W=True,
                              strict_convex=self.strict_convex, max_iter=max_iter, tol=self.tol)

        self.Z = Z
        self.W = W
        self.D = W @ A
        self._set_fitted()

        return Z, self.D

    def get_dictionary(self):
        """
        Return the learned dictionary components from Convex NMF.

        Returns
        -------
        torch.Tensor
            Dictionary components D = A W.
        """
        self._assert_fitted()
        return self.D

    def init_semi_nmf(self, A, max_snmf_iters=100):
        """
        Initialize the Convex NMF model using Semi-NMF.

        Use the procedure (B) in the paper:
        "Convex and Semi-Nonnegative Matrix Factorizations"
        by Chris Ding, Tao Li, and Michael I. Jordan

        Parameters
        ----------
        A : torch.Tensor
            Input tensor of shape (n_samples, n_features).
        max_snmf_iters : int, optional
            Maximum number of iterations for Semi-NMF, by default 100.

        Returns
        -------
        Z : torch.Tensor
            Initialized codes tensor.
        D : torch.Tensor
            Initialized dictionary tensor.
        """
        snmf = SemiNMF(n_components=self.n_components, device=self.device)
        Z, _ = snmf.fit(A, max_iter=max_snmf_iters)

        Z = Z + 0.2  # see P.12, method B.B
        # now we need to solve A = W A Z for W
        # which resolve to W = Zt(Z Zt)^-1
        # we then ensure positivity of W
        W = Z.T @ torch.linalg.pinv(Z @ Z.T)
        W = pos_part(W) + 0.2

        if self.strict_convex:
            W = W / (torch.sum(W, dim=1, keepdim=True) + 1e-10)

        return Z, W

    def init_random_z(self, A):
        """
        Initialize the codes Z using non negative random values.

        Parameters
        ----------
        A : torch.Tensor
            Input tensor of shape (batch_size, n_features).

        Returns
        -------
        Z : torch.Tensor
            Initialized codes tensor.
        """
        mu = torch.sqrt(torch.mean(torch.abs(A)) / self.n_components)

        Z = torch.randn(A.shape[0], self.n_components, device=self.device) * mu
        Z = torch.abs(Z)

        return Z

    def init_random_w(self, A):
        """
        Initialize the coefficient matrix W with non-negative random values.

        Parameters
        ----------
        A : torch.Tensor
            Input tensor of shape (n_samples, n_features).

        Returns
        -------
        W : torch.Tensor
            Initialized coefficient tensor.
        """
        mu = torch.sqrt(torch.mean(torch.abs(A)) / self.n_components)

        W = torch.randn(self.n_components, A.shape[0], device=self.device) * mu
        W = torch.abs(W)

        if self.strict_convex:
            W = W / (torch.sum(W, dim=1, keepdim=True) + 1e-10)

        return W
