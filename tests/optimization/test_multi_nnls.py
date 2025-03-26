import pytest
import numpy as np
import torch
from scipy.optimize import nnls
from overcomplete.optimization.utils import batched_matrix_nnls


def make_ground_truth(n, k, d, rank=None, noise=0.0, device="cpu"):
    if rank is None:
        rank = min(k, d)

    # create low-rank dictionary D
    base = torch.randn(n, k, rank, device=device)
    D = torch.randn(n, rank, d, device=device)
    D = torch.bmm(base, D)  # shape (n, k, d)

    # generate sparse non-negative ground truth Z
    Z_true = torch.rand(n, k, device=device)
    Z_true *= (torch.rand_like(Z_true) > 0.5).float()  # sparsify

    # compute X = Z @ D + optional noise
    X = torch.bmm(Z_true.unsqueeze(1), D).squeeze(1)
    if noise > 0.0:
        X += noise * torch.randn_like(X)

    return D, X, Z_true


# test 1: exact recovery in noiseless case
def test_exact_recovery():
    n, k, d = 32, 16, 12
    D, X, Z_true = make_ground_truth(n, k, d, noise=0.0)

    Z_est = batched_matrix_nnls(D, X, max_iter=200, tol=1e-5)

    recon = torch.bmm(Z_est.unsqueeze(1), D).squeeze(1)
    err = torch.norm(recon - X) / torch.norm(X)

    assert err < 0.05, f"reconstruction error too high: {err.item()}"
    assert torch.all(Z_est >= -1e-6), "non-negativity violated"


# test 2: with random noise
def test_noisy_case():
    n, k, d = 32, 16, 12
    D, X, Z_true = make_ground_truth(n, k, d, noise=0.01)

    Z_est = batched_matrix_nnls(D, X, max_iter=300)
    recon = torch.bmm(Z_est.unsqueeze(1), D).squeeze(1)
    err = torch.norm(recon - X) / torch.norm(X)

    assert err < 0.05, f"noisy reconstruction error too high: {err.item()}"
    assert torch.all(Z_est >= -1e-6)


# test 3: test tolerance effect
def test_tolerance_control():
    n, k, d = 16, 8, 6
    D, X, _ = make_ground_truth(n, k, d)

    Z_low_tol = batched_matrix_nnls(D, X, max_iter=300, tol=1e-1)
    Z_high_tol = batched_matrix_nnls(D, X, max_iter=500, tol=1e-6)

    err_low = torch.norm(torch.bmm(Z_low_tol.unsqueeze(1), D).squeeze(1) - X)
    err_high = torch.norm(torch.bmm(Z_high_tol.unsqueeze(1), D).squeeze(1) - X)

    assert err_high < err_low, "high tolerance didn't improve error"


# test 4: test z_init
def test_z_init_usage():
    n, k, d = 16, 8, 6
    D, X, Z0 = make_ground_truth(n, k, d)

    Z_custom = batched_matrix_nnls(D, X, Z_init=Z0, max_iter=200)

    # Z_custom should be Z0 or better
    assert torch.norm(Z_custom - Z0) < 1e-3, "z_init not respected"
    assert torch.all(Z_custom >= -1e-6), "non-negativity violated in z_init test"


# test 5: batched shape check
def test_batch_shape():
    n, k, d = 64, 10, 8
    D, X, _ = make_ground_truth(n, k, d)
    Z = batched_matrix_nnls(D, X)

    assert Z.shape == (n, k), f"unexpected shape: {Z.shape}"


def test_scipy_nnls_vs_pgd():
    n, k, d = 16, 8, 6
    tol = 1e-2
    D, X, Z_true = make_ground_truth(n, k, d, noise=0.1)

    # our pgd nnls
    Z_est = batched_matrix_nnls(D, X, max_iter=300, tol=tol)

    # checking scipy solution (calling for each element in batch)
    Z_scipy = []
    for i in range(n):
        A = D[i].cpu().T.numpy()
        b = X[i].cpu().numpy()

        z_i, _ = nnls(A, b, maxiter=150, atol=tol)
        Z_scipy.append(z_i)

    Z_scipy = torch.from_numpy(np.array(Z_scipy)).float().to(D.device)

    # compare the errors
    recon_est = torch.bmm(Z_est.unsqueeze(1), D).squeeze(1)
    err_est = torch.norm(recon_est - X) / torch.norm(X)

    recon_sp = torch.bmm(Z_scipy.unsqueeze(1), D).squeeze(1)
    err_sp = torch.norm(recon_sp - X) / torch.norm(X)

    # check that our batched nnls and scipy are in the same ballpark.
    assert err_est < 2.0 * err_sp + tol, (
        f"PGD error {err_est} is unexpectedly larger than SciPy error {err_sp}"
    )
