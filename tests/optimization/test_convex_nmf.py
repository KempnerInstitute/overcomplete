import pytest
import torch
from sklearn.decomposition import NMF as SkNMF

from overcomplete.optimization import ConvexNMF
from overcomplete.metrics import relative_avg_l2_loss, l0_eps

data_shape = (50, 10)
nb_concepts = 5

A = torch.rand(data_shape, dtype=torch.float32)

sk_model = SkNMF(n_components=nb_concepts, init='random', solver='mu', max_iter=1000)
W_sk = sk_model.fit_transform(A.numpy())
D_sk = sk_model.components_
sk_error = relative_avg_l2_loss(A, W_sk @ D_sk)


solvers = ['mu', 'pgd']


@pytest.mark.parametrize("solver", solvers)
def test_convex_nmf_initialization(solver):
    """Test that the ConvexNMF class initializes properly."""
    model = ConvexNMF(nb_concepts=nb_concepts, solver=solver)
    assert model.nb_concepts == nb_concepts, "Incorrect number of components."


@pytest.mark.parametrize("solver", solvers)
def test_convex_nmf_fit(solver):
    """Test that the ConvexNMF model can fit to the data."""
    model = ConvexNMF(nb_concepts=nb_concepts, max_iter=2, solver=solver)
    Z, D = model.fit(A)
    assert (Z >= 0).all(), "Negative values in Z."
    assert (model.W >= 0).all(), "Negative values in W."
    assert D.shape == (nb_concepts, A.shape[1]), "Incorrect shape for D."
    assert Z.shape == (A.shape[0], nb_concepts), "Incorrect shape for Z."
    assert model.W.shape == (nb_concepts, A.shape[0]), "Incorrect shape for W."
    assert torch.isnan(Z).sum() == 0, "NaN values in Z."


@pytest.mark.parametrize("solver", solvers)
def test_convex_nmf_encode_decode(solver):
    """Test the encode and decode methods of the ConvexNMF model."""
    model = ConvexNMF(nb_concepts=nb_concepts, max_iter=2, solver=solver)
    model.fit(A)

    Z = model.encode(A)
    assert (Z >= 0).all(), "Negative values in encoded data."
    assert Z.shape == (A.shape[0], nb_concepts), "Incorrect shape for encoded data."

    A_hat = model.decode(Z)
    assert A_hat.shape == A.shape, "Incorrect shape for decoded data."


@pytest.mark.parametrize("solver", solvers)
def test_convex_nmf_reconstruction_error(solver):
    """Test that the reconstruction error decreases after fitting."""
    model = ConvexNMF(nb_concepts=nb_concepts, max_iter=100, solver=solver)
    initial_Z = model.init_random_z(A)
    initial_W = model.init_random_w(A)
    initial_D = initial_W @ A
    initial_error = torch.norm(A - initial_Z @ initial_D, 'fro')
    Z, _ = model.fit(A)
    D = model.get_dictionary()
    A_hat = Z @ D
    final_error = torch.norm(A - A_hat, 'fro')
    assert (Z >= 0).all(), "Negative values in Z."
    assert (model.W >= 0).all(), "Negative values in W."
    assert final_error < initial_error, "Reconstruction error did not decrease."


@pytest.mark.parametrize("solver", solvers)
def test_convex_nmf_zero_data(solver):
    """Test how the model handles data with zeros."""
    zero_data = torch.zeros_like(A)
    model = ConvexNMF(nb_concepts=nb_concepts, solver=solver)
    Z, D = model.fit(zero_data)
    reconstruction_error = torch.norm(zero_data - Z @ D, 'fro')
    assert (Z >= 0).all(), "Negative values in Z."
    assert (model.W >= 0).all(), "Negative values in W."
    assert reconstruction_error < 1e-5, "Model did not reconstruct zero data correctly."


@pytest.mark.parametrize("solver", solvers)
def test_convex_nmf_large_number_of_components(solver):
    """Test the ConvexNMF model with varying number of components."""
    small_model = ConvexNMF(nb_concepts=1, solver=solver)
    small_model.fit(A)
    error_small = torch.square(A - small_model.decode(small_model.encode(A))).sum()

    large_model = ConvexNMF(nb_concepts=100, solver=solver)
    large_model.fit(A)
    error_large = torch.square(A - large_model.decode(large_model.encode(A))).sum()

    assert error_large <= error_small, "Higher error with more components."


@pytest.mark.parametrize("solver", solvers)
def test_convex_nmf_strict_convexity(solver):
    """Test the ConvexNMF model with strict convexity enforced."""
    model = ConvexNMF(nb_concepts=nb_concepts, strict_convex=True, solver=solver)
    Z, D = model.fit(A)
    W = model.W
    sum_of_alphas_per_components = torch.sum(W, dim=1)
    assert torch.allclose(sum_of_alphas_per_components, torch.ones(
        nb_concepts), atol=1e-5), "Columns of W do not sum to 1."


@pytest.mark.parametrize("solver", solvers)
def test_compare_to_sklearn(solver, repetitions=10):
    """
    Test that ConvexNMF achieves similar performance to sklearn NMF.
    """
    is_ok = False
    for _ in range(repetitions):
        our_model = ConvexNMF(nb_concepts=nb_concepts, max_iter=1000, solver=solver)
        Z, D = our_model.fit(A)
        our_error = relative_avg_l2_loss(A, Z @ D)

        assert (Z >= 0).all(), "Negative values in Z."
        assert (our_model.W >= 0).all(), "Negative values in W."

        if our_error < 2.0 * sk_error:
            is_ok = True
            break

    assert is_ok, "ConvexNMF did not match sklearn NMF performance."


@pytest.mark.parametrize("solver", solvers)
def test_compare_strict_convex_to_sklearn(solver, repetition=10):
    """
    Test that ConvexNMF achieves similar performance to sklearn NMF.
    """
    is_ok = False
    for _ in range(repetition):
        our_model = ConvexNMF(nb_concepts=nb_concepts, max_iter=1000, strict_convex=True, solver=solver)
        Z, D = our_model.fit(A)
        our_error = relative_avg_l2_loss(A, Z @ D)

        assert (Z >= 0).all(), "Negative values in Z."
        assert (our_model.W >= 0).all(), "Negative values in W."

        if our_error < 2.0 * sk_error:
            is_ok = True
            break

    assert is_ok, "ConvexNMF did not match sklearn NMF performance."


def test_sparsity():
    """Ensure the l1 penalty induce better sparsity score for ConvexNMF 'pgd' solver."""
    model1 = ConvexNMF(nb_concepts=nb_concepts, max_iter=1000, solver='pgd', l1_penalty=0.0)
    Z1, D1 = model1.fit(A)

    model2 = ConvexNMF(nb_concepts=nb_concepts, max_iter=1000, solver='pgd', l1_penalty=0.5)
    Z2, D2 = model2.fit(A)

    s1 = l0_eps(Z1)
    s2 = l0_eps(Z2)

    assert s2 < s1, "Stronger penalty should induce better sparsity."


@pytest.mark.parametrize("solver", solvers)
def test_cnmf_to_device(solver):
    """Test that the CNMF model can move to the 'meta' device and handle lightweight initialization."""
    model = ConvexNMF(nb_concepts=nb_concepts, solver=solver)

    model.to('meta')

    for attr_name in dir(model):
        attr = getattr(model, attr_name)
        if isinstance(attr, torch.Tensor):
            assert attr.device.type == 'meta', f"Tensor {attr_name} is not on the 'meta' device"
