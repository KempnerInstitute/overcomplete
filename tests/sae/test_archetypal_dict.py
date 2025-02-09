import torch
import pytest

from overcomplete.sae import RelaxedArchetypalDictionary
from ..utils import epsilon_equal


def test_relaxed_archetypal_dictionary_initialization():
    nb_concepts = 10
    dimensions = 20
    nb_points = 30
    device = 'cpu'
    delta = 1.0

    points = torch.randn(nb_points, dimensions, device=device)
    dictionary = RelaxedArchetypalDictionary(
        dimensions, nb_concepts, points, delta=delta, device=device
    )

    assert dictionary.nb_concepts == nb_concepts
    assert dictionary.in_dimensions == dimensions
    assert dictionary.device == device
    assert dictionary.delta == delta
    assert dictionary.W.shape == (nb_concepts, nb_points)
    assert dictionary.Relax.shape == (nb_concepts, dimensions)
    assert dictionary.C.shape == (nb_points, dimensions)
    assert dictionary.get_dictionary().shape == (nb_concepts, dimensions)


def test_dictionary_layer_forward():
    nb_concepts = 5
    dimensions = 10
    nb_points = 15
    batch_size = 3

    points = torch.randn(nb_points, dimensions)
    layer = RelaxedArchetypalDictionary(dimensions, nb_concepts, points)

    z = torch.randn(batch_size, nb_concepts)
    x_hat = layer.forward(z)

    assert x_hat.shape == (batch_size, dimensions)


def test_dictionary_layer_get_dictionary():
    nb_concepts = 5
    dimensions = 10
    nb_points = 15

    points = torch.randn(nb_points, dimensions)
    layer = RelaxedArchetypalDictionary(dimensions, nb_concepts, points)
    dictionary = layer.get_dictionary()

    assert dictionary.shape == (nb_concepts, dimensions)


def test_weight_constraints():
    nb_concepts = 5
    dimensions = 10
    nb_points = 15

    points = torch.randn(nb_points, dimensions)
    layer = RelaxedArchetypalDictionary(dimensions, nb_concepts, points)
    dictionary = layer.get_dictionary()

    # Ensure W remains row-stochastic (each row sums to 1)
    W = torch.relu(layer.W)
    W /= W.sum(dim=-1, keepdim=True)

    assert epsilon_equal(W.sum(dim=-1), torch.ones(nb_concepts))

    # Ensure W is non-negative
    assert torch.all(W >= 0)


def test_relax_constraints():
    nb_concepts = 5
    dimensions = 10
    nb_points = 15
    delta = 1.0

    points = torch.randn(nb_points, dimensions)
    layer = RelaxedArchetypalDictionary(dimensions, nb_concepts, points, delta=delta)

    # Ensure ||Relax|| does not exceed delta
    norm_Lambda = layer.Relax.norm(dim=-1)
    assert torch.all(norm_Lambda <= delta + 1e-4)


def test_multiplier_initialization():
    nb_concepts = 5
    dimensions = 10
    nb_points = 15

    points = torch.randn(nb_points, dimensions)

    # With multiplier
    layer = RelaxedArchetypalDictionary(dimensions, nb_concepts, points, use_multiplier=True)
    assert layer.multiplier.requires_grad

    # Without multiplier
    layer_no_multiplier = RelaxedArchetypalDictionary(dimensions, nb_concepts, points, use_multiplier=False)
    assert not layer_no_multiplier.multiplier.requires_grad
    assert epsilon_equal(torch.exp(layer_no_multiplier.multiplier), torch.tensor(1.0))


def test_dictionary_scaling_with_multiplier():
    nb_concepts = 5
    dimensions = 10
    nb_points = 15

    points = torch.randn(nb_points, dimensions)
    layer = RelaxedArchetypalDictionary(dimensions, nb_concepts, points, use_multiplier=True)

    new_value = 0.7  # Arbitrary multiplier value
    with torch.no_grad():
        layer.multiplier.copy_(torch.tensor(new_value, device=layer.device))

    dictionary = layer.get_dictionary()
    expected = (layer.W @ layer.C + layer.Relax) * torch.exp(torch.tensor(new_value, device=layer.device))
    assert epsilon_equal(dictionary, expected)


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_device_compatibility(device):
    nb_concepts = 5
    dimensions = 10
    nb_points = 15

    points = torch.randn(nb_points, dimensions, device=device)
    layer = RelaxedArchetypalDictionary(dimensions, nb_concepts, points, device=device)

    z = torch.randn(2, nb_concepts, device=device)
    x_hat = layer.forward(z)

    assert x_hat.device == torch.device(device)
    assert layer.get_dictionary().device == torch.device(device)


def test_trainable_parameters():
    nb_concepts = 5
    dimensions = 10
    nb_points = 15

    points = torch.randn(nb_points, dimensions)
    layer = RelaxedArchetypalDictionary(dimensions, nb_concepts, points)

    assert layer.W.requires_grad
    assert layer.Relax.requires_grad


def test_gradient_flow():
    nb_concepts = 5
    dimensions = 10
    nb_points = 15
    batch_size = 3

    points = torch.randn(nb_points, dimensions)
    layer = RelaxedArchetypalDictionary(dimensions, nb_concepts, points, use_multiplier=True)

    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)

    z = torch.randn(batch_size, nb_concepts)
    x_hat = layer.forward(z)
    target = torch.zeros_like(x_hat)

    loss = ((x_hat - target) ** 2).mean()
    loss.backward()

    assert layer.W.grad is not None
    assert layer.Relax.grad is not None
    assert layer.multiplier.grad is not None


def test_optimizer_step():
    nb_concepts = 5
    dimensions = 10
    nb_points = 15
    batch_size = 3

    points = torch.randn(nb_points, dimensions)
    layer = RelaxedArchetypalDictionary(dimensions, nb_concepts, points, use_multiplier=True)
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)

    init_W = layer.W.clone().detach()
    init_Relax = layer.Relax.clone().detach()
    init_multiplier = layer.multiplier.clone().detach()

    z = torch.randn(batch_size, nb_concepts)
    x_hat = layer.forward(z)
    target = torch.zeros_like(x_hat)

    loss = ((x_hat - target) ** 2).mean()
    loss.backward()
    optimizer.step()

    assert not torch.allclose(init_W, layer.W.detach(), atol=1e-6)
    assert not torch.allclose(init_Relax, layer.Relax.detach(), atol=1e-6)
    assert not torch.allclose(init_multiplier, layer.multiplier.detach(), atol=1e-6)

    # continue training
    for _ in range(10):
        optimizer.zero_grad()
        x_hat = layer.forward(z)
        loss = ((x_hat - target) ** 2).mean()
        loss.backward()
        optimizer.step()

    # Ensure ||Relax|| does not exceed delta
    norm_Lambda = layer.Relax.norm(dim=-1)
    assert torch.all(norm_Lambda <= layer.delta + 1e-4)

    # Ensure W remains row-stochastic (each row sums to 1)
    layer.get_dictionary()
    W = torch.relu(layer.W)
    W /= (W.sum(dim=-1, keepdim=True) + 1e-10)
    assert epsilon_equal(W.sum(dim=-1), torch.ones(nb_concepts))


@pytest.mark.parametrize("batch_size", [1, 10, 100])
def test_batch_processing(batch_size):
    nb_concepts = 5
    dimensions = 10
    nb_points = 15

    points = torch.randn(nb_points, dimensions)
    layer = RelaxedArchetypalDictionary(dimensions, nb_concepts, points)

    z = torch.randn(batch_size, nb_concepts)
    x_hat = layer.forward(z)

    assert x_hat.shape == (batch_size, dimensions)
