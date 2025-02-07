import torch
import pytest

from overcomplete.sae.trackers import DeadCodeTracker


def test_initial_dead_ratio():
    """Test that the tracker initially reports all features as dead."""
    nb_concepts = 5
    tracker = DeadCodeTracker(nb_concepts=nb_concepts, device='cpu')
    # No update has been performed, so all features should be dead.
    assert tracker.get_dead_ratio() == 1.0, "Expected initial dead ratio to be 1.0"


def test_update_activation():
    """Test that updating with a batch that has positive activations works."""
    nb_concepts = 4
    tracker = DeadCodeTracker(nb_concepts=nb_concepts, device='cpu')
    # Create a batch with 2 samples and 4 features.
    # In this batch:
    #   - Feature 0: activated in the first sample.
    #   - Feature 1: activated in the second sample.
    #   - Features 2 and 3: not activated.
    z = torch.tensor([
        [1.0,  0.0, -1.0,  0.0],
        [0.0,  1.0,  0.0, -1.0]
    ])
    tracker.update(z)
    # Two out of four features are alive, so dead ratio = 1 - 2/4 = 0.5.
    expected_dead_ratio = 0.5
    assert tracker.get_dead_ratio() == pytest.approx(expected_dead_ratio), \
        f"Expected dead ratio to be {expected_dead_ratio}"


def test_accumulated_updates():
    """Test that multiple updates accumulate activated features."""
    nb_concepts = 3
    tracker = DeadCodeTracker(nb_concepts=nb_concepts, device='cpu')

    # First update: only feature 0 is activated.
    z1 = torch.tensor([[1.0, 0.0, 0.0]])
    tracker.update(z1)
    # Only 1 feature alive out of 3, so dead ratio = 1 - 1/3.
    expected_ratio_after_z1 = 1 - (1 / 3)
    assert tracker.get_dead_ratio() == pytest.approx(expected_ratio_after_z1), \
        f"Expected dead ratio after first update to be {expected_ratio_after_z1}"

    # Second update: activate feature 1.
    z2 = torch.tensor([[0.0, 2.0, 0.0]])
    tracker.update(z2)
    # Now features 0 and 1 are alive: dead ratio = 1 - (2/3).
    expected_ratio_after_z2 = 1 - (2 / 3)
    assert tracker.get_dead_ratio() == pytest.approx(expected_ratio_after_z2), \
        f"Expected dead ratio after second update to be {expected_ratio_after_z2}"

    # Third update: activate feature 2.
    z3 = torch.tensor([[0.0, 0.0, 3.0]])
    tracker.update(z3)
    # Now all 3 features are activated, so dead ratio should be 0.
    assert tracker.get_dead_ratio() == 0.0, "Expected dead ratio to be 0 after all features activated"


def test_update_with_no_activation():
    """Test that an update with no positive activations does not change the tracker."""
    nb_concepts = 2
    tracker = DeadCodeTracker(nb_concepts=nb_concepts, device='cpu')

    # Update with a batch of zeros: no feature is activated.
    z_zeros = torch.zeros((3, nb_concepts))
    tracker.update(z_zeros)
    assert tracker.get_dead_ratio() == 1.0, "Dead ratio should remain 1.0 after a zero update"

    # Update with a batch of negative values.
    z_neg = -torch.ones((2, nb_concepts))
    tracker.update(z_neg)
    assert tracker.get_dead_ratio() == 1.0, "Dead ratio should remain 1.0 after a negative update"
