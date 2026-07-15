"""Tests for the class-weight computation used for the balancing rubric item."""

import torch

from training.losses import compute_class_weights


class FakeImageFolder:
    """Minimal stand-in for torchvision.datasets.ImageFolder's relevant attributes."""

    def __init__(self, classes, targets):
        self.classes = classes
        self.targets = targets


def test_compute_class_weights_balanced_dataset_returns_near_uniform_weights():
    dataset = FakeImageFolder(
        classes=["Apple", "Banana", "Grape"],
        targets=[0, 0, 1, 1, 2, 2],
    )

    weights = compute_class_weights(dataset)

    assert torch.allclose(weights, torch.ones(3), atol=1e-6)


def test_compute_class_weights_imbalanced_dataset_favors_minority_class():
    dataset = FakeImageFolder(
        classes=["Apple", "Banana", "Grape"],
        targets=[0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
    )

    weights = compute_class_weights(dataset)

    assert weights[1] > weights[0]
    assert weights[2] > weights[0]
