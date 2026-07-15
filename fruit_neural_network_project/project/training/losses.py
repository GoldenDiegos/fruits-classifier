"""Loss function factory."""

from typing import Optional

import torch
import torch.nn as nn


def compute_class_weights(dataset) -> torch.Tensor:
    """Compute inverse-frequency class weights from an ImageFolder-style dataset.

    Uses the dataset's `.targets` (list of integer class indices) and `.classes`
    (list of class names) attributes, both exposed by torchvision.datasets.ImageFolder.
    Weights are normalized so the mean weight is 1.0: a perfectly balanced dataset
    yields weights of ~1.0 for every class.
    """
    targets = torch.as_tensor(dataset.targets, dtype=torch.long)
    counts = torch.bincount(targets, minlength=len(dataset.classes)).float()
    weights = counts.sum() / (len(counts) * counts)
    return weights


def get_loss_function(name: str, weight: Optional[torch.Tensor] = None, device=None):
    """Return loss function by name.

    `weight` is only used by loss functions that support per-class weighting
    (currently `cross_entropy`); it is ignored for the others.
    """
    normalized_name = name.lower().strip()

    if normalized_name == "cross_entropy":
        class_weight = weight.to(device) if weight is not None else None
        return nn.CrossEntropyLoss(weight=class_weight)

    losses = {
        "mse": nn.MSELoss(),
        "mae": nn.L1Loss(),
        "binary_cross_entropy": nn.BCEWithLogitsLoss(),
        "huber": nn.HuberLoss(),
    }

    if normalized_name not in losses:
        supported = ", ".join(sorted(losses.keys() | {"cross_entropy"}))
        raise ValueError(f"Unsupported loss '{name}'. Supported: {supported}")

    return losses[normalized_name]
