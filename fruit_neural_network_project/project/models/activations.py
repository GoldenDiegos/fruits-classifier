"""Activation function factory for neural network experiments."""

import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    """Return a PyTorch activation layer by name."""
    normalized_name = name.lower().strip()

    activations = {
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "softmax": nn.Softmax(dim=1),
        "leaky_relu": nn.LeakyReLU(negative_slope=0.01),
        "gelu": nn.GELU(),
        "swish": nn.SiLU(),
    }

    if normalized_name not in activations:
        supported = ", ".join(sorted(activations.keys()))
        raise ValueError(f"Unsupported activation '{name}'. Supported: {supported}")

    return activations[normalized_name]
