"""Optimizer factory."""

import torch.optim as optim


def get_optimizer(
    name: str,
    parameters,
    learning_rate: float,
    weight_decay: float = 0.0,
):
    """Return optimizer by name."""
    normalized_name = name.lower().strip()

    if normalized_name == "sgd":
        return optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )

    if normalized_name == "adam":
        return optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if normalized_name == "adamw":
        return optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if normalized_name == "rmsprop":
        return optim.RMSprop(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if normalized_name == "lion":
        raise NotImplementedError(
            "Lion is not available in native torch.optim. "
            "Use a third-party implementation after validating it."
        )

    raise ValueError(f"Unsupported optimizer: {name}")
