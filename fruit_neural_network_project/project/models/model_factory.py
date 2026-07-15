"""Model factory for selecting neural network architectures."""

from models.fruit_cnn import FruitCNN
from models.simple_dense_net import SimpleDenseFruitNet


def build_model(
    model_name: str,
    num_classes: int,
    activation: str = "relu",
    dropout_rate: float = 0.3,
    base_channels: int = 8,
):
    """Build a model from a simple name."""
    normalized_name = model_name.lower().strip()

    if normalized_name in {"fruit_cnn", "cnn"}:
        return FruitCNN(
            num_classes=num_classes,
            activation=activation,
            dropout_rate=dropout_rate,
            base_channels=base_channels,
        )

    if normalized_name in {"simple_dense", "dense"}:
        return SimpleDenseFruitNet(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )

    raise ValueError(f"Unsupported model name: {model_name}")
