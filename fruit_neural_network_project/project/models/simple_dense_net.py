"""Educational dense neural network for fruit image classification."""

import torch
import torch.nn as nn


class SimpleDenseFruitNet(nn.Module):
    """
    Dense neural network used to understand basic neural network mechanics.

    This model flattens the full image and applies fully connected layers.
    It is useful for education, but CNNs are a better choice for images.
    """

    def __init__(
        self,
        num_classes: int = 5,
        hidden_size_1: int = 512,
        hidden_size_2: int = 256,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        input_features = 3 * 224 * 224

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_features, hidden_size_1),
            nn.BatchNorm1d(hidden_size_1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.BatchNorm1d(hidden_size_2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size_2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits."""
        return self.network(x)
