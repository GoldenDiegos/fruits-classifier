"""CNN architecture for fruit image classification."""

import torch
import torch.nn as nn

from models.activations import get_activation


class ConvBlock(nn.Module):
    """Conv2D -> BatchNorm -> Activation -> MaxPool -> Dropout block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "relu",
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            get_activation(activation),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FruitCNN(nn.Module):
    """
    Baseline CNN for 224x224 RGB fruit images.

    This is intentionally compact so it can run on CPU for learning and inside
    VS Code. For a production experiment, increase base_channels or move to
    transfer learning with ResNet/MobileNet/EfficientNet.

    Input shape:
        [batch_size, 3, 224, 224]

    Output:
        raw logits [batch_size, num_classes]
    """

    def __init__(
        self,
        num_classes: int = 5,
        activation: str = "relu",
        dropout_rate: float = 0.3,
        base_channels: int = 8,
    ) -> None:
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.features = nn.Sequential(
            ConvBlock(3, c1, activation=activation, dropout_rate=0.1),
            ConvBlock(c1, c2, activation=activation, dropout_rate=0.2),
            ConvBlock(c2, c3, activation=activation, dropout_rate=0.3),
            ConvBlock(c3, c4, activation=activation, dropout_rate=0.3),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c4, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
