"""ResNet18 transfer-learning backbone for fruit image classification."""

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(num_classes: int, freeze_backbone: bool = True, pretrained: bool = True) -> nn.Module:
    """Build a ResNet18 with its classifier head replaced for `num_classes`.

    `pretrained=False` skips downloading ImageNet weights, used by tests that
    only need to check the architecture assembles and produces the right shape.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
