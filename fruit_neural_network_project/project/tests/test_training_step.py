"""Basic training step test."""

import torch

from models.fruit_cnn import FruitCNN
from training.losses import get_loss_function
from training.optimizers import get_optimizer


def test_single_training_step_runs():
    model = FruitCNN(num_classes=5)
    loss_function = get_loss_function("cross_entropy")
    optimizer = get_optimizer("adamw", model.parameters(), learning_rate=1e-3)

    images = torch.randn(2, 3, 224, 224)
    labels = torch.tensor([0, 1])

    logits = model(images)
    loss = loss_function(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert loss.item() > 0
