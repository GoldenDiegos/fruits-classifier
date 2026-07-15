"""Model forward-pass tests."""

import torch

from models.fruit_cnn import FruitCNN
from models.model_factory import build_model
from models.simple_dense_net import SimpleDenseFruitNet


def test_fruit_cnn_forward_shape():
    model = FruitCNN(num_classes=5)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 5)


def test_simple_dense_forward_shape():
    model = SimpleDenseFruitNet(num_classes=5)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 5)


def test_fruit_cnn_forward_shape_with_custom_base_channels():
    model = FruitCNN(num_classes=5, base_channels=16)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 5)


def test_model_factory_forwards_base_channels_to_fruit_cnn():
    model = build_model(model_name="fruit_cnn", num_classes=5, base_channels=16)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 5)
