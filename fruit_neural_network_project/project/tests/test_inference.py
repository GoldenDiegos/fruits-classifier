"""Inference tests for FruitPredictor."""

import torch
from PIL import Image

from deployment.inference import FruitPredictor
from models.fruit_cnn import FruitCNN


def test_fruit_predictor_loads_non_default_base_channels(tmp_path):
    class_names = ["Apple", "Banana", "Grape", "Mango", "Strawberry"]
    model = FruitCNN(num_classes=len(class_names), base_channels=16)

    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": {
                "class_names": class_names,
                "model_name": "fruit_cnn",
                "activation": "relu",
                "dropout_rate": 0.3,
                "base_channels": 16,
            },
        },
        checkpoint_path,
    )

    image_path = tmp_path / "sample.png"
    Image.new("RGB", (224, 224), color=(200, 50, 50)).save(image_path)

    predictor = FruitPredictor(model_path=str(checkpoint_path))
    result = predictor.predict(str(image_path))

    assert result["class"] in class_names
    assert 0.0 <= result["confidence"] <= 1.0
    assert set(result["probabilities"].keys()) == set(class_names)
