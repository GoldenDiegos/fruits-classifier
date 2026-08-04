"""Single-image inference utilities."""

from pathlib import Path
from typing import Dict

import torch
from PIL import Image
from torchvision import transforms

from models.model_factory import build_model


class FruitPredictor:
    """Load a trained model and predict one image.

    The model architecture is reconstructed entirely from the checkpoint's own
    `metadata` (class names, model name, and every architecture hyperparameter),
    so a checkpoint always loads with the exact configuration it was trained
    with instead of the caller having to know/pass it.
    """

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self.device = torch.device(device)

        checkpoint = torch.load(model_path, map_location=self.device)
        if "metadata" not in checkpoint:
            raise ValueError(f"Checkpoint at {model_path} is missing metadata.")
        metadata = checkpoint["metadata"]

        self.class_names = metadata["class_names"]

        self.model = build_model(
            model_name=metadata.get("model_name", "fruit_cnn"),
            num_classes=len(self.class_names),
            activation=metadata.get("activation", "relu"),
            dropout_rate=metadata.get("dropout_rate", 0.3),
            base_channels=metadata.get("base_channels", 8),
            # Only affects the optimizer during training, not inference correctness.
            freeze_backbone=metadata.get("freeze_backbone", True),
            # The real trained weights come from the checkpoint right below, so
            # there's no need to also download ImageNet weights just to overwrite them.
            pretrained=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def predict(self, image_path: str) -> Dict:
        """Predict class and probabilities for one image."""
        image = Image.open(Path(image_path)).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_index = torch.max(probabilities, dim=1)

        return {
            "class": self.class_names[predicted_index.item()],
            "confidence": float(confidence.item()),
            "probabilities": {
                class_name: float(probabilities[0][idx].item())
                for idx, class_name in enumerate(self.class_names)
            },
        }
