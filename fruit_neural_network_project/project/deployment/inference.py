"""Single-image inference utilities."""

from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from torchvision import transforms

from models.model_factory import build_model


class FruitPredictor:
    """Load a trained model and predict one image."""

    def __init__(
        self,
        model_path: str,
        class_names: List[str],
        model_name: str = "fruit_cnn",
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.class_names = class_names

        self.model = build_model(model_name=model_name, num_classes=len(class_names))
        checkpoint = torch.load(model_path, map_location=self.device)

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

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
