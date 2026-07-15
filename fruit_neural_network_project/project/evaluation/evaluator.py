"""Model evaluation utilities."""

import torch

from evaluation.metrics import compute_classification_metrics


class Evaluator:
    """Evaluate a classification model."""

    def __init__(self, model, device) -> None:
        self.model = model.to(device)
        self.device = device

    def predict_all(self, dataloader):
        """Return labels and predictions for a full dataloader."""
        self.model.eval()
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                logits = self.model(images)
                predictions = torch.argmax(logits, dim=1).cpu().tolist()

                all_predictions.extend(predictions)
                all_labels.extend(labels.tolist())

        return all_labels, all_predictions

    def evaluate(self, dataloader):
        """Compute metrics for a dataloader."""
        labels, predictions = self.predict_all(dataloader)
        return compute_classification_metrics(labels, predictions)
