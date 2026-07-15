"""Reusable training loop."""

from pathlib import Path
from typing import Dict, Tuple

import torch


class Trainer:
    """Trainer that implements forward, loss, backpropagation and update."""

    def __init__(self, model, loss_function, optimizer, device) -> None:
        self.model = model.to(device)
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = device

    def train_one_epoch(self, dataloader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        return self._run_epoch(dataloader, training=True)

    def validate_one_epoch(self, dataloader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        with torch.no_grad():
            return self._run_epoch(dataloader, training=False)

    def _run_epoch(self, dataloader, training: bool) -> Tuple[float, float]:
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            loss = self.loss_function(logits, labels)

            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

        average_loss = total_loss / max(total_samples, 1)
        accuracy = correct_predictions / max(total_samples, 1)
        return average_loss, accuracy

    def save_checkpoint(self, output_path: str, metadata: Dict) -> None:
        """Save model checkpoint."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "metadata": metadata,
            },
            output_path,
        )
