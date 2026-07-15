"""Tests for training curve plotting."""

from pathlib import Path

from evaluation.curves import save_training_curves


def test_save_training_curves_creates_png(tmp_path: Path):
    history = {
        "train_loss": [1.0, 0.8, 0.6],
        "val_loss": [1.1, 0.9, 0.7],
        "train_acc": [0.3, 0.5, 0.7],
        "val_acc": [0.2, 0.4, 0.6],
    }
    output_path = tmp_path / "curves.png"

    save_training_curves(history, str(output_path))

    assert output_path.exists()
