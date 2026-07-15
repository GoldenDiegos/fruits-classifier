"""Training curve visualization."""

from pathlib import Path

import matplotlib.pyplot as plt


def save_training_curves(history: dict, output_path: str) -> None:
    """Save train/validation loss and accuracy curves as a single PNG.

    `history` must contain the keys: train_loss, val_loss, train_acc, val_acc,
    each a list with one value per epoch.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(12, 5))

    loss_ax.plot(epochs, history["train_loss"], label="Train")
    loss_ax.plot(epochs, history["val_loss"], label="Validation")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.set_title("Loss Curve")
    loss_ax.legend()

    acc_ax.plot(epochs, history["train_acc"], label="Train")
    acc_ax.plot(epochs, history["val_acc"], label="Validation")
    acc_ax.set_xlabel("Epoch")
    acc_ax.set_ylabel("Accuracy")
    acc_ax.set_title("Accuracy Curve")
    acc_ax.legend()

    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
