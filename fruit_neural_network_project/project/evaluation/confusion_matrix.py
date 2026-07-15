"""Confusion matrix visualization."""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def save_confusion_matrix(y_true, y_pred, class_names, output_path: str) -> None:
    """Save a confusion matrix image."""
    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(8, 8))
    display.plot(ax=ax, xticks_rotation=45)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
