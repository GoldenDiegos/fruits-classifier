"""Main executable for training the Fruit CNN in VS Code, terminal, or Colab."""

import argparse
import json
from pathlib import Path

import torch

from evaluation.confusion_matrix import save_confusion_matrix
from evaluation.curves import save_training_curves
from evaluation.metrics import compute_classification_metrics
from models.model_factory import build_model
from training.callbacks import EarlyStopping
from training.dataset import create_dataloader
from training.losses import compute_class_weights, get_loss_function
from training.optimizers import get_optimizer
from training.train import set_seed
from training.trainer import Trainer


def get_device() -> torch.device:
    """Return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Fruit neural network training pipeline")
    parser.add_argument("--train-dir", default="data/split/train")
    parser.add_argument("--val-dir", default="data/split/val")
    parser.add_argument("--test-dir", default="data/split/test")
    parser.add_argument("--model-name", default="fruit_cnn", choices=["fruit_cnn", "simple_dense"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--dropout-rate", type=float, default=0.3)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-class-weights", action="store_true", default=True)
    parser.add_argument("--output", default="models/checkpoints/best_model.pt")
    parser.add_argument("--metrics-output", default="reports/parcial_2/base_model_metrics.json")
    parser.add_argument("--curves-output", default="reports/parcial_2/base_model_curves.png")
    parser.add_argument(
        "--confusion-matrix-output",
        default="reports/parcial_2/base_model_confusion_matrix.png",
    )
    return parser.parse_args()


def run_training(args) -> dict:
    """Train one full configuration end to end and return its results.

    Shared by the CLI entry point (`main`) and `scripts/tune.py`, which builds an
    equivalent `args`-like object from the winning Optuna hyperparameters. Keeping
    a single implementation guarantees the base run and the tuned retrain follow
    the exact same training/evaluation logic.
    """
    set_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    train_dataset, train_loader = create_dataloader(
        data_dir=args.train_dir,
        batch_size=args.batch_size,
        training=True,
        num_workers=args.num_workers,
    )

    val_dataset, val_loader = create_dataloader(
        data_dir=args.val_dir,
        batch_size=args.batch_size,
        training=False,
        num_workers=args.num_workers,
    )

    test_dataset, test_loader = create_dataloader(
        data_dir=args.test_dir,
        batch_size=args.batch_size,
        training=False,
        num_workers=args.num_workers,
    )

    if train_dataset.classes != val_dataset.classes or train_dataset.classes != test_dataset.classes:
        raise ValueError("Train, validation and test class folders do not match.")

    class_names = train_dataset.classes
    print(f"Detected classes: {class_names}")

    model = build_model(
        model_name=args.model_name,
        num_classes=len(class_names),
        activation=args.activation,
        dropout_rate=args.dropout_rate,
        base_channels=args.base_channels,
    )

    class_weights = None
    if getattr(args, "use_class_weights", True):
        class_weights = compute_class_weights(train_dataset)
        print(f"Class weights ({dict(zip(class_names, class_weights.tolist()))})")

    loss_function = get_loss_function("cross_entropy", weight=class_weights, device=device)
    optimizer = get_optimizer(
        name="adamw",
        parameters=model.parameters(),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    trainer = Trainer(
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        device=device,
    )

    early_stopping = EarlyStopping(patience=args.patience)
    best_validation_loss = float("inf")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(args.epochs):
        train_loss, train_acc = trainer.train_one_epoch(train_loader)
        val_loss, val_acc = trainer.validate_one_epoch(val_loader)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs:03d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            trainer.save_checkpoint(
                args.output,
                metadata={
                    "class_names": class_names,
                    "model_name": args.model_name,
                    "activation": args.activation,
                    "dropout_rate": args.dropout_rate,
                    "base_channels": args.base_channels,
                    "validation_loss": val_loss,
                    "validation_accuracy": val_acc,
                },
            )
            print(f"Saved best model to: {Path(args.output).resolve()}")

        if early_stopping.step(val_loss):
            print("Early stopping triggered.")
            break

    save_training_curves(history, args.curves_output)

    # Reload the best checkpoint (not necessarily the last epoch trained) before
    # the final test-set evaluation.
    checkpoint = torch.load(args.output, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    evaluator_model = model.to(device)
    evaluator_model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = evaluator_model(images)
            predictions = torch.argmax(logits, dim=1).cpu().tolist()
            y_pred.extend(predictions)
            y_true.extend(labels.tolist())

    test_metrics = compute_classification_metrics(y_true, y_pred)
    save_confusion_matrix(y_true, y_pred, class_names, args.confusion_matrix_output)

    result = {
        "checkpoint_path": str(Path(args.output).resolve()),
        "class_names": class_names,
        "best_validation_loss": best_validation_loss,
        "test_metrics": test_metrics,
        "history": history,
        "hyperparameters": {
            "model_name": args.model_name,
            "activation": args.activation,
            "dropout_rate": args.dropout_rate,
            "base_channels": args.base_channels,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
        },
    }

    metrics_output_path = Path(args.metrics_output)
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_metrics": test_metrics,
                "best_validation_loss": best_validation_loss,
                "hyperparameters": result["hyperparameters"],
                "class_names": class_names,
            },
            f,
            indent=2,
        )
    print(f"Saved metrics to: {metrics_output_path.resolve()}")

    return result


def main() -> None:
    args = parse_args()
    result = run_training(args)
    print("Final test metrics:", result["test_metrics"])


if __name__ == "__main__":
    main()
