"""Hyperparameter search (Optuna) and final tuned-model retrain/export.

Usage:
    python scripts/tune.py --n-trials 20

Requires `reports/parcial_2/base_model_metrics.json` to already exist (produced by
running `main.py` first) so the comparison table can be generated.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import get_device, run_training  # noqa: E402
from training.tuning import run_study  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for FruitCNN")
    parser.add_argument("--train-dir", default="data/split/train")
    parser.add_argument("--val-dir", default="data/split/val")
    parser.add_argument("--test-dir", default="data/split/test")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--search-epochs", type=int, default=12)
    parser.add_argument("--final-epochs", type=int, default=30)
    parser.add_argument("--final-patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-metrics", default="reports/parcial_2/base_model_metrics.json")
    parser.add_argument("--output", default="models/checkpoints/tuned_model.pt")
    parser.add_argument("--metrics-output", default="reports/parcial_2/tuned_model_metrics.json")
    parser.add_argument("--curves-output", default="reports/parcial_2/tuned_model_curves.png")
    parser.add_argument(
        "--confusion-matrix-output",
        default="reports/parcial_2/tuned_model_confusion_matrix.png",
    )
    parser.add_argument("--comparison-output", default="reports/parcial_2/comparison_table.md")
    parser.add_argument("--trials-output", default="reports/parcial_2/optuna_trials.csv")
    return parser.parse_args()


class TunedTrainingArgs:
    """Minimal args-like object accepted by `main.run_training`."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def build_comparison_table(base_metrics_path: str, tuned_metrics_path: str, output_path: str) -> None:
    with open(base_metrics_path, encoding="utf-8") as f:
        base = json.load(f)
    with open(tuned_metrics_path, encoding="utf-8") as f:
        tuned = json.load(f)

    base_metrics = base["test_metrics"]
    tuned_metrics = tuned["test_metrics"]

    lines = [
        "# Parcial 2 — Base vs Tuned Model Comparison",
        "",
        "| Metric | Base Model | Tuned Model |",
        "|---|---:|---:|",
        f"| Accuracy | {base_metrics['accuracy']:.4f} | {tuned_metrics['accuracy']:.4f} |",
        f"| Precision (macro) | {base_metrics['precision_macro']:.4f} | {tuned_metrics['precision_macro']:.4f} |",
        f"| Recall (macro) | {base_metrics['recall_macro']:.4f} | {tuned_metrics['recall_macro']:.4f} |",
        f"| F1 (macro) | {base_metrics['f1_macro']:.4f} | {tuned_metrics['f1_macro']:.4f} |",
        f"| Best validation loss | {base['best_validation_loss']:.4f} | {tuned['best_validation_loss']:.4f} |",
        "",
        "## Winning hyperparameters (Optuna)",
        "",
        "```json",
        json.dumps(tuned["hyperparameters"], indent=2),
        "```",
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved comparison table to: {Path(output_path).resolve()}")


def main() -> None:
    args = parse_args()

    base_metrics_path = Path(args.base_metrics)
    if not base_metrics_path.exists():
        raise FileNotFoundError(
            f"Base model metrics not found at {base_metrics_path}. Run `python main.py` first."
        )

    device = get_device()
    print(f"Using device: {device}")

    print(f"Starting Optuna search: {args.n_trials} trials, {args.search_epochs} epochs each")
    study = run_study(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        device=device,
        n_trials=args.n_trials,
        epochs=args.search_epochs,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    print(f"Best trial value (val_loss): {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    trials_output_path = Path(args.trials_output)
    trials_output_path.parent.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(trials_output_path, index=False)
    print(f"Saved trial history to: {trials_output_path.resolve()}")

    best_params = study.best_params
    tuned_args = TunedTrainingArgs(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        model_name="fruit_cnn",
        epochs=args.final_epochs,
        batch_size=best_params["batch_size"],
        learning_rate=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"],
        activation=best_params["activation"],
        dropout_rate=best_params["dropout_rate"],
        base_channels=best_params["base_channels"],
        patience=args.final_patience,
        num_workers=args.num_workers,
        seed=args.seed,
        use_class_weights=True,
        output=args.output,
        metrics_output=args.metrics_output,
        curves_output=args.curves_output,
        confusion_matrix_output=args.confusion_matrix_output,
    )

    print("Retraining final tuned model with the winning hyperparameters...")
    result = run_training(tuned_args)

    # Enrich the exported checkpoint's metadata with the tuning evidence.
    checkpoint = torch.load(args.output, map_location="cpu")
    checkpoint["metadata"]["optuna_best_params"] = study.best_params
    checkpoint["metadata"]["optuna_best_val_loss"] = study.best_value
    checkpoint["metadata"]["test_metrics"] = result["test_metrics"]
    torch.save(checkpoint, args.output)
    print(f"Updated checkpoint metadata at: {Path(args.output).resolve()}")

    build_comparison_table(args.base_metrics, args.metrics_output, args.comparison_output)


if __name__ == "__main__":
    main()
