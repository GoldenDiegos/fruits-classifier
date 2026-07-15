"""Smoke test for the Optuna objective function used by scripts/tune.py."""

from pathlib import Path

import optuna
from PIL import Image

from training.tuning import objective

CLASSES = ["Apples", "Bananas"]


def _make_synthetic_split(root: Path, count_per_class: int) -> None:
    for class_name in CLASSES:
        folder = root / class_name
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(count_per_class):
            image = Image.new("RGB", (224, 224), color=(index * 10, 50, 100))
            image.save(folder / f"{class_name.lower()}_{index}.png")


def test_objective_runs_a_single_trial(tmp_path: Path):
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    _make_synthetic_split(train_dir, count_per_class=4)
    _make_synthetic_split(val_dir, count_per_class=2)

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(
            trial,
            train_dir=str(train_dir),
            val_dir=str(val_dir),
            device="cpu",
            epochs=1,
            num_workers=0,
            trial_patience=1,
        ),
        n_trials=1,
    )

    assert len(study.trials) == 1
    assert study.best_value != float("inf")
