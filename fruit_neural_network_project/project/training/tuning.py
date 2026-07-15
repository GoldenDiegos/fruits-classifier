"""Optuna hyperparameter search for FruitCNN.

Builds every component through the existing factories (`build_model`,
`get_loss_function`, `get_optimizer`, `Trainer`, `EarlyStopping`) so a trial is
just a short-budget version of the same training loop used by `main.py`.
"""

import optuna

from models.model_factory import build_model
from training.callbacks import EarlyStopping
from training.dataset import create_dataloader
from training.losses import compute_class_weights, get_loss_function
from training.optimizers import get_optimizer
from training.trainer import Trainer

ACTIVATIONS = ["relu", "leaky_relu", "gelu", "swish"]
BASE_CHANNELS_CHOICES = [8, 16, 32]
BATCH_SIZE_CHOICES = [16, 32, 64]


def objective(
    trial: optuna.Trial,
    train_dir: str,
    val_dir: str,
    device,
    epochs: int = 12,
    num_workers: int = 0,
    trial_patience: int = 3,
) -> float:
    """Train one hyperparameter combination for a short budget and return best val loss."""
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    activation = trial.suggest_categorical("activation", ACTIVATIONS)
    base_channels = trial.suggest_categorical("base_channels", BASE_CHANNELS_CHOICES)
    batch_size = trial.suggest_categorical("batch_size", BATCH_SIZE_CHOICES)

    train_dataset, train_loader = create_dataloader(
        data_dir=train_dir, batch_size=batch_size, training=True, num_workers=num_workers
    )
    val_dataset, val_loader = create_dataloader(
        data_dir=val_dir, batch_size=batch_size, training=False, num_workers=num_workers
    )

    model = build_model(
        model_name="fruit_cnn",
        num_classes=len(train_dataset.classes),
        activation=activation,
        dropout_rate=dropout_rate,
        base_channels=base_channels,
    )

    class_weights = compute_class_weights(train_dataset)
    loss_function = get_loss_function("cross_entropy", weight=class_weights, device=device)
    optimizer = get_optimizer(
        name="adamw", parameters=model.parameters(), learning_rate=learning_rate, weight_decay=weight_decay
    )

    trainer = Trainer(model=model, loss_function=loss_function, optimizer=optimizer, device=device)
    early_stopping = EarlyStopping(patience=trial_patience)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        trainer.train_one_epoch(train_loader)
        val_loss, val_acc = trainer.validate_one_epoch(val_loader)
        best_val_loss = min(best_val_loss, val_loss)

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if early_stopping.step(val_loss):
            break

    trial.set_user_attr("final_val_accuracy", val_acc)
    return best_val_loss


def run_study(
    train_dir: str,
    val_dir: str,
    device,
    n_trials: int = 20,
    epochs: int = 12,
    num_workers: int = 0,
    seed: int = 42,
) -> optuna.Study:
    """Run the Optuna study and return it (holds `best_params`/`best_value`/trials)."""
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(
        lambda trial: objective(
            trial, train_dir=train_dir, val_dir=val_dir, device=device, epochs=epochs, num_workers=num_workers
        ),
        n_trials=n_trials,
    )
    return study
