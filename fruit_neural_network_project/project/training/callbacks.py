"""Training callbacks."""


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, validation_loss: float) -> bool:
        """Update early stopping state and return whether training should stop."""
        if validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True

        return self.should_stop
