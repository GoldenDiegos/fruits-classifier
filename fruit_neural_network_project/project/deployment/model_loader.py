"""Model loading helpers for deployment."""

from pathlib import Path


def validate_model_path(model_path: str) -> Path:
    """Validate model path exists."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file does not exist: {path}")
    return path
