"""Dataset tests."""

from pathlib import Path

import pytest

from training.dataset import create_dataloader


def test_missing_data_dir_raises_error(tmp_path: Path):
    missing_path = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        create_dataloader(str(missing_path), batch_size=2)
