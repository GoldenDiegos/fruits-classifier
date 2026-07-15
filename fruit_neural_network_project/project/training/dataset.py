"""Dataset and dataloader utilities."""

from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_transforms(image_size: int = 224, training: bool = True):
    """Build image transformations for training or evaluation."""
    if training:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def create_dataloader(
    data_dir: str,
    batch_size: int,
    image_size: int = 224,
    training: bool = True,
    num_workers: int = 0,
) -> Tuple[datasets.ImageFolder, DataLoader]:
    """Create an ImageFolder dataset and DataLoader."""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {root}")

    dataset = datasets.ImageFolder(
        root=root,
        transform=build_transforms(image_size=image_size, training=training),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataset, dataloader
