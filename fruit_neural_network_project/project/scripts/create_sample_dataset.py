"""Create a tiny synthetic dataset to verify that the pipeline runs.

This is not for real training. It only creates simple RGB images so the code
can be tested without downloading the real fruit dataset.

Usage:
    python scripts/create_sample_dataset.py
"""

from pathlib import Path
from PIL import Image, ImageDraw

CLASSES = ["Apples", "Bananas", "Grapes", "Mangoes", "Strawberries"]
SPLITS = ["train", "val", "test"]
COLORS = {
    "Apples": (220, 30, 30),
    "Bananas": (240, 220, 40),
    "Grapes": (120, 50, 180),
    "Mangoes": (240, 150, 40),
    "Strawberries": (220, 40, 80),
}


def create_image(path: Path, color: tuple[int, int, int], text: str) -> None:
    image = Image.new("RGB", (224, 224), color=color)
    draw = ImageDraw.Draw(image)
    draw.text((20, 100), text, fill=(255, 255, 255))
    image.save(path)


def main() -> None:
    for split in SPLITS:
        for class_name in CLASSES:
            folder = Path("data/split") / split / class_name
            folder.mkdir(parents=True, exist_ok=True)
            count = 8 if split == "train" else 3
            for index in range(count):
                path = folder / f"{class_name.lower()}_{index:03d}.png"
                create_image(path, COLORS[class_name], class_name)

    print("Synthetic sample dataset created in data/split")


if __name__ == "__main__":
    main()
