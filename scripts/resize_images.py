import shutil
from pathlib import Path
from PIL import Image

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
IMAGE_SIZE = (224, 224)
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]


def make_unique_output_path(output_dir, img_path):
    output_path = output_dir / img_path.name

    if not output_path.exists():
        return output_path

    parent_name = img_path.parent.parent.name
    unique_name = f"{parent_name}_{img_path.name}"
    output_path = output_dir / unique_name

    counter = 1
    while output_path.exists():
        unique_name = f"{parent_name}_{counter}_{img_path.name}"
        output_path = output_dir / unique_name
        counter += 1

    return output_path


def resize_images():
    if not RAW_DIR.exists():
        print(f"ERROR: RAW_DIR not found: {RAW_DIR}")
        return

    class_dirs = sorted([
        d for d in RAW_DIR.rglob("*")
        if d.is_dir() and any(
            f.suffix.lower() in IMAGE_EXTENSIONS for f in d.iterdir() if f.is_file()
        )
    ])

    if not class_dirs:
        print("ERROR: No class directories found in RAW_DIR.")
        return

    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / ".gitkeep").touch()

    total_processed = 0
    total_errors = 0

    for class_dir in class_dirs:
        class_name = class_dir.name
        output_dir = PROCESSED_DIR / class_name
        output_dir.mkdir(parents=True, exist_ok=True)

        images = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]

        for img_path in images:
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    img = img.resize(IMAGE_SIZE, Image.LANCZOS)
                    img.save(make_unique_output_path(output_dir, img_path))
                total_processed += 1
            except Exception as e:
                print(f"  ERROR: {img_path.name} - {e}")
                total_errors += 1

        print(f"  {class_name}: {len(images)} images processed.")

    print(f"\nResize complete.")
    print(f"  Total processed : {total_processed}")
    print(f"  Total errors    : {total_errors}")


if __name__ == "__main__":
    resize_images()
