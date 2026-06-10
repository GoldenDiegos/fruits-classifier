import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

PROCESSED_DIR = Path("data/processed")
SPLIT_DIR = Path("data/split")
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]


def split_dataset():
    if not PROCESSED_DIR.exists():
        print(f"ERROR: PROCESSED_DIR not found: {PROCESSED_DIR}")
        return

    assert round(TRAIN_RATIO + VAL_RATIO + TEST_RATIO, 2) == 1.0, "Split ratios must sum to 1.0"

    class_dirs = sorted([d for d in PROCESSED_DIR.iterdir() if d.is_dir()])

    if not class_dirs:
        print("ERROR: No class directories found in PROCESSED_DIR.")
        return

    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)

    for subset in ["train", "val", "test"]:
        (SPLIT_DIR / subset).mkdir(parents=True, exist_ok=True)

    total_train = total_val = total_test = 0

    for class_dir in class_dirs:
        class_name = class_dir.name
        images = sorted([
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ])

        if len(images) < 3:
            print(f"  WARNING: {class_name} has fewer than 3 images, skipping.")
            continue

        train_files, temp_files = train_test_split(
            images, test_size=(1 - TRAIN_RATIO), random_state=RANDOM_SEED, shuffle=True
        )
        val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
        val_files, test_files = train_test_split(
            temp_files, test_size=(1 - val_ratio_adjusted), random_state=RANDOM_SEED, shuffle=True
        )

        for subset, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            dest_dir = SPLIT_DIR / subset / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, dest_dir / f.name)

        total_train += len(train_files)
        total_val += len(val_files)
        total_test += len(test_files)

        print(f"  {class_name}: {len(train_files)} train / {len(val_files)} val / {len(test_files)} test")

    total = total_train + total_val + total_test
    print(f"\nSplit complete.")
    print(f"  Train : {total_train} ({100 * total_train / total:.2f}%)")
    print(f"  Val   : {total_val} ({100 * total_val / total:.2f}%)")
    print(f"  Test  : {total_test} ({100 * total_test / total:.2f}%)")
    print(f"  Total : {total}")


if __name__ == "__main__":
    split_dataset()
