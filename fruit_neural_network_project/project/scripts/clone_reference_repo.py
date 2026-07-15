"""Clone the reference GitHub repository locally.

Usage:
    python scripts/clone_reference_repo.py
"""

from pathlib import Path
import subprocess

REPO_URL = "https://github.com/GoldenDiegos/fruits-classifier.git"
TARGET_DIR = Path("external/fruits-classifier")


def main() -> None:
    TARGET_DIR.parent.mkdir(parents=True, exist_ok=True)

    if TARGET_DIR.exists():
        print(f"Repository already exists at: {TARGET_DIR}")
        return

    subprocess.run(
        ["git", "clone", REPO_URL, str(TARGET_DIR)],
        check=True,
    )
    print(f"Repository cloned into: {TARGET_DIR}")


if __name__ == "__main__":
    main()
