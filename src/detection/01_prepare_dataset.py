"""
01_prepare_dataset.py
---------------------
1. The dataset has been manually uploaded into: datasets/raw/wound-yolo/
2. This script just verifies that the dataset exists and prints basic info.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "datasets" / "raw" / "wound-yolo"
DATA_YAML = DATASET_DIR / "data.yaml"

def main():
    if not DATA_YAML.exists():
        print(f"data.yaml not found in {DATASET_DIR}.")
        print("Please export YOLOv8 dataset from Roboflow and place it there.")
    else:
        print("Dataset ready at:", DATASET_DIR)
        print("Contents:")
        for sub in ["train", "valid", "test"]:
            path = DATASET_DIR / sub / "images"
            print(f" - {sub}: {len(list(path.glob('*')))} images")

if __name__ == "__main__":
    main()
