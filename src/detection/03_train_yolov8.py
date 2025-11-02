"""
03_train_yolov8.py
------------------
1. This script is for anyone want to retrain the model
"""

from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_YAML = BASE_DIR / "datasets" / "balanced" / "wound-yolo" / "data.yaml"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("Training YOLOv8 segmentation model...")
    model = YOLO("yolov8s-seg.pt")  # or 'yolo11s-seg.pt'

    results = model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=8,
        name="wound_segmentation_balanced",
        project=str(MODEL_DIR),
        patience=10,
    )

    print("Training complete.")
    print("Metrics:", results.results_dict)

if __name__ == "__main__":
    main()
