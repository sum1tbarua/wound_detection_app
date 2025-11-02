"""
Runtime YOLO inference used by the Streamlit app.

- loads the trained YOLO segmentation model
- runs prediction on a single image
- postprocesses detections (low-confidence → 'wound_unknown')
- saves/returns visualization image (bbox/segmented)
- also returns a simple XAI placeholder image for the UI
"""


from pathlib import Path
from ultralytics import YOLO
import cv2
import os
import numpy as np  # 👈 add this

from src.config import YOLO_MODEL_PATH, DETECTION_CONF_THRESHOLD

# classes you actually trained
KNOWN_CLASSES = {
    "healthy_hand",
    "healthy_leg",
    "healthy_arm",
    "hand_cut",
    "hand_burn",
    "leg_cut",
    "leg_burn",
    "arm_cut",
    "arm_burn",
}

def normalize_detections(raw_dets):
    """
    Normalize and clean YOLO detections.
    - Converts low-confidence detections to 'wound_unknown'
    - Prevents obvious cross-body mislabels like leg_* on hand
    """
    cleaned = []
    for d in raw_dets:
        label = d.get("label", "wound_unknown")
        conf = float(d.get("confidence", 0))
        bbox = d.get("bbox")

        # rule 1: too low confidence → unknown
        if conf < 0.25:
            label = "wound_unknown"

        # rule 2: leg_* at low confidence → often misclassified
        if label.startswith("leg_") and conf < 0.4:
            label = "wound_unknown"

        # rule 3: sanity — keep only known classes or unknown
        if label not in KNOWN_CLASSES:
            label = "wound_unknown"

        cleaned.append({
            "label": label,
            "confidence": round(conf, 3),
            "bbox": bbox,
        })
    return cleaned


def make_xai_overlay(img_path: str, detections: list, save_path: Path):
    """
    Lightweight XAI: tint the detected regions red so the UI can
    show 'where the model looked'. This is NOT full Grad-CAM,
    but it's good for the prototype.
    """
    img = cv2.imread(img_path)
    if img is None:
        return

    heat = np.zeros_like(img, dtype=np.uint8)

    for det in detections:
        bbox = det.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        # red rectangle region
        heat[y1:y2, x1:x2, :] = (0, 0, 255)

    # blend original + heat
    overlay = cv2.addWeighted(img, 0.55, heat, 0.45, 0)
    cv2.imwrite(str(save_path), overlay)


def run_detection(image_path: str):
    model_path = Path(YOLO_MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = YOLO(str(model_path))

    # 1) run YOLO fairly open so we don’t miss burn-like areas
    #    (we'll filter ourselves)
    results = model(image_path, conf=0.05, verbose=False)

    detections = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        label = results[0].names[cls_id]

        # if it's below our real threshold → call it unknown
        if conf < DETECTION_CONF_THRESHOLD:
            label = "wound_unknown"

        # if the label is not something we trained → call it unknown
        if label not in KNOWN_CLASSES:
            label = "wound_unknown"

        detections.append({
            "label": label,
            "confidence": round(conf, 3),
            "bbox": [x1, y1, x2, y2],
        })

    # 2) if YOLO returned literally nothing → create a default unknown
    if not detections:
        detections = [{
            "label": "wound_unknown",
            "confidence": 0.0,
            "bbox": None
        }]

    # 3) save detection image (YOLO-plotted)
    save_dir = Path("runs/detect")
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "tmp_detected.jpg"
    results[0].plot(save=True, filename=str(out_path))

    # 4) make an XAI overlay from detections 👇 (this is the change)
    xai_path = Path("runs/detect/tmp_xai.jpg")
    make_xai_overlay(image_path, detections, xai_path)

    # 5) normalize detections before returning
    detections = normalize_detections(detections)
    
    return detections, out_path, xai_path
