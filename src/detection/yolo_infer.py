"""
Runtime YOLO inference used by the Streamlit app.

- loads the trained YOLO segmentation model
- runs prediction on a single image
- postprocesses detections (low-confidence → 'wound_unknown')
- saves/returns visualization image (bbox/segmented)
- also returns a simple XAI placeholder image for the UI
"""

# src/detection/yolo_infer.py
"""
Run YOLO on an image, normalize low-confidence / weird labels, and
(optionally) generate a Grad-CAM image for the top detection.

Returns:
    detections: list[dict]
    detection_image_path: Path
    xai_image_path: Path
"""

from pathlib import Path
import cv2
from ultralytics import YOLO

from src.config import YOLO_MODEL_PATH, DETECTION_CONF_THRESHOLD
from src.xai.gradcam import make_gradcam  # <-- new util

# classes actually used in your project
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
    "wound_unknown",
}


def _normalize_detections(raw, ui_conf: float):
    cleaned = []
    for d in raw:
        label = d["label"]
        conf = float(d["confidence"])
        bbox = d["bbox"]

        # 1) UI confidence wins
        if conf < ui_conf:
            label = "wound_unknown"

        # 2) sanity
        if label not in KNOWN_CLASSES:
            label = "wound_unknown"

        cleaned.append(
            {
                "label": label,
                "confidence": round(conf, 3),
                "bbox": bbox,
            }
        )
    return cleaned


def run_detection(image_path: str, conf_threshold: float = None):
    model_path = Path(YOLO_MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = YOLO(str(model_path))

    # if UI did not send anything, fall back to config
    ui_conf = conf_threshold if conf_threshold is not None else DETECTION_CONF_THRESHOLD

    # run quite open; we will post-filter
    results = model(image_path, conf=0.05, verbose=False)
    res0 = results[0]

    detections = []
    for box in res0.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        label = res0.names[cls_id]

        detections.append(
            {
                "label": label,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
            }
        )

    # normalize
    detections = _normalize_detections(detections, ui_conf)

    # if nothing at all → synthesize unknown
    if not detections:
        detections = [{"label": "wound_unknown", "confidence": 0.0, "bbox": None}]

    # save yolo render
    save_dir = Path("runs/detect")
    save_dir.mkdir(parents=True, exist_ok=True)
    det_img_path = save_dir / "tmp_detected.jpg"
    res0.plot(save=True, filename=str(det_img_path))

    # real-ish gradcam (Tier 3)
    xai_path = save_dir / "tmp_xai.jpg"
    try:
        make_gradcam(model, image_path, str(xai_path))
    except Exception:
        # fallback: copy original if cam fails
        img = cv2.imread(image_path)
        cv2.imwrite(str(xai_path), img)

    return detections, det_img_path, xai_path
