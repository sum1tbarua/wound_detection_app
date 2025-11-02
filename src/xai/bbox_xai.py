# src/xai/bbox_xai.py
"""
Simple XAI fallback that highlights the detected region.

- takes YOLO bbox/segmentation output
- draws a soft/red overlay over the detected area
- used when full Grad-CAM is not available or not implemented
"""

from pathlib import Path
import cv2
import numpy as np


def generate_bbox_heatmap(image_path: str, detections: list, save_path: str):
    """
    Simple, robust XAI: highlight detected boxes with a heatmap.
    This does NOT use gradients, so it never crashes.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]

    # start with black heatmap
    heatmap = np.zeros((h, w), dtype=np.float32)

    for det in detections:
        bbox = det.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = map(int, bbox)

        # clamp
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w - 1, x2); y2 = min(h - 1, y2)

        # draw a bright box area
        heatmap[y1:y2, x1:x2] = 1.0

        # optional: blur for nicer look
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)

    # normalize
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # convert to color
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # overlay
    overlay = (0.45 * heatmap_color + 0.55 * img).astype(np.uint8)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), overlay)

    return str(save_path)
