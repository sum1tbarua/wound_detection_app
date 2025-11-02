# src/xai/gradcam.py
"""
Grad-CAM utilities (work-in-progress).

- hook a vision model layer
- compute activation map
- overlay heatmap on original image
- can be called from yolo_infer or directly from the UI
"""


from pathlib import Path
import cv2
import numpy as np
import torch


def _overlay(img_bgr: np.ndarray, cam: np.ndarray, save_path: Path):
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.5 * heatmap + 0.5 * img_bgr).astype(np.uint8)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), overlay)


def generate_yolo_gradcam(model, image_path: str, save_path: str):
    """
    Best-effort Grad-CAM for Ultralytics YOLO.
    If we can't get grads (because YOLO ran under no_grad), we fall back
    to saving the original image so the app doesn't crash.
    """
    save_path = Path(save_path)
    img_bgr = cv2.imread(image_path)

    # fallback in case *anything* goes wrong
    def _fallback():
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), img_bgr)
        return str(save_path)

    try:
        # we need a tensor that REQUIRES grad
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (640, 640))
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(model.device)
        img_tensor.requires_grad_(True)

        # pick a mid/high layer
        target_layer = model.model.model[-2]

        activations = {}
        gradients = {}

        def fwd_hook(m, i, o):
            activations["val"] = o

        def bwd_hook(m, gi, go):
            gradients["val"] = go[0]

        h1 = target_layer.register_forward_hook(fwd_hook)
        h2 = target_layer.register_full_backward_hook(bwd_hook)

        # forward with grads enabled
        with torch.enable_grad():
            preds = model.model(img_tensor)  # raw forward, NOT model(...)
            # preds is usually a list/tuple; make a scalar to backprop
            if isinstance(preds, (list, tuple)):
                loss_like = 0
                for p in preds:
                    loss_like = loss_like + p.mean()
            else:
                loss_like = preds.mean()

        # backward
        model.model.zero_grad(set_to_none=True)
        loss_like.backward()

        acts = activations["val"]          # [B, C, H, W]
        grads = gradients["val"]           # [B, C, H, W]

        # global average pool grads
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        cam = (weights * acts).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        cam = torch.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()

        # normalize
        cam -= cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # resize to original
        cam = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))

        _overlay(img_bgr, cam, save_path)

        # cleanup hooks
        h1.remove()
        h2.remove()

        return str(save_path)

    except Exception as e:
        # print to console so dev can see it
        print("[GradCAM warning]:", e)
        return _fallback()
