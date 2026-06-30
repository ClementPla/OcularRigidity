"""Inference: predict the per-frame foveal x used for lateral registration.

`predict_fovea_x` runs the keypoint model over a video cube and returns the
foveal x-coordinate in the *original* frame width (resolution-independent: the
model runs on a downscaled copy, the normalized coordinate maps back to full
width). It also returns a per-frame confidence (heatmap peak sharpness), useful
for gating / QC.

Integration with the rigid registration: replace the cross-correlation step in
``register_masks_by_displacement`` with::

    fovea_x, conf = predict_fovea_x(model, raw_frames)
    global_dx = fovea_to_dx(fovea_x, ref_idx=0)   # then smooth_translations(...)
"""

import numpy as np
import torch
import torch.nn.functional as F

from ocularrigidity.fovea.dsnt import dsnt, flat_softmax, normalized_to_pixel


@torch.inference_mode()
def predict_fovea_x(
    model,
    frames: np.ndarray,
    img_size: tuple[int, int] = (256, 512),
    batch_size: int = 64,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Predict foveal x per frame.

    Args:
        model: a trained `FoveaKeypointModule` (or its inner heatmap model).
        frames: (T, H, W) uint8 video cube.
        img_size: (H, W) the model was trained at (frames are resized to this).
        batch_size: frames per forward pass.

    Returns:
        fovea_x: (T,) float, x-coordinate in ORIGINAL frame pixels.
        confidence: (T,) float, heatmap peak value in [0, 1] (sharper = higher).
    """
    model = model.to(device).eval()
    # Accept either a FoveaKeypointModule or a bare heatmap-logit network.
    net = model.model if hasattr(model, "model") else model
    T, H, W = frames.shape
    fovea_x = np.empty(T, dtype=np.float32)
    confidence = np.empty(T, dtype=np.float32)

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        chunk = torch.from_numpy(frames[start:end]).to(device).float().div_(255.0)
        chunk = (chunk - 0.5) / 0.5
        chunk = chunk.unsqueeze(1)  # (b, 1, H, W)
        chunk = F.interpolate(
            chunk, size=img_size, mode="bilinear", align_corners=False
        )

        heatmap = flat_softmax(net(chunk))  # (b, 1, h, w)
        coords = dsnt(heatmap)  # (b, 1, 2) normalized
        # Map normalized x back to the ORIGINAL width (resolution-independent).
        x_px = normalized_to_pixel(coords[:, 0, 0], W)
        peak = heatmap.amax(dim=(-1, -2))[:, 0]

        fovea_x[start:end] = x_px.float().cpu().numpy()
        confidence[start:end] = peak.float().cpu().numpy()

    return fovea_x, confidence


def fovea_to_dx(fovea_x: np.ndarray, ref_idx: int = 0) -> np.ndarray:
    """Lateral shift per frame relative to the reference frame's fovea."""
    return fovea_x - fovea_x[ref_idx]
