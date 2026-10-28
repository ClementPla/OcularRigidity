import cv2
import numpy as np
from ocularrigidity.consts import (
    AXIAL_PIXEL_SIZE_MM,
    TRANVERSAL_PIXEL_SIZE_MM,
    BSCAN_WIDTH_PX,
)


def to_plex_grid(frames, axial_mm, lateral_mm, width=BSCAN_WIDTH_PX):
    """Resample B-scans to the PLEX pixel size the networks were trained on."""
    T, H, W = frames.shape
    new_h = int(round(H * axial_mm / AXIAL_PIXEL_SIZE_MM / 32)) * 32
    new_w = int(round(W * lateral_mm / TRANVERSAL_PIXEL_SIZE_MM))
    out = np.stack(
        [cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_CUBIC) for f in frames]
    )
    width = min(width, new_w // 32 * 32)
    x0 = (new_w - width) // 2
    out = np.ascontiguousarray(out[:, :, x0 : x0 + width])
    return out, H * axial_mm / new_h, W * lateral_mm / new_w
