import numpy as np

import pandas as pd

from scipy.signal import butter, filtfilt
import torch

from ocularrigidity.pipeline_config import AXIAL_PIXEL_SIZE_MM
from ocularrigidity.segmentation.postprocess.blob import (
    keep_largest_connected_component,
)
from ocularrigidity.segmentation.postprocess.interfaces import (
    clean_boundaries,
    extract_boundaries_fast,
    rebuild_mask,
)


def estimate_fovea_from_ilm(
    ilm: np.ndarray, margin: int = 75, center_bias: float | None = 1.0
) -> np.ndarray:
    """Vectorised ``estimate_fovea_from_ilm`` over a batch of ILM curves.

    Parameters
    ----------
    ilm : (B, W) array
        One ILM boundary per row (NaN where unsegmented).
    margin : int
        Half-width (px) of the parabolic window fitted around the detected peak.
    center_bias : float or None
        Bias the peak toward the image center — the macula is central, while the
        edges can carry artefactual extrema. A quadratic penalty (0 at the center,
        growing to ``center_bias * dynamic_range`` at the edges) is subtracted
        from the peak score, so a border candidate must beat a central one by that
        much to win. Larger = stronger pull to the center; ``0``/``None`` disables
        it (plain ``argmax``).

    Returns
    -------
    (B, 2) array
        ``[fovea_x, fovea_y]`` per curve; ``[nan, nan]`` for rows with < 2 valid
        samples.
    """
    ilm = np.asarray(ilm, dtype=float)
    if ilm.ndim == 1:
        ilm = ilm[None, :]
    B, W = ilm.shape
    x = np.arange(W, dtype=float)

    mask = ~np.isnan(ilm)  # (B, W)
    valid = mask.sum(axis=1) >= 2  # rows we can actually fit

    # 1. Per-row linear detrend (masked OLS, closed form == np.polyfit deg=1).
    n = mask.sum(axis=1)
    Sx = np.where(mask, x, 0.0).sum(axis=1)
    Sxx = np.where(mask, x * x, 0.0).sum(axis=1)
    Sy = np.nansum(ilm, axis=1)
    Sxy = np.nansum(ilm * x, axis=1)
    denom = n * Sxx - Sx**2
    denom = np.where(denom == 0, np.nan, denom)
    slope = (n * Sxy - Sx * Sy) / denom
    intercept = (Sy - slope * Sx) / n
    trendline = slope[:, None] * x[None, :] + intercept[:, None]  # (B, W)
    baseline = trendline.mean(axis=1, keepdims=True)  # (B, 1)
    ilm_flat = ilm - trendline + baseline  # (B, W)

    # 2. Fill gaps (linear interp + edge fill) along each row.
    clean = (
        pd.DataFrame(ilm_flat)
        .interpolate(method="linear", axis=1, limit_direction="both")
        .to_numpy()
    )

    # 3. Low-pass filter each row (filtfilt operates along the given axis).
    b, a = butter(N=3, Wn=0.005, btype="lowpass", fs=1.0)
    filtered = filtfilt(b, a, clean, axis=1)
    filtered[~mask] = np.nan  # restore original gaps before the peak search

    # 4. Peak per row. Optionally subtract a quadratic penalty (0 at the center,
    #    center_bias * dynamic_range at the edges) so central (macular) extrema
    #    are favoured over artefactual ones at the edges.
    score = np.where(np.isnan(filtered), -np.inf, filtered)  # (B, W)
    if center_bias:
        center = (W - 1) / 2.0
        rng = np.nanmax(filtered, axis=1, keepdims=True) - np.nanmin(
            filtered, axis=1, keepdims=True
        )  # (B, 1) per-row scale, makes the penalty amplitude-independent
        dist2 = (((x - center) / (0.5 * W)) ** 2)[None, :]  # 0 at center, 1 at edges
        score = np.where(np.isfinite(score), score - center_bias * rng * dist2, -np.inf)
    peak = np.argmax(score, axis=1)  # (B,)

    # 5. Parabolic vertex in a ±margin window centred on each peak. The window
    #    offsets are the same for every row, so the deg-2 design matrix is
    #    constant and the fit is one pseudo-inverse applied to the gathered rows.
    t = np.arange(-margin, margin + 1, dtype=float)  # (K,) centred offsets
    V_pinv = np.linalg.pinv(np.stack([t**2, t, np.ones_like(t)], axis=1))  # (3, K)
    idx = np.clip(peak[:, None] + t[None, :].astype(int), 0, W - 1)  # (B, K)
    region = np.take_along_axis(ilm_flat, idx, axis=1)  # (B, K)
    a2, b2, c2 = (region @ V_pinv.T).T  # (B,) each, for a2*t^2 + b2*t + c2

    with np.errstate(divide="ignore", invalid="ignore"):
        vertex_t = -b2 / (2.0 * a2)
    fovea_x = peak + vertex_t
    fovea_y = a2 * vertex_t**2 + b2 * vertex_t + c2

    out = np.stack([fovea_x, fovea_y], axis=1)  # (B, 2)
    out[~valid] = np.nan
    return out


def estimate_fovea(
    frames: torch.Tensor | np.ndarray, masks: torch.Tensor | np.ndarray
) -> np.ndarray:
    if isinstance(frames, torch.Tensor):
        frames = frames.cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    bm, csi = clean_boundaries(*extract_boundaries_fast(masks))
    margin = 75
    axial_pixel_size = AXIAL_PIXEL_SIZE_MM
    max_thickness_um = 425
    upper_retinal_bbox = bm - (
        max_thickness_um / axial_pixel_size
    )  # upper retinal boundary

    roi_mask = rebuild_mask(upper_retinal_bbox, bm, masks.shape[1])
    roi_mask = roi_mask.astype(bool) & (frames > 25)
    roi_mask = keep_largest_connected_component(
        roi_mask
    )  # keep largest connected component

    ilm, bm = clean_boundaries(*extract_boundaries_fast(roi_mask))

    return estimate_fovea_from_ilm(ilm, margin=margin)
