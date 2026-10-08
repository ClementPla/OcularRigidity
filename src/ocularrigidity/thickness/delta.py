"""Measured pulsatile choroidal-thickness change (ΔCT)."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d

from ocularrigidity.motion.displacement import extract_displacement_at_boundaries
from ocularrigidity.consts import AXIAL_PIXEL_SIZE_MM, TRANVERSAL_PIXEL_SIZE_MM
from ocularrigidity.pipeline_config import DELTA_A
from ocularrigidity.segmentation.postprocess.interfaces import (
    clean_boundaries,
    extract_boundaries_fast,
    smooth_boundary_2d,
)


@dataclass
class DeltaCTResult:
    """Result of a ΔCT measurement."""

    deltaCT_mm: float
    deltaCT_um: float
    ct_series_mm: np.ndarray
    baseline_ct_mm: float
    ct_abs_series_mm: np.ndarray
    min_ct_mm: float
    max_ct_mm: float
    n_csi_anchors: int
    rpe_residual_um: float


@dataclass
class CycleRates:
    """How fast the choroid thickens and thins within one cardiac cycle."""

    thickening_um_s: float
    thinning_um_s: float
    asymmetry: float
    thickening_fraction: float


def cycle_rates(
    ct_series_mm: np.ndarray, period_s: float, n_harm: int = 4
) -> CycleRates:
    """Peak thickening / thinning rates over one *folded* cardiac cycle."""
    y = np.asarray(ct_series_mm, dtype=float) * 1000.0  # -> µm
    n = y.size
    nan = CycleRates(np.nan, np.nan, np.nan, np.nan)
    if n < 2 * n_harm + 1 or not np.isfinite(period_s) or period_s <= 0:
        return nan
    ok = np.isfinite(y)
    if ok.sum() < n // 2:
        return nan
    if not ok.all():  # periodic gap fill, as elsewhere in the pipeline
        idx = np.flatnonzero(ok)
        y = np.interp(np.arange(n), idx, y[idx], period=n)

    spec = np.fft.rfft(y - y.mean())
    spec[n_harm + 1 :] = 0
    k = np.arange(spec.size)  # harmonic number = cycles per period
    dy = np.fft.irfft(spec * (2j * np.pi * k / period_s), n)  # µm/s

    up, down = float(dy.max()), float(-dy.min())
    return CycleRates(
        thickening_um_s=up,
        thinning_um_s=down,
        asymmetry=up / down if down > 0 else np.nan,
        thickening_fraction=float((dy > 0).mean()),
    )


def _boundary_slope(boundary: np.ndarray, window: int) -> np.ndarray:
    """Per-column ``dy/dx`` (px/px) from a local least-squares line fit."""
    half = max(1, int(window) // 2)
    size = 2 * half + 1
    x = np.arange(boundary.size, dtype=np.float64)
    w = np.isfinite(boundary).astype(np.float64)
    y = np.where(w > 0, boundary, 0.0)

    def box(a: np.ndarray) -> np.ndarray:  # windowed sum, zero-padded at edges
        return uniform_filter1d(a, size=size, mode="constant") * size

    s0, s1, s2 = box(w), box(w * x), box(w * x * x)
    sy, sxy = box(w * y), box(w * x * y)
    den = s0 * s2 - s1 * s1  # == s0^2 * Var(x) over the valid window points
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = (s0 * sxy - s1 * sy) / den
    return np.where((s0 >= 2) & (den > 0), slope, np.nan)


def _csi_unit_normal_mm(
    csi_ref: np.ndarray,
    axial_mm_per_px: float,
    transversal_mm_per_px: float,
    smooth_sigma: float = 0.0,
    slope_window: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-column unit CSI normal in *physical* (mm) space, as (nx, ny)."""
    if smooth_sigma > 0:
        csi_ref = smooth_boundary_2d(
            csi_ref[None, :], sigma_time=0.0, sigma_col=smooth_sigma
        )[0]
    if slope_window >= 3:
        slope = _boundary_slope(csi_ref, slope_window)
    else:
        slope = np.gradient(csi_ref)  # dy/dx in px, on the de-noised boundary
    nx = axial_mm_per_px * slope
    ny = np.full_like(slope, -transversal_mm_per_px)
    norm = np.hypot(nx, ny)
    norm[norm == 0] = 1.0
    return nx / norm, ny / norm


def mask_ct_series_mm(
    masks: np.ndarray,
    axial_mm_per_px: float = AXIAL_PIXEL_SIZE_MM,
    transversal_mm_per_px: float = TRANVERSAL_PIXEL_SIZE_MM,
    normal_slope_window: int = DELTA_A.csi_normal_slope_window,
) -> np.ndarray:
    """Per-frame mean choroidal thickness (mm), measured on the masks alone."""
    rpe, csi = extract_boundaries_fast(np.asarray(masks, dtype=bool))
    rpe, csi = clean_boundaries(rpe, csi)
    cos_tilt = np.stack(
        [
            np.abs(
                _csi_unit_normal_mm(
                    c,
                    axial_mm_per_px,
                    transversal_mm_per_px,
                    slope_window=normal_slope_window,
                )[1]
            )
            for c in csi
        ]
    )  # (T, W)
    ct = (csi - rpe) * axial_mm_per_px * cos_tilt  # (T, W) mm
    valid = np.isfinite(ct)
    counts = valid.sum(axis=1)
    sums = np.where(valid, ct, 0.0).sum(axis=1)
    return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def measure_delta_ct(
    frames: np.ndarray,
    masks: np.ndarray,
    *,
    reference_frame_idx: int = 0,
    method: str = DELTA_A.method,
    smooth_window: int = DELTA_A.smooth_window,
    lk_window: int = DELTA_A.lk_window,
    subtract_rpe_motion: bool = True,
    axial_mm_per_px: float = AXIAL_PIXEL_SIZE_MM,
    transversal_mm_per_px: float = TRANVERSAL_PIXEL_SIZE_MM,
    normal_smooth_sigma: float = DELTA_A.csi_normal_smooth_sigma,
    normal_slope_window: int = DELTA_A.csi_normal_slope_window,
) -> DeltaCTResult:
    """Measure the peak-to-peak choroidal thickness change (ΔCT)."""
    frames = np.asarray(frames)
    masks = np.asarray(masks)
    if frames.shape != masks.shape:
        raise ValueError(
            f"`frames` and `masks` must share shape; got {frames.shape} vs {masks.shape}."
        )
    disp, ref_xy = extract_displacement_at_boundaries(
        frames,
        masks,
        reference_frame_idx=reference_frame_idx,
        smooth_window=smooth_window,
        method=method,
        lk_window=lk_window,
    )
    return measure_delta_ct_from_disp(
        disp,
        ref_xy,
        masks,
        reference_frame_idx=reference_frame_idx,
        subtract_rpe_motion=subtract_rpe_motion,
        axial_mm_per_px=axial_mm_per_px,
        transversal_mm_per_px=transversal_mm_per_px,
        normal_smooth_sigma=normal_smooth_sigma,
        normal_slope_window=normal_slope_window,
    )


def measure_delta_ct_from_disp(
    disp: np.ndarray,
    ref_xy: np.ndarray,
    masks: np.ndarray,
    reference_frame_idx: int = 0,
    subtract_rpe_motion: bool = True,
    axial_mm_per_px: float = AXIAL_PIXEL_SIZE_MM,
    transversal_mm_per_px: float = TRANVERSAL_PIXEL_SIZE_MM,
    normal_smooth_sigma: float = DELTA_A.csi_normal_smooth_sigma,
    normal_slope_window: int = DELTA_A.csi_normal_slope_window,
):
    rpe, csi = extract_boundaries_fast(masks.astype(bool))
    rpe, csi = clean_boundaries(rpe, csi)
    rpe_ref, csi_ref = rpe[reference_frame_idx], csi[reference_frame_idx]
    W = masks.shape[2]
    xi = np.clip(np.round(ref_xy[:, 0]).astype(int), 0, W - 1)
    yi = ref_xy[:, 1]
    d_csi = np.abs(yi - csi_ref[xi])
    d_rpe = np.abs(yi - rpe_ref[xi])
    _d_csi = np.nan_to_num(d_csi, nan=np.inf)
    _d_rpe = np.nan_to_num(d_rpe, nan=np.inf)
    is_csi = np.isfinite(d_csi) & (_d_csi <= _d_rpe)
    is_rpe = np.isfinite(d_rpe) & (_d_rpe < _d_csi)
    if not is_csi.any():
        raise ValueError("No tracked boundary anchors were classified as CSI.")

    # Physical (mm) displacement vectors and per-anchor physical CSI normal.
    disp_mm = np.stack(
        [disp[..., 0] * transversal_mm_per_px, disp[..., 1] * axial_mm_per_px],
        axis=-1,
    )  # (T, N, 2)
    nx_col, ny_col = _csi_unit_normal_mm(
        csi_ref,
        axial_mm_per_px,
        transversal_mm_per_px,
        normal_smooth_sigma,
        normal_slope_window,
    )
    nx, ny = nx_col[xi], ny_col[xi]  # (N,), pointing into the choroid (ny < 0)

    # Remove residual rigid RPE motion (common-mode) for robustness.
    rpe_residual_um = 0.0
    if subtract_rpe_motion and is_rpe.any():
        rpe_mean = np.nanmean(disp_mm[:, is_rpe, :], axis=1)  # (T, 2) mm
        disp_mm = disp_mm - rpe_mean[:, None, :]
        rpe_residual_um = float(
            np.nanmax(np.hypot(rpe_mean[:, 0], rpe_mean[:, 1])) * 1000.0
        )

    # Signed across-interface displacement per anchor (mm), CSI anchors only.
    proj = -(disp_mm[..., 0] * nx[None, :] + disp_mm[..., 1] * ny[None, :])  # (T, N)
    proj_csi = proj[:, is_csi]  # (T, Nc)

    # Signed spatial mean per frame (NaN-aware), then peak-to-peak over time.
    valid = np.isfinite(proj_csi)
    sums = np.where(valid, proj_csi, 0.0).sum(axis=1)
    counts = valid.sum(axis=1)
    ct_series = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)  # (T,) mm

    delta_ct_mm = float(np.nanmax(ct_series) - np.nanmin(ct_series))

    # Absolute thickness: add the reference-frame baseline to the smooth change.
    csi_x = xi[is_csi]
    cos_tilt = np.abs(ny[is_csi])  # = cos(physical CSI tilt), per CSI anchor
    baseline_col_mm = (csi_ref[csi_x] - rpe_ref[csi_x]) * axial_mm_per_px * cos_tilt
    baseline_ct_mm = float(np.nanmean(baseline_col_mm))
    ct_abs_series = baseline_ct_mm + ct_series  # (T,) mm

    return DeltaCTResult(
        deltaCT_mm=delta_ct_mm,
        deltaCT_um=delta_ct_mm * 1000.0,
        ct_series_mm=ct_series,
        baseline_ct_mm=baseline_ct_mm,
        ct_abs_series_mm=ct_abs_series,
        min_ct_mm=float(np.nanmin(ct_abs_series)),
        max_ct_mm=float(np.nanmax(ct_abs_series)),
        n_csi_anchors=int(is_csi.sum()),
        rpe_residual_um=rpe_residual_um,
    )
