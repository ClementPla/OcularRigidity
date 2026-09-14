"""Measured pulsatile choroidal-thickness change (ΔCT).

Estimates the peak-to-peak change in choroidal thickness across one (folded)
cardiac cycle by tracking the choroid boundary with optical flow and following
the displacement of the moving choroid-sclera interface (CSI) along its own
normal.

Rationale
---------
The frames are registered so that the RPE (upper choroid boundary) is
stationary; the choroidal thickness therefore changes only through motion of
the CSI (lower boundary). We track the whole boundary (optical flow + temporal
smoothing, via :func:`extract_displacement_at_boundaries`), keep the anchors
that sit on the CSI, project their *signed* displacement onto the local CSI
normal, and average over the CSI to obtain a per-frame thickness change
``ct(t)`` relative to the reference frame.

``ΔCT = max_t ct(t) - min_t ct(t)`` (peak-to-peak). Using the *signed* per-frame
mean rather than the mean of magnitudes makes ΔCT invariant to the choice of
reference frame: shifting the reference only adds a constant to ``ct(t)``, which
cancels in the peak-to-peak. (The mean-of-magnitudes estimator equals the true
peak-to-peak only when the reference frame happens to sit at a pulsation
extreme, and underestimates by up to 2x otherwise.)

Units
-----
The projection is done in physical space: the axial and transversal pixel sizes
differ by ~3x (:data:`AXIAL_PIXEL_SIZE_MM` vs :data:`TRANVERSAL_PIXEL_SIZE_MM`),
so displacements and the CSI normal are converted to mm before projecting. ΔCT
is returned in both mm and µm; the µm value is what
:func:`ocularrigidity.friedenwald.friedenwald_K_from_deltaCT` consumes.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d

from ocularrigidity.motion.displacement import extract_displacement_at_boundaries
from ocularrigidity.pipeline_config import (
    DELTA_A,
    AXIAL_PIXEL_SIZE_MM,
    TRANVERSAL_PIXEL_SIZE_MM,
)
from ocularrigidity.segmentation.postprocess.interfaces import (
    clean_boundaries,
    extract_boundaries_fast,
    smooth_boundary_2d,
)


@dataclass
class DeltaCTResult:
    """Result of a ΔCT measurement. Lengths in mm unless the name says µm.

    ``ct_series_mm`` is the *signed* per-frame change relative to the reference
    frame (zero at ``reference_frame_idx``); ``ct_abs_series_mm`` adds
    ``baseline_ct_mm`` to give absolute thickness, and is smooth because it is
    driven by the optical-flow change rather than the raw per-frame mask.

    ``baseline_ct_mm`` is the mean RPE→CSI distance over the CSI anchor columns
    measured perpendicular to the CSI, so it is tilt-corrected the same way the
    change is projected. A large ``rpe_residual_um`` flags imperfect RPE
    alignment: it is the peak residual RPE motion removed as common-mode.
    """

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
    """How fast the choroid thickens and thins within one cardiac cycle.

    Attributes
    ----------
    thickening_um_s, thinning_um_s :
        Peak rate of increase / decrease of the thickness (µm/s). Both are
        reported as positive magnitudes.
    asymmetry :
        ``thickening / thinning``. 1.0 means the two limbs are equally fast; a
        departure is the viscoelastic signature (the choroid filling faster
        than it drains, or the reverse).
    thickening_fraction :
        Fraction of the cycle spent thickening. 0.5 for a symmetric waveform.
    """

    thickening_um_s: float
    thinning_um_s: float
    asymmetry: float
    thickening_fraction: float


def cycle_rates(
    ct_series_mm: np.ndarray, period_s: float, n_harm: int = 4
) -> CycleRates:
    """Peak thickening / thinning rates over one *folded* cardiac cycle.

    The folded cycle is periodic, so the derivative comes from a truncated
    Fourier series -- exact for the retained harmonics. A central difference
    over ~30 bins is both biased (0.43 error on a 3-harmonic test signal) and
    amplifies noise ~2.7x more, which matters because ``ct_series_mm`` is a
    sub-pixel signal.

    ``period_s`` is the cardiac period, ``60 / HR``. ``n_harm`` caps the
    harmonics kept; the cardiac waveform lives in the first few, and going
    higher just differentiates noise. Returns all-NaN when the series is too
    gappy or the period is unknown, rather than raising -- cohort loops call
    this per cycle.
    """
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
    """Per-column unit CSI normal in *physical* (mm) space, as (nx, ny).

    In pixel space the CSI is ``y = f(x)`` with slope ``f'``. Mapping to mm
    (``X = x·tx``, ``Y = y·ax``) the physical tangent is ``(tx, ax·f')`` so the
    normal is proportional to ``±(-ax·f', tx)``. We return the branch pointing
    *into* the choroid, i.e. up-image toward the RPE (``ny < 0``, image y grows
    downward): ``(ax·f', -tx)``. Callers that want a thickness *increase* to be
    positive must therefore negate the along-normal projection -- see
    :func:`measure_delta_ct_from_disp`.

    ``slope_window`` (columns) is the extent of the local least-squares line fit
    used for ``f'``; <3 keeps the legacy 2-point ``np.gradient``.
    ``smooth_sigma`` (px) additionally column-wise de-noises the boundary before
    differentiating; 0 disables it. Both exist because a wrong local normal
    leaks tangential motion into the across-interface projection. NaN-aware
    (edge/gap columns).
    """
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
    """Measure the peak-to-peak choroidal thickness change (ΔCT).

    Parameters
    ----------
    frames : (T, H, W) uint8
        Grayscale frames, RPE-registered (RPE stationary).
    masks : (T, H, W)
        Binary choroid masks (bool or 0/1).
    reference_frame_idx :
        Frame the displacements are measured against.
    method, smooth_window, lk_window :
        Passed to :func:`extract_displacement_at_boundaries` (optical flow +
        temporal savgol smoothing). Defaults follow ``DELTA_A``.
    subtract_rpe_motion :
        If True, subtract the mean physical displacement of the RPE anchors from
        every anchor (common-mode removal) before projecting, making ΔCT robust
        to residual RPE misalignment.
    axial_mm_per_px, transversal_mm_per_px :
        OCT pixel scales used to convert to physical units.
    normal_smooth_sigma :
        Column-wise Gaussian sigma (px) used to de-noise the CSI boundary
        before differentiating it for the interface normal. 0 disables it.
    normal_slope_window :
        Number of columns the local least-squares line fit for the CSI
        orientation spans. <3 falls back to a 2-point ``np.gradient``.
    """
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
    # Reference RPE (top) and CSI (bottom) boundaries, used to label anchors and
    # give the CSI normal direction.
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
    # The normal points into the choroid, so a positive projection is a CSI that
    # moved toward the RPE, i.e. a *thinner* choroid: negate to get the signed
    # thickness change.
    proj = -(disp_mm[..., 0] * nx[None, :] + disp_mm[..., 1] * ny[None, :])  # (T, N)
    proj_csi = proj[:, is_csi]  # (T, Nc)

    # Signed spatial mean per frame (NaN-aware), then peak-to-peak over time.
    valid = np.isfinite(proj_csi)
    sums = np.where(valid, proj_csi, 0.0).sum(axis=1)
    counts = valid.sum(axis=1)
    ct_series = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)  # (T,) mm

    delta_ct_mm = float(np.nanmax(ct_series) - np.nanmin(ct_series))

    # Absolute thickness: add the reference-frame baseline to the smooth change.
    # The baseline is the RPE->CSI gap measured *perpendicular* to the CSI, not
    # vertically: project the vertical gap onto the CSI normal, i.e. multiply by
    # cos(tilt). That factor is exactly ``ny`` (the normalized, physical-space
    # y-component of the CSI normal), so the baseline and the change are measured
    # along the same direction -- important when the interfaces are steeply
    # sloped, where the vertical gap overestimates the true thickness by 1/cos.
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
