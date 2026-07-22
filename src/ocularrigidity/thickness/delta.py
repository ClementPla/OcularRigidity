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
    """Result of a ΔCT measurement.

    Attributes
    ----------
    deltaCT_mm, deltaCT_um :
        Peak-to-peak choroidal thickness change over the tracked frames.
    ct_series_mm :
        (T,) signed per-frame thickness *change* relative to the reference frame
        (mm); ``ct_series_mm[reference_frame_idx]`` is 0.
    baseline_ct_mm :
        Absolute choroidal thickness at the reference frame (mm): mean RPE→CSI
        distance over the CSI anchor columns, measured perpendicular to the CSI
        (tilt-corrected), consistent with how the change is projected.
    ct_abs_series_mm :
        (T,) absolute choroidal thickness per frame (mm) = ``baseline_ct_mm +
        ct_series_mm``. Smooth (driven by the optical-flow change), unlike the
        raw per-frame mask thickness.
    min_ct_mm, max_ct_mm :
        Minimum / maximum absolute thickness across the sequence (mm).
    n_csi_anchors :
        Number of tracked boundary anchors classified as CSI.
    rpe_residual_um :
        Peak residual RPE motion (µm) that was removed as common-mode. A large
        value flags imperfect RPE alignment.
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


def _csi_unit_normal_mm(
    csi_ref: np.ndarray,
    axial_mm_per_px: float,
    transversal_mm_per_px: float,
    smooth_sigma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-column unit CSI normal in *physical* (mm) space, as (nx, ny).

    In pixel space the CSI is ``y = f(x)`` with slope ``f'``. Mapping to mm
    (``X = x·tx``, ``Y = y·ax``) the physical tangent is ``(tx, ax·f')`` so the
    normal is proportional to ``(-ax·f', tx)``. The sign is irrelevant to the
    peak-to-peak (flipping it negates ``ct(t)``, leaving max-min unchanged).

    ``smooth_sigma`` (px) column-wise de-noises the boundary before
    differentiating: the CSI normal is a low-frequency property of a smooth
    interface, so raw per-column mask jitter should not steer it (a wrong local
    normal leaks tangential motion into the across-interface projection). 0
    keeps the legacy bare ``np.gradient``. NaN-aware (edge/gap columns).
    """
    if smooth_sigma > 0:
        csi_ref = smooth_boundary_2d(
            csi_ref[None, :], sigma_time=0.0, sigma_col=smooth_sigma
        )[0]
    slope = np.gradient(csi_ref)  # dy/dx in px, on the de-noised boundary
    nx = -axial_mm_per_px * slope
    ny = np.full_like(slope, transversal_mm_per_px)
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
        csi_ref, axial_mm_per_px, transversal_mm_per_px, normal_smooth_sigma
    )
    nx, ny = nx_col[xi], ny_col[xi]  # (N,)

    # Remove residual rigid RPE motion (common-mode) for robustness.
    rpe_residual_um = 0.0
    if subtract_rpe_motion and is_rpe.any():
        rpe_mean = np.nanmean(disp_mm[:, is_rpe, :], axis=1)  # (T, 2) mm
        disp_mm = disp_mm - rpe_mean[:, None, :]
        rpe_residual_um = float(
            np.nanmax(np.hypot(rpe_mean[:, 0], rpe_mean[:, 1])) * 1000.0
        )

    # Signed across-interface displacement per anchor (mm), CSI anchors only.
    proj = disp_mm[..., 0] * nx[None, :] + disp_mm[..., 1] * ny[None, :]  # (T, N)
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
    cos_tilt = ny[is_csi]  # = cos(physical CSI tilt), per CSI anchor
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
