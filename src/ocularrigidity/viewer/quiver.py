"""Shared displacement-quiver rendering (no I/O, no Gradio / Streamlit deps).

A quiver render is always the same three steps — get boundary anchors, get their
displacement over time, draw arrows — and only the *middle* step differs between
the two callers:

* :func:`ocularrigidity.viewer.gif.render_mask_quiver` tracks the anchors itself
  with Lucas–Kanade optical flow (:func:`track_anchors`), starting from the mask
  boundary of a reference frame;
* :func:`ocularrigidity.viewer.render.render_quiver` replays the displacements
  already stored per cardiac cycle in ``deltaA_per_cycle.pkl``.

Both then hand ``(ref_xy, disp)`` to :func:`draw_quiver`, so the drawing options
— arrow scaling and colour, the magnitude floor, ``only_y`` / across-interface
projection, the CSI summary arrow, side-by-side — live in one place
(:class:`QuiverStyle`) and behave identically wherever they are exposed.

Coordinates are ``(x, y)`` = (column, row) throughout, matching
``extract_displacement_at_boundaries``.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import cv2
import numpy as np
from matplotlib import cm
from scipy.signal import savgol_filter

from ocularrigidity.segmentation.postprocess.interfaces import (
    clean_boundaries,
    extract_boundaries_fast,
)

_POLYORDER = 3


class QuiverStyle(NamedTuple):
    """How the arrows are drawn. A tuple, so it can key a Streamlit cache.

    Attributes
    ----------
    stride :
        Draw every ``stride``-th anchor.
    arrow_scale :
        Displacements are magnified by this factor (pulsation is sub-pixel).
    min_magnitude :
        Anchors moving less than this (px, before scaling) are not drawn.
    arrow_cmap, arrow_thickness, tip_length :
        Arrow colour ramp (by magnitude) and shape.
    smooth_window, cyclic :
        Savitzky–Golay temporal smoothing of the displacement; ``cyclic`` wraps
        the filter around the loop, which is right for a folded cardiac cycle.
    only_y :
        Keep the axial component only — the pulsation is mostly axial, and the
        lateral component is dominated by residual registration jitter.
    only_orthogonal_to_border :
        Keep the component across the boundary (projected on the local mask
        normal). Requires the masks.
    border_normal_sigma :
        Blur (px) applied to the mask before differentiating it for that normal.
    show_csi_summary :
        Draw one arrow (top-right) whose length is the mean across-interface
        displacement of the CSI anchors — the thickness-change signal. Requires
        the masks.
    show_only_csi_anchors :
        Drop the anchors sitting on the RPE, keeping the choroid-sclera
        interface. Requires the masks.
    annotate_scale :
        Print the arrow magnification on each frame.
    side_by_side :
        Concatenate the untouched frame next to the overlay.
    """

    stride: int = 8
    arrow_scale: float = 20.0
    min_magnitude: float = 0.05
    arrow_cmap: str = "viridis"
    arrow_thickness: int = 1
    tip_length: float = 0.35
    smooth_window: int = 0
    cyclic: bool = True
    only_y: bool = False
    only_orthogonal_to_border: bool = False
    border_normal_sigma: float = 2.0
    show_csi_summary: bool = False
    show_only_csi_anchors: bool = False
    annotate_scale: bool = True
    side_by_side: bool = False


def smooth_disp(disp: np.ndarray, smooth_window: int, cyclic: bool) -> np.ndarray:
    """Savitzky–Golay temporal smoothing of a (T, N, 2) displacement.

    Lost tracks (NaN) are linearly interpolated along time first — anchors that
    never came back are pinned to zero displacement — so a single dropped frame
    cannot punch a hole through the filter.
    """
    disp = np.array(disp, dtype=np.float32, copy=True)
    T = disp.shape[0]
    if smooth_window <= _POLYORDER or T <= _POLYORDER + 1:
        return np.nan_to_num(disp)

    t_axis = np.arange(T)
    for n in range(disp.shape[1]):
        valid = np.isfinite(disp[:, n, 0])
        n_valid = int(valid.sum())
        if n_valid == T:
            continue
        if n_valid < 2:
            disp[:, n, :] = 0.0
            continue
        idx = np.where(valid)[0]
        for c in (0, 1):
            disp[:, n, c] = np.interp(
                t_axis, idx, disp[idx, n, c], period=T if cyclic else None
            )

    win = min(int(smooth_window) | 1, T if T % 2 else T - 1)  # savgol wants odd
    if win <= _POLYORDER:
        return disp
    return savgol_filter(
        disp,
        window_length=win,
        polyorder=_POLYORDER,
        axis=0,
        mode="wrap" if cyclic else "interp",
    )


def boundary_anchors(
    mask: np.ndarray, *, stride: int = 8, border_kernel_size: int = 2
) -> np.ndarray:
    """Anchor points (N, 2) in ``(x, y)`` on a mask's boundary, every ``stride``-th."""
    kernel = np.ones((border_kernel_size, border_kernel_size), dtype=np.uint8)
    border = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    ys, xs = np.nonzero(border)
    if ys.size == 0:
        raise ValueError("Reference frame has an empty boundary.")
    return np.stack([xs[::stride], ys[::stride]], axis=1).astype(np.float32)


def track_anchors(
    frames: np.ndarray,
    ref_xy: np.ndarray,
    *,
    reference: int = 0,
    lk_window: int = 21,
    lk_levels: int = 3,
) -> np.ndarray:
    """Track ``ref_xy`` across ``frames`` with Lucas–Kanade -> (T, N, 2) displacement.

    Every frame is tracked against the reference (not chained), so errors do not
    accumulate. Lost tracks come back as NaN.
    """
    T = frames.shape[0]
    p0 = ref_xy.astype(np.float32).reshape(-1, 1, 2)
    lk_params = dict(
        winSize=(lk_window, lk_window),
        maxLevel=lk_levels,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        minEigThreshold=1e-4,
    )
    positions = np.full((T, ref_xy.shape[0], 2), np.nan, dtype=np.float32)
    positions[reference] = ref_xy
    for t in range(T):
        if t == reference:
            continue
        p1, status, _ = cv2.calcOpticalFlowPyrLK(
            frames[reference], frames[t], p0, None, **lk_params
        )
        ok = status[:, 0].astype(bool)
        positions[t, ok] = p1[ok, 0, :]
    return positions - ref_xy[None, :, :]


def _csi_geometry(
    masks: np.ndarray, ref_xy: np.ndarray, reference: int, width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-anchor CSI membership and the unit CSI normal at each anchor's column.

    An anchor belongs to the choroid-sclera interface when it sits closer to the
    bottom boundary than to the RPE on top.
    """
    rpe, csi = clean_boundaries(*extract_boundaries_fast(masks.astype(bool)))
    rpe_ref, csi_ref = rpe[reference], csi[reference]  # (W,) each

    slope = np.gradient(csi_ref)  # dy/dx per column
    den = np.sqrt(1.0 + slope**2)
    nx_col, ny_col = -slope / den, 1.0 / den

    xi = np.clip(np.round(ref_xy[:, 0]).astype(int), 0, width - 1)
    d_csi = np.abs(ref_xy[:, 1] - csi_ref[xi])
    d_rpe = np.abs(ref_xy[:, 1] - rpe_ref[xi])
    is_csi = np.isfinite(d_csi) & (
        np.nan_to_num(d_csi, nan=np.inf) <= np.nan_to_num(d_rpe, nan=np.inf)
    )
    normal = np.nan_to_num(np.stack([nx_col[xi], ny_col[xi]], axis=1))  # (N, 2)
    return is_csi, normal


def _border_normals(mask: np.ndarray, ref_xy: np.ndarray, sigma: float) -> np.ndarray:
    """Unit normal of the mask boundary at each anchor, from the blurred mask gradient."""
    blur = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=sigma)
    gy, gx = np.gradient(blur)
    xi = np.clip(np.round(ref_xy[:, 0]).astype(int), 0, mask.shape[1] - 1)
    yi = np.clip(np.round(ref_xy[:, 1]).astype(int), 0, mask.shape[0] - 1)
    normals = np.stack([gx[yi, xi], gy[yi, xi]], axis=1)
    norm = np.hypot(normals[:, 0], normals[:, 1])
    norm[norm == 0] = 1.0
    return normals / norm[:, None]


def draw_quiver(
    frames: np.ndarray,
    ref_xy: np.ndarray,
    disp: np.ndarray,
    *,
    masks: np.ndarray | None = None,
    reference: int = 0,
    style: QuiverStyle = QuiverStyle(),
    labels: Sequence[str] | None = None,
    vrange: tuple[float, float] | None = None,
) -> np.ndarray:
    """Draw the displacement quiver over ``frames`` -> (T, H, W[·2], 3) uint8.

    Parameters
    ----------
    frames : (T, H, W) uint8
        Grayscale frames the arrows are drawn on.
    ref_xy : (N, 2)
        Anchor coordinates ``(x, y)`` in *this* frame's pixel space.
    disp : (T, N, 2)
        Displacement of each anchor per frame, relative to the reference frame.
    masks : (T, H, W), optional
        Choroid masks, in the same pixel space as ``frames``. Required by the
        options that need the interfaces (``only_orthogonal_to_border``,
        ``show_csi_summary``, ``show_only_csi_anchors``).
    labels :
        Per-frame caption; defaults to ``"<i> / <T>"``.
    vrange :
        Magnitude range the arrow colours are normalised over. Computed from
        ``disp`` when omitted — pass it to keep the colours comparable across
        several clips (e.g. the cycles of one video).
    """
    if frames.ndim != 3:
        raise ValueError(f"`frames` must have shape (T, H, W); got {frames.shape}.")
    T, H, W = frames.shape
    needs_masks = (
        style.only_orthogonal_to_border
        or style.show_csi_summary
        or style.show_only_csi_anchors
    )
    if needs_masks and masks is None:
        raise ValueError(
            "`masks` are required by only_orthogonal_to_border / show_csi_summary "
            "/ show_only_csi_anchors."
        )

    ref_xy = np.asarray(ref_xy, dtype=np.float32)
    disp = smooth_disp(disp, style.smooth_window, style.cyclic)

    is_csi = csi_normal = border_normal = None
    if masks is not None:
        if style.show_csi_summary or style.show_only_csi_anchors:
            is_csi, csi_normal = _csi_geometry(masks, ref_xy, reference, W)
        if style.only_orthogonal_to_border:
            border_normal = _border_normals(
                masks[reference], ref_xy, style.border_normal_sigma
            )

    if style.show_only_csi_anchors:
        keep = is_csi
        ref_xy, disp = ref_xy[keep], disp[:, keep, :]
        csi_normal = csi_normal[keep]
        is_csi = is_csi[keep]
        if border_normal is not None:
            border_normal = border_normal[keep]

    # Drawn magnitude — the same rule the arrows use, so the colour ramp spans
    # exactly what is on screen.
    if style.only_y:
        mags = np.abs(disp[..., 1])
    elif border_normal is not None:
        proj = (
            disp[..., 0] * border_normal[None, :, 0]
            + disp[..., 1] * border_normal[None, :, 1]
        )
        mags = np.abs(proj)
    else:
        mags = np.hypot(disp[..., 0], disp[..., 1])

    if vrange is None:
        vrange = magnitude_range(mags, style.min_magnitude)
    vmin, vmax = vrange

    # Mean across-interface motion of the CSI anchors — the thickness signal.
    csi_mean = np.zeros(T, dtype=np.float32)
    if style.show_csi_summary and is_csi is not None and is_csi.any():
        proj = (
            disp[:, is_csi, 0] * csi_normal[None, is_csi, 0]
            + disp[:, is_csi, 1] * csi_normal[None, is_csi, 1]
        )
        finite = np.isfinite(proj)
        counts = finite.sum(axis=1)
        csi_mean = np.where(
            counts > 0,
            np.where(finite, np.abs(proj), 0.0).sum(axis=1) / np.maximum(counts, 1),
            0.0,
        )

    cmap = cm.get_cmap(style.arrow_cmap)
    out_frames: list[np.ndarray] = []
    for t in range(T):
        base = cv2.cvtColor(frames[t], cv2.COLOR_GRAY2RGB)
        out = base.copy()

        valid = np.isfinite(mags[t]) & (mags[t] >= style.min_magnitude)
        idx = np.nonzero(valid)[0][:: style.stride]
        for n in idx:
            x0, y0 = ref_xy[n]
            dx, dy = disp[t, n]
            mag = mags[t, n]
            if style.only_y:
                dx = 0.0
            elif border_normal is not None:
                d_n = float(dx * border_normal[n, 0] + dy * border_normal[n, 1])
                dx, dy = d_n * border_normal[n, 0], d_n * border_normal[n, 1]

            norm = 0.0 if vmax == vmin else (mag - vmin) / (vmax - vmin)
            cv2.arrowedLine(
                out,
                (int(round(x0)), int(round(y0))),
                (
                    int(round(x0 + dx * style.arrow_scale)),
                    int(round(y0 + dy * style.arrow_scale)),
                ),
                tuple(int(v * 255) for v in cmap(float(norm))[:3]),
                style.arrow_thickness,
                line_type=cv2.LINE_AA,
                tipLength=style.tip_length,
            )

        cv2.putText(
            out,
            labels[t] if labels is not None else f"{t:04d} / {T}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        if style.annotate_scale:
            cv2.putText(
                out,
                f"x{style.arrow_scale:g}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
        if style.show_csi_summary:
            # Vertical arrow (top-right), scaled like the quiver and clamped so
            # it never leaves the panel.
            base_x, base_y = W - 20, 75
            length = int(
                round(min(float(csi_mean[t]) * style.arrow_scale, base_y - 20))
            )
            cv2.arrowedLine(
                out,
                (base_x, base_y),
                (base_x, base_y - length),
                (0, 255, 255),
                2,
                line_type=cv2.LINE_AA,
                tipLength=0.3,
            )

        out_frames.append(
            np.concatenate([base, out], axis=1) if style.side_by_side else out
        )

    return np.stack(out_frames)


def magnitude_range(mags: np.ndarray, min_magnitude: float) -> tuple[float, float]:
    """Colour-scale range over the magnitudes that clear ``min_magnitude``."""
    finite = mags[np.isfinite(mags)]
    sel = finite[finite >= min_magnitude]
    if not sel.size:
        return 0.0, 1.0
    return float(sel.min()), float(sel.max())
