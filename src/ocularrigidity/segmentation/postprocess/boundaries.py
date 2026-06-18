"""

Boundary-aware post-processing with robust handling of fragmented/missing segmentations.
"""

import numpy as np
from scipy.ndimage import label as nd_label
from scipy.signal import convolve
from tqdm.auto import tqdm
import cv2


def postprocess_boundaries(
    masks: np.ndarray,
    mode: str = "parabolic",
    fit_window: int = 40,
    fit_sigma: float = 15.0,
    max_extrap: int = 300,
    min_valid: int = 15,
    min_main_region_width: int = 50,
    max_extrap_slope: float = 3.0,
    max_extrap_out_of_bounds: int = 10,
    temporal_window: int = 5,
    temporal_max_gap: int = 10,
) -> dict:
    """
    Post-process a (T, H, W) boolean mask video with robust boundary extrapolation.

    Parameters
    ----------
    masks : (T, H, W) bool
    mode : "parabolic" or "elliptical"
    fit_window : columns used for local parabolic fit
    fit_sigma : gaussian weight sigma for weighted fit
    max_extrap : hard cap on columns to extrapolate per side
    min_valid : min valid points needed to fit a parabola
    min_main_region_width : reject frames where main blob spans fewer columns
    max_extrap_slope : reject extrapolation if |dy/dx| exceeds this
    max_extrap_out_of_bounds : reject extrapolation if it goes this many px beyond image
    temporal_window : frames to consider for temporal interpolation
    temporal_max_gap : max spatial gap width that allows temporal fill

    Returns
    -------
    dict with:
        'masks'       : (T, H, W) bool — post-processed masks
        'bm', 'csi'   : (T, W) float
        'bm_valid', 'csi_valid' : (T, W) bool — originally observed
        'filled_by'   : (T, W) int — 0=observed, 1=temporal, 2=spatial, 3=failed
    """
    T, H, W = masks.shape

    bm, csi, valid = _extract_boundaries_batch(masks)
    bm_raw_valid = valid.copy()
    csi_raw_valid = valid.copy()

    # Additionally filter out frames where the main valid region is too small
    valid = _filter_tiny_fragments(valid, min_main_region_width)

    filled_by = np.zeros((T, W), dtype=np.int8)

    # Temporal fill
    bm, csi, valid_tmp = _fill_temporal(
        bm, csi, valid, temporal_window, temporal_max_gap
    )
    filled_by[valid_tmp & ~bm_raw_valid] = 1

    # Spatial extrapolation
    fit_fn = _fit_ellipse_boundaries if mode == "elliptical" else _fit_parabola_weighted

    for t in tqdm(range(T), desc="Spatial extrapolation", leave=False):
        bm[t], valid_bm_t = _extrapolate_boundary(
            bm[t],
            valid_tmp[t],
            W,
            H,
            fit_window,
            fit_sigma,
            max_extrap,
            min_valid,
            max_extrap_slope,
            max_extrap_out_of_bounds,
            fit_fn,
        )
        csi[t], valid_csi_t = _extrapolate_boundary(
            csi[t],
            valid_tmp[t],
            W,
            H,
            fit_window,
            fit_sigma,
            max_extrap,
            min_valid,
            max_extrap_slope,
            max_extrap_out_of_bounds,
            fit_fn,
        )
        both_valid = valid_bm_t & valid_csi_t
        newly_spatial = both_valid & ~valid_tmp[t]
        filled_by[t][newly_spatial] = 2
        # Columns where extrapolation failed: neither observed nor filled
        failed = ~both_valid
        filled_by[t][failed & (filled_by[t] == 0) & ~valid_tmp[t]] = 3

    # Enforce monotonicity
    both_defined = ~(np.isnan(bm) | np.isnan(csi))
    violation = both_defined & (csi < bm + 2)
    csi = np.where(violation, bm + 2, csi)

    # Rebuild masks
    out_masks = _rebuild_masks(bm, csi, H)

    return {
        "masks": out_masks,
        "bm": bm,
        "csi": csi,
        "bm_valid": bm_raw_valid,
        "csi_valid": csi_raw_valid,
        "filled_by": filled_by,
    }


# --- Boundary extraction ---------------------------------------------------


def _extract_boundaries_batch(masks):
    T, H, W = masks.shape
    valid = masks.any(axis=1)
    bm = np.argmax(masks, axis=1).astype(np.float32)
    csi = H - 1 - np.argmax(masks[:, ::-1, :], axis=1).astype(np.float32)
    bm[~valid] = np.nan
    csi[~valid] = np.nan
    return bm, csi, valid


def _rebuild_masks(bm, csi, H):
    T, W = bm.shape
    y_grid = np.arange(H)[None, :, None]
    bm_b = bm[:, None, :]
    csi_b = csi[:, None, :]
    mask = (y_grid >= bm_b) & (y_grid <= csi_b)
    invalid = np.isnan(bm) | np.isnan(csi)
    mask = mask & ~invalid[:, None, :]
    return mask


# --- Fragment filtering (part of Fix 3) ------------------------------------


def _filter_tiny_fragments(valid, min_width):
    """
    For each frame, keep only the longest contiguous valid region IF it's
    wider than min_width. Drop all smaller fragments as noise.
    """
    T, W = valid.shape
    out = np.zeros_like(valid)
    for t in range(T):
        main = _find_main_valid_region(valid[t], min_length=min_width)
        if main is not None:
            s, e = main
            out[t, s:e] = valid[t, s:e]
    return out


def _find_main_valid_region(valid, min_length=30):
    """Find the longest contiguous True run. Return (start, end) exclusive or None."""
    runs = _find_runs(valid)
    if not runs:
        return None
    longest = max(runs, key=lambda r: r[1] - r[0])
    if (longest[1] - longest[0]) < min_length:
        return None
    return longest


def _find_runs(b):
    if not b.any():
        return []
    padded = np.concatenate([[False], b, [False]])
    edges = np.diff(padded.astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return list(zip(starts, ends))


# --- Temporal interpolation (unchanged) ------------------------------------


def _fill_temporal(bm, csi, valid, window, max_spatial_gap):
    T, W = bm.shape
    bm_out = bm.copy()
    csi_out = csi.copy()
    valid_out = valid.copy()

    weights = valid.astype(np.float32)
    kernel_len = 2 * window + 1
    kernel = np.ones(kernel_len, dtype=np.float32) / kernel_len

    bm_filled = np.nan_to_num(bm, nan=0.0)
    csi_filled = np.nan_to_num(csi, nan=0.0)

    bm_num = convolve(bm_filled, kernel[:, None], mode="same")
    csi_num = convolve(csi_filled, kernel[:, None], mode="same")
    w_sum = convolve(weights, kernel[:, None], mode="same")

    min_valid_neighbors = max(2, kernel_len // 4)
    can_fill = (~valid) & (w_sum >= min_valid_neighbors / kernel_len)

    bm_out[can_fill] = bm_num[can_fill] / np.maximum(w_sum[can_fill], 1e-6)
    csi_out[can_fill] = csi_num[can_fill] / np.maximum(w_sum[can_fill], 1e-6)
    valid_out[can_fill] = True

    # Don't fill deep interiors of large gaps
    for t in range(T):
        runs = _find_runs(~valid[t])
        for start, end in runs:
            if (end - start) > max_spatial_gap:
                edge_margin = max_spatial_gap // 2
                interior_start = start + edge_margin
                interior_end = end - edge_margin
                if interior_end > interior_start:
                    reset_slice = slice(interior_start, interior_end)
                    valid_out[t, reset_slice] = valid[t, reset_slice]
                    bm_out[t, reset_slice] = np.where(
                        valid[t, reset_slice],
                        bm_out[t, reset_slice],
                        np.nan,
                    )
                    csi_out[t, reset_slice] = np.where(
                        valid[t, reset_slice],
                        csi_out[t, reset_slice],
                        np.nan,
                    )

    return bm_out, csi_out, valid_out


# --- Spatial extrapolation (Fixes 1 + 3) -----------------------------------


def _extrapolate_boundary(
    y,
    valid,
    W,
    H,
    fit_window,
    fit_sigma,
    max_extrap,
    min_valid,
    max_slope,
    max_out_of_bounds,
    fit_fn,
):
    """
    Robust single-pass extrapolation: identify the main valid region, then
    extrapolate once leftward and once rightward from its boundaries, with
    plausibility checks. Fragments outside the main region are discarded.
    """
    y_out = y.copy()
    was_valid = np.zeros(W, dtype=bool)

    main = _find_main_valid_region(valid, min_length=max(min_valid, 30))
    if main is None:
        # No usable data in this frame — return all-NaN
        return np.full_like(y_out, np.nan), was_valid

    main_start, main_end = main

    # Mark the main region as valid and copy original values
    was_valid[main_start:main_end] = valid[main_start:main_end]
    # Inside the main region there can still be small sub-gaps; handle those
    # with a simple pass of local fits. (Usually minimal work.)
    y_out[main_start:main_end] = _fill_interior_gaps(
        y[main_start:main_end],
        valid[main_start:main_end],
        fit_window,
        fit_sigma,
        min_valid,
        fit_fn,
        max_slope,
        max_out_of_bounds,
        H,
    )
    was_valid[main_start:main_end] = ~np.isnan(y_out[main_start:main_end])

    # Outside the main region, set to NaN (ignoring any fragments)
    y_out[:main_start] = np.nan
    y_out[main_end:] = np.nan

    # Extrapolate rightward from main_end
    if main_end < W:
        model = _fit_side(
            y_out,
            was_valid,
            main_end,
            "left",
            fit_window,
            fit_sigma,
            min_valid,
            fit_fn,
        )
        if model is not None:
            stop = min(W, main_end + max_extrap)
            xs = np.arange(main_end, stop)
            y_extrap = model(xs)
            if _is_plausible(y_extrap, H, max_slope, max_out_of_bounds):
                y_out[main_end:stop] = np.clip(y_extrap, 0, H - 1)
                was_valid[main_end:stop] = True

    # Extrapolate leftward from main_start
    if main_start > 0:
        model = _fit_side(
            y_out,
            was_valid,
            main_start,
            "right",
            fit_window,
            fit_sigma,
            min_valid,
            fit_fn,
        )
        if model is not None:
            start_ext = max(0, main_start - max_extrap)
            xs = np.arange(start_ext, main_start)
            y_extrap = model(xs)
            if _is_plausible(y_extrap, H, max_slope, max_out_of_bounds):
                y_out[start_ext:main_start] = np.clip(y_extrap, 0, H - 1)
                was_valid[start_ext:main_start] = True

    return y_out, was_valid


def _fill_interior_gaps(
    y, valid, fit_window, fit_sigma, min_valid, fit_fn, max_slope, max_oob, H
):
    """Fill small NaN gaps inside an already-isolated valid region by blending."""
    y_out = y.copy()
    invalid = ~valid
    if not invalid.any():
        return y_out

    n = len(y)
    gaps = _find_runs(invalid)
    for start, end in gaps:
        has_left = start > 0 and valid[start - 1]
        has_right = end < n and valid[end]
        if not (has_left or has_right):
            continue

        left_model = (
            _fit_side(
                y_out,
                valid,
                start,
                "left",
                fit_window,
                fit_sigma,
                min_valid,
                fit_fn,
            )
            if has_left
            else None
        )
        right_model = (
            _fit_side(
                y_out,
                valid,
                end,
                "right",
                fit_window,
                fit_sigma,
                min_valid,
                fit_fn,
            )
            if has_right
            else None
        )

        if left_model is None and right_model is None:
            continue

        xs = np.arange(start, end)
        if left_model is not None and right_model is not None:
            yl = left_model(xs)
            yr = right_model(xs)
            w = (xs - start) / max(1, end - start - 1)
            y_blend = (1 - w) * yl + w * yr
            if _is_plausible(y_blend, H, max_slope, max_oob):
                y_out[start:end] = np.clip(y_blend, 0, H - 1)
        elif left_model is not None:
            yl = left_model(xs)
            if _is_plausible(yl, H, max_slope, max_oob):
                y_out[start:end] = np.clip(yl, 0, H - 1)
        else:
            yr = right_model(xs)
            if _is_plausible(yr, H, max_slope, max_oob):
                y_out[start:end] = np.clip(yr, 0, H - 1)

    return y_out


def _fit_side(y, valid, boundary, side, window, sigma, min_valid, fit_fn):
    if side == "left":
        lo = max(0, boundary - window)
        xs = np.arange(lo, boundary)
        ys = y[lo:boundary]
        mask = valid[lo:boundary] & ~np.isnan(ys)
        anchor = boundary - 1
    else:
        hi = min(len(y), boundary + window)
        xs = np.arange(boundary, hi)
        ys = y[boundary:hi]
        mask = valid[boundary:hi] & ~np.isnan(ys)
        anchor = boundary

    if mask.sum() < min_valid:
        return None
    return fit_fn(xs[mask], ys[mask], anchor, sigma)


def _fit_parabola_weighted(xs, ys, anchor_x, sigma):
    weights = np.exp(-((xs - anchor_x) ** 2) / (2 * sigma**2))
    coeffs = np.polyfit(xs, ys, deg=2, w=weights)
    return lambda x: np.polyval(coeffs, x)


def _fit_ellipse_boundaries(xs, ys, anchor_x, sigma):
    # Same as parabolic at local scale; see _fit_ellipse_shared for true ellipse
    return _fit_parabola_weighted(xs, ys, anchor_x, sigma)


def _is_plausible(y_vals, H, max_slope, max_out_of_bounds):
    """
    Reject extrapolations that:
    - contain NaN/Inf
    - go too far out of image bounds
    - have excessive frame-to-frame slope
    """
    if np.any(~np.isfinite(y_vals)):
        return False
    if y_vals.min() < -max_out_of_bounds:
        return False
    if y_vals.max() > H - 1 + max_out_of_bounds:
        return False
    if len(y_vals) > 1:
        max_step = np.abs(np.diff(y_vals)).max()
        if max_step > max_slope:
            return False
    return True


def get_masks_edges(masks):
    """
    Get the edges of the masks using morphological gradient
    masks: (T, H, W) bool or (H, W) bool
    returns: (T, H, W) bool or (H, W) bool
    """
    kernel = np.ones((3, 3), dtype=np.uint8)
    edges = np.zeros_like(masks, dtype=bool)
    if masks.ndim == 2:
        return cv2.morphologyEx(
            masks.astype(np.uint8), cv2.MORPH_GRADIENT, kernel
        ).astype(bool)

    for t in range(masks.shape[0]):
        edges[t] = cv2.morphologyEx(
            masks[t].astype(np.uint8), cv2.MORPH_GRADIENT, kernel
        ).astype(bool)
    return edges


def get_masks_contours(masks):
    """
    Get the contours of the masks using OpenCV findContours
    masks: (T, H, W) bool or (H, W) bool
    returns: list of contours for each frame
    """
    if masks.ndim == 2:
        contours, _ = cv2.findContours(
            masks.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        main_contour = max(contours, key=cv2.contourArea)
        return main_contour
    contours_list = []
    for t in range(masks.shape[0]):
        contours, _ = cv2.findContours(
            masks[t].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        main_contour = max(contours, key=cv2.contourArea)
        contours_list.append(main_contour)
    return contours_list
