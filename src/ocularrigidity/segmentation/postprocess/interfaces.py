import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.interpolate import interp1d
from numba import njit, prange
import torch
import cv2


@njit(parallel=True)
def extract_boundaries_fast(masks):
    T, H, W = masks.shape
    bm = np.full((T, W), np.nan, dtype=np.float32)
    csi = np.full((T, W), np.nan, dtype=np.float32)

    # Parallelize over the T dimension
    for t in prange(T):
        for w in range(W):
            # 1. Find Top Boundary (BM)
            first = -1
            for h in range(H):
                if masks[t, h, w]:
                    first = h
                    break

            if first != -1:
                bm[t, w] = float(first)

                # Only scan if we found a top; scan backwards
                for h in range(H - 1, first - 1, -1):
                    if masks[t, h, w]:
                        csi[t, w] = float(h)
                        break
    return bm, csi


def extract_boundaries_gpu(
    masks: np.ndarray,
    batch_size: int = 128,
    to_numpy: bool = True,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-accelerated extraction of top (BM) and bottom (CSI) boundaries per column.

    Args:
        masks: (T, H, W) boolean numpy array.
        batch_size: number of frames per GPU batch.

    Returns:
        (bm, csi), each (T, W) float32 with NaN for columns containing no True.
    """
    T, H, W = masks.shape
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    if isinstance(masks, torch.Tensor):
        masks_torch = masks.to(device)
    else:
        masks_torch = torch.from_numpy(masks).to(device)
    bm = torch.full((T, W), float("nan"), dtype=torch.float32, device=device)
    csi = torch.full((T, W), float("nan"), dtype=torch.float32, device=device)

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        batch = masks_torch[start:end]  # (B, H, W) bool
        batch_int = batch.to(torch.uint8)  # argmax needs non-bool

        # Top boundary: first True from the top.
        top_idx = torch.argmax(batch_int, dim=1)  # (B, W)
        has_top = batch.any(dim=1)  # (B, W)
        bm[start:end] = torch.where(has_top, top_idx.float(), bm[start:end])

        # Bottom boundary: first True from the bottom == last True overall.
        # argmax on the flipped tensor gives the index from the bottom; convert.
        bot_idx = H - 1 - torch.argmax(batch_int.flip(dims=[1]), dim=1)  # (B, W)
        has_bot = has_top  # same condition: column has any True
        csi[start:end] = torch.where(has_bot, bot_idx.float(), csi[start:end])

    if to_numpy:
        bm = bm.cpu().numpy()
        csi = csi.cpu().numpy()
    return bm, csi


def reject_column_outliers(
    boundary: np.ndarray,
    median_window: int = 11,
    max_residual: float = 15.0,
) -> np.ndarray:
    """
    Set per-column outliers in a (T, W) boundary to NaN.

    A column is flagged when it deviates from a robust local median baseline
    (computed along W) by more than `max_residual` pixels. This removes the
    single-column spikes produced when the argmax-based extraction latches onto
    a stray segmented pixel near an image edge.

    Args:
        boundary: (T, W) float array, NaN for missing columns.
        median_window: width (in columns) of the median baseline filter.
        max_residual: max allowed deviation (px) from the baseline.

    Returns:
        (T, W) float32 copy with outlier columns replaced by NaN.
    """
    out = boundary.astype(np.float32, copy=True)
    valid = ~np.isnan(out)
    idx = np.arange(out.shape[1])

    # Fill holes per row so the median baseline is gap-free, then flag deviations.
    filled = out.copy()
    for t in range(out.shape[0]):
        v = valid[t]
        if v.sum() < 2:
            continue
        filled[t] = np.interp(idx, idx[v], out[t, v])

    baseline = median_filter(filled, size=(1, median_window), mode="mirror")
    outliers = np.abs(filled - baseline) > max_residual
    out[outliers] = np.nan
    return out


def impute_small_holes(boundary: np.ndarray, max_hole: int = 5) -> np.ndarray:
    """
    Linearly interpolate interior NaN gaps of width <= `max_hole` columns in a
    (T, W) boundary. Larger gaps and leading/trailing gaps are left as NaN so
    downstream extrapolation/smoothing can handle them.

    Args:
        boundary: (T, W) float array, NaN for missing columns.
        max_hole: maximum gap width (in columns) eligible for imputation.

    Returns:
        (T, W) float32 copy with small interior holes filled.
    """
    out = boundary.astype(np.float32, copy=True)
    T, W = out.shape
    idx = np.arange(W)
    for t in range(T):
        row = out[t]
        valid = ~np.isnan(row)
        if valid.sum() < 2:
            continue
        interp = np.interp(idx, idx[valid], row[valid])
        nan = ~valid
        padded = np.concatenate([[False], nan, [False]])
        d = np.diff(padded.astype(np.int8))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        for s, e in zip(starts, ends):
            # Only fill bounded interior gaps; np.interp would otherwise
            # extrapolate leading/trailing gaps as a flat line.
            if (e - s) <= max_hole and s > 0 and e < W:
                row[s:e] = interp[s:e]
    return out


def clean_boundaries(
    bm: np.ndarray,
    csi: np.ndarray,
    max_hole: int = 5,
    median_window: int = 11,
    max_residual: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Robustify raw (T, W) boundaries: reject column outliers (spikes), then
    impute the small holes that rejection leaves behind.

    Returns:
        (bm, csi) cleaned float32 arrays.
    """
    bm = reject_column_outliers(bm, median_window, max_residual)
    csi = reject_column_outliers(csi, median_window, max_residual)
    bm = impute_small_holes(bm, max_hole)
    csi = impute_small_holes(csi, max_hole)
    return bm, csi


def smooth_boundary_2d(
    boundary: np.ndarray, sigma_time: float = 3.0, sigma_col: float = 1.0
) -> np.ndarray:
    """
    Smooth a (T, W) boundary curve anisotropically.
    Handles NaN via a weighted gaussian trick.
    """
    valid = ~np.isnan(boundary)
    filled = np.where(valid, boundary, 0.0)

    # Smooth numerator and denominator separately, then divide.
    # This is the standard NaN-aware gaussian: only valid pixels contribute.
    num = gaussian_filter(filled, sigma=(sigma_time, sigma_col), mode="nearest")
    denom = gaussian_filter(
        valid.astype(np.float32), sigma=(sigma_time, sigma_col), mode="nearest"
    )

    smoothed = np.where(denom > 1e-6, num / denom, np.nan)
    return smoothed


def smooth_boundary_2d_non_uniform(
    boundary: np.ndarray,
    timestamps: np.ndarray,
    sigma_time: float = 3.0,
    sigma_col: float = 1.0,
    resample_factor: int = 1,
) -> np.ndarray:
    """
    Smooth a (T, W) boundary curve with non-uniform timestamps.

    Args:
        boundary: (T, W) array.
        timestamps: (T,) array of actual time values.
        sigma_time: Standard deviation for Gaussian kernel in time units.
        sigma_col: Standard deviation for Gaussian kernel in column indices.
    """
    T, W = boundary.shape

    # 1. Create a uniform time grid
    # We use the average spacing to define the new 'pixel' size
    t_min, t_max = timestamps[0], timestamps[-1]
    uniform_t = np.linspace(t_min, t_max, T * resample_factor)
    dt = (t_max - t_min) / (len(uniform_t) - 1)

    # Convert sigma_time (in time units) to sigma_pixels (for the uniform grid)
    sigma_time_pixels = sigma_time / dt

    # 2. Interpolate data and valid-mask onto the uniform grid
    # We handle each column to create a (T_uniform, W) grid
    valid = (~np.isnan(boundary)).astype(np.float32)
    filled = np.nan_to_num(boundary, nan=0.0)

    # Vectorized interpolation across columns
    f_data = interp1d(
        timestamps, filled, axis=0, kind="linear", fill_value="extrapolate"
    )
    f_valid = interp1d(
        timestamps, valid, axis=0, kind="linear", fill_value="extrapolate"
    )

    grid_data = f_data(uniform_t)
    grid_valid = f_valid(uniform_t)

    # 3. Apply the standard NaN-aware Gaussian trick on the uniform grid
    num = gaussian_filter(
        grid_data, sigma=(sigma_time_pixels, sigma_col), mode="nearest"
    )
    denom = gaussian_filter(
        grid_valid, sigma=(sigma_time_pixels, sigma_col), mode="nearest"
    )

    # Avoid division by zero
    grid_smoothed = np.where(denom > 1e-6, num / denom, np.nan)

    # 4. Interpolate back to the original non-uniform timestamps
    f_final = interp1d(
        uniform_t, grid_smoothed, axis=0, kind="linear", fill_value="extrapolate"
    )
    return f_final(timestamps)


def bandpass_boundary_2d_non_uniform(
    boundary: np.ndarray,
    timestamps: np.ndarray,
    sigma_time_low: float,  # LARGE sigma  -> sets the low-freq cutoff
    sigma_time_high: float,  # SMALL sigma  -> sets the high-freq cutoff
    sigma_col: float = 1.0,
    resample_factor: int = 1,
) -> np.ndarray:
    """
    Band-pass a (T, W) boundary along the time axis using a
    Difference-of-Gaussians, NaN-aware and non-uniform-timestamp-aware.

    Passband (approx, -3 dB):
        f_low  ~ sqrt(ln 2) / (2 pi * sigma_time_low)   ~  0.133 / sigma_time_low
        f_high ~ sqrt(ln 2) / (2 pi * sigma_time_high)  ~  0.133 / sigma_time_high
    in 1 / (units of `timestamps`).
    """
    assert sigma_time_low > sigma_time_high, (
        "sigma_time_low (slow cutoff) must be larger than sigma_time_high (fast cutoff)"
    )

    T, W = boundary.shape

    # 1. Uniform time grid
    t_min, t_max = timestamps[0], timestamps[-1]
    uniform_t = np.linspace(t_min, t_max, T * resample_factor)
    dt = (t_max - t_min) / (len(uniform_t) - 1)

    sigma_low_pix = sigma_time_low / dt
    sigma_high_pix = sigma_time_high / dt

    valid = (~np.isnan(boundary)).astype(np.float32)
    filled = np.nan_to_num(boundary, nan=0.0)

    grid_data = interp1d(
        timestamps, filled, axis=0, kind="linear", fill_value="extrapolate"
    )(uniform_t)
    grid_valid = interp1d(
        timestamps, valid, axis=0, kind="linear", fill_value="extrapolate"
    )(uniform_t)

    # 3. NaN-aware Gaussian at both scales
    def _nan_aware_gauss(sigma_t_pix):
        num = gaussian_filter(grid_data, sigma=(sigma_t_pix, sigma_col), mode="nearest")
        den = gaussian_filter(
            grid_valid, sigma=(sigma_t_pix, sigma_col), mode="nearest"
        )
        return np.where(den > 1e-6, num / den, np.nan)

    slow = _nan_aware_gauss(sigma_low_pix)  # only the slow drift survives
    fast = _nan_aware_gauss(sigma_high_pix)  # slow drift + the mid band

    grid_bp = fast - slow  # DoG: isolate the mid band

    # 4. Interpolate back to original timestamps
    return interp1d(
        uniform_t, grid_bp, axis=0, kind="linear", fill_value="extrapolate"
    )(timestamps)


def get_masks_contours(masks):
    """
    Get the contours of the masks using OpenCV findContours.

    masks: (T, H, W) bool or (H, W) bool
    returns: the largest-area contour for a single frame, or a list of them
             (one per frame) for a (T, H, W) stack.
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


def rebuild_mask(upper: np.ndarray, lower: np.ndarray, H: int) -> np.ndarray:
    """
    Rebuild (T, H, W) bool mask from (T, W) boundary curves.
    """
    T, W = upper.shape
    y_grid = np.arange(H)[None, :, None]  # (1, H, 1)
    bm_grid = upper[:, None, :]  # (T, 1, W)
    csi_grid = lower[:, None, :]

    mask = (y_grid >= bm_grid) & (y_grid <= csi_grid)

    # Handle NaN columns (leave as False everywhere)
    valid = ~(np.isnan(upper) | np.isnan(lower))
    mask = mask & valid[:, None, :]
    return mask


def rebuild_mask_torch(bm: torch.Tensor, csi: torch.Tensor, H: int) -> torch.Tensor:
    """
    Rebuild (T, H, W) bool mask from (T, W) boundary curves.
    """
    T, W = bm.shape
    device = bm.device
    y_grid = torch.arange(H, device=device)[None, :, None]  # (1, H, 1)
    bm_grid = bm[:, None, :]  # (T, 1, W)
    csi_grid = csi[:, None, :]

    mask = (y_grid >= bm_grid) & (y_grid <= csi_grid)

    # Handle NaN columns (leave as False everywhere)
    valid = ~(torch.isnan(bm) | torch.isnan(csi))
    mask = mask & valid[:, None, :]
    return mask
