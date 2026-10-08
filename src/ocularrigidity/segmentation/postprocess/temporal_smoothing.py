import numpy as np

from ocularrigidity.segmentation.postprocess.blob import (
    keep_largest_connected_component,
)
from ocularrigidity.segmentation.postprocess.interfaces import (
    clean_boundaries,
    extract_boundaries_fast,
    rebuild_mask,
    smooth_boundary_2d_non_uniform,
)


def smooth_masks_temporal(mask, timestamps, sigma_time=3.0, sigma_col=0):
    """Smooth a 3D mask along the temporal axis using a Gaussian filter."""
    T, H, W = mask.shape
    rpe, csi = extract_boundaries_fast(mask)

    rpe, csi = clean_boundaries(rpe, csi)

    csi = smooth_boundary_2d_non_uniform(
        csi, timestamps, sigma_time=sigma_time, sigma_col=sigma_col
    )

    mask = rebuild_mask(rpe, csi, H)
    return mask


def postprocess_cycles(
    raw_mask: np.ndarray, cardiac_freq: float, n_cycles: int, keep_largest_cc: bool = True
) -> np.ndarray:
    """Largest-CC (optional) + temporal smoothing, the mask as it is written to disk."""
    mask = keep_largest_connected_component(raw_mask) if keep_largest_cc else raw_mask

    T = mask.shape[0]
    timestamps = (np.linspace(0, 1.0, T) / cardiac_freq) * n_cycles
    one_cardiac_period = 1.0 / cardiac_freq
    # We filter to a window of 1/5 of a cardiac period around each timestamp
    return smooth_masks_temporal(
        mask, timestamps, sigma_time=one_cardiac_period / 5.0, sigma_col=0
    )
