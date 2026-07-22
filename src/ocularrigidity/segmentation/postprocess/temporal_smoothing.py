import numpy as np
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

    mask = rebuild_mask(rpe, csi)
    return mask
