import numpy as np


def trim_choroid(mask, trimming):
    """Zero ``trimming`` columns from each side of a ``(T, H, W)`` mask."""
    trimmed_mask = np.copy(mask)
    if trimming > 0:
        trimmed_mask[:, :, :trimming] = 0
        trimmed_mask[:, :, -trimming:] = 0
    return trimmed_mask


# TODO: Instead of a vertical trim, it would probably be better to follow the structure within the choroid from the B-scan, which would be more robust to variations.
