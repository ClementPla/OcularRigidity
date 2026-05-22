import numpy as np


def trim_choroid(mask, trimming):
    """
    Trim the choroid from the mask by removing a certain number of pixels from the left and right sides.

    Parameters:
    - mask: 3D numpy array representing the binary mask of the retina.
    - trimming: Number of pixels to trim from the left and right sides of the mask.

    Returns:
    - trimmed_mask: 3D numpy array representing the trimmed binary mask.
    """
    trimmed_mask = np.copy(mask)
    if trimming > 0:
        trimmed_mask[:, :, :trimming] = 0
        trimmed_mask[:, :, -trimming:] = 0
    return trimmed_mask


# TODO: Instead of a vertical trim, it would probably be better to follow the structure within the choroid from the B-scan, which would be more robust to variations.
