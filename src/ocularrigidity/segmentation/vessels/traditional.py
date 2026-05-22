import numpy as np
from skimage.filters import threshold_niblack
from tqdm.auto import tqdm


def niblack_vessels(frames, choroid_masks, k=-0.2, window_size=15, verbose=True):
    vessel_masks = []
    for frame, mask in tqdm(
        zip(frames, choroid_masks), total=len(frames), disable=not verbose
    ):
        # frame = frame  # invert for niblack to detect dark vessels
        thresh_niblack = threshold_niblack(frame, window_size=window_size, k=k)
        binary_niblack = frame < thresh_niblack
        vessel_mask = binary_niblack & mask.astype(bool)
        vessel_masks.append(vessel_mask.astype(np.uint8))
    return np.array(vessel_masks)
