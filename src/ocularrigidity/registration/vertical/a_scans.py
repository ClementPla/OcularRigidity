import numpy as np
import torch
from ocularrigidity.segmentation.postprocess.interfaces import extract_boundaries_fast


def align_ascans(frames, masks, margin=15):
    """


    Args:
        frames (_type_): TxHxW array of frames
        masks (_type_): TxHxW array of masks
        margin (int, optional): Margin around the BM to limit the search for the RPE. Defaults to 15.
    """
    if isinstance(masks, np.ndarray):
        masks = torch.from_numpy(masks)
    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(frames)

    bm, csi = extract_boundaries_fast(masks.cpu().numpy())
