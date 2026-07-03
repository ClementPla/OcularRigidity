import cc3d
import numpy as np
from tqdm.auto import tqdm


def keep_largest_connected_component(masks):
    """Per-frame largest CC. masks: (n, H, W) bool."""
    out = np.empty_like(masks)
    for i in tqdm(range(masks.shape[0]), desc="CC Analysis", position=1, leave=False):
        out[i] = cc3d.largest_k(masks[i].astype(np.uint8), k=1, connectivity=4) > 0
    return out
