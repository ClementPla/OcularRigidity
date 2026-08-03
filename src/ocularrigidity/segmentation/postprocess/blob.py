from concurrent.futures import ThreadPoolExecutor
import os
import cc3d
import numpy as np


def keep_largest_connected_component(
    masks: np.ndarray, max_workers: int | None = None
) -> np.ndarray:
    """Per-frame largest CC using multi-threaded execution.

    Args:
        masks: (N, H, W) boolean numpy array.
        max_workers: Number of CPU threads to use (defaults to CPU core count).

    Returns:
        (N, H, W) boolean numpy array containing only the largest component per frame.
    """
    n, h, w = masks.shape
    out = np.empty((n, h, w), dtype=bool)

    # 1. Single conversion up front (prevents N inner-loop allocations)
    masks_u8 = masks.astype(np.uint8, copy=False)

    if max_workers is None:
        max_workers = min(32, os.cpu_count() or 4)

    # 2. Worker function: process individual slice in-place
    def _process_slice(i: int) -> None:
        out[i] = cc3d.largest_k(masks_u8[i], k=1, connectivity=4) > 0

    # 3. Parallelize across CPU threads (cc3d releases GIL)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(_process_slice, range(n)))

    return out
