from scipy.ndimage import distance_transform_edt
import torch
from tqdm.auto import tqdm
import numpy as np

import cupy as cp

from cupyx.scipy.ndimage import distance_transform_edt as gdt


def extract_thickness_gpu(masks_cpu, verbose: bool = False) -> np.ndarray:
    T, H, W = masks_cpu.shape
    results = np.zeros((T, W), dtype=np.float32)

    for t in tqdm(range(T), disable=not verbose, leave=False):
        # 1. Move only ONE slice (1.5 MB) to VRAM
        frame_gpu = cp.array(masks_cpu[t])

        # 2. Process on GPU
        dt = gdt(frame_gpu)
        thickness_gpu = 2 * dt.max(axis=0)

        # 3. Pull only the 1D result (6 KB) back to RAM
        results[t] = cp.asnumpy(thickness_gpu)

        # 4. Clear VRAM for the next iteration
        del frame_gpu, dt, thickness_gpu
        # cp.get_default_memory_pool().free_all_blocks() # Force clear if needed

    return results


def compute_deltaY_masks(mask: np.ndarray) -> np.ndarray:
    """
    Extract the deltaY (thickness) feature from a (T, H, W) mask.
    Returns a (T, W) array of floats, with NaN where the column has no mask content.
    """

    return np.sum(mask, axis=1).astype(np.float32)


def compute_deltaY_boundaries(bm: np.ndarray, csi: np.ndarray) -> np.ndarray:
    """
    Extract the deltaY (thickness) feature from boundary masks.
    Returns a (T, W) array of floats, with NaN where the column has no mask content.
    """
    if isinstance(bm, torch.Tensor):
        bm = bm.cpu().numpy()
    if isinstance(csi, torch.Tensor):
        csi = csi.cpu().numpy()
    return (csi - bm).astype(np.float32)


def extract_thickness_distance(mask: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    Per-frame mean thickness via distance transform.
    For each mask pixel, compute distance to the nearest boundary.
    The skeleton ridge has the maximum value; 2× that is the local thickness.
    Returns (T, W) of mean local thickness per column.
    """
    T, H, W = mask.shape
    out = np.zeros((T, W), dtype=np.float32)
    for t in tqdm(range(T), disable=not verbose, leave=False):
        dt = distance_transform_edt(mask[t])
        # Local thickness at each pixel = 2 × distance to nearest boundary
        # For column stats, take the max along y (effectively at the medial axis)
        out[t] = 2 * dt.max(axis=0)
    return out
