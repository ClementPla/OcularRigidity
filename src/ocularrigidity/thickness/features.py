from scipy.ndimage import distance_transform_edt
import torch
from tqdm.auto import tqdm
import numpy as np

import cupy as cp

from cupyx.scipy.ndimage import distance_transform_edt as gdt

from ocularrigidity.motion.one_cycle import estimate_cardiac_amplitude


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
    """Extract the deltaY (thickness) feature from a (T, H, W) mask."""

    return np.sum(mask, axis=1).astype(np.float32)


def compute_deltaY_boundaries(bm: np.ndarray, csi: np.ndarray) -> np.ndarray:
    """Extract the deltaY (thickness) feature from boundary masks."""
    if isinstance(bm, torch.Tensor):
        bm = bm.cpu().numpy()
    if isinstance(csi, torch.Tensor):
        csi = csi.cpu().numpy()
    return (csi - bm).astype(np.float32)


def extract_thickness_distance(mask: np.ndarray, verbose: bool = False) -> np.ndarray:
    """Per-frame mean thickness via distance transform."""
    T, H, W = mask.shape
    out = np.zeros((T, W), dtype=np.float32)
    for t in tqdm(range(T), disable=not verbose, leave=False):
        dt = distance_transform_edt(mask[t])
        # Local thickness at each pixel = 2 × distance to nearest boundary
        out[t] = 2 * dt.max(axis=0)
    return out


def measure_deltaY(masks: np.ndarray, video: str, n_cycles: int, config) -> list[dict]:
    """Per-cycle cardiac amplitude of one segmented clip."""
    thickness = compute_deltaY_masks(masks)
    T = thickness.shape[0]

    rows = []
    for cycle in range(n_cycles):
        current_cycle = thickness[cycle * T // n_cycles : (cycle + 1) * T // n_cycles]
        fits, _ = estimate_cardiac_amplitude(
            current_cycle,
            n_harmonics=config.n_harmonics,
            residual_threshold_percentile=config.residual_threshold_percentile,
            amplitude_threshold_percentile=config.amplitude_threshold_percentile,
        )
        amplitude = fits.max(axis=0) - fits.min(axis=0)
        rows.append(
            {
                "video": video,
                "cycle": cycle,
                "deltaY": float(np.mean(amplitude)),
                "Amplitudes": amplitude,
                "Fits": fits,
            }
        )
    return rows
