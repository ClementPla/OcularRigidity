"""Flags folded cycles whose registration looks unreliable."""

import numpy as np
import torch

from ocularrigidity.segmentation.postprocess.interfaces import extract_boundaries_gpu

# --- Flagging thresholds (pixels / fractions) --------------------------------
BM_JITTER_P95_PX = 3.0
# A "jump" is a frame transition exceeding this many pixels.
JUMP_PX = 5.0
MAX_JUMP_FRAC = 0.05
# Coverage: fraction of columns that carry mask content.
MIN_FRAME_COVERAGE = 0.60
MAX_FRAC_EMPTY_COLUMNS = 0.20
MAX_EMPTY_FRAMES = 0


def _temporal_jitter(curve: torch.Tensor) -> tuple[float, float, float, float]:
    """Robust frame-to-frame jitter of a (T, W) boundary curve (NaN-aware)."""
    d = (curve[1:] - curve[:-1]).abs()
    d = d[~torch.isnan(d)]
    if d.numel() == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    med = torch.median(d).item()
    p95 = torch.quantile(d, 0.95).item()
    mx = d.max().item()
    jump_frac = (d > JUMP_PX).float().mean().item()
    return med, p95, mx, jump_frac


def compute_qc_metrics(mask: np.ndarray, device: str = "cuda") -> dict:
    """Compute registration QC metrics for one (T, H, W) bool mask."""
    T, H, W = mask.shape
    # (T, W) boundary positions, NaN where a column has no mask content.
    bm, _ = extract_boundaries_gpu(mask, to_numpy=False)
    bm = bm.to(device)

    valid_col = ~torch.isnan(bm)  # (T, W)
    coverage_per_frame = valid_col.float().mean(dim=1)  # (T,)

    bm_med, bm_p95, bm_max, bm_jump = _temporal_jitter(bm)

    return {
        "T": T,
        "H": H,
        "W": W,
        "n_empty_frames": int((coverage_per_frame == 0).sum().item()),
        "min_frame_coverage": float(coverage_per_frame.min().item()),
        "mean_frame_coverage": float(coverage_per_frame.mean().item()),
        "frac_empty_columns": float(1.0 - valid_col.float().mean().item()),
        "bm_jitter_med": bm_med,
        "bm_jitter_p95": bm_p95,
        "bm_jitter_max": bm_max,
        "bm_jump_frac": bm_jump,
    }


def evaluate_flag(m: dict) -> tuple[bool, str]:
    """Turn metrics into a boolean flag + reason string."""
    reasons = []
    if m["n_empty_frames"] > MAX_EMPTY_FRAMES:
        reasons.append(f"{m['n_empty_frames']} empty frame(s)")
    if m["min_frame_coverage"] < MIN_FRAME_COVERAGE:
        reasons.append(f"low coverage ({m['min_frame_coverage']:.2f})")
    if m["frac_empty_columns"] > MAX_FRAC_EMPTY_COLUMNS:
        reasons.append(f"empty columns ({m['frac_empty_columns']:.2f})")
    if m["bm_jitter_p95"] > BM_JITTER_P95_PX:
        reasons.append(f"BM jitter ({m['bm_jitter_p95']:.1f}px)")
    if m["bm_jump_frac"] > MAX_JUMP_FRAC:
        reasons.append("frequent BM jumps")
    return (len(reasons) > 0), "; ".join(reasons)
