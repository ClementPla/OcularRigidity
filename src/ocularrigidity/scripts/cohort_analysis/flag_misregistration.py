import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from ocularrigidity.consts import ROOT_CARDIAC_PIPELINE
from ocularrigidity.pipeline_config import MISREGISTRATION
from ocularrigidity.data.io import load_mask
from ocularrigidity.segmentation.postprocess.interfaces import extract_boundaries_gpu

# --- Flagging thresholds (pixels / fractions) --------------------------------
# Sourced from pipeline_config.MISREGISTRATION so every stage shares one config.
# A well-registered cycle has a smooth Bruch's membrane (BM): cardiac motion is
# only a few pixels over ~30 frames, so robust frame-to-frame jumps stay
# sub-pixel. BM is the stable, trustworthy boundary; the deeper CSI is too noisy
# to threshold reliably, so we ignore it here and key only on BM + coverage.
BM_JITTER_P95_PX = MISREGISTRATION.bm_jitter_p95_px
# A "jump" is a frame transition exceeding this many pixels.
JUMP_PX = MISREGISTRATION.jump_px
MAX_JUMP_FRAC = MISREGISTRATION.max_jump_frac
# Coverage: fraction of columns that carry mask content.
MIN_FRAME_COVERAGE = MISREGISTRATION.min_frame_coverage
MAX_FRAC_EMPTY_COLUMNS = MISREGISTRATION.max_frac_empty_columns
MAX_EMPTY_FRAMES = MISREGISTRATION.max_empty_frames


def _temporal_jitter(curve: torch.Tensor) -> tuple[float, float, float, float]:
    """Robust frame-to-frame jitter of a (T, W) boundary curve (NaN-aware).

    Returns (median |Δ|, p95 |Δ|, max |Δ|, fraction of jumps > JUMP_PX).
    """
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
    # Only BM (top boundary) is used; CSI is discarded.
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


def find_segmentations(root: Path) -> list[Path]:
    """Locate every segmented_cycles.npz under the cohort root."""
    return sorted(root.rglob("segmented_cycles.npz"))


def process_cohort(root: Path, output_csv: Path, device: str = "cuda") -> pd.DataFrame:
    paths = find_segmentations(root)
    if not paths:
        raise FileNotFoundError(f"No segmented_cycles.npz found under {root}")

    rows = []
    # Overlap disk IO + zstd decompression with GPU compute: a few worker
    # threads prefetch masks while the main thread runs the GPU metrics.
    def _load(p):
        # Faults are carried through as a value: raising inside pool.map would
        # surface on iteration and take the whole cohort down.
        try:
            return p, load_mask(p), None
        except Exception as e:
            return p, None, e

    with ThreadPoolExecutor(max_workers=4) as pool:
        loaded = pool.map(_load, paths)
        for path, mask, load_error in tqdm(loaded, total=len(paths), desc="QC"):
            rel = path.relative_to(root)
            if load_error is not None:
                rows.append(
                    {
                        "source": rel.parts[0],
                        "video": str(Path(*rel.parts[1:-1])),
                        "flag": True,
                        "reasons": f"unreadable: {type(load_error).__name__}: {load_error}",
                    }
                )
                continue
            try:
                metrics = compute_qc_metrics(mask, device=device)
            except Exception as e:  # keep going across the cohort
                rows.append(
                    {
                        "source": str(rel.parents[-2]) if len(rel.parts) > 1 else "",
                        "video": str(rel.parent),
                        "flag": True,
                        "reasons": f"error: {e}",
                    }
                )
                continue
            flag, reasons = evaluate_flag(metrics)
            rows.append(
                {
                    # top-level dir == measures_<method>_<phase>
                    "source": rel.parts[0],
                    "video": str(Path(*rel.parts[1:-1])),
                    **metrics,
                    "flag": flag,
                    "reasons": reasons,
                }
            )

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    n_flagged = int(df["flag"].sum())
    print(f"Processed {len(df)} videos, flagged {n_flagged} -> {output_csv}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT_CARDIAC_PIPELINE,
        help="Root holding the measures_* dirs with segmented_cycles.npz files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <root>/misregistration_flags.csv).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_csv = args.output or (args.root / "misregistration_flags.csv")
    process_cohort(args.root, output_csv, device=args.device)
