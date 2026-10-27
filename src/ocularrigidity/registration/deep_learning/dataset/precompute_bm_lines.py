"""Precompute the BM line of every registration frame, for ``train_registration --bm-lines``.

The BM term of the registration loss compares where the BM (top of the choroid
mask) sits in the fixed and in the warped moving frame. Instead of segmenting
the warped frame at every step, the frozen segmentation model is run once per
frame here, at full resolution, and the line is warped analytically during
training (``losses.warp_bm_line``).

For each frame, per column: the soft BM row (``losses.soft_bm_rows`` of the
sigmoid mask, in full-frame rows) and whether the column has choroid (mask max
> 0.5, as the loss's valid columns). One file per video::

    <out>/<video>.npz   stems (n,) str, rows (n, W) float32, present (n, W) bool
    <out>/meta.json     segmentation checkpoint the lines come from

The trainer refuses lines made with another segmentation model. The frames are
the PNGs the trainer reads, uncropped and unaugmented: the trainer crops the
lines with the same window as the frames.

    # Both GPUs, one shard each
    CUDA_VISIBLE_DEVICES=0 python -m ocularrigidity.registration.deep_learning.dataset.precompute_bm_lines \\
        --segmentation-checkpoint <ckpt> --shard 0 --num-shards 2 &
    CUDA_VISIBLE_DEVICES=1 python -m ocularrigidity.registration.deep_learning.dataset.precompute_bm_lines \\
        --segmentation-checkpoint <ckpt> --shard 1 --num-shards 2
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from ocularrigidity.registration.deep_learning.models.losses import soft_bm_rows
from ocularrigidity.registration.deep_learning.train_registration import (
    _list_frames,
    build_segmenter,
)

DEFAULT_ROOT = Path("/home/clement/Documents/data/OcularRigidityAnnotated/dataset/Registration/")


def default_out(root: Path, checkpoint) -> Path:
    """``<root>/bm_lines/<segmentation run>_<checkpoint stem>`` (or ``hub``)."""
    if checkpoint is None:
        return root / "bm_lines" / "hub"
    checkpoint = Path(checkpoint)
    return root / "bm_lines" / f"{checkpoint.parent.name}_{checkpoint.stem}"


def checkpoint_id(checkpoint) -> str:
    return "hub" if checkpoint is None else str(Path(checkpoint).resolve())


@torch.no_grad()
def lines_of(seg, frames: np.ndarray, device: str, batch_size: int):
    """``(rows (n, W) float32, present (n, W) bool)`` of uint8 frames ``(n, H, W)``."""
    rows, present = [], []
    for s in range(0, len(frames), batch_size):
        x = torch.from_numpy(frames[s : s + batch_size]).to(device).unsqueeze(1)
        # The encoder's normalisation ((x/255 - 0.5) / 0.5), in fp32 as in training
        x = x.float().mul_(1.0 / 255.0).sub_(0.5).div_(0.5)
        prob = torch.sigmoid(seg(x))
        rows.append(soft_bm_rows(prob).float().cpu())
        present.append((prob.amax(dim=2) > 0.5).squeeze(1).cpu())
    return torch.cat(rows).numpy(), torch.cat(present).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--segmentation-checkpoint", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="default: <root>/bm_lines/<model>")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--read-threads", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = args.out or default_out(args.root, args.segmentation_checkpoint)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"segmentation_checkpoint": checkpoint_id(args.segmentation_checkpoint)}
    meta_path = out / "meta.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old["segmentation_checkpoint"] != meta["segmentation_checkpoint"]:
            raise SystemExit(f"{out} holds lines of {old['segmentation_checkpoint']}")
    else:
        meta_path.write_text(json.dumps(meta, indent=2))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg = build_segmenter(device, args.segmentation_checkpoint)

    frames = _list_frames(args.root)
    videos = sorted(frames)[args.shard :: args.num_shards]
    pool = ThreadPoolExecutor(args.read_threads)
    for vid in tqdm(videos, desc=f"BM lines (shard {args.shard}/{args.num_shards})"):
        path = out / f"{vid}.npz"
        if path.exists() and not args.overwrite:
            continue
        paths = [p for p, _ in frames[vid]]
        imgs = list(pool.map(lambda p: cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), paths))
        if any(im is None for im in imgs):
            raise FileNotFoundError(f"unreadable frame in {vid}")
        rows, present = lines_of(seg, np.stack(imgs), device, args.batch_size)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, stems=np.array([p.stem for p in paths]), rows=rows, present=present)
        tmp.rename(path)  # a killed run never leaves a half-written file


if __name__ == "__main__":
    main()
