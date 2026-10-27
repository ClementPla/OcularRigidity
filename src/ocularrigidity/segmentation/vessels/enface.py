"""The en face grid on which the choroidal vessels are annotated and segmented.

An OCTA volume is flattened on Bruch's membrane and binned along the depth into
``(slices, B-scans, A-scans)``; slice ``k`` holds, at B-scan ``t`` and A-scan
``w``, the native rows ``rint(bm[t, w] + z0) + DEPTH_BIN * k`` (``DEPTH_BIN`` of
them). The Didascalie export of ``notebooks/to_labelmed`` shows these slices as
frames; the 3D nnU-Net trains and predicts on the same grid.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

from ocularrigidity.data.io import load_mask, save_mask_atomic
from ocularrigidity.segmentation.inference import infer
from ocularrigidity.segmentation.postprocess.interfaces import (
    clean_boundaries,
    extract_boundaries_fast,
)
from ocularrigidity.segmentation.vessels.choroid3d import (
    _bin_depth,
    _fill_nan,
    flatten_on_boundary,
)

# Native pixels kept above the BM and below the deepest CSI of the volume.
MARGIN_TOP = 12
MARGIN_BOTTOM = 90
# Native pixels averaged in one en face slice: 6 -> 11.7 um, about the
# spacing of the A-scans and of the B-scans (12 um).
DEPTH_BIN = 6

# Voxel size of the grid in mm, (slice, B-scan, A-scan), for a 6x6 mm Angio.
AXIAL_MM = 1.95e-3
SCAN_MM = 6.0


def grid_spacing(n_bscans: int, n_ascans: int, depth_bin: int = DEPTH_BIN) -> tuple:
    return (depth_bin * AXIAL_MM, SCAN_MM / n_bscans, SCAN_MM / n_ascans)


def choroid_boundaries(
    amp: np.ndarray, cache: Path | None = None, model=None, device: str = "cuda:0"
) -> tuple:
    """Smoothed BM and CSI of a ``(B-scans, depth, A-scans)`` volume, ``(T, W)`` each in native pixels.

    ``model`` is called to get the choroid segmentation model, only when the
    masks are not in ``cache``.
    """
    if cache is not None and cache.exists():
        masks = load_mask(cache)
    else:
        masks = infer(
            model(),
            amp,
            scale_factor=(1, 2.0),
            batch_size=8,
            device=device,
            post_process=True,
        ).astype(bool)
        if cache is not None:
            save_mask_atomic(masks, cache)
    bm, csi = clean_boundaries(*extract_boundaries_fast(masks))
    bm, csi = _fill_nan(bm), _fill_nan(csi)
    # Same flattening boundary as choroid_vesselness
    bm = gaussian_filter(median_filter(bm, size=(3, 9)), sigma=(1, 3))
    return bm, csi


def grid_depth(bm: np.ndarray, csi: np.ndarray) -> int:
    """Native rows below the BM covered by the grid."""
    depth = int(np.ceil(np.percentile(csi - bm, 99.5))) + MARGIN_BOTTOM
    return depth + -(MARGIN_TOP + depth) % DEPTH_BIN


def enface_volume(
    amp: np.ndarray, bm: np.ndarray, n_slices: int, z0: int = -MARGIN_TOP
) -> np.ndarray:
    """``(n_slices, T, W)`` float32 mean amplitude of each slice of the grid."""
    flat = flatten_on_boundary(amp, bm, (z0, z0 + n_slices * DEPTH_BIN))
    return _bin_depth(flat.astype(np.float32), DEPTH_BIN).transpose(1, 0, 2)


def stretch_slices(volume: np.ndarray) -> np.ndarray:
    """uint8 frames for annotation, contrast set per slice (the amplitude falls off with depth)."""
    lo, hi = np.percentile(volume, [0.5, 99.5], axis=(1, 2), keepdims=True)
    return (np.clip((volume - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def to_native(
    grid: np.ndarray, bm: np.ndarray, z0: int, depth: int, depth_bin: int = DEPTH_BIN
) -> np.ndarray:
    """A ``(slices, T, W)`` grid array in the native ``(T, depth, W)`` frame, 0 outside the grid."""
    n, T, W = grid.shape
    out = np.zeros((T, depth, W), dtype=grid.dtype)
    top = np.rint(bm + z0).astype(np.int64)  # (T, W) native row of slice 0
    y = np.arange(depth)
    for t in range(T):
        k = (y[:, None] - top[t][None, :]) // depth_bin  # (depth, W)
        valid = (k >= 0) & (k < n)
        out[t] = np.where(
            valid, grid[np.clip(k, 0, n - 1), t, np.arange(W)[None, :]], 0
        )
    return out


def to_native_linear(
    grid: np.ndarray, bm: np.ndarray, z0: int, depth: int, depth_bin: int = DEPTH_BIN
) -> np.ndarray:
    """``to_native`` of a float grid (e.g. probabilities), interpolated linearly along the depth.

    Each slice is taken at its centre row; the rows between two centres blend
    them, so that thresholding the result places a boundary at native-row
    precision rather than on the ``depth_bin`` steps of the grid.
    """
    n, T, W = grid.shape
    out = np.zeros((T, depth, W), dtype=np.float32)
    top = np.rint(bm + z0).astype(np.int64)
    y = np.arange(depth)[:, None]
    w = np.arange(W)[None, :]
    for t in range(T):
        rel = y - top[t][None, :]  # (depth, W) native rows below the top of slice 0
        u = (rel - (depth_bin - 1) / 2) / depth_bin  # position in slices, 0 at slice 0's centre
        k0 = np.floor(u).astype(np.int64)
        f = (u - k0).astype(np.float32)
        lo = grid[np.clip(k0, 0, n - 1), t, w]
        hi = grid[np.clip(k0 + 1, 0, n - 1), t, w]
        valid = (rel >= 0) & (rel < n * depth_bin)
        out[t] = np.where(valid, (1 - f) * lo + f * hi, 0)
    return out


def read_dida_masks(path: Path, label: str = "vessel") -> tuple[dict, dict]:
    """``label`` masks of the fully reviewed sequences of a Didascalie project.

    Returns ``{sequence: (frames, H, W) bool}``, a reviewed frame without a mask
    being empty, and ``{sequence: (reviewed, frames)}`` for the sequences left
    out because only some of their frames are reviewed.

    Reads the database directly, so it works with projects saved by versions
    of the application newer than the ``didascalie`` package.
    """
    from didascalie.project import _decode_mask

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    (label_id,) = conn.execute(
        "SELECT id FROM labels WHERE name = ?", (label,)
    ).fetchone()
    if "frame_reviews" in tables:
        reviewed = {r[0] for r in conn.execute("SELECT frame_id FROM frame_reviews")}
    else:
        reviewed = {r[0] for r in conn.execute("SELECT id FROM frames WHERE reviewed")}
    columns = {r[1] for r in conn.execute("PRAGMA table_info(annotations)")}
    if "user_id" in columns:
        users = conn.execute(
            "SELECT COUNT(DISTINCT user_id) FROM annotations WHERE label_id = ?",
            (label_id,),
        ).fetchone()[0]
        if users > 1:
            raise ValueError(f"{path}: '{label}' is annotated by {users} users")

    masks, partial = {}, {}
    for seq_id, name in conn.execute("SELECT id, name FROM sequences ORDER BY name"):
        frames = conn.execute(
            "SELECT id, width, height FROM frames WHERE sequence_id = ? ORDER BY frame_index",
            (seq_id,),
        ).fetchall()
        n_reviewed = sum(f[0] in reviewed for f in frames)
        if n_reviewed < len(frames):
            if n_reviewed:
                partial[name] = (n_reviewed, len(frames))
            continue
        stack = np.zeros((len(frames), frames[0][2], frames[0][1]), dtype=bool)
        for i, (frame_id, width, height) in enumerate(frames):
            row = conn.execute(
                "SELECT encoding, mask_data FROM annotations WHERE frame_id = ? AND label_id = ?",
                (frame_id, label_id),
            ).fetchone()
            if row is not None:
                stack[i] = _decode_mask(row[0], row[1], width, height, raw=False) > 0
        masks[name] = stack
    conn.close()
    return masks, partial
