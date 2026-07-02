"""Shared rendering toolkit for the viewer UIs (no Gradio / Streamlit deps).

Turns precomputed cohort artifacts — the folded one-cycle ``.mkv``, the saved
segmentation ``.npz`` and the stored boundary displacements — into small,
browser-friendly mp4s. Reads are cached and frames are downscaled before
encoding, since for browsing the encode time and file size (not fidelity) are
what matter. Used by both :mod:`ocularrigidity.viewer.explorer` (Gradio) and the
Streamlit viewer page.
"""

from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from matplotlib import cm
from scipy.signal import savgol_filter
import streamlit as st
from ocularrigidity.data.compression import read_gray
from ocularrigidity.data.io import load_mask

# Shared temp dir for the mp4s we transcode for the browser.
WORKDIR = Path(tempfile.mkdtemp(prefix="ocular_viewer_"))

# Fast libx264 preset: browsing favours quick encodes over the smallest files.
PRESET = "veryfast"


# --- encoding ----------------------------------------------------------------


def to_uint8(cube: np.ndarray) -> np.ndarray:
    """Clip + cast a (possibly float) frame stack to display-ready uint8."""
    if cube.dtype == np.uint8:
        return cube
    return np.clip(np.nan_to_num(cube), 0, 255).astype(np.uint8)


def write_mp4(
    frames: np.ndarray,
    path: str,
    fps: int,
    *,
    quality: int = 8,
    preset: str | None = None,
) -> str:
    """Write a (T, H, W) gray or (T, H, W, 3) RGB stack to a browser-playable mp4.

    ``preset`` is forwarded to libx264 (e.g. ``"ultrafast"`` for fast, larger
    files when browsing); ``quality`` trades size for fidelity (0–10).
    """
    frames = to_uint8(frames)
    if frames.ndim == 3:
        # gray (T, H, W) -> RGB: repeat the new trailing channel axis, NOT width.
        frames = np.repeat(frames[..., None], 3, axis=-1)
    imageio.mimwrite(
        path,
        list(frames),
        fps=fps,
        codec="libx264",
        quality=quality,
        macro_block_size=16,  # pad odd dims so libx264 doesn't choke
        output_params=["-preset", preset] if preset else None,
    )
    return path


def overlay_video(
    frames: np.ndarray, masks: np.ndarray, alpha: float = 0.4
) -> np.ndarray:
    """Composite a red mask overlay onto grayscale frames -> (T, H, W, 3) uint8."""
    frames = to_uint8(frames)
    out = np.repeat(frames[..., None], 3, axis=3).astype(np.float32)
    color = np.array([255, 85, 85], dtype=np.float32)
    m = masks.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


# --- cached reads + cheap resizing -------------------------------------------
# Reads are cached so the one-cycle, mask overlay and quiver renders for one
# case share a single decode. Downscaling before encoding is the main speed/size
# lever — full-res (1536×1024) mp4s are slow to write and heavy to ship.


@st.cache_data(max_entries=4)
def read_cube(mkv_path: str, _indices=None) -> np.ndarray:
    return read_gray(mkv_path, indices=_indices)


@st.cache_data(max_entries=4)
def read_masks(npz_path: str, _indices=None) -> np.ndarray:
    mask = load_mask(npz_path)
    if _indices is not None:
        mask = mask[_indices]
    return mask


def resize_cube(cube: np.ndarray, factor: int, *, nearest: bool = False) -> np.ndarray:
    """Downscale a (T, H, W) stack by an integer ``factor`` (1 = no-op)."""
    if factor <= 1:
        return cube
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    h, w = cube.shape[1:3]
    nh, nw = max(1, h // factor), max(1, w // factor)
    return np.stack(
        [cv2.resize(f, (nw, nh), interpolation=interp) for f in cube.astype(np.uint8)]
    )


def square_crop_offset(h: int, w: int, size: int | None = None) -> tuple[int, int, int]:
    """Centered-square crop box for a (H, W) frame -> ``(x0, y0, side)``.

    ``side`` is clamped to ``min(h, w)`` so the box always fits; ``size=None``
    takes the largest centered square.
    """
    side = min(h, w) if size is None else min(int(size), h, w)
    return (w - side) // 2, (h - side) // 2, side


def center_crop_square(cube: np.ndarray, size: int | None = None) -> np.ndarray:
    """Center-crop a (T, H, W[, C]) stack to a square (``min(h, w)`` by default)."""
    x0, y0, side = square_crop_offset(cube.shape[1], cube.shape[2], size)
    return cube[:, y0 : y0 + side, x0 : x0 + side]


# --- displacement quiver -----------------------------------------------------


def smooth_disp(disp: np.ndarray, smooth_window: int, cyclic: bool) -> np.ndarray:
    """Savitzky–Golay temporal smoothing of a (T, N, 2) displacement, à la gif.py."""
    disp = np.nan_to_num(np.asarray(disp, dtype=np.float32))
    if smooth_window <= 2 or disp.shape[0] <= 3:
        return disp
    win = int(smooth_window) | 1  # savgol needs an odd window
    win = min(win, disp.shape[0] if disp.shape[0] % 2 == 1 else disp.shape[0] - 1)
    if win <= 3:
        return disp
    return savgol_filter(
        disp,
        window_length=win,
        polyorder=3,
        axis=0,
        mode="wrap" if cyclic else "interp",
    )


def render_quiver(
    frames: np.ndarray,
    displacement_per_cycle: list[np.ndarray],
    reference_coordinates_per_cycle: list[np.ndarray],
    output_path: str | Path,
    *,
    cycle: int | None = None,
    fps: int = 10,
    stride: int = 1,
    arrow_scale: float = 20.0,
    min_magnitude: float = 0.05,
    arrow_cmap: str = "viridis",
    arrow_thickness: int = 1,
    tip_length: float = 0.35,
    smooth_window: int = 0,
    cyclic: bool = True,
    only_y: bool = False,
    side_by_side: bool = False,
    annotate_scale: bool = True,
    coord_scale: float = 1.0,
    crop_offset: tuple[float, float] = (0.0, 0.0),
    quality: int = 8,
    preset: str | None = None,
) -> str:
    """Animate the *stored* boundary displacements as a quiver over the frames.

    Uses ``deltaA_per_cycle.pkl`` arrays directly — no optical flow is run here.
    The one-cycle video is the concatenation of ``N`` cardiac cycles; pass
    ``cycle`` to render just one (``None`` renders the whole loop). Reference
    coordinates and displacements are in ``(x, y)`` (column, row), matching
    ``extract_displacement_at_boundaries``; ``coord_scale`` maps them onto
    downscaled frames. Filtering (``smooth_window``/``cyclic``, ``min_magnitude``,
    ``only_y``, ``stride``, ``arrow_scale``, ``arrow_cmap``, ``side_by_side``)
    mirrors :func:`ocularrigidity.viewer.gif.render_mask_quiver`.
    """
    T = frames.shape[0]
    n_cycles = len(displacement_per_cycle)
    frame_per_cycle = T // n_cycles
    cmap = cm.get_cmap(arrow_cmap)

    # Pre-smooth + scale each cycle's vectors into the (downscaled) frame space.
    disps = [
        smooth_disp(displacement_per_cycle[c], smooth_window, cyclic) * coord_scale
        for c in range(n_cycles)
    ]
    # Shift reference coords into the cropped frame, then into downscaled space.
    offset = np.asarray(crop_offset, dtype=np.float32)
    refs = [
        (np.asarray(reference_coordinates_per_cycle[c], dtype=np.float32) - offset)
        * coord_scale
        for c in range(n_cycles)
    ]

    if cycle is None:
        cycle_list = list(range(n_cycles))
    else:
        cycle_list = [max(0, min(int(cycle), n_cycles - 1))]

    # Colour scale over the selected cycle(s) for good contrast.
    mags = []
    for c in cycle_list:
        d = disps[c]
        m = np.abs(d[..., 1]) if only_y else np.hypot(d[..., 0], d[..., 1])
        mags.append(m[np.isfinite(m)])
    mags = np.concatenate(mags) if mags else np.array([0.0, 1.0])
    sel = mags[mags >= min_magnitude]
    vmin, vmax = (sel.min(), sel.max()) if sel.size else (0.0, 1.0)

    overlay: list[np.ndarray] = []
    for c in cycle_list:
        ref, d_cycle = refs[c], disps[c]
        for local_t in range(d_cycle.shape[0]):
            gt = c * frame_per_cycle + local_t
            if gt >= T:
                break
            base = cv2.cvtColor(frames[gt], cv2.COLOR_GRAY2RGB)
            out = base.copy()
            d = d_cycle[local_t]  # (N, 2)
            m = np.abs(d[:, 1]) if only_y else np.hypot(d[:, 0], d[:, 1])
            valid = np.isfinite(m) & (m >= min_magnitude)
            for (x0, y0), (dx, dy), mag in zip(
                ref[valid][::stride], d[valid][::stride], m[valid][::stride]
            ):
                if only_y:
                    dx = 0.0
                ex, ey = x0 + dx * arrow_scale, y0 + dy * arrow_scale
                norm = 0.0 if vmax == vmin else (mag - vmin) / (vmax - vmin)
                color = tuple(int(v * 255) for v in cmap(norm)[:3])
                cv2.arrowedLine(
                    out,
                    (int(round(x0)), int(round(y0))),
                    (int(round(ex)), int(round(ey))),
                    color,
                    arrow_thickness,
                    line_type=cv2.LINE_AA,
                    tipLength=tip_length,
                )
            cv2.putText(
                out,
                f"cycle {c}  {local_t:02d}/{d_cycle.shape[0]}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            if annotate_scale:
                cv2.putText(
                    out,
                    f"x{arrow_scale:g}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
            if side_by_side:
                out = np.concatenate([base, out], axis=1)
            overlay.append(out)

    # mp4 (libx264) keeps these lightweight for the browser; a gif would be huge.
    return write_mp4(
        np.stack(overlay), str(output_path), fps=fps, quality=quality, preset=preset
    )
