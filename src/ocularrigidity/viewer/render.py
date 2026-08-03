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
import streamlit as st
from ocularrigidity.data.compression import read_gray
from ocularrigidity.data.io import load_mask
from ocularrigidity.viewer.quiver import (
    QuiverStyle,
    draw_quiver,
    magnitude_range,
    smooth_disp,
)

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


def render_quiver(
    frames: np.ndarray,
    displacement_per_cycle: list[np.ndarray],
    reference_coordinates_per_cycle: list[np.ndarray],
    output_path: str | Path,
    *,
    masks: np.ndarray | None = None,
    cycle: int | None = None,
    style: QuiverStyle = QuiverStyle(),
    fps: int = 10,
    coord_scale: float = 1.0,
    crop_offset: tuple[float, float] = (0.0, 0.0),
    quality: int = 8,
    preset: str | None = None,
) -> str:
    """Animate the *stored* boundary displacements as a quiver over the frames.

    Uses the ``deltaA_per_cycle.pkl`` arrays directly — no optical flow is run
    here, unlike :func:`ocularrigidity.viewer.gif.render_mask_quiver`, which
    tracks the boundary itself. The drawing (and every :class:`QuiverStyle`
    option) is the shared :func:`ocularrigidity.viewer.quiver.draw_quiver`.

    The one-cycle video is the concatenation of ``N`` cardiac cycles, each with
    its own anchors; pass ``cycle`` to render just one (``None`` renders the
    whole loop, cycle after cycle). Stored coordinates and displacements are in
    the *full-resolution* frame, so ``crop_offset`` and ``coord_scale`` map them
    onto the cropped, downscaled frames actually being drawn — and ``masks``,
    when given (required by the CSI options), must already be in that same space.
    """
    T = frames.shape[0]
    n_cycles = len(displacement_per_cycle)
    frame_per_cycle = T // n_cycles
    cycles = (
        list(range(n_cycles))
        if cycle is None
        else [max(0, min(int(cycle), n_cycles - 1))]
    )

    # Stored coords/displacements live in the full frame: shift into the crop,
    # then scale into the downscaled frame the arrows are drawn on.
    offset = np.asarray(crop_offset, dtype=np.float32)
    disps = [np.asarray(displacement_per_cycle[c]) * coord_scale for c in cycles]
    refs = [
        (np.asarray(reference_coordinates_per_cycle[c], dtype=np.float32) - offset)
        * coord_scale
        for c in cycles
    ]

    # One colour scale across the rendered cycles, so arrows stay comparable.
    smoothed = [smooth_disp(d, style.smooth_window, style.cyclic) for d in disps]
    mags = [
        np.abs(d[..., 1]) if style.only_y else np.hypot(d[..., 0], d[..., 1])
        for d in smoothed
    ]
    vrange = magnitude_range(
        np.concatenate([m.ravel() for m in mags]), style.min_magnitude
    )

    overlay: list[np.ndarray] = []
    for c, ref, d in zip(cycles, refs, disps):
        start = c * frame_per_cycle
        stop = min(start + d.shape[0], T)
        d = d[: stop - start]
        overlay.append(
            draw_quiver(
                frames[start:stop],
                ref,
                d,
                masks=None if masks is None else masks[start:stop],
                reference=0,
                style=style,
                labels=[f"cycle {c}  {t:02d}/{d.shape[0]}" for t in range(d.shape[0])],
                vrange=vrange,
            )
        )

    # mp4 (libx264) keeps these lightweight for the browser; a gif would be huge.
    return write_mp4(
        np.concatenate(overlay),
        str(output_path),
        fps=fps,
        quality=quality,
        preset=preset,
    )
