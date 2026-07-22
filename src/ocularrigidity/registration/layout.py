"""Frame-layout helpers, so the registration works on gray *and* colour videos.

The pipeline speaks three layouts:

* ``(T, H, W)`` — grayscale, what the OCT gives us and what every boundary /
  correlation routine expects;
* ``(T, H, W, C)`` — colour, channels last, how videos are decoded and displayed;
* ``(T, C, H, W)`` — channels first, what ``grid_sample`` needs.

Rather than sprinkle ``if is_color`` through every warp, the registration
converts once with :func:`to_bchw`, does everything channels-first (a gray video
simply carries ``C = 1``), and converts back with :func:`restore_layout`.
Anything that measures a *displacement* — correlation, fovea, BM boundaries —
runs on the luminance instead, via :func:`to_gray`: the shift is a property of
the scene, not of the channel it is measured in.
"""

from __future__ import annotations

import numpy as np
import torch

# A trailing axis this small is a channel, never an image width.
_CHANNEL_SIZES = (1, 3, 4)

# ITU-R BT.601 luma weights.
_LUMA = (0.299, 0.587, 0.114)

GRAY = "gray"  # (T, H, W)
CHANNELS_LAST = "hwc"  # (T, H, W, C)
CHANNELS_FIRST = "chw"  # (T, C, H, W)


def frame_layout(frames) -> str:
    """Which of the three layouts ``frames`` is in.

    A 4-D stack is read as channels-last when its trailing axis is 1/3/4 — an
    image is never that narrow — and as channels-first otherwise.
    """
    if frames.ndim == 3:
        return GRAY
    if frames.ndim == 4:
        return CHANNELS_LAST if frames.shape[-1] in _CHANNEL_SIZES else CHANNELS_FIRST
    raise ValueError(
        f"Expected a (T, H, W[, C]) stack; got shape {tuple(frames.shape)}."
    )


def to_bchw(frames) -> tuple[torch.Tensor, str]:
    """Convert any layout to ``(T, C, H, W)``; returns it with the layout to restore."""
    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(frames)
    layout = frame_layout(frames)
    if layout == GRAY:
        return frames.unsqueeze(1), layout
    if layout == CHANNELS_LAST:
        return frames.permute(0, 3, 1, 2), layout
    return frames, layout


def restore_layout(frames: torch.Tensor, layout: str) -> torch.Tensor:
    """Inverse of :func:`to_bchw`: ``(T, C, H, W)`` back to ``layout``."""
    if layout == GRAY:
        return frames.squeeze(1)
    if layout == CHANNELS_LAST:
        return frames.permute(0, 2, 3, 1).contiguous()
    return frames


def to_gray(frames) -> torch.Tensor:
    """Luminance ``(T, H, W)`` of a stack in any layout (a no-op on gray input).

    Three channels are combined with the BT.601 luma weights; any other channel
    count is averaged.
    """
    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(frames)
    if frame_layout(frames) == GRAY:
        return frames
    bchw, _ = to_bchw(frames)
    if bchw.shape[1] == 3:
        w = torch.tensor(_LUMA, dtype=torch.float32, device=bchw.device)
        return (bchw.to(torch.float32) * w.view(1, 3, 1, 1)).sum(dim=1)
    return bchw.to(torch.float32).mean(dim=1)
