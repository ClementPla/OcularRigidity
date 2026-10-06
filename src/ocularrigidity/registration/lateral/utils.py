import torch
import torch.nn.functional as F
import numpy as np


def robust_temporal_dx(
    dx: torch.Tensor,
    conf: torch.Tensor = None,
    conf_z: float = 1.0,
    k: float = 2.5,
    win: int = 15,
    max_velocity: float = 4.0,
) -> torch.Tensor:
    """Reject temporally inconsistent lateral shifts and re-interpolate them."""
    dx = dx.float()
    good = torch.ones_like(dx, dtype=torch.bool)
    if conf is not None:
        good &= conf >= conf_z
    med = _median_filter_1d(dx, win)
    resid = (dx - med).abs()
    # Primary consistency test: deviation from the local median, robust-scaled.
    good &= resid <= k * (torch.median(resid) + 1e-6)
    # Hard velocity cap: reject anything that strays too far from the trend.
    if max_velocity is not None:
        good &= resid <= max_velocity
    if good.all() or good.sum() < 2:
        return dx
    idx = torch.arange(dx.numel(), device=dx.device, dtype=torch.float32)
    interp = _interp1d(idx, idx[good], dx[good])
    return torch.where(good, dx, interp)


def smooth_translations(dx: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    """Applies a 1D Gaussian filter to smooth out high-frequency jitter in dx."""
    if sigma <= 0:
        return dx

    radius = int(4 * sigma + 0.5)

    x = torch.arange(-radius, radius + 1, device=dx.device, dtype=torch.float32)
    kernel = torch.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(
        1, 1, -1
    )  # Formatted for conv1d: (out_channels, in_channels, width)

    # Pad dx to handle edges cleanly
    dx_padded = F.pad(dx.view(1, 1, -1), (radius, radius), mode="replicate")

    # Convolve
    dx_smoothed = F.conv1d(dx_padded, kernel)
    return dx_smoothed.view(-1)


def _median_filter_1d(x: torch.Tensor, win: int = 11) -> torch.Tensor:
    """Rolling median (replicate-padded)."""
    n = x.numel()
    if n < 3:
        return x
    win = min(win, n if n % 2 else n - 1)
    if win < 3:
        return x
    pad = win // 2
    xp = F.pad(x.view(1, 1, -1), (pad, pad), mode="replicate").view(-1)
    return xp.unfold(0, win, 1).median(dim=1).values


def _interp1d(query: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """1D linear interpolation of (xp, fp) at `query`, with flat extrapolation."""
    idx = torch.searchsorted(xp, query).clamp(1, xp.numel() - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = fp[idx - 1], fp[idx]
    y = y0 + (query - x0) / (x1 - x0 + 1e-12) * (y1 - y0)
    y = torch.where(query <= xp[0], fp[0], y)
    y = torch.where(query >= xp[-1], fp[-1], y)
    return y
