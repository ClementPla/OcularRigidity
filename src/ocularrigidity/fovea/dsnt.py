"""Differentiable Spatial-to-Numerical Transform (DSNT) for single-landmark
sub-pixel localization, implemented inline (no `dsntnn` dependency).

A model outputs a 1-channel heatmap; `flat_softmax` turns it into a spatial
probability map, and `dsnt` takes its expectation to get a sub-pixel coordinate.
Training combines a coordinate (euclidean) loss with a Jensen-Shannon regularizer
that keeps the predicted heatmap tight around the target.

Coordinate convention (matches the linspace DSNT grid): a pixel index ``i`` over
a size-``n`` axis maps to the normalized value ``(2*i - (n - 1)) / n`` in ~[-1, 1].
Use `pixel_to_normalized` / `normalized_to_pixel` so dataset targets and inference
decoding stay consistent with the grid used here.
"""

import torch
import torch.nn.functional as F

EPS = 1e-12


def pixel_to_normalized(coord_px: torch.Tensor | float, size: int):
    """Pixel index -> normalized coordinate consistent with the DSNT grid."""
    return (2.0 * coord_px - (size - 1)) / size


def normalized_to_pixel(coord_norm: torch.Tensor | float, size: int):
    """Normalized coordinate -> pixel index (inverse of `pixel_to_normalized`)."""
    return (coord_norm * size + (size - 1)) / 2.0


def _coord_grid(n: int, device, dtype) -> torch.Tensor:
    return (2.0 * torch.arange(n, device=device, dtype=dtype) - (n - 1)) / n


def flat_softmax(logits: torch.Tensor) -> torch.Tensor:
    """Spatial softmax over (H, W). Input/Output: (B, C, H, W)."""
    b, c, h, w = logits.shape
    flat = F.softmax(logits.reshape(b, c, h * w), dim=-1)
    return flat.reshape(b, c, h, w)


def dsnt(heatmap: torch.Tensor) -> torch.Tensor:
    """Expected coordinate of a spatial-prob heatmap.

    Args:
        heatmap: (B, C, H, W), each (H, W) slice sums to 1.
    Returns:
        coords: (B, C, 2) normalized (x, y) in ~[-1, 1].
    """
    b, c, h, w = heatmap.shape
    xs = _coord_grid(w, heatmap.device, heatmap.dtype)  # (W,)
    ys = _coord_grid(h, heatmap.device, heatmap.dtype)  # (H,)
    px = heatmap.sum(dim=2)  # marginal over rows -> (B, C, W)
    py = heatmap.sum(dim=3)  # marginal over cols -> (B, C, H)
    exp_x = (px * xs).sum(dim=-1)
    exp_y = (py * ys).sum(dim=-1)
    return torch.stack([exp_x, exp_y], dim=-1)


def euclidean_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-keypoint euclidean distance. pred/target: (B, C, 2) -> (B, C)."""
    return torch.linalg.norm(pred - target, dim=-1)


def _render_gaussian(coords: torch.Tensor, h: int, w: int, sigma_px: float) -> torch.Tensor:
    """Normalized Gaussian heatmaps centered at `coords` (B, C, 2) -> (B, C, H, W)."""
    device, dtype = coords.device, coords.dtype
    xs = _coord_grid(w, device, dtype).view(1, 1, 1, w)
    ys = _coord_grid(h, device, dtype).view(1, 1, h, 1)
    cx = coords[..., 0].view(*coords.shape[:2], 1, 1)
    cy = coords[..., 1].view(*coords.shape[:2], 1, 1)
    # One pixel ~ 2/size in normalized units.
    sx = sigma_px * 2.0 / w
    sy = sigma_px * 2.0 / h
    g = torch.exp(-((xs - cx) ** 2) / (2 * sx**2) - ((ys - cy) ** 2) / (2 * sy**2))
    g = g / (g.sum(dim=(2, 3), keepdim=True) + EPS)
    return g


def js_reg_loss(
    heatmap: torch.Tensor, target_coords: torch.Tensor, sigma_px: float = 1.0
) -> torch.Tensor:
    """Jensen-Shannon divergence between the predicted heatmap and a Gaussian
    rendered at the target coordinate. Keeps the heatmap unimodal and tight.

    Returns (B, C).
    """
    target = _render_gaussian(target_coords, heatmap.shape[2], heatmap.shape[3], sigma_px)
    m = 0.5 * (heatmap + target)

    def _kl(p, q):
        return (p * (torch.log(p + EPS) - torch.log(q + EPS))).sum(dim=(2, 3))

    return 0.5 * _kl(heatmap, m) + 0.5 * _kl(target, m)
