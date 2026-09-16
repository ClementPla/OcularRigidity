from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ocularrigidity.registration.deep_learning.models.losses import warp

__all__ = [
    "CorrelationBaseline",
    "estimate_transform",
    "cost_volume",
    "normalize_features",
]


def normalize_features(x: torch.Tensor, center: bool = True) -> torch.Tensor:
    """Channel-centre over the map, then L2-normalise per position.

    Centring is not optional in practice: raw MiT features carry a large DC
    component, and a cosine similarity computed on it sits near 1 everywhere
    regardless of alignment, flattening the very peak we are trying to find.
    """
    if center:
        x = x - x.mean(dim=(2, 3), keepdim=True)
    return F.normalize(x, dim=1)


def cost_volume(
    fixed: torch.Tensor, moving: torch.Tensor, ry: int, rx: int
) -> torch.Tensor:
    """Matching cost for every candidate shift in a ``(2ry+1, 2rx+1)`` window.

    ``cost[b, i, j, h, w]`` correlates ``fixed`` at ``(h, w)`` with ``moving`` at
    ``(h + i - ry, w + j - rx)``. Inputs are expected normalised.
    """
    B, _, h, w = fixed.shape
    padded = F.pad(moving, (rx, rx, ry, ry))
    out = fixed.new_empty(B, 2 * ry + 1, 2 * rx + 1, h, w)
    for i in range(2 * ry + 1):
        for j in range(2 * rx + 1):
            out[:, i, j] = (fixed * padded[:, :, i : i + h, j : j + w]).sum(dim=1)
    return out


def _parabolic(profile: torch.Tensor, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sub-cell peak of ``profile`` along ``dim``: (offset from centre, peak value).

    Same three-point fit the classical estimators use, and the same guard: a peak
    on the boundary has no parabola through it, so it is left at integer
    position.
    """
    n = profile.shape[dim]
    peak = profile.argmax(dim=dim, keepdim=True)
    clamped = peak.clamp(1, n - 2)
    ym1 = profile.gather(dim, clamped - 1)
    y0 = profile.gather(dim, clamped)
    yp1 = profile.gather(dim, clamped + 1)
    denom = ym1 - 2 * y0 + yp1
    off = torch.where(
        denom.abs() > 1e-8, 0.5 * (ym1 - yp1) / denom, torch.zeros_like(y0)
    ).clamp(-1.0, 1.0)
    # On the boundary the fit is meaningless; keep the integer peak there.
    off = torch.where(peak == clamped, off, torch.zeros_like(off))
    pos = clamped.to(profile.dtype) + off - (n // 2)
    return pos.squeeze(dim), y0.squeeze(dim)


def _aggregate_w(profile: torch.Tensor, k: int) -> torch.Tensor:
    """Box-smooth a ``(B, D, w)`` cost profile along the column axis.

    Aggregating the *cost* rather than smoothing the resulting displacement is
    the standard stereo trick and is strictly better here: a column whose peak is
    ambiguous borrows evidence from its neighbours before committing, instead of
    committing to noise and having it averaged away afterwards.
    """
    if k <= 1:
        return profile
    return F.avg_pool1d(profile, k, stride=1, padding=k // 2, count_include_pad=False)


@torch.no_grad()
def estimate_transform(
    fixed_feats: Sequence[torch.Tensor],
    moving_feats: Sequence[torch.Tensor],
    img_shape: Tuple[int, int],
    *,
    dy_radius: int = 6,
    dx_radius: int = 3,
    aggregate: int = 9,
    center: bool = True,
    scales: Optional[Sequence[int]] = None,
    return_trace: bool = False,
):
    """Estimate ``(dx, dy)`` in full-resolution pixels from two feature pyramids.

    Args:
        fixed_feats/moving_feats: pyramids, fine->coarse, each ``(B, C, h, w)``.
        img_shape: ``(H, W)`` of the full-resolution frame.
        dy_radius/dx_radius: search window per scale, in cells of that scale.
            At stride 32 a radius of 6 reaches 192 px, which covers the bulk
            axial offset in one step; finer scales then see only a residual.
        aggregate: cost-aggregation width along W (cells). 0/1 disables.
        scales: which pyramid levels to use, coarse-to-fine order is enforced.
            Defaults to all of them.
        return_trace: also return the per-scale estimates, for diagnosis.

    Returns:
        ``(dx (B,), dy (B, W))``, plus a list of per-scale dicts if requested.
    """
    H, W = img_shape
    n = len(fixed_feats)
    levels = sorted(scales if scales is not None else range(n), reverse=True)

    B = fixed_feats[0].shape[0]
    dev, dtype = fixed_feats[0].device, torch.float32
    dx = torch.zeros(B, device=dev, dtype=dtype)
    dy = torch.zeros(B, W, device=dev, dtype=dtype)
    trace: List[dict] = []

    for pos, i in enumerate(levels):
        f = fixed_feats[i].to(dtype)
        m = moving_feats[i].to(dtype)
        stride = H / f.shape[-2]
        if pos > 0:  # the first level starts from the identity: no warp needed
            m, _ = warp(m, dx, dy, (H, W))
        f = normalize_features(f, center)
        m = normalize_features(m, center)

        vol = cost_volume(f, m, dy_radius, dx_radius)  # (B, Dy, Dx, h, w)

        # --- global lateral shift ------------------------------------------
        # Pool over the whole map, then take the best row shift for each column
        # shift: dx must not be read off a slice that assumes dy is already
        # right, since at this point it is not.
        glob = vol.mean(dim=(3, 4))  # (B, Dy, Dx)
        dx_profile = glob.max(dim=1).values  # (B, Dx)
        ddx, _ = _parabolic(dx_profile, dim=1)
        # cost_volume indexes `moving` at w + j - rx, while the warp convention
        # samples moving at x - dx: the peak offset is the negation of dx.
        ddx = -ddx

        # --- per-column vertical shift -------------------------------------
        # Evaluated at the column shift just found, so the two are consistent.
        jstar = (glob.max(dim=1).values.argmax(dim=1)).view(B, 1, 1, 1, 1)
        sel = vol.gather(2, jstar.expand(B, vol.shape[1], 1, *vol.shape[3:]))
        col = sel.squeeze(2).mean(dim=2)  # (B, Dy, w)
        col = _aggregate_w(col, aggregate)
        ddy, peak = _parabolic(col, dim=1)  # (B, w)

        dx = dx + ddx * stride
        ddy_full = F.interpolate(
            ddy.unsqueeze(1), size=W, mode="linear", align_corners=False
        ).squeeze(1)
        dy = dy + ddy_full * stride

        if return_trace:
            trace.append(
                {
                    "level": i,
                    "stride": stride,
                    "ddx_px": float(ddx.mean() * stride),
                    "ddy_px": float(ddy_full.mean() * stride),
                    "peak": float(peak.mean()),
                }
            )

    return (dx, dy, trace) if return_trace else (dx, dy)


class CorrelationBaseline(nn.Module):
    """``(dx, dy)`` read straight off the correlation peak. No training, no
    parameters.

    A drop-in for :class:`RegistrationRegressor`: the same ``forward``
    signature, so it can be handed to ``fused.register`` or to the trainer's
    evaluation pass anywhere the regressor goes, and scored on the same ruler.

    This is the one comparison the ablation arms cannot make. All twelve share
    the same backbone, so they only ever measure learned against learned; this
    measures against no learning at all -- the frozen pyramid and an argmax,
    which is precisely what :class:`CorrelationVolume`'s docstring contrasts
    its learned head with ("the peak is picked by a learned, context-aware head
    instead of an ``argmax``").

    ``scales=None`` walks the whole pyramid coarse-to-fine, the training-free
    analogue of the cascade. ``scales=[3]`` uses the coarsest level alone, which
    is what the cascade is worth with no learned head anywhere in the
    comparison -- unlike the ``no_cascade`` arm, whose dx head collapsed
    (``val/shift`` pinned at 4.0, the value a dx-invariant model scores).

    Parameter-free, so ``.to(device)`` and ``.eval()`` are no-ops kept only for
    interface compatibility.
    """

    def __init__(
        self,
        img_shape: Tuple[int, int] = (1536, 1024),
        *,
        dy_radius: int = 6,
        dx_radius: int = 3,
        aggregate: int = 9,
        center: bool = True,
        scales: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.img_shape = tuple(img_shape)
        self.dy_radius = int(dy_radius)
        self.dx_radius = int(dx_radius)
        self.aggregate = int(aggregate)
        self.center = bool(center)
        self.scales = None if scales is None else list(scales)
        # Mirrors the checkpoint fields fused.py and the notebooks report on.
        self.cascade = self.scales is None or len(self.scales) > 1
        self.use_correlation = True

    def forward(
        self,
        fixed_feats: Sequence[torch.Tensor],
        moving_feats: Sequence[torch.Tensor],
        img_shape: Optional[Tuple[int, int]] = None,
        detach_dy: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(dx (B,), dy (B, W))`` in pixels of ``img_shape``.

        ``detach_dy`` is accepted and ignored: there is no gradient here for it
        to cut, but the trainer passes it on the triplet and shift batches.
        """
        return estimate_transform(
            fixed_feats,
            moving_feats,
            img_shape if img_shape is not None else self.img_shape,
            dy_radius=self.dy_radius,
            dx_radius=self.dx_radius,
            aggregate=self.aggregate,
            center=self.center,
            scales=self.scales,
        )
