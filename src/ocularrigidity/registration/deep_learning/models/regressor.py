from typing import List, Optional, Sequence, Tuple

from huggingface_hub import PyTorchModelHubMixin
import torch
import torch.nn as nn
import torch.nn.functional as F

from ocularrigidity.registration.deep_learning.models.losses import warp


# --------------------------------------------------------------------------- #
# Small building blocks
# --------------------------------------------------------------------------- #
def conv_norm_act(
    in_ch: int, out_ch: int, k=3, s=1, p=1, d=1, groups_gn: int = 8
) -> nn.Sequential:
    """Conv -> GroupNorm -> GELU. GroupNorm (not BatchNorm) because
    registration is often trained with small batch sizes."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p, dilation=d, bias=False),
        nn.GroupNorm(groups_gn, out_ch),
        nn.GELU(),
    )


class ResidualConv1d(nn.Module):
    """Dilated 1D residual block, run along the W axis to share evidence
    between neighbouring columns."""

    def __init__(self, ch: int, dilation: int = 1, groups_gn: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(groups_gn, ch),
            nn.GELU(),
            nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(groups_gn, ch),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


# --------------------------------------------------------------------------- #
# Correlation volume: the displacement evidence, made explicit
# --------------------------------------------------------------------------- #
_MAX_CORR_DY_RADIUS = 16


class CorrelationVolume(nn.Module):
    """Normalised dot products between ``fixed`` and ``moving`` over a small
    window of candidate shifts, one channel per candidate.

    ``[fixed, moving, fixed - moving]`` asks the network to *infer* a
    displacement from a signed difference — a linearisation that only holds
    while the shift is well under one feature cell. A correlation volume hands
    it the matching cost directly, which is what makes PWC-Net/RAFT-style
    networks trainable without a regression target. It is also what the
    classical estimator computes (``lateral/correlation.py`` is a phase
    correlation, ``axial/median_registration.py`` a per-column one) — here the
    peak is picked by a learned, context-aware head instead of an ``argmax``.

    Radii are in cells of *this* scale, so the same radius covers 32 image
    pixels at stride 32 and 4 at stride 4: range where the estimate is coarse,
    precision where it is refined.
    """

    def __init__(
        self, in_ch: int, dim: int = 32, dy_radius: int = 6, dx_radius: int = 2
    ):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=1, bias=False)
        self.dy_radius = dy_radius
        self.dx_radius = dx_radius

    @property
    def out_channels(self) -> int:
        return (2 * self.dy_radius + 1) * (2 * self.dx_radius + 1)

    def compile_(self, dynamic: bool = True, **kwargs) -> "CorrelationVolume":
        """Replace the shift loop with a compiled version."""
        self._corr = torch.compile(self._corr, dynamic=dynamic, **kwargs)
        return self

    def _corr(self, f: torch.Tensor, m: torch.Tensor, h: int, w: int) -> torch.Tensor:
        ry, rx = self.dy_radius, self.dx_radius
        cost = [
            (f * m[:, :, dy : dy + h, dx : dx + w]).sum(dim=1, keepdim=True)
            for dy in range(2 * ry + 1)
            for dx in range(2 * rx + 1)
        ]
        return torch.cat(cost, dim=1)

    def forward(self, fixed: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
        h, w = fixed.shape[-2:]
        ry, rx = self.dy_radius, self.dx_radius
        # L2-normalised per position: a cosine cost, so the volume does not
        # simply track how bright the region is.
        f = F.normalize(self.proj(fixed), dim=1)
        m = F.normalize(self.proj(moving), dim=1)
        m = F.pad(m, (rx, rx, ry, ry))
        return self._corr(f, m, h, w)  # (B, (2ry+1)(2rx+1), h, w)


# --------------------------------------------------------------------------- #
# Scale fusion:  [fixed, moving, fixed-moving] (+ correlation) -> single trunk
# --------------------------------------------------------------------------- #
class ScalePyramidFusion(nn.Module):
    """SegFormer-style all-conv decode, specialised for registration.

    Per scale it stacks [fixed, moving, fixed - moving] (the signed diff is
    the linearised displacement readout), optionally adds a correlation volume
    (the matching cost, which does not rely on that linearisation), projects to
    a common embed dim, upsamples every scale to the finest resolution,
    concatenates and fuses.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        embed: int = 128,
        use_correlation: bool = True,
        corr_dim: int = 32,
        corr_dy_radius: int = 6,
        corr_dx_radius: int = 2,
    ):
        super().__init__()
        self.proj = nn.ModuleList(
            [nn.Conv2d(3 * c, embed, kernel_size=1) for c in in_channels]
        )
        self.use_correlation = use_correlation
        if use_correlation:
            # Without a cascade every scale must resolve the *absolute*
            # displacement, so a fixed radius in cells means the finest scale
            # sees the smallest window in pixels — exactly backwards. A stride-4
            # map with radius 6 saturates at 24 px while the bulk axial offsets
            # in this data reach 80. Widen the radius as the stride shrinks, to
            # a cap that keeps the volume affordable at full resolution.
            n = len(in_channels)
            radii = [
                min(corr_dy_radius * 2 ** (n - 1 - i), _MAX_CORR_DY_RADIUS)
                for i in range(n)
            ]
            self.corr = nn.ModuleList(
                [
                    CorrelationVolume(c, corr_dim, r, corr_dx_radius)
                    for c, r in zip(in_channels, radii)
                ]
            )
            self.corr_proj = nn.ModuleList(
                [nn.Conv2d(cv.out_channels, embed, kernel_size=1) for cv in self.corr]
            )
        self.fuse = conv_norm_act(embed * len(in_channels), embed, k=1, p=0)

    def forward(
        self, fixed: List[torch.Tensor], moving: List[torch.Tensor]
    ) -> torch.Tensor:
        tgt = fixed[0].shape[-2:]  # finest scale defines trunk size
        feats = []
        for i, (ff, mm) in enumerate(zip(fixed, moving)):
            x = torch.cat([ff, mm, ff - mm], dim=1)  # (B, 3C_i, h, w)
            x = self.proj[i](x)  # (B, E,   h, w)
            if self.use_correlation:
                x = x + self.corr_proj[i](self.corr[i](ff, mm))
            if x.shape[-2:] != tgt:
                x = F.interpolate(x, size=tgt, mode="bilinear", align_corners=False)
            feats.append(x)
        return self.fuse(torch.cat(feats, dim=1))  # (B, E, H4, W4)


# --------------------------------------------------------------------------- #
# dx head: one global scalar
# --------------------------------------------------------------------------- #
class DxHead(nn.Module):
    """Convs (on fine features, so the ~2% shift is resolvable) -> global
    average pool -> MLP -> scalar. GAP averages many local estimates of the
    single global quantity, reducing variance."""

    def __init__(self, embed: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            conv_norm_act(embed, embed, s=2),  # 384x256 -> 192x128
            conv_norm_act(embed, embed, s=2),  # 192x128 ->  96x 64
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(embed, embed // 2),
            nn.GELU(),
            nn.Linear(embed // 2, 1),
        )

    def forward(self, trunk: torch.Tensor) -> torch.Tensor:
        x = self.body(trunk)
        x = self.pool(x).flatten(1)  # (B, E)
        return self.mlp(x).squeeze(-1)  # (B,)


# --------------------------------------------------------------------------- #
# dy head: length-W per-column vector
# --------------------------------------------------------------------------- #
class DyHead(nn.Module):
    """Collapse H (stride only in H, W preserved) so each column yields one
    feature vector; run a dilated 1D context stack along W; project to a
    single channel; upsample to full width.

    H is collapsed by a few stride-in-H convs (which encode the vertical
    shift into channels) followed by an adaptive pool over the residual H.

    The output is split into a bulk term and a per-column residual::

        dy(x) = bulk + residual(x)

    which is how the motion is actually generated: the eye moves axially as a
    whole (tens of pixels across an acquisition) and the BM then deforms by a
    few pixels on top. Folding both into one per-column output makes the
    conditioning terrible — under direct optimisation the field has to travel
    the whole bulk offset through a 1x1 conv whose step size is set by the
    learning rate, so the large, easy part of the answer is also the slowest to
    arrive. Separating them lets each be scaled to its own range
    (``dy_scale`` vs ``residual_scale`` in the trainer).
    """

    def __init__(
        self,
        embed: int = 128,
        out_width: int = 1024,
        ctx_dilations: Tuple[int, ...] = (1, 2, 4, 8),
    ):
        super().__init__()
        self.h_reduce = nn.Sequential(
            conv_norm_act(embed, embed, k=3, s=(2, 1), p=1),  # H/2, W kept
            conv_norm_act(embed, embed, k=3, s=(2, 1), p=1),  # H/4
            conv_norm_act(embed, embed, k=3, s=(2, 1), p=1),  # H/8
        )
        self.h_pool = nn.AdaptiveAvgPool2d((1, None))  # H -> 1, W kept
        self.ctx = nn.Sequential(
            *[ResidualConv1d(embed, dilation=d) for d in ctx_dilations]
        )
        self.out = nn.Conv1d(embed, 1, kernel_size=1)
        self.bulk = nn.Sequential(
            nn.Linear(embed, embed // 2),
            nn.GELU(),
            nn.Linear(embed // 2, 1),
        )
        self.out_width = out_width

    def forward(self, trunk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(bulk (B,), residual (B, W))``, both dimensionless."""
        x = self.h_reduce(trunk)  # (B, E, H', W4)
        x = self.h_pool(x).squeeze(2)  # (B, E, W4)
        x = self.ctx(x)  # (B, E, W4)
        bulk = self.bulk(x.mean(dim=2)).squeeze(-1)  # (B,)
        r = self.out(x)  # (B, 1, W4)
        r = F.interpolate(r, size=self.out_width, mode="linear", align_corners=False)
        # Mean-free by construction, so the two terms cannot fight over the DC.
        r = r.squeeze(1)  # (B, W)
        return bulk, r - r.mean(dim=1, keepdim=True)


# --------------------------------------------------------------------------- #
# Cascade stage: one coarse-to-fine refinement step
# --------------------------------------------------------------------------- #
class CascadeStage(nn.Module):
    """Predict the *residual* displacement at one scale, given the moving
    features already warped by the estimate carried down from coarser scales.

    This is the structural fix for a correlation volume that saturates. A flat
    pyramid asks every scale to resolve the absolute displacement, so the finest
    map — the one with the best localisation — has the narrowest window in
    pixels and clips exactly the large offsets it is least able to guess. Warping
    first means each stage only ever sees a small residual, so a radius of a few
    cells is enough everywhere and the window is always centred on the current
    best estimate.

    Outputs are in *cells of this scale*; the caller multiplies by the stride.
    That is the natural normalisation — every stage predicts a number of order
    one regardless of where it sits in the pyramid — and it is what lets the
    coarsest stage cover an 80 px bulk offset in a single step of ~2.5 units
    instead of crawling there through a head scaled in raw pixels.
    """

    def __init__(
        self,
        in_ch: int,
        embed: int = 128,
        corr_dim: int = 32,
        dy_radius: int = 6,
        dx_radius: int = 2,
    ):
        super().__init__()
        self.corr = CorrelationVolume(in_ch, corr_dim, dy_radius, dx_radius)
        self.proj = nn.Conv2d(2 * in_ch, embed, kernel_size=1)
        self.mix = conv_norm_act(embed + self.corr.out_channels, embed, k=3)
        # Collapse H (stride in H only, W preserved) so each column ends up with
        # one feature vector; the vertical shift is encoded into channels first.
        self.h_reduce = nn.Sequential(
            conv_norm_act(embed, embed, k=3, s=(2, 1), p=1),
            conv_norm_act(embed, embed, k=3, s=(2, 1), p=1),
        )
        self.h_pool = nn.AdaptiveAvgPool2d((1, None))
        self.ctx = nn.Sequential(
            *[ResidualConv1d(embed, dilation=d) for d in (1, 2, 4)]
        )
        self.out_res = nn.Conv1d(embed, 1, kernel_size=1)
        self.out_global = nn.Linear(embed, 2)  # (ddx, d_bulk)

    def forward(
        self, fixed: torch.Tensor, moving_warped: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(ddx (B,), d_bulk (B,), d_residual (B, w))``, in cells."""
        c = self.corr(fixed, moving_warped)
        x = self.proj(torch.cat([fixed, moving_warped], dim=1))
        x = self.mix(torch.cat([x, c], dim=1))  # (B, E, h, w)
        x = self.h_pool(self.h_reduce(x)).squeeze(2)  # (B, E, w)
        x = self.ctx(x)
        g = self.out_global(x.mean(dim=2))  # (B, 2)
        r = self.out_res(x).squeeze(1)  # (B, w)
        return g[:, 0], g[:, 1], r - r.mean(dim=1, keepdim=True)


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #
class RegistrationRegressor(nn.Module, PyTorchModelHubMixin):
    """Predict ``(dx, dy)`` in **pixels** from a pair of frozen encoder pyramids.

    Two modes:

    ``cascade=True`` (default)
        Coarse-to-fine refinement. Start from the identity at stride 32, warp
        the moving features by the running estimate, let a :class:`CascadeStage`
        read the residual off a correlation volume centred on it, accumulate,
        repeat at the next finer scale. Each stage works in its own cells, so
        the same architecture covers a 200 px bulk offset and a sub-pixel
        correction without either being badly conditioned.

    ``cascade=False``
        The original single-shot fusion: all scales projected to a common
        embedding, upsampled to the finest, concatenated, and read by one pair
        of heads. Kept for comparison; its correlation volumes have to resolve
        absolute displacement, which is what the cascade exists to avoid.

    Args:
        in_channels: per-scale channel counts of the MiT pyramid, fine->coarse.
                     MiT-B1..B5: (64, 128, 320, 512); MiT-B0: (32, 64, 160, 256).
        embed:       common working width.
        img_shape:   ``(H, W)`` of the full-resolution frame. Needed to convert
                     between feature cells and pixels, and to size ``dy``.
        scales:      ``(dx, dy_bulk, dy_residual)`` output scales in pixels, used
                     only when ``cascade=False`` (the cascade derives its dy
                     scales from the strides).
        dx_step:     per-stage lateral step in pixels, for the cascade.
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (64, 128, 320, 512),
        embed: int = 128,
        img_shape: Tuple[int, int] = (1536, 1024),
        cascade: bool = True,
        scales: Tuple[float, float, float] = (16.0, 128.0, 8.0),
        dx_step: float = 4.0,
        use_correlation: bool = True,
        corr_dim: int = 32,
        corr_dy_radius: int = 6,
        corr_dx_radius: int = 2,
    ):
        super().__init__()
        self.config = dict(
            in_channels=in_channels,
            embed=embed,
            img_shape=img_shape,
            cascade=cascade,
            scales=scales,
            dx_step=dx_step,
            use_correlation=use_correlation,
            corr_dim=corr_dim,
            corr_dy_radius=corr_dy_radius,
            corr_dx_radius=corr_dx_radius,
        )
        self.img_shape = tuple(img_shape)
        self.cascade = cascade
        self.scales = scales
        self.dx_step = dx_step
        if cascade:
            self.stages = nn.ModuleList(
                [
                    CascadeStage(c, embed, corr_dim, corr_dy_radius, corr_dx_radius)
                    for c in in_channels
                ]
            )
        else:
            self.fusion = ScalePyramidFusion(
                in_channels,
                embed,
                use_correlation=use_correlation,
                corr_dim=corr_dim,
                corr_dy_radius=corr_dy_radius,
                corr_dx_radius=corr_dx_radius,
            )
            self.dx_head = DxHead(embed)
            self.dy_head = DyHead(embed, img_shape[1])

    def compile_correlation(
        self, dynamic: bool = True, **kwargs
    ) -> "RegistrationRegressor":
        """Compile every correlation volume in the model. See CorrelationVolume."""
        for mod in self.modules():
            if isinstance(mod, CorrelationVolume):
                mod.compile_(dynamic=dynamic, **kwargs)
        return self

    def forward(
        self,
        fixed_feats: List[torch.Tensor],
        moving_feats: List[torch.Tensor],
        img_shape: Optional[Tuple[int, int]] = None,
        detach_dy: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(dx (B,), dy (B, W))``, both in pixels of the frame the features
        came from.

        ``img_shape`` overrides ``self.img_shape`` for this call. The cascade
        reads its per-stage stride as ``H / f.shape[-2]``, so that number has
        to describe the frames actually encoded -- pass the crop's shape when
        training on crops, and the full frame's at inference. Everything else
        is convolutional or adaptively pooled, so the same weights serve both;
        getting this wrong does not raise, it silently mis-scales every stride.

        ``detach_dy`` cuts the gradient that reaches ``dy`` through the warp.
        Each stage warps the moving features by the running estimate before the
        next one reads ``ddx`` off them, so ``dx`` depends on ``dy`` and a
        *dx-only* objective can lower its error by moving ``dy`` -- with no dy
        term present to object.
        """
        H, W = img_shape if img_shape is not None else self.img_shape
        if not self.cascade:
            trunk = self.fusion(fixed_feats, moving_feats)
            s_dx, s_bulk, s_res = self.scales
            dy_bulk, dy_res = self.dy_head(trunk)
            return (
                self.dx_head(trunk) * s_dx,
                dy_bulk.unsqueeze(1) * s_bulk + dy_res * s_res,
            )

        B = fixed_feats[0].shape[0]
        dev, dtype = fixed_feats[0].device, fixed_feats[0].dtype
        dx = torch.zeros(B, device=dev, dtype=dtype)
        dy = torch.zeros(B, W, device=dev, dtype=dtype)

        n = len(fixed_feats)
        for i in range(n - 1, -1, -1):  # coarse -> fine
            f, m = fixed_feats[i], moving_feats[i]
            stride = H / f.shape[-2]

            if i < n - 1:
                m, _ = warp(m, dx, dy.detach() if detach_dy else dy, (H, W))
            ddx, d_bulk, d_res = self.stages[i](f, m)
            dx = dx + ddx * self.dx_step
            d_res = F.interpolate(
                d_res.unsqueeze(1), size=W, mode="linear", align_corners=False
            ).squeeze(1)
            dy = dy + (d_bulk.unsqueeze(1) + d_res) * stride
        return dx, dy
