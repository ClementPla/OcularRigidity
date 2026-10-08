from typing import List, Optional, Sequence, Tuple

from huggingface_hub import PyTorchModelHubMixin
import torch
import torch.nn as nn
import torch.nn.functional as F

from ocularrigidity.registration.deep_learning.models.losses import warp


# --------------------------------------------------------------------------- #
# Small building blocks
def conv_norm_act(
    in_ch: int, out_ch: int, k=3, s=1, p=1, d=1, groups_gn: int = 8
) -> nn.Sequential:
    """Conv -> GroupNorm -> GELU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p, dilation=d, bias=False),
        nn.GroupNorm(groups_gn, out_ch),
        nn.GELU(),
    )


class ResidualConv1d(nn.Module):
    """Dilated 1D residual block, run along the W axis to share evidence between neighbouring columns."""

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
_MAX_CORR_DY_RADIUS = 16


class CorrelationVolume(nn.Module):
    """Normalised dot products between ``fixed`` and ``moving`` over a small window of candidate shifts, one channel per candidate."""

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
        f = F.normalize(self.proj(fixed), dim=1)
        m = F.normalize(self.proj(moving), dim=1)
        m = F.pad(m, (rx, rx, ry, ry))
        return self._corr(f, m, h, w)  # (B, (2ry+1)(2rx+1), h, w)


# --------------------------------------------------------------------------- #
# Scale fusion:  [fixed, moving, fixed-moving] (+ correlation) -> single trunk
class ScalePyramidFusion(nn.Module):
    """SegFormer-style all-conv decode, specialised for registration."""

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
class DxHead(nn.Module):
    """Convs (on fine features, so the ~2% shift is resolvable) -> global average pool -> MLP -> scalar."""

    def __init__(self, embed: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            conv_norm_act(embed, embed, s=2),  # 384x256 -> 192x128
            conv_norm_act(embed, embed, s=2),
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
class DyHead(nn.Module):
    """Collapse H (stride only in H, W preserved) so each column yields one feature vector"""

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
        r = r.squeeze(1)  # (B, W)
        return bulk, r - r.mean(dim=1, keepdim=True)


# --------------------------------------------------------------------------- #
# Cascade stage: one coarse-to-fine refinement step
class CascadeStage(nn.Module):
    """Predict the *residual* displacement at one scale, given the moving features already warped by the estimate carried down from coarser scales."""

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
class RegistrationRegressor(nn.Module, PyTorchModelHubMixin):
    """Predict ``(dx, dy)`` in **pixels** from a pair of frozen encoder pyramids."""

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
        """Compile every correlation volume in the model."""
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
        bulk_only: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(dx (B,), dy (B, W))``, both in pixels of the frame the features came from."""
        H, W = img_shape if img_shape is not None else self.img_shape
        if not self.cascade:
            trunk = self.fusion(fixed_feats, moving_feats)
            s_dx, s_bulk, s_res = self.scales
            dy_bulk, dy_res = self.dy_head(trunk)
            if bulk_only:
                dy_res = torch.zeros_like(dy_res)
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
            if bulk_only:
                dy = dy + d_bulk.unsqueeze(1) * stride
                continue
            d_res = F.interpolate(
                d_res.unsqueeze(1), size=W, mode="linear", align_corners=False
            ).squeeze(1)
            dy = dy + (d_bulk.unsqueeze(1) + d_res) * stride
        return dx, dy
