"""Boundary map → one pulse waveform.

Input is the ``(B, T, 2, W)`` pooled BM/CSI map with NaN on gaps and holes,
plus a ``(B, T)`` frame-validity mask. The network:

1. normalises each clip (per-column mean removed, one global scale);
2. runs a 2-D conv stack over (time, column) with dilated temporal kernels,
   halving the column axis as it goes;
3. pools columns with learned attention — a learned version of the pipeline's
   "which A-scans carry the pulse" selection;
4. finishes with a short dilated 1-D stack and emits one sample per frame.

Fully convolutional in time, so the same weights run on a 10-s clip or a whole
video. Receptive field is ~1.5 s: enough to shape a pulse, far too short to
invent a periodicity the input does not carry.
"""

import torch
from torch import nn


def normalise(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``(B, T, 2, W)`` + ``(B, T)`` → ``(B, 4, T, W)`` network input.

    Channels: BM, CSI, thickness (CSI − BM), frame validity. Invalid entries
    are zero after normalisation.
    """
    x = torch.cat([x, x[:, :, 1:2] - x[:, :, 0:1]], dim=2)  # (B, T, 3, W)
    valid = ~torch.isnan(x) & mask[:, :, None, None]
    v = valid.float()
    x = torch.where(valid, x, 0.0)
    n = v.sum(1, keepdim=True).clamp_min(1.0)
    x = (x - x.sum(1, keepdim=True) / n) * v
    scale = (
        x.square().sum((1, 2, 3), keepdim=True)
        / v.sum((1, 2, 3), keepdim=True).clamp_min(1.0)
    ).sqrt()
    x = x / scale.clamp_min(1e-6)
    m = mask.float()[:, :, None, None].expand(-1, -1, 1, x.shape[-1])
    return torch.cat([x, m], dim=2).permute(0, 2, 1, 3)


class Block2d(nn.Module):
    def __init__(self, c_in, c_out, k_t=7, dilation=1, pool_w=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c_in,
            c_out,
            (k_t, 3),
            padding=(dilation * (k_t // 2), 1),
            dilation=(dilation, 1),
        )
        self.norm = nn.GroupNorm(8, c_out)
        self.act = nn.GELU()
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.pool = nn.AvgPool2d((1, 2)) if pool_w else nn.Identity()

    def forward(self, x):
        return self.pool(self.act(self.norm(self.conv(x))) + self.skip(x))


class Block1d(nn.Module):
    def __init__(self, c, k=5, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(c, c, k, padding=dilation * (k // 2), dilation=dilation)
        self.norm = nn.GroupNorm(8, c)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.act(self.norm(self.conv(x)))


class PulseNet(nn.Module):
    def __init__(self, width: int = 32, head_width: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            Block2d(4, width, dilation=1),  # W 256 → 128
            Block2d(width, width, dilation=2),  # → 64
            Block2d(width, head_width, dilation=4),  # → 32
            Block2d(head_width, head_width, dilation=8),  # → 16
        )
        self.attn = nn.Conv2d(head_width, 1, 1)
        self.head = nn.Sequential(
            Block1d(head_width, dilation=1),
            Block1d(head_width, dilation=2),
            Block1d(head_width, dilation=4),
            Block1d(head_width, dilation=8),
            nn.Conv1d(head_width, 1, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``(B, T, 2, W)``, ``(B, T)`` → waveform ``(B, T)``."""
        h = self.encoder(normalise(x, mask))  # (B, C, T, W')
        a = torch.softmax(self.attn(h), dim=-1)
        h = (h * a).sum(-1)  # (B, C, T)
        return self.head(h).squeeze(1)

    @torch.no_grad()
    def column_attention(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean attention over time per pooled column, ``(B, W')`` — diagnostics."""
        h = self.encoder(normalise(x, mask))
        return torch.softmax(self.attn(h), dim=-1).mean(2).squeeze(1)
