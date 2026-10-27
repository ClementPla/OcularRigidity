"""Dense pulsatile displacement of a folded B-scan video, fitted as cardiac harmonics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, map_coordinates

__all__ = ["HarmonicField", "harmonic_lk", "fit_harmonic_field", "phases_of_folded", "inject_thickening"]


def phases_of_folded(n_frames: int, n_cycles: int) -> np.ndarray:
    """Cardiac phase of each frame of a folded video (``n_cycles`` equal cycles)."""
    fpc = n_frames // n_cycles
    return (np.arange(n_frames) % fpc) / fpc


def _basis(phases: np.ndarray, n_harmonics: int) -> np.ndarray:
    """(T, 2K): cos(2 pi k t), sin(2 pi k t) for k = 1..K."""
    k = np.arange(1, n_harmonics + 1)
    arg = 2 * np.pi * phases[:, None] * k[None]
    return np.concatenate([np.cos(arg), np.sin(arg)], axis=1).astype(np.float32)


@dataclass
class HarmonicField:
    """Periodic displacement ``d(p, t)`` of the tissue, in pixels."""

    coeffs: np.ndarray
    row0: int
    frame_shape: tuple[int, int]
    loss: np.ndarray  # loss per iteration

    @property
    def n_harmonics(self) -> int:
        return self.coeffs.shape[1] // 2

    def dense(self, phases: np.ndarray) -> np.ndarray:
        """(T, 2, H, W) displacement over the fitted rows."""
        B = _basis(np.asarray(phases, float), self.n_harmonics)
        return np.einsum("tk,ckhw->tchw", B, self.coeffs)

    def at(self, xy: np.ndarray, phases: np.ndarray) -> np.ndarray:
        """(T, N, 2) displacement (x, y) at points ``xy`` (N, 2) of the full frame."""
        B = _basis(np.asarray(phases, float), self.n_harmonics)
        coords = np.stack([xy[:, 1] - self.row0, xy[:, 0]])
        c = np.stack(
            [
                np.stack([map_coordinates(self.coeffs[i, j], coords, order=1, mode="nearest") for j in range(B.shape[1])])
                for i in range(2)
            ]
        )  # (2, 2K, N)
        return np.einsum("tk,ckn->tnc", B, c)

    def harmonic(self, k: int = 1) -> np.ndarray:
        """Complex amplitude (2, H, W) of harmonic ``k``: d_k(t) = Re(A e^{-2i pi k t})."""
        K = self.n_harmonics
        return self.coeffs[:, k - 1] + 1j * self.coeffs[:, K + k - 1]


def _gauss_blur(x: torch.Tensor, sigma: tuple[float, float]) -> torch.Tensor:
    """Separable Gaussian blur over the last two axes of (N, h, w)."""
    for axis, s in zip((-2, -1), sigma):
        if s <= 0:
            continue
        r = int(np.ceil(3 * s))
        k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) / s) ** 2)
        k = (k / k.sum()).view(1, 1, -1)
        y = x.movedim(axis, -1)
        shape = y.shape
        y = F.conv1d(F.pad(y.reshape(-1, 1, shape[-1]), (r, r), mode="replicate"), k)
        x = y.reshape(shape).movedim(-1, axis)
    return x


def harmonic_lk(
    frames: np.ndarray,
    phases: np.ndarray,
    roi: np.ndarray,
    n_harmonics: int = 3,
    window: tuple[float, float] = (6.0, 8.0),
    lateral: bool = True,
    sigma: tuple[float, float] = (1.0, 1.0),
    n_iter: int = 3,
    ridge: float = 1e-3,
    margin: int = 32,
    device: str = "cuda:0",
    return_steps: bool = False,
) -> HarmonicField | tuple[HarmonicField, list[dict]]:
    """Groupwise Lucas-Kanade on the cardiac harmonics (closed form)."""
    T, H, W = frames.shape
    rows = np.flatnonzero(roi.any(axis=1))
    r0, r1 = max(rows[0] - margin, 0), min(rows[-1] + margin + 1, H)
    h = r1 - r0
    dev = torch.device(device)
    I = torch.from_numpy(frames[:, r0:r1].astype(np.float32)).to(dev)
    I = _gauss_blur(I, sigma)
    I = (I - I.mean()) / I.std()
    B = torch.from_numpy(_basis(np.asarray(phases, float), n_harmonics)).to(dev)  # (T, 2K)
    K2 = B.shape[1]
    norm = (B**2).sum(0)  # = T/2 for whole cycles

    yy, xx = torch.meshgrid(torch.arange(h, device=dev), torch.arange(W, device=dev), indexing="ij")
    to_norm = torch.tensor([2.0 / (W - 1), 2.0 / (h - 1)], device=dev, dtype=torch.float32)
    base = torch.stack([xx, yy], dim=-1).float() * to_norm - 1

    C = torch.zeros(2, K2, h, W, device=dev)
    M = torch.from_numpy(roi[r0:r1]).to(dev)
    losses, steps = [], []
    for _ in range(n_iter):
        d = torch.einsum("tk,ckhw->tchw", B, C)
        grid = base[None] + d.permute(0, 2, 3, 1) * to_norm
        Wt = F.grid_sample(I[:, None], grid, mode="bicubic", padding_mode="border", align_corners=True)[:, 0]
        R = Wt.mean(0)
        losses.append((Wt - R).abs()[:, M].mean().item())
        E = torch.einsum("tk,thw->khw", B, Wt - R) / norm[:, None, None]  # (2K, h, W)
        gy, gx = torch.gradient(R)
        if lateral:
            Sxx, Sxy, Syy = _gauss_blur(torch.stack([gx * gx, gx * gy, gy * gy]), window)
            bx = _gauss_blur(gx[None] * E, window)
            by = _gauss_blur(gy[None] * E, window)
            eps = ridge * (Sxx + Syy).mean()
            Sxx, Syy = Sxx + eps, Syy + eps
            det = Sxx * Syy - Sxy**2
            dC = torch.stack([-(Syy * bx - Sxy * by) / det, -(Sxx * by - Sxy * bx) / det])
        else:
            Sxx, Sxy = torch.zeros_like(gy), torch.zeros_like(gy)
            Syy = _gauss_blur(gy[None] ** 2, window)[0]
            Syy = Syy + ridge * Syy.mean()
            bx, by = torch.zeros_like(E), _gauss_blur(gy[None] * E, window)
            dC = torch.stack([torch.zeros_like(by), -by / Syy])
        C += dC
        if return_steps:
            K = n_harmonics
            first = [0, K]  # cos_1, sin_1
            steps.append(
                {
                    k: v.cpu().numpy()
                    for k, v in dict(
                        R=R, gx=gx, gy=gy, E=E[first],
                        S=torch.stack([Sxx, Sxy, Syy]),
                        b=torch.stack([bx[first], by[first]]),
                        dC=dC[:, first],
                    ).items()
                }
            )
    field = HarmonicField(coeffs=C.cpu().numpy(), row0=int(r0), frame_shape=(H, W), loss=np.array(losses))
    return (field, steps) if return_steps else field


def fit_harmonic_field(
    frames: np.ndarray,
    phases: np.ndarray,
    roi: np.ndarray,
    n_harmonics: int = 3,
    spacing: tuple[int, int] = (8, 16),
    lateral: bool = True,
    smooth: float = 0.0,
    sigma: tuple[float, float] = (1.0, 1.0),
    n_iter: int = 400,
    lr: float = 0.02,
    margin: int = 16,
    interp: str = "bicubic",
    device: str = "cuda:0",
) -> HarmonicField:
    """Fit the periodic displacement field of a folded video."""
    T, H, W = frames.shape
    rows = np.flatnonzero(roi.any(axis=1))
    r0, r1 = max(rows[0] - margin, 0), min(rows[-1] + margin + 1, H)
    img = frames[:, r0:r1].astype(np.float32)
    img = gaussian_filter(img, (0, *sigma))
    img = ((img - img.mean()) / img.std()).astype(np.float32)
    h = r1 - r0

    dev = torch.device(device)
    I = torch.from_numpy(img)[:, None].to(dev)  # (T, 1, h, W)
    M = torch.from_numpy(roi[r0:r1].astype(np.float32)).to(dev)
    B = torch.from_numpy(_basis(np.asarray(phases, float), n_harmonics)).to(dev)  # (T, 2K)
    gh, gw = int(np.ceil((h - 1) / spacing[0])) + 1, int(np.ceil((W - 1) / spacing[1])) + 1
    C = torch.zeros(1, 2 * 2 * n_harmonics, gh, gw, device=dev, requires_grad=True)
    comp_mask = torch.ones(2, 1, 1, 1, device=dev)
    if not lateral:
        comp_mask[0] = 0

    # identity sampling grid, normalised to [-1, 1] (align_corners=True)
    yy, xx = torch.meshgrid(torch.arange(h, device=dev), torch.arange(W, device=dev), indexing="ij")
    to_norm = torch.tensor([2.0 / (W - 1), 2.0 / (h - 1)], device=dev, dtype=torch.float32)
    base = torch.stack([xx, yy], dim=-1).float() * to_norm - 1  # (h, W, 2) in (x, y)

    def dense(C):
        c = F.interpolate(C, size=(h, W), mode="bicubic", align_corners=True)[0]
        c = c.view(2, 2 * n_harmonics, h, W) * comp_mask
        return torch.einsum("tk,ckhw->tchw", B, c)  # (T, 2, h, W)

    opt = torch.optim.Adam([C], lr=lr)
    losses = []
    for _ in range(n_iter):
        opt.zero_grad()
        d = dense(C)
        grid = base[None] + d.permute(0, 2, 3, 1) * to_norm
        warped = F.grid_sample(I, grid, mode=interp, padding_mode="border", align_corners=True)[:, 0]
        resid = warped - warped.mean(dim=0, keepdim=True)
        loss = (torch.sqrt(resid**2 + 1e-2) * M).sum() / (M.sum() * T)
        if smooth > 0:
            loss = loss + smooth * (
                (C[..., 1:, :] - C[..., :-1, :]).pow(2).mean() + (C[..., :, 1:] - C[..., :, :-1]).pow(2).mean()
            )
        loss.backward()
        opt.step()
        losses.append(loss.item())

    with torch.no_grad():
        c = F.interpolate(C, size=(h, W), mode="bicubic", align_corners=True)[0]
        c = (c.view(2, 2 * n_harmonics, h, W) * comp_mask).cpu().numpy()
    return HarmonicField(coeffs=c, row0=int(r0), frame_shape=(H, W), loss=np.array(losses))


def inject_thickening(
    frames: np.ndarray,
    phases: np.ndarray,
    rpe: np.ndarray,
    csi: np.ndarray,
    amp_px: np.ndarray,
    phase0: float = 0.0,
) -> np.ndarray:
    """Add a known periodic choroidal thickening to a video (for recovery tests)."""
    T, H, W = frames.shape
    z = np.arange(H, dtype=np.float32)[:, None]
    th = np.maximum(csi - rpe, 1)[None]
    f = np.clip((z - rpe[None]) / th, 0, 1)  # (H, W)
    f = np.nan_to_num(f) * np.nan_to_num(amp_px)[None]
    rr, cc = np.mgrid[0:H, 0:W].astype(np.float32)
    out = np.empty_like(frames)
    for t in range(T):
        d = f * np.cos(2 * np.pi * (phases[t] - phase0))
        # the tissue at z - d lands at z (d is small and smooth, so d(z - d) ~ d(z))
        warped = map_coordinates(frames[t].astype(np.float32), [rr - d, cc], order=3, mode="nearest")
        out[t] = np.clip(np.round(warped), 0, 255).astype(frames.dtype)
    return out
