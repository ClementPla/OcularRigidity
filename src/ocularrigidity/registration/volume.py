"""Motion correction across the B-scans of a raster volume.

In a raster (e.g. 6x6 mm angio) volume every B-scan is a new position, so the
video tools — which align repeats of one position to a median — do not apply.
What does hold: the retina and choroid change smoothly from one B-scan to the
next (~12 um apart), while eye motion does not. An axial jump of tens of pixels
between neighbours is motion, not anatomy.

Measured on a PLEX Elite 6x6 angio, the motion is axial: the BM jumps by up to
~80 px across a few B-scans (saccades, head motion), while the lateral shift
between neighbours is 0 +/- 1 px (the retinal-vessel shadows stay continuous).
So this module corrects the axial position and tilt of each B-scan and flags
the frames that are corrupted rather than displaced.

The split between anatomy and motion is a modelling choice: the BM's offset
and tilt along the slow axis are fitted with a robust low-order polynomial (the
eye's curvature; degree 2 by default), and the residual is taken as motion.
Slow drift over the ~6 s acquisition is indistinguishable from curvature with a
single raster, so it stays in the "anatomy" — only an orthogonal scan could
separate the two.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

__all__ = [
    "AxialMotion",
    "estimate_axial_motion",
    "detect_bad_bscans",
    "apply_axial_shift",
    "correct_volume_motion",
]


@dataclass
class AxialMotion:
    shift: np.ndarray  # (T, W) px; corrected[t, y, x] = volume[t, y + shift[t, x], x]
    offset: np.ndarray  # (T,) motion of each B-scan's BM at the centre column
    tilt: np.ndarray  # (T,) motion of each B-scan's BM slope (px per A-scan)
    bm_offset: np.ndarray  # (T,) measured BM depth at the centre column
    anatomy_offset: np.ndarray  # (T,) the smooth fit kept as anatomy
    valid: np.ndarray  # (T,) bool, B-scans used for the fit (not corrupted)

    def correct_boundary(self, b: np.ndarray) -> np.ndarray:
        """A ``(T, W)`` boundary expressed in the corrected volume."""
        return b - self.shift


def _robust_polyfit(t, y, w0, degree, n_iter=10, c=4.685):
    """Tukey-biweight IRLS polynomial fit; returns the fitted curve at ``t``."""
    w = w0.astype(float).copy()
    fit = np.zeros_like(y)
    for _ in range(n_iter):
        coef = np.polyfit(t[w > 0], y[w > 0], degree, w=np.sqrt(w[w > 0]))
        fit = np.polyval(coef, t)
        r = y - fit
        s = 1.4826 * np.median(np.abs(r[w0 > 0])) + 1e-9
        u = r / (c * s)
        w = np.where(np.abs(u) < 1, (1 - u**2) ** 2, 0.0) * w0
    return fit


def _line_fit(bm: np.ndarray, xc: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-B-scan least-squares line ``bm[t, x] = a + b (x - xc)`` over valid columns."""
    x = np.arange(bm.shape[1], dtype=float) - xc
    m = ~np.isnan(bm)
    n = m.sum(1)
    Sx = np.where(m, x, 0).sum(1)
    Sxx = np.where(m, x * x, 0).sum(1)
    Sy = np.nansum(bm, 1)
    Sxy = np.nansum(bm * x, 1)
    den = n * Sxx - Sx**2
    with np.errstate(divide="ignore", invalid="ignore"):
        b = (n * Sxy - Sx * Sy) / den
        a = (Sy - b * Sx) / n
    return a, b


def detect_bad_bscans(
    flattened: np.ndarray, slab: tuple[int, int] = (10, 120), n_mad: float = 6.0
) -> np.ndarray:
    """Flag B-scans whose content does not continue their neighbours'.

    ``flattened`` is ``(T, Z, W)`` flattened on the BM (row 0 = BM). The mean
    over ``slab`` rows is the retinal-vessel shadow pattern, which is continuous
    across a good raster; a corrupted frame (blink, fly-back, saccade mid-scan)
    correlates with neither neighbour. Returns ``(T,)`` bool, True = bad.
    """
    p = np.log1p(flattened[:, slab[0] : slab[1]].astype(np.float32)).mean(1)
    p = p - gaussian_filter1d(p, 15, axis=1)
    p = (p - p.mean(1, keepdims=True)) / (p.std(1, keepdims=True) + 1e-6)
    c = (p[1:] * p[:-1]).mean(1)  # zero-lag correlation of neighbours
    best = np.maximum(np.r_[c, -1.0], np.r_[-1.0, c])  # best neighbour
    med = np.median(best)
    mad = 1.4826 * np.median(np.abs(best - med))
    return best < med - n_mad * mad


def estimate_axial_motion(
    bm: np.ndarray,
    degree: int = 2,
    bad: np.ndarray | None = None,
    correct_tilt: bool = True,
) -> AxialMotion:
    """Axial motion of each B-scan from the BM boundary ``(T, W)`` (NaN allowed).

    ``degree`` is the polynomial kept as anatomy along the slow axis (2 = the
    eye's curvature). ``bad`` B-scans are excluded from the fit and their
    motion is interpolated from their neighbours'.
    """
    T, W = bm.shape
    xc = (W - 1) / 2
    a, b = _line_fit(bm, xc)
    ok = np.isfinite(a) & np.isfinite(b) & (np.isfinite(bm).mean(1) > 0.5)
    if bad is not None:
        ok &= ~bad
    t = np.arange(T, dtype=float)

    anat_a = _robust_polyfit(t, np.where(ok, a, 0), ok, degree)
    anat_b = _robust_polyfit(t, np.where(ok, b, 0), ok, degree)
    off = np.where(ok, a - anat_a, np.nan)
    tilt = np.where(ok, b - anat_b, np.nan) if correct_tilt else np.zeros(T)
    # Corrupted frames: borrow the motion of the nearest good neighbours.
    for v in (off, tilt):
        m = np.isnan(v)
        if m.any():
            v[m] = np.interp(t[m], t[~m], v[~m])

    shift = off[:, None] + tilt[:, None] * (np.arange(W) - xc)[None, :]
    return AxialMotion(
        shift=shift.astype(np.float32),
        offset=off,
        tilt=tilt,
        bm_offset=a,
        anatomy_offset=anat_a,
        valid=ok,
    )


@torch.no_grad()
def apply_axial_shift(
    volume: np.ndarray,
    shift: np.ndarray,
    order: int = 1,
    device: str = "cuda",
    batch: int = 32,
) -> np.ndarray:
    """``out[t, y, x] = volume[t, y + shift[t, x], x]``, per A-scan.

    ``order=1`` interpolates linearly (images), ``order=0`` takes the nearest
    sample (masks). Samples outside the volume are 0. Keeps the input dtype.
    """
    T, H, W = volume.shape
    out = np.empty_like(volume)
    y = torch.arange(H, device=device, dtype=torch.float32)[None, :, None]
    for s in range(0, T, batch):
        e = min(s + batch, T)
        v = torch.from_numpy(np.ascontiguousarray(volume[s:e])).to(device).float()
        src = y + torch.from_numpy(shift[s:e]).to(device)[:, None, :]
        if order == 0:
            src = src.round()
        y0 = src.floor()
        f = src - y0
        y0 = y0.long()
        inside0 = (y0 >= 0) & (y0 < H)
        inside1 = (y0 + 1 >= 0) & (y0 + 1 < H)
        v0 = torch.gather(v, 1, y0.clamp(0, H - 1)) * inside0
        v1 = torch.gather(v, 1, (y0 + 1).clamp(0, H - 1)) * inside1
        r = v0 * (1 - f) + v1 * f
        if volume.dtype == bool:
            r = r > 0.5
        elif np.issubdtype(volume.dtype, np.integer):
            r = r.round().clamp(
                np.iinfo(volume.dtype).min, np.iinfo(volume.dtype).max
            )
        out[s:e] = r.to(torch.from_numpy(out[:0]).dtype).cpu().numpy()
    return out


def correct_volume_motion(
    bm: np.ndarray,
    *volumes: np.ndarray,
    masks: np.ndarray | None = None,
    degree: int = 2,
    flag_bad: bool = True,
    device: str = "cuda",
):
    """Estimate the axial motion from ``bm`` and remove it from every volume.

    Returns ``(motion, corrected_volumes, corrected_masks)``. The same shift is
    applied to all volumes (amplitude and flow of one acquisition share it);
    ``masks`` are shifted with nearest-neighbour sampling.
    """
    bad = None
    if flag_bad and volumes:
        bmf = np.where(np.isnan(bm), np.nanmedian(bm), bm)
        z = np.arange(0, 130)
        idx = np.clip(
            np.rint(bmf[:, None, :] + z[None, :, None]), 0, volumes[0].shape[1] - 1
        )
        flat = np.take_along_axis(np.asarray(volumes[0]), idx.astype(np.intp), axis=1)
        bad = detect_bad_bscans(flat)
    motion = estimate_axial_motion(bm, degree=degree, bad=bad)
    corrected = [apply_axial_shift(v, motion.shift, 1, device) for v in volumes]
    cmasks = (
        None if masks is None else apply_axial_shift(masks, motion.shift, 0, device)
    )
    return motion, corrected, cmasks
