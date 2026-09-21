"""3D segmentation of the main choroidal vessels from an OCT/OCTA volume.

The per-B-scan Niblack in ``traditional.py`` sees one noisy 2D slice at a time.
Here the whole volume is used, which is what makes the large vessels separable
from speckle:

1. **Flatten** amplitude and flow on the BM (the choroid mask's top boundary).
   The two channels come from the same acquisition, so the same per-A-scan
   shift aligns both; no warp between them is needed.
2. **Bin** the depth axis so voxels are roughly cubic (axial pixels are ~6x
   finer than the 6 mm / 500 lateral sampling) — isotropy for the Hessian and a
   free 6x average against speckle.
3. **Normalise** each channel per B-scan and depth (removes the motion stripes
   and the depth attenuation), then against a broad lateral background, and fuse
   the two in log space. Vessel lumens are dark in both.
4. **Remove retinal shadows**: their thin detail is the same at every depth,
   so its median over depth is subtracted from every layer.
5. **Dark-tube response**: multiscale Hessian on the GPU, large scales only,
   because the target is the main (Haller/Sattler) vessels.
6. **Threshold**: hysteresis on the locally normalised response, gated by a 3D
   Niblack (``F < m + k s``), then small components are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import (
    binary_erosion,
    gaussian_filter,
    median_filter,
    uniform_filter,
)
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import remove_small_objects

__all__ = [
    "ChoroidVesselConfig",
    "ChoroidVesselResult",
    "segment_choroid_vessels",
    "flatten_on_boundary",
    "dark_tube_response",
]


@dataclass
class ChoroidVesselConfig:
    # Depth binning: 6 x ~2 um axial ~= the ~12 um lateral pitch of a 6x6 mm,
    # 500x500 scan. Set to 1 for volumes that are already isotropic.
    depth_bin: int = 6
    # Depth kept below the BM, in native pixels. Voxels deeper than the CSI are
    # masked out anyway; this only bounds the memory.
    max_depth: int = 330
    # Hessian scales, in binned voxels (~12 um). 3-8 = radii of ~35-100 um,
    # i.e. the large vessels; add 2 to pick up medium ones (and more noise).
    sigmas: tuple[float, ...] = (3.0, 4.5, 6.0, 8.0)
    # Hysteresis on the locally normalised tube response, as percentiles of
    # its values inside the choroid.
    high_percentile: float = 95.0
    low_percentile: float = 85.0
    # 3D Niblack gate: a vessel voxel must be darker than m + k*s over
    # `niblack_window` (T, Z, W) binned voxels.
    niblack_k: float = -0.2
    niblack_window: tuple[int, int, int] = (61, 21, 61)
    # Smallest kept component, in binned voxels (~1.7e-6 mm^3 each).
    min_size: int = 20_000
    # Voxels eroded from the choroid mask before thresholding: the BM and CSI
    # edges otherwise read as dark tubes.
    edge_erosion: int = 2
    # Weight of amplitude vs flow in the fused signal (flow gets 1 - w).
    amplitude_weight: float = 0.5
    remove_shadows: bool = True
    device: str = "cuda"


@dataclass
class ChoroidVesselResult:
    """Everything in the flattened, binned frame ``(T, Zb, W)``.

    ``z`` index ``k`` is ``k * depth_bin`` to ``(k + 1) * depth_bin`` native
    pixels below ``bm`` (the smoothed boundary actually used for flattening).
    """

    mask: np.ndarray  # (T, Zb, W) bool, the vessels
    choroid: np.ndarray  # (T, Zb, W) bool, voxels between BM and CSI
    response: np.ndarray  # (T, Zb, W) float32, locally normalised tube response
    fused: np.ndarray  # (T, Zb, W) float32, the denoised fused signal
    shadows: np.ndarray | None  # (T, W) retinal-shadow detail that was removed
    bm: np.ndarray  # (T, W) float, flattening boundary in native pixels
    depth_bin: int
    config: ChoroidVesselConfig = field(repr=False, default=None)

    def to_native(self, depth: int) -> np.ndarray:
        """Vessel mask back in the native ``(T, depth, W)`` frame."""
        T, Zb, W = self.mask.shape
        y = np.arange(depth)[None, :, None]
        k = np.floor((y - self.bm[:, None, :]) / self.depth_bin).astype(np.int64)
        valid = (k >= 0) & (k < Zb)
        k = np.clip(k, 0, Zb - 1)
        return valid & np.take_along_axis(self.mask, k, axis=1)

    def enface_depth(self) -> np.ndarray:
        """``(T, W)`` depth below BM of the shallowest vessel voxel (native px), NaN where none."""
        top = np.argmax(self.mask, axis=1).astype(float) * self.depth_bin
        return np.where(self.mask.any(axis=1), top, np.nan)


def _fill_nan(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=float).copy()
    m = np.isnan(b)
    if m.all():
        raise ValueError("boundary is entirely NaN")
    if m.any():
        flat = b.ravel()
        flat[m.ravel()] = np.interp(
            np.flatnonzero(m.ravel()), np.flatnonzero(~m.ravel()), flat[~m.ravel()]
        )
    return b


def flatten_on_boundary(
    volume: np.ndarray, boundary: np.ndarray, z_range: tuple[int, int]
) -> np.ndarray:
    """Resample ``(T, H, W)`` so that ``boundary`` (T, W) is a flat row.

    Row ``j`` of the output is native depth ``boundary + z_range[0] + j``.
    """
    T, H, W = volume.shape
    z = np.arange(*z_range)
    idx = np.clip(np.rint(boundary[:, None, :] + z[None, :, None]), 0, H - 1)
    return np.take_along_axis(np.asarray(volume), idx.astype(np.intp), axis=1)


def _bin_depth(v: np.ndarray, b: int) -> np.ndarray:
    T, Z, W = v.shape
    Z = Z // b * b
    return v[:, :Z].reshape(T, Z // b, b, W).mean(axis=2, dtype=np.float32)


def _normalise(v: np.ndarray) -> np.ndarray:
    """Per-(B-scan, depth) median, then a broad lateral background; log."""
    v = v / (np.median(v, axis=2, keepdims=True) + 1e-3)
    v = v / (gaussian_filter(v, sigma=(30, 0, 30)) + 1e-6)
    # Speckle is multiplicative; the log makes it additive for the Hessian.
    # The offset keeps the (frequent) zero-valued voxels finite.
    return np.log(v + 0.05)


def _eigvalsh3(H: torch.Tensor) -> torch.Tensor:
    """Closed-form eigenvalues of symmetric 3x3 matrices, ascending.

    cuSOLVER's batched ``eigvalsh`` refuses batches of ~1e7; this is exact to
    float precision and runs in one elementwise pass.
    """
    a, b, c = H[..., 0, 0], H[..., 1, 1], H[..., 2, 2]
    d, e, f = H[..., 0, 1], H[..., 1, 2], H[..., 0, 2]
    q = (a + b + c) / 3
    p = torch.sqrt(
        ((a - q) ** 2 + (b - q) ** 2 + (c - q) ** 2 + 2 * (d * d + e * e + f * f)) / 6
    ).clamp_min(1e-12)
    B11, B22, B33 = (a - q) / p, (b - q) / p, (c - q) / p
    B12, B23, B13 = d / p, e / p, f / p
    det = (
        B11 * (B22 * B33 - B23 * B23)
        - B12 * (B12 * B33 - B23 * B13)
        + B13 * (B12 * B23 - B22 * B13)
    )
    phi = torch.acos((det / 2).clamp(-1, 1)) / 3
    l_max = q + 2 * p * torch.cos(phi)
    l_min = q + 2 * p * torch.cos(phi + 2 * torch.pi / 3)
    return torch.stack([l_min, 3 * q - l_max - l_min, l_max], dim=-1)


@torch.no_grad()
def dark_tube_response(
    volume: np.ndarray, sigmas: tuple[float, ...], device: str = "cuda"
) -> np.ndarray:
    """Max over scales of a Frangi-like dark-tube measure, sigma^2-normalised.

    With eigenvalues sorted by magnitude ``|e1| <= |e2| <= |e3|``, a dark tube
    has ``e2, e3 > 0`` (bright walls on both sides across the vessel) and
    ``e1 ~ 0`` (constant along it). Response is ``sqrt(e2 e3)``, damped when
    ``e1`` is not small relative to ``e2`` — which rejects dark blobs.
    """
    v = torch.from_numpy(np.ascontiguousarray(volume, dtype=np.float32)).to(device)
    v = v[None, None]
    best = torch.zeros(volume.shape, device=device)

    for s in sigmas:
        r = int(3 * s + 1)
        x = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
        g0 = torch.exp(-(x**2) / (2 * s * s))
        g0 /= g0.sum()
        g1 = -x / (s * s) * g0
        g2 = (x**2 / s**4 - 1 / (s * s)) * g0

        def conv(k0, k1, k2):
            out = v
            for axis, k in enumerate((k0, k1, k2)):
                shape = [1, 1, 1, 1, 1]
                shape[2 + axis] = -1
                pad = [0] * 6
                pad[2 * (2 - axis)] = pad[2 * (2 - axis) + 1] = r
                out = F.conv3d(F.pad(out, pad, mode="replicate"), k.view(shape))
            return out[0, 0]

        hxx, hyy, hzz = conv(g2, g0, g0), conv(g0, g2, g0), conv(g0, g0, g2)
        hxy, hxz, hyz = conv(g1, g1, g0), conv(g1, g0, g1), conv(g0, g1, g1)
        H = torch.stack(
            [
                torch.stack([hxx, hxy, hxz], -1),
                torch.stack([hxy, hyy, hyz], -1),
                torch.stack([hxz, hyz, hzz], -1),
            ],
            -2,
        ) * (s * s)
        del hxx, hyy, hzz, hxy, hxz, hyz

        lam = _eigvalsh3(H)
        del H
        lam = torch.gather(lam, -1, lam.abs().argsort(-1))
        e1, e2, e3 = lam.unbind(-1)
        tube = (e2 > 0) & (e3 > 0)
        resp = torch.sqrt((e2 * e3).clamp_min(0)) * torch.exp(
            -(e1**2) / (2 * (0.5 * e2).clamp_min(1e-3) ** 2)
        )
        best = torch.maximum(best, torch.where(tube, resp, torch.zeros_like(resp)))
        del lam, e1, e2, e3, resp

    out = best.cpu().numpy()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def segment_choroid_vessels(
    amplitude: np.ndarray,
    flow: np.ndarray | None,
    bm: np.ndarray,
    csi: np.ndarray,
    config: ChoroidVesselConfig | None = None,
) -> ChoroidVesselResult:
    """Segment the main choroidal vessels of one volume.

    Args:
        amplitude: ``(T, H, W)`` structural OCT, B-scan major (as in
            ``volume.ipynb`` after the transpose).
        flow: ``(T, H, W)`` OCTA channel (``phase.bin``) of the *same*
            acquisition, or ``None`` to use the amplitude alone.
        bm, csi: ``(T, W)`` choroid top and bottom boundaries, in the same
            (native, unregistered) frame as the volumes, e.g. from
            ``extract_boundaries_fast(raw_masks)``. NaNs are interpolated.
    """
    cfg = config or ChoroidVesselConfig()
    b = cfg.depth_bin

    bm = _fill_nan(bm)
    csi = _fill_nan(csi)
    # The mask's boundary jitters by a pixel or two per A-scan; flattening on
    # the raw trace would print that jitter into every layer.
    bm_s = gaussian_filter(median_filter(bm, size=(3, 9)), sigma=(1, 3))
    thickness = gaussian_filter(csi - bm, sigma=2)

    def prep(vol):
        flat = flatten_on_boundary(vol, bm_s, (0, cfg.max_depth))
        return _normalise(_bin_depth(flat.astype(np.float32), b))

    fused_log = prep(amplitude)
    zb = (np.arange(fused_log.shape[1]) + 0.5) * b
    choroid = zb[None, :, None] < thickness[:, None, :]

    def zscore(v):
        return (v - v[choroid].mean()) / v[choroid].std()

    fused = zscore(fused_log)
    if flow is not None:
        w = cfg.amplitude_weight
        fused = w * fused + (1 - w) * zscore(prep(flow))
    fused[~choroid] = 0.0

    shadows = None
    if cfg.remove_shadows:
        detail = fused - gaussian_filter(fused, sigma=(4, 0, 4))
        with np.errstate(all="ignore"):
            shadows = np.nanmedian(np.where(choroid, detail, np.nan), axis=1)
        shadows = np.nan_to_num(shadows).astype(np.float32)
        fused -= shadows[:, None, :] * choroid

    response = dark_tube_response(fused, cfg.sigmas, cfg.device)
    core = binary_erosion(choroid, iterations=cfg.edge_erosion)
    response[~core] = 0
    # The thick subfoveal choroid is darker and noisier, so a global threshold
    # only keeps the periphery; normalise by the local response level instead.
    local = gaussian_filter(response, sigma=(40, 0, 40))
    response = response / (local + 0.5 * local.mean())

    smooth = gaussian_filter(fused, 2.0)
    m = uniform_filter(smooth, cfg.niblack_window)
    s = np.sqrt(np.maximum(uniform_filter(smooth**2, cfg.niblack_window) - m**2, 0))
    dark = smooth < m + cfg.niblack_k * s

    hi, lo = np.percentile(response[core], [cfg.high_percentile, cfg.low_percentile])
    mask = apply_hysteresis_threshold(response, lo, hi) & dark & core
    mask = remove_small_objects(mask, min_size=cfg.min_size)

    return ChoroidVesselResult(
        mask=mask,
        choroid=choroid,
        response=response.astype(np.float32),
        fused=smooth.astype(np.float32),
        shadows=shadows,
        bm=bm_s,
        depth_bin=b,
        config=cfg,
    )
