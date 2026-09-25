"""3D vesselness of the main choroidal vessels from an OCT/OCTA volume.

The output is a dark-tube response, not a segmentation: thresholding is left to
the caller. Each step is there for one reason:

1. **Flatten** amplitude and flow on the BM (the choroid mask's top boundary),
   so a depth index means "distance below the BM". Both channels come from the
   same acquisition, so the same per-A-scan shift aligns them.
2. **Bin** the depth axis by ``depth_bin``. The smallest Hessian scale is
   several binned voxels, so a box average over ``depth_bin`` native pixels
   removes nothing the filter would see; it only makes the voxels ~cubic and
   the volume ``depth_bin`` times cheaper on the GPU.
3. **Normalise** each channel by its per-(B-scan, depth) median over choroid
   voxels, then take the log. The median removes the B-scan brightness stripes
   and the depth attenuation; the log makes the multiplicative speckle additive
   for the Hessian.
4. **Inpaint retinal shadows.** Large retinal vessels shadow the amplitude and
   leave projection artifacts in the flow, in the same (B-scan, A-scan) columns
   at every depth. Those columns are the retinal vessels, mapped by a ridge
   filter on the retinal flow MIP (where they have far more contrast than
   their shadows have on the amplitude). Each shadowed voxel copies a random
   unshadowed neighbour at the same depth, which keeps the speckle texture:
   a smooth fill would respond less to the Hessian than its surroundings and
   print the retinal network back into the output.
5. **Fuse** the two channels (z-scored over the choroid) and compute a
   multiscale Hessian dark-tube response on the GPU.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, gaussian_filter, median_filter
from skimage.filters import sato
from skimage.morphology import disk, remove_small_objects

__all__ = [
    "ChoroidVesselnessConfig",
    "ChoroidVesselness",
    "choroid_vesselness",
    "shadow_mask",
    "inpaint_columns",
    "flatten_on_boundary",
    "dark_tube_response",
]


@dataclass
class ChoroidVesselnessConfig:
    # Depth binning: 6 x ~2 um axial ~= the ~12 um lateral pitch of a 6x6 mm,
    # 500x500 scan. Set to 1 for volumes that are already isotropic.
    depth_bin: int = 6
    # Depth kept below the BM, in native pixels. Voxels deeper than the CSI are
    # zeroed anyway; this only bounds the memory.
    max_depth: int = 330
    # Hessian scales, in binned voxels (~12 um). 3-8 = radii of ~35-100 um,
    # i.e. the large vessels; add 2 to pick up medium ones (and more noise).
    sigmas: tuple[float, ...] = (3.0, 4.5, 6.0, 8.0)
    # Weight of amplitude vs flow in the fused signal (flow gets 1 - w).
    amplitude_weight: float = 0.5
    # Retinal slab for the flow MIP, native pixels relative to the BM. The
    # lower end stays clear of the RPE and choriocapillaris.
    retina_range: tuple[int, int] = (-200, -15)
    # Ridge scales on the MIP, lateral px (~12 um): radii of ~18-48 um, above
    # the capillary bed, which casts no shadow.
    shadow_sigmas: tuple[float, ...] = (1.5, 2.5, 4.0)
    # A column is a vessel when its ridge response exceeds median + k * MAD,
    # i.e. is calibrated on the background whatever the vessel density.
    shadow_k: float = 1.5
    # Shadows are a little wider than the lumen that casts them (lateral px).
    shadow_dilation: int = 1
    # Lateral scale (px) of the offsets at which shadowed voxels pick donors.
    inpaint_sigma: float = 6.0
    inpaint_shadows: bool = True
    device: str = "cuda"


@dataclass
class ChoroidVesselness:
    """Everything in the flattened, binned frame ``(T, Zb, W)``.

    ``z`` index ``k`` is ``k * depth_bin`` to ``(k + 1) * depth_bin`` native
    pixels below ``bm`` (the smoothed boundary actually used for flattening).
    """

    response: np.ndarray  # (T, Zb, W) float32, dark-tube response, 0 outside the choroid
    fused: np.ndarray  # (T, Zb, W) float32, the signal the Hessian saw
    choroid: np.ndarray  # (T, Zb, W) bool, voxels between BM and CSI
    shadow: np.ndarray | None  # (T, W) bool, inpainted columns
    bm: np.ndarray  # (T, W) float, flattening boundary in native pixels
    depth_bin: int
    config: ChoroidVesselnessConfig = field(repr=False, default=None)

    def to_native(self, volume: np.ndarray, depth: int) -> np.ndarray:
        """Any ``(T, Zb, W)`` array of this frame back in the native ``(T, depth, W)`` frame.

        Nearest bin; voxels above the BM or below the last bin are 0 / False.
        """
        T, Zb, W = volume.shape
        y = np.arange(depth)[None, :, None]
        k = np.floor((y - self.bm[:, None, :]) / self.depth_bin).astype(np.int64)
        valid = (k >= 0) & (k < Zb)
        out = np.take_along_axis(volume, np.clip(k, 0, Zb - 1), axis=1)
        return np.where(valid, out, np.zeros((), dtype=volume.dtype))


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


def _normalise(v: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """log of ``v`` over its per-(B-scan, depth) median on ``valid`` voxels."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows, handled below
        med = np.nanmedian(np.where(valid, v, np.nan), axis=2, keepdims=True)
    # Rows with no valid voxel lie entirely outside the choroid; any scale will do.
    med = np.where(np.isfinite(med) & (med > 0), med, 1.0)
    # The offset keeps the zero-valued voxels finite.
    return np.log(v / med + 0.05).astype(np.float32)


def shadow_mask(
    flow_enface: np.ndarray,
    sigmas: tuple[float, ...] = (1.5, 2.5, 4.0),
    k: float = 1.5,
    dilation: int = 1,
    min_size: int = 50,
) -> np.ndarray:
    """``(T, W)`` columns under retinal vessels, from the retinal flow MIP.

    Each B-scan is divided by its median (the same brightness stripes as in the
    choroid), then a bright-ridge filter at ``sigmas`` keeps the vessels large
    enough to cast a shadow. A pixel is kept when its response is ``k`` robust
    SDs above the median; components under ``min_size`` px are noise.
    """
    e = np.log1p(np.asarray(flow_enface, dtype=np.float32))
    e -= np.median(e, axis=1, keepdims=True)
    ridge = sato(e, sigmas=sigmas, black_ridges=False)
    med = np.median(ridge)
    mad = 1.4826 * np.median(np.abs(ridge - med))
    m = remove_small_objects(ridge > med + k * mad, min_size=min_size)
    return binary_dilation(m, disk(dilation)) if dilation else m


def inpaint_columns(
    v: np.ndarray,
    hole: np.ndarray,
    valid: np.ndarray,
    sigma: float = 6.0,
    seed: int = 0,
    tries: int = 30,
) -> np.ndarray:
    """Fill ``hole`` (T, W) at every depth of ``v`` (T, Z, W) with nearby voxels.

    Each filled voxel copies one ``valid``, unshadowed voxel at the same depth,
    drawn at a Gaussian lateral offset of scale ``sigma``. On average this is
    the normalized convolution of the neighbours, but unlike that mean it keeps
    their speckle variance, so the Hessian sees the same texture inside the
    shadow as around it. Voxels without a donor after ``tries`` draws get the
    Gaussian-weighted mean.
    """
    rng = np.random.default_rng(seed)
    T, _, W = v.shape
    donor = valid & ~hole[:, None, :]
    t, z, w = np.nonzero(hole[:, None, :] & valid)
    out = v.copy()
    for _ in range(tries):
        if not t.size:
            break
        tt = np.clip(t + np.rint(rng.normal(0, sigma, t.size)).astype(np.intp), 0, T - 1)
        ww = np.clip(w + np.rint(rng.normal(0, sigma, t.size)).astype(np.intp), 0, W - 1)
        ok = donor[tt, z, ww]
        out[t[ok], z[ok], w[ok]] = v[tt[ok], z[ok], ww[ok]]
        t, z, w = t[~ok], z[~ok], w[~ok]
    if t.size:
        wt = donor.astype(np.float32)
        num = gaussian_filter(v * wt, sigma=(sigma, 0, sigma))
        den = gaussian_filter(wt, sigma=(sigma, 0, sigma))
        out[t, z, w] = num[t, z, w] / np.maximum(den[t, z, w], 1e-6)
    return out


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


def choroid_vesselness(
    amplitude: np.ndarray,
    flow: np.ndarray | None,
    bm: np.ndarray,
    csi: np.ndarray,
    config: ChoroidVesselnessConfig | None = None,
    shadow: np.ndarray | None = None,
) -> ChoroidVesselness:
    """Dark-tube response of the main choroidal vessels of one volume.

    Args:
        amplitude: ``(T, H, W)`` structural OCT, B-scan major (as in
            ``volume.ipynb`` after the transpose).
        flow: ``(T, H, W)`` OCTA channel (``phase.bin``) of the *same*
            acquisition, or ``None`` to use the amplitude alone.
        bm, csi: ``(T, W)`` choroid top and bottom boundaries, in the same
            (native, unregistered) frame as the volumes, e.g. from
            ``extract_boundaries_fast(raw_masks)``. NaNs are interpolated.
        shadow: ``(T, W)`` bool columns to inpaint. By default they are found
            on the retinal flow MIP; without ``flow`` nothing is inpainted
            unless a mask is given here.
    """
    cfg = config or ChoroidVesselnessConfig()
    b = cfg.depth_bin

    bm = _fill_nan(bm)
    csi = _fill_nan(csi)
    # The mask's boundary jitters by a pixel or two per A-scan; flattening on
    # the raw trace would print that jitter into every layer.
    bm_s = gaussian_filter(median_filter(bm, size=(3, 9)), sigma=(1, 3))
    thickness = gaussian_filter(csi - bm, sigma=2)

    Zb = cfg.max_depth // b
    zb = (np.arange(Zb) + 0.5) * b
    choroid = zb[None, :, None] < thickness[:, None, :]

    if not cfg.inpaint_shadows:
        shadow = None
    elif shadow is None and flow is not None:
        retina = flatten_on_boundary(flow, bm_s, cfg.retina_range)
        shadow = shadow_mask(
            retina.max(axis=1),
            sigmas=cfg.shadow_sigmas,
            k=cfg.shadow_k,
            dilation=cfg.shadow_dilation,
        )
    valid = choroid if shadow is None else choroid & ~shadow[:, None, :]

    def prep(vol):
        flat = flatten_on_boundary(vol, bm_s, (0, cfg.max_depth))
        v = _normalise(_bin_depth(flat.astype(np.float32), b), valid)
        if shadow is not None:
            v = inpaint_columns(v, shadow, choroid, cfg.inpaint_sigma)
        # z-scored so that `amplitude_weight` means the same for both channels
        return (v - v[valid].mean()) / v[valid].std()

    fused = prep(amplitude)
    if flow is not None:
        w = cfg.amplitude_weight
        fused = w * fused + (1 - w) * prep(flow)
    # 0 is the choroid's typical level: outside voxels are neutral rather than
    # a bright sclera or a dark vitreous that would read as a tube wall.
    fused[~choroid] = 0.0

    response = dark_tube_response(fused, cfg.sigmas, cfg.device)
    response[~choroid] = 0.0

    return ChoroidVesselness(
        response=response.astype(np.float32),
        fused=fused.astype(np.float32),
        choroid=choroid,
        shadow=shadow,
        bm=bm_s,
        depth_bin=b,
        config=cfg,
    )
