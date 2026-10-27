"""Map radial line scans onto an OCTA volume, for a 3-D choroidal displacement field."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.ndimage import (
    binary_closing,
    gaussian_filter,
    gaussian_filter1d,
    map_coordinates,
    uniform_filter,
)
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import ball, remove_small_objects

from ocularrigidity.consts import AXIAL_PIXEL_SIZE_MM, TRANVERSAL_PIXEL_SIZE_MM
from ocularrigidity.segmentation.fovea.from_ilm import estimate_fovea_from_ilm
from ocularrigidity.segmentation.postprocess.blob import (
    keep_largest_connected_component,
)
from ocularrigidity.segmentation.postprocess.interfaces import (
    clean_boundaries,
    extract_boundaries_fast,
    rebuild_mask,
)
from ocularrigidity.thickness.delta import _csi_unit_normal_mm

AXIAL_UM = AXIAL_PIXEL_SIZE_MM * 1e3
LINE_UM = TRANVERSAL_PIXEL_SIZE_MM * 1e3  # line scan A-scan pitch

__all__ = [
    "AXIAL_UM",
    "LINE_UM",
    "scan_direction",
    "retina_ilm",
    "line_fovea",
    "volume_fovea",
    "LinePlacement",
    "shadow_profile",
    "enface_highpass",
    "placement_score",
    "refine_placement",
    "refine_center",
    "flip_vote",
    "local_thickness_change",
    "bin_columns",
    "first_harmonic_ptp",
    "PolarInterpolator",
    "threshold_vessels",
    "flow_slab",
    "vessel_binary",
    "masked_smooth",
    "octa_layers",
]


# --------------------------------------------------------------------------- #
# Line geometry from the Fundus.bmp overlay


def _magenta(img: np.ndarray) -> np.ndarray:
    # Same colour rule as fundus_utils.get_line_orientation.
    bw = (img[..., 0] == img[..., 1]) & (img[..., 1] == img[..., 2])
    return (img[..., 0] > 0) & (img[..., 1] < 200) & (img[..., 2] > 0) & ~bw


def scan_direction(fundus_path) -> tuple[float, np.ndarray]:
    """Orientation and scan direction of the line drawn on a Fundus.bmp."""
    img = np.asarray(Image.open(fundus_path).convert("RGB")).astype(int)
    ys, xs = np.nonzero(_magenta(img))
    P = np.stack([xs, ys], axis=1).astype(float)
    c = P.mean(axis=0)
    d = np.linalg.svd(P - c, full_matrices=False)[2][0]
    t = (P - c) @ d
    n = np.abs((P - c) @ np.array([-d[1], d[0]]))
    lo, hi = np.quantile(t, [0.1, 0.9])
    if n[t < lo].mean() > n[t > hi].mean():
        d = -d
    angle = np.rad2deg(np.arctan2(d[1], d[0])) % 180.0
    return float(angle), d


# --------------------------------------------------------------------------- #
# Fovea


def retina_ilm(
    frames: np.ndarray, masks: np.ndarray, threshold: float = 25, max_um: float = 450
) -> tuple[np.ndarray, np.ndarray]:
    """``(ilm, bm)`` per B-scan, each ``(T, W)``, as in ``estimate_fovea``."""
    bm, csi = clean_boundaries(*extract_boundaries_fast(masks))
    roi = rebuild_mask(bm - max_um / AXIAL_UM, bm, masks.shape[1])
    roi &= frames > threshold
    roi = keep_largest_connected_component(roi)
    ilm, _ = clean_boundaries(*extract_boundaries_fast(roi))
    return ilm, bm


def line_fovea(frame: np.ndarray, mask: np.ndarray) -> float:
    """A-scan index of the foveal pit on one (averaged) B-scan."""
    ilm, _ = retina_ilm(frame[None], mask[None])
    return float(estimate_fovea_from_ilm(ilm)[0, 0])


def volume_fovea(
    ilm: np.ndarray, bm: np.ndarray, sigma: float = 4.0, search: float = 0.3
) -> tuple[float, float, np.ndarray]:
    """``(row, col, thickness)``: minimum of the retinal thickness map."""
    th = (bm - ilm) * AXIAL_UM
    th = np.where(np.isfinite(th), th, np.nanmedian(th))
    th = gaussian_filter(th, sigma)
    H, W = th.shape
    r0, c0 = int(H * (0.5 - search / 2)), int(W * (0.5 - search / 2))
    sub = th[r0 : H - r0, c0 : W - c0]
    r, c = np.unravel_index(np.argmin(sub), sub.shape)
    return float(r + r0), float(c + c0), th


# --------------------------------------------------------------------------- #
# Placement of a line in the volume en-face frame


@dataclass
class LinePlacement:
    """Where the A-scans of one line fall in the volume's en-face frame."""

    center_rc: tuple[float, float]  # en-face (row, col) of the line's fovea
    direction: np.ndarray  # unit (dcol, drow) = fundus (dx, dy)
    fovea_ascan: float  # A-scan index of the fovea on the line
    px_um: tuple[float, float]  # en-face pixel size (row, col), um

    def s_um(self, ascan) -> np.ndarray:
        return (np.asarray(ascan, dtype=float) - self.fovea_ascan) * LINE_UM

    def rc(self, ascan) -> tuple[np.ndarray, np.ndarray]:
        s = self.s_um(ascan)
        row = self.center_rc[0] + s * self.direction[1] / self.px_um[0]
        col = self.center_rc[1] + s * self.direction[0] / self.px_um[1]
        return row, col

    def moved(self, drow=0.0, dcol=0.0, dtheta_deg=0.0) -> "LinePlacement":
        a = np.deg2rad(dtheta_deg)
        d = self.direction
        rot = np.array([np.cos(a) * d[0] - np.sin(a) * d[1], np.sin(a) * d[0] + np.cos(a) * d[1]])
        return LinePlacement(
            (self.center_rc[0] + drow, self.center_rc[1] + dcol), rot, self.fovea_ascan, self.px_um
        )


def shadow_profile(frame: np.ndarray, sigma: float = 40.0) -> np.ndarray:
    """High-passed A-scan mean of a B-scan: the retinal vessels and their shadows."""
    p = frame.astype(float).mean(axis=0)
    return p - gaussian_filter1d(p, sigma)


def enface_highpass(enface: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    e = enface.astype(float)
    return e - gaussian_filter(e, sigma)


def placement_score(
    placement: LinePlacement, profile: np.ndarray, enface_hp: np.ndarray, margin: int = 80
) -> float:
    """Correlation of the line's shadow profile with the en-face map sampled on it."""
    x = np.arange(profile.size)[margin:-margin]
    r, c = placement.rc(x)
    samp = map_coordinates(enface_hp, [r, c], order=1, mode="nearest")
    return float(np.corrcoef(samp, profile[margin:-margin])[0, 1])


def refine_placement(
    placement: LinePlacement,
    profile: np.ndarray,
    enface_hp: np.ndarray,
    max_shift: int = 12,
    step: int = 2,
    max_rot: float = 4.0,
    rot_step: float = 1.0,
) -> tuple[LinePlacement, float]:
    """Small grid search around ``placement`` maximising :func:`placement_score`."""
    best, best_r = placement, placement_score(placement, profile, enface_hp)
    shifts = np.arange(-max_shift, max_shift + 1, step)
    for dth in np.arange(-max_rot, max_rot + 1e-9, rot_step):
        for dr in shifts:
            for dc in shifts:
                p = placement.moved(dr, dc, dth)
                r = placement_score(p, profile, enface_hp)
                if r > best_r:
                    best, best_r = p, r
    return best, best_r


def refine_center(
    placements: list[LinePlacement],
    profiles: list[np.ndarray],
    enface_hp: np.ndarray,
    max_shift: int = 20,
    step: int = 1,
) -> tuple[tuple[float, float], float]:
    """One en-face shift shared by all lines, maximising their mean correlation."""
    best, best_r = (0.0, 0.0), -np.inf
    shifts = np.arange(-max_shift, max_shift + 1, step)
    for dr in shifts:
        for dc in shifts:
            r = np.mean(
                [placement_score(p.moved(dr, dc), prof, enface_hp) for p, prof in zip(placements, profiles)]
            )
            if r > best_r:
                best, best_r = (float(dr), float(dc)), float(r)
    return best, best_r


def flip_vote(
    lines: list[tuple[np.ndarray, float, np.ndarray]],
    enface: np.ndarray,
    fovea_rc: tuple[float, float],
    px_um: tuple[float, float],
) -> tuple[bool, float, float]:
    """Does the volume's B-scan axis run bottom-to-top on the fundus?"""
    H = enface.shape[0]
    scores = []
    for flip in (False, True):
        e_hp = enface_highpass(enface[::-1] if flip else enface)
        center = (H - 1 - fovea_rc[0], fovea_rc[1]) if flip else fovea_rc
        scores.append(
            sum(
                refine_placement(
                    LinePlacement(center, d, f, px_um), prof, e_hp, max_shift=20, step=4
                )[1]
                for d, f, prof in lines
            )
        )
    return bool(scores[1] > scores[0]), float(scores[0]), float(scores[1])


# --------------------------------------------------------------------------- #
# Local thickness change along a line


def local_thickness_change(
    disp: np.ndarray,
    ref_xy: np.ndarray,
    ref_mask: np.ndarray,
    normal_smooth_sigma: float = 0.0,
    normal_slope_window: int = 51,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-anchor version of ``measure_delta_ct_from_disp``."""
    ax, tx = AXIAL_PIXEL_SIZE_MM, TRANVERSAL_PIXEL_SIZE_MM
    rpe, csi = clean_boundaries(*extract_boundaries_fast(ref_mask[None].astype(bool)))
    rpe, csi = rpe[0], csi[0]
    W = ref_mask.shape[1]
    xi = np.clip(np.round(ref_xy[:, 0]).astype(int), 0, W - 1)
    yi = ref_xy[:, 1]
    d_csi = np.nan_to_num(np.abs(yi - csi[xi]), nan=np.inf)
    d_rpe = np.nan_to_num(np.abs(yi - rpe[xi]), nan=np.inf)
    is_csi = np.isfinite(d_csi) & (d_csi <= d_rpe)
    is_rpe = np.isfinite(d_rpe) & (d_rpe < d_csi)

    disp_mm = np.stack([disp[..., 0] * tx, disp[..., 1] * ax], axis=-1)
    if is_rpe.any():
        disp_mm = disp_mm - np.nanmean(disp_mm[:, is_rpe, :], axis=1)[:, None, :]
    nx_col, ny_col = _csi_unit_normal_mm(csi, ax, tx, normal_smooth_sigma, normal_slope_window)
    nx, ny = nx_col[xi], ny_col[xi]
    proj = -(disp_mm[..., 0] * nx[None] + disp_mm[..., 1] * ny[None])

    x = xi[is_csi]
    ct = (csi[x] - rpe[x]) * ax * np.abs(ny[is_csi])
    return x, proj[:, is_csi] * 1e3, ct * 1e3


def bin_columns(x: np.ndarray, values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """NaN-mean of ``values (..., N)`` over anchors whose A-scan ``x`` falls in each bin."""
    idx = np.digitize(x, edges) - 1
    out = np.full(values.shape[:-1] + (len(edges) - 1,), np.nan)
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.any():
            with np.errstate(all="ignore"):
                out[..., b] = np.nanmean(values[..., sel], axis=-1)
    return out


def first_harmonic_ptp(series: np.ndarray, axis: int = 0) -> np.ndarray:
    """Peak-to-peak of the fundamental of a one-period series (``2 * 2|X1| / T``)."""
    s = np.moveaxis(series, axis, 0)
    T = s.shape[0]
    X1 = np.fft.rfft(np.nan_to_num(s), axis=0)[1]
    return 4.0 * np.abs(X1) / T


# --------------------------------------------------------------------------- #
# Interpolation between the lines


class PolarInterpolator:
    """Interpolate values known on radial half-lines onto an en-face grid."""

    def __init__(
        self,
        center_rc: tuple[float, float],
        shape: tuple[int, int],
        px_um: tuple[float, float],
        ray_angles: np.ndarray,
        r_um: np.ndarray,
    ):
        self.order = np.argsort(ray_angles)
        a = np.asarray(ray_angles)[self.order]
        rows, cols = np.mgrid[: shape[0], : shape[1]]
        dy = (rows - center_rc[0]) * px_um[0]
        dx = (cols - center_rc[1]) * px_um[1]
        r = np.hypot(dx, dy)
        phi = np.arctan2(dy, dx) % (2 * np.pi)
        k1 = np.searchsorted(a, phi) % len(a)
        k0 = (k1 - 1) % len(a)
        gap = (a[k1] - a[k0]) % (2 * np.pi)
        self.w = ((phi - a[k0]) % (2 * np.pi)) / np.where(gap == 0, 1, gap)
        self.k0, self.k1 = k0, k1
        self.ir = np.interp(r, r_um, np.arange(len(r_um)), right=np.nan)
        self.r = r

    def __call__(self, values: np.ndarray) -> np.ndarray:
        """``values (..., n_rays, n_r)`` -> ``(..., H, W)``."""
        v = np.asarray(values, dtype=float)[..., self.order, :]
        n_r = v.shape[-1]
        ok = np.isfinite(self.ir)
        ir = np.where(ok, self.ir, 0)
        i0 = np.clip(np.floor(ir).astype(int), 0, n_r - 1)
        i1 = np.clip(i0 + 1, 0, n_r - 1)
        f = ir - i0

        def ray(k):
            a, b = v[..., k, i0], v[..., k, i1]
            return np.where(np.isfinite(b), a * (1 - f) + b * f, a)

        v0, v1 = ray(self.k0), ray(self.k1)
        w = self.w
        out = np.where(
            np.isfinite(v0) & np.isfinite(v1),
            v0 * (1 - w) + v1 * w,
            np.where(np.isfinite(v0), v0, v1),
        )
        return np.where(ok, out, np.nan)


def threshold_vessels(
    v,
    hi_p: float = 90.0,
    lo_p: float = 80.0,
    niblack_k: float | None = -0.1,
    window: tuple[int, int, int] = (31, 11, 31),
    close_radius: int = 1,
    smooth: float = 1.0,
    min_size: int = 1000,
) -> np.ndarray:
    """Binary vessels from a ChoroidVesselness, with the walls smoothed in 3D."""
    r, f, c = v.response, v.fused, v.choroid

    # Hysteresis: seeds above hi_p, grown through anything above lo_p
    hi, lo = np.percentile(r[c], [hi_p, lo_p])
    mask = apply_hysteresis_threshold(r, lo, hi) & c

    if niblack_k is not None:
        # A vessel voxel must also be dark for its neighbourhood
        m = uniform_filter(f, window)
        s = np.sqrt(np.maximum(uniform_filter(f**2, window) - m**2, 0))
        mask &= f < m + niblack_k * s

    if close_radius:
        mask = binary_closing(mask, ball(close_radius))
    if smooth:
        mask = gaussian_filter(mask.astype(np.float32), smooth) > 0.5

    mask &= c
    return remove_small_objects(mask, min_size=int(min_size))


# --------------------------------------------------------------------------- #
# Other vascular layers of the OCTA


def flow_slab(flow, top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """Mean flow between two ``(T, W)`` depth boundaries of a ``(T, W, depth)`` volume."""
    T, W, D = flow.shape
    z = np.arange(D)[None, :]
    out = np.zeros((T, W), np.float32)
    for t in range(T):
        m = (z >= top[t][:, None]) & (z < bottom[t][:, None])
        f = np.asarray(flow[t], np.float32)
        out[t] = (f * m).sum(1) / np.maximum(m.sum(1), 1)
    return out


def _destripe(enface: np.ndarray) -> np.ndarray:
    """Divide each B-scan by its median: removes the per-B-scan brightness stripes."""
    return enface / np.maximum(np.median(enface, axis=1, keepdims=True), 1e-6)


def vessel_binary(enface: np.ndarray, block: float = 8.0, k: float = 0.5) -> np.ndarray:
    """Vessel pixels of a plexus slab: brighter than their neighbourhood by ``k`` local SDs."""
    e = np.log(_destripe(enface) + 0.05)
    m = gaussian_filter(e, block)
    s = np.sqrt(np.maximum(gaussian_filter(e**2, block) - m**2, 1e-12))
    return e > m + k * s


def masked_smooth(values: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian mean of ``values`` over ``valid`` pixels only (normalised convolution)."""
    w = valid.astype(np.float32)
    return gaussian_filter(np.where(valid, values, 0) * w, sigma) / np.maximum(
        gaussian_filter(w, sigma), 1e-6
    )


def octa_layers(flow, ilm: np.ndarray, bm: np.ndarray, sigma: float = 6.0) -> dict:
    """En-face vascular maps of the retina and choriocapillaris, ``(T, W)`` each."""
    from ocularrigidity.segmentation.vessels.choroid3d import _fill_nan, shadow_mask

    ilm = gaussian_filter(_fill_nan(ilm), 2)
    bm = gaussian_filter(_fill_nan(bm), 2)
    th = bm - ilm
    svp = flow_slab(flow, ilm, ilm + 0.35 * th)
    dvp = flow_slab(flow, ilm + 0.35 * th, ilm + 0.6 * th)
    cc = flow_slab(flow, bm + 4, bm + 12)
    large = shadow_mask(flow_slab(flow, ilm, bm - 8))
    return dict(
        svp=svp,
        dvp=dvp,
        cc=cc,
        large=large,
        svp_density=gaussian_filter(vessel_binary(svp).astype(np.float32), sigma),
        dvp_density=masked_smooth(vessel_binary(dvp).astype(np.float32), ~large, sigma),
        large_vessels=gaussian_filter(large.astype(np.float32), sigma),
        cc_flow=masked_smooth(_destripe(cc), ~large, sigma),
    )

