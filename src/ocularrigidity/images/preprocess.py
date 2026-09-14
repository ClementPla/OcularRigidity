"""
Depth-dependent attenuation compensation for OCT B-scans.

Implements the model of Wang, Wei, Guo & Kang, "Optimized OCT-based depth-resolved
model for attenuation compensation using point-spread-function calibration",
Proc. SPIE 11635, 116350E (2021), Eqs. 2-5.

    Eq. 2  I_D(z) = A INT sqrt(R_R R_S(z)) exp(-4 (z_R - z)^2 dk^2) dz
           the coherence-gated source term (Gaussian axial PSF of width set by dk)

    Eq. 3  dL/dz = -2 mu(z) L(z) + I_D(z)
           Beer-Lambert with the source term retained

    Eq. 4  mu(n) ~ 1/(2a) [ i[n] / SUM_{n+1}^inf i[n]
                            - a I_D(n) exp(2 INT_0^n mu du)
                              / (L0 + a SUM_0^n I_D(n) exp(2 INT_0^n mu du)) ]
           first term = Vermeer et al. 2014; second term corrects for the source

    Eq. 5  di(n) = 2 mu(n) (1 - exp(-2 a SUM_0^n mu(n))) a SUM_0^inf I(n)
           the energy lost above depth n, added back to the measured signal

Two structural notes:

  * Eq. 4 is implicit -- mu appears inside exp(2 INT mu) -- so it is solved by
    fixed-point iteration seeded with the Vermeer term alone.
  * The Eq. 4 correction term depends only on the dimensionless ratio a*A/L0
    (`source_gain` below) once numerator and denominator are divided by L0. The
    paper bounds this term with R_R = R_S = N = 1, n = 0 rather than evaluating
    it, so the default here is 0.0, i.e. the term is dropped.

INPUT ASSUMPTION: `bscan` is *linear intensity*, shape (n_depth, n_ascans) by
default. Zeiss cube.bin 8-bit values are not linear intensity -- invert the
vendor display transform first, or every number below is meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

__all__ = [
    "SystemCalibration",
    "to_linear",
    "to_db",
    "delta_k_from_source",
    "confocal_function",
    "calibrate",
    "coherence_kernel",
    "source_term",
    "depth_resolved_mu",
    "compensate_eq5",
    "compensate_bscan",
]


# --------------------------------------------------------------------------- #
# Calibration container
# --------------------------------------------------------------------------- #
@dataclass
class SystemCalibration:
    """System parameters. All depths in mm *in tissue* (optical path / n).

    dz          : axial pixel size, `a` in Eqs. 4-5 [mm/pixel] (3.6 um in the paper)
    delta_k     : spectral bandwidth dk = 2 pi dlambda / lambda^2 [mm^-1]; sets the
                  width of the Gaussian coherence gate in Eq. 2. See
                  delta_k_from_source(). Only needed if source_gain != 0.
    z_focus     : depth of the beam focus from row 0 [mm]
    z_rayleigh  : apparent Rayleigh length in tissue [mm]
    rolloff     : measured sensitivity roll-off, one linear value per depth pixel,
                  normalised to 1 at row 0. None -> no roll-off correction.
    noise_floor : per-depth-pixel noise floor in the same raw units as the B-scan;
                  scalar or length-n_depth array.
    """

    dz: float
    delta_k: Optional[float] = None
    z_focus: Optional[float] = None
    z_rayleigh: Optional[float] = None
    rolloff: Optional[np.ndarray] = None
    noise_floor: Optional[np.ndarray | float] = None


def delta_k_from_source(lambda0_mm: float, delta_lambda_mm: float) -> float:
    """dk = 2 pi dlambda / lambda^2, in mm^-1.

    e.g. 1060 nm centre, 100 nm bandwidth -> delta_k_from_source(1.06e-3, 1e-4).
    Note the resulting Gaussian gate exp(-4 z^2 dk^2) has FWHM sqrt(ln2)/dk,
    which for typical SS-OCT sources is a few um -- comparable to or below the
    3.6 um pixel size, so the discretised kernel is only 1-3 taps wide.
    """
    return 2.0 * np.pi * delta_lambda_mm / lambda0_mm**2


# --------------------------------------------------------------------------- #
# Scale conversions
# --------------------------------------------------------------------------- #
def to_linear(
    bscan: np.ndarray, scale: Literal["linear", "db", "amplitude"]
) -> np.ndarray:
    b = np.asarray(bscan, dtype=np.float64)
    if scale == "linear":
        return b
    if scale == "db":
        return 10.0 ** (b / 10.0)
    if scale == "amplitude":
        return b**2
    raise ValueError(f"unknown scale {scale!r}")


def to_db(intensity: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(intensity, floor))


# --------------------------------------------------------------------------- #
# PSF calibration (roll-off + confocal), applied before Eq. 4
# --------------------------------------------------------------------------- #
def confocal_function(n_depth: int, cal: SystemCalibration) -> np.ndarray:
    """h(z) = 1 / ( ((z - z_f)/z_R)^2 + 1 ), the point-scanning confocal function.

    Distinct from the Eq. 2 coherence gate: this is the lateral/defocus envelope,
    the coherence gate is axial. Both are "PSF" terms; they are not the same term.
    """
    if cal.z_focus is None or cal.z_rayleigh is None:
        return np.ones(n_depth)
    z = np.arange(n_depth, dtype=np.float64) * cal.dz
    return 1.0 / (((z - cal.z_focus) / cal.z_rayleigh) ** 2 + 1.0)


def calibrate(
    bscan: np.ndarray,
    cal: SystemCalibration,
    noise_rows: Optional[slice] = None,
    min_intensity: float = 1e-12,
) -> np.ndarray:
    """Subtract the noise floor, then divide out roll-off and confocal terms.

    Order matters: the noise floor is in raw detector units, so removing it after
    the division would amplify it by exactly the factor being corrected.
    """
    I = np.asarray(bscan, dtype=np.float64)
    n_depth = I.shape[0]

    if cal.noise_floor is not None:
        nf = np.asarray(cal.noise_floor, dtype=np.float64)
        nf = np.full(n_depth, float(nf)) if nf.ndim == 0 else nf
    elif noise_rows is not None:
        nf = np.full(n_depth, float(np.median(I[noise_rows])))
    else:
        nf = np.zeros(n_depth)
    I = I - nf[:, None]

    h = confocal_function(n_depth, cal)
    if cal.rolloff is not None:
        r = np.asarray(cal.rolloff, dtype=np.float64)
        if r.shape[0] != n_depth:
            raise ValueError(
                f"rolloff has {r.shape[0]} samples, B-scan has {n_depth} rows"
            )
        h = h * r
    I = I / h[:, None]

    return np.maximum(I, min_intensity)


# --------------------------------------------------------------------------- #
# Eq. 2: coherence-gated source term
# --------------------------------------------------------------------------- #
def coherence_kernel(dz: float, delta_k: float, n_sigma: float = 4.0) -> np.ndarray:
    """Discretised Gaussian gate exp(-4 z^2 dk^2) from Eq. 2, normalised to sum 1.

    Width sigma = 1/(2 sqrt(2) dk). Returns an odd-length 1-D kernel; for typical
    parameters this is a single tap, in which case Eq. 2 reduces to
    I_D(n) ~ A sqrt(R_R R_S(n)).
    """
    sigma = 1.0 / (2.0 * np.sqrt(2.0) * delta_k)
    half = max(int(np.ceil(n_sigma * sigma / dz)), 0)
    d = np.arange(-half, half + 1, dtype=np.float64) * dz
    k = np.exp(-4.0 * d**2 * delta_k**2)
    return k / k.sum()


def source_term(
    I: np.ndarray, cal: SystemCalibration, r_ref: float = 1.0
) -> np.ndarray:
    """I_D(n) up to the scalar A, following Eq. 2 with R_S taken from the data.

    Uses sqrt(R_R * R_S(z)) with R_S the calibrated intensity normalised so its
    maximum is 1 (matching the paper's R_S = 1 upper-bound convention). The
    unknown scalar A is folded into `source_gain` in depth_resolved_mu().
    """
    if cal.delta_k is None:
        raise ValueError("cal.delta_k is required to evaluate the Eq. 2 source term")
    R_s = I / np.maximum(I.max(), 1e-300)
    amp = np.sqrt(r_ref * R_s)
    k = coherence_kernel(cal.dz, cal.delta_k)
    if k.size == 1:
        return amp
    pad = k.size // 2
    padded = np.pad(amp, ((pad, pad), (0, 0)), mode="edge")
    return np.apply_along_axis(lambda col: np.convolve(col, k, mode="valid"), 0, padded)


# --------------------------------------------------------------------------- #
# Eq. 4: depth-resolved attenuation coefficient
# --------------------------------------------------------------------------- #
def depth_resolved_mu(
    I: np.ndarray,
    cal: SystemCalibration,
    source_gain: float = 0.0,
    n_iter: int = 8,
    mu_max: float = 1e3,
    eps: float = 1e-30,
) -> np.ndarray:
    """Eq. 4. Returns mu [mm^-1], same shape as I.

    source_gain = a * A / L0, the only free parameter left in the Eq. 4
    correction term after dividing through by L0. With source_gain = 0 this is
    exactly Vermeer's estimator, which is what the paper's upper-bound argument
    justifies when the bound is small. Non-zero values require cal.delta_k.

    The correction term is *subtracted*, so it lowers mu; it is clipped at 0 to
    keep mu physical, and mu is capped at mu_max.

    Solved by fixed-point iteration since exp(2 INT_0^n mu du) depends on mu.
    """
    a = cal.dz
    S_below = np.cumsum(I[::-1, :], axis=0)[::-1, :] - I  # SUM_{m>n} i[m]
    vermeer = I / (S_below + eps)  # first term of Eq. 4

    mu = np.clip(vermeer / (2.0 * a), 0.0, mu_max)
    if source_gain == 0.0:
        return mu

    I_D = source_term(I, cal)  # Eq. 2, up to A
    for _ in range(n_iter):
        # exp(2 INT_0^n mu du), midpoint rule to avoid double-counting pixel n
        gain = np.exp(np.minimum(2.0 * a * (np.cumsum(mu, axis=0) - 0.5 * mu), 700.0))
        w = source_gain * I_D * gain  # (a A / L0) I_D exp(...)
        correction = w / (1.0 + np.cumsum(w, axis=0))  # L0 divided out
        mu = np.clip((vermeer - correction) / (2.0 * a), 0.0, mu_max)
    return mu


# --------------------------------------------------------------------------- #
# Eq. 5: compensation
# --------------------------------------------------------------------------- #
def compensate_eq5(I: np.ndarray, mu: np.ndarray, cal: SystemCalibration) -> np.ndarray:
    """i_c(n) = i(n) + di(n), with di from Eq. 5 (rho = 1).

        di(n) = 2 mu(n) (1 - exp(-2 a SUM_0^n mu)) a SUM_0^inf I(n)

    This adds back the energy already absorbed/scattered above depth n. Because
    i[n] = 2 a mu[n] SUM_{m>n} i[m], the compensated line equals
    2 a mu[n] (SUM_{m>n} i + (1 - T[n]) SUM_all i), which is depth-independent
    for a homogeneous sample -- see the self-test in __main__.
    """
    a = cal.dz
    E_total = a * I.sum(axis=0, keepdims=True)  # a SUM_0^inf I(n)
    lost = 1.0 - np.exp(-2.0 * a * np.cumsum(mu, axis=0))  # fraction lost by n
    return I + 2.0 * mu * lost * E_total


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #
def compensate_bscan(
    bscan: np.ndarray,
    cal: SystemCalibration,
    scale: Literal["linear", "db", "amplitude"] = "linear",
    axis: int = 0,
    noise_rows: Optional[slice] = None,
    source_gain: float = 0.0,
    n_iter: int = 8,
    return_db: bool = True,
) -> dict[str, np.ndarray]:
    """Full pipeline on one B-scan: calibrate -> Eq. 4 -> Eq. 5.

    axis=0 means depth runs down the rows; pass axis=1 for (n_ascans, n_depth)
    and the arrays are transposed internally and on output.

    Returns dict with 'mu' [mm^-1], 'compensated', 'calibrated', and (optionally)
    'compensated_db'.
    """
    b = np.asarray(bscan)
    if b.ndim != 2:
        raise ValueError(f"expected a 2-D B-scan, got shape {b.shape}")
    if axis == 1:
        b = b.T
    elif axis != 0:
        raise ValueError("axis must be 0 or 1")

    I = calibrate(to_linear(b, scale), cal, noise_rows=noise_rows)
    mu = depth_resolved_mu(I, cal, source_gain=source_gain, n_iter=n_iter)
    Ic = compensate_eq5(I, mu, cal)

    out = {"mu": mu, "compensated": Ic, "calibrated": I}
    if return_db:
        out["compensated_db"] = to_db(Ic)
    if axis == 1:
        out = {k: v.T for k, v in out.items()}
    return out


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_depth, n_a = 512, 64
    a = 3.6e-3  # mm, the paper's axial pixel size

    z = np.arange(n_depth) * a
    cal = SystemCalibration(
        dz=a,
        delta_k=delta_k_from_source(1.06e-3, 1e-4),
        z_focus=0.8,
        z_rayleigh=0.6,
        rolloff=np.exp(-((z / 2.5) ** 2)),
        noise_floor=2e-5,
    )

    def synth(mu_true, rho=0.6, noise=2e-7):
        L = np.exp(-2.0 * np.cumsum(mu_true) * a)
        ideal = 2.0 * rho * mu_true * L
        h = confocal_function(n_depth, cal) * cal.rolloff
        m = (ideal * h)[:, None] * np.ones((1, n_a))
        return m + 2e-5 + rng.normal(0, noise, m.shape), ideal

    # 1. homogeneous sample: Eq. 5 should flatten the exponential decay
    mu_hom = np.full(n_depth, 4.0)
    meas, _ = synth(mu_hom)
    res = compensate_bscan(meas, cal)
    prof = res["compensated"].mean(axis=1)[10:400]
    print(
        f"[homogeneous] mu recovered = {res['mu'][10:400].mean():.2f} mm^-1 (true 4.00)"
    )
    print(
        f"[homogeneous] compensated profile spread = "
        f"{prof.std() / prof.mean() * 100:.1f}% of mean "
        f"(raw decay spans {10 * np.log10(meas[10].mean() / meas[400].mean()):.0f} dB)"
    )

    # 2. layered sample
    mu_lay = np.piecewise(
        z, [z < 0.5, (z >= 0.5) & (z < 1.2), z >= 1.2], [3.0, 6.0, 2.0]
    )
    meas, _ = synth(mu_lay)
    res = compensate_bscan(meas, cal)
    mu_est = res["mu"].mean(axis=1)
    for lo, hi, truth in [(30, 120, 3.0), (150, 290, 6.0), (330, 430, 2.0)]:
        print(
            f"[layered] z=[{z[lo]:.2f},{z[hi]:.2f}] mm  true={truth:.1f}  "
            f"estimated={mu_est[lo:hi].mean():.2f} mm^-1"
        )

    # 3. effect of the Eq. 4 source-term correction
    for g in (0.0, 1e-4, 1e-3):
        m = depth_resolved_mu(res["calibrated"], cal, source_gain=g)
        print(
            f"[Eq.4 term] source_gain={g:.0e}  mean mu = {m[30:430].mean():.3f} mm^-1"
        )
