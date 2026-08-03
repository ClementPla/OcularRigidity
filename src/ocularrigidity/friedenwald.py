import numpy as np

from ocularrigidity.pipeline_config import FRIEDENWALD, FriedenwaldConfig


def cycle_amplitude(delta_a) -> float:
    """Peak-to-peak area change (px²) of one cycle's ``deltaA`` time series.

    ``delta_a`` is one element of ``deltaA_per_cycle`` (shape ``(T,)``); NaNs
    from lost tracking are ignored.
    """
    delta_a = np.asarray(delta_a, dtype=float)
    return float(np.nanmax(delta_a) - np.nanmin(delta_a))


def _pressures(iop, opa, cfg: FriedenwaldConfig = FRIEDENWALD):
    """Diastolic and systolic IOP (mmHg) under the configured convention.

    Works elementwise, so ``iop``/``opa`` may be scalars or pandas Series.
    """
    if cfg.pressure_mode == "diastolic":
        return iop, iop + opa
    elif cfg.pressure_mode == "mean":
        return iop - opa / 2.0, iop + opa / 2.0
    else:
        raise ValueError(cfg.pressure_mode)


def inner_radius_mm(
    axial_length, choroidal_thickness=None, cfg: FriedenwaldConfig = FRIEDENWALD
):
    """Vitreous-chamber radius (mm) ≈ choroid-vitreous interface radius.

    ``choroidal_thickness`` (mm) is the measured absolute choroidal thickness;
    when given, the shell sits on top of the choroid rather than on the bare
    vitreous chamber. Omitting it keeps the plain axial-fraction radius.
    """
    r = (axial_length * cfg.vitreous_chamber_frac) / 2.0
    if choroidal_thickness is not None:
        r = r + choroidal_thickness
    return r


def deltaA_to_deltaV_uL(
    deltaA_px2,
    axial_length,
    choroidal_thickness_mm=None,
    cfg: FriedenwaldConfig = FRIEDENWALD,
):
    """Convert a choroidal area change (px²) to a shell volume change (µL).

    1. mean axial thickness change: ``ΔA / w_px`` (px) → mm via the axial scale
       (the lateral scale and focal length cancel in the area/width ratio).
    2. spherical-shell volume between radius ``R`` and ``R + dt`` over the
       covered fraction of the sphere (mm³ == µL).
    """
    dt_mm = (deltaA_px2 / cfg.w_px) * cfg.s_axial_mm_per_px
    R = inner_radius_mm(axial_length, choroidal_thickness_mm, cfg=cfg)
    return (4.0 / 3.0) * np.pi * ((R + dt_mm) ** 3 - R**3) * cfg.surface_coverage


def deltaCT_to_deltaV_uL(
    deltaCT_um,
    axial_length,
    choroidal_thickness_mm=None,
    cfg: FriedenwaldConfig = FRIEDENWALD,
):
    """Convert a choroidal thickness change (µm) to a shell volume change (µL).

    1. convert to mm.
    2. spherical-shell volume between radius ``R`` and ``R + dt`` over the
       covered fraction of the sphere (mm³ == µL).

    ``choroidal_thickness_mm`` is the measured absolute choroidal thickness (mm,
    e.g. ``DeltaCTResult.min_ct_mm``); when given, the shell sits on top of the
    choroid rather than on the bare vitreous chamber.
    """
    dt_mm = deltaCT_um / 1000.0
    R = inner_radius_mm(axial_length, choroidal_thickness_mm, cfg=cfg)
    return (4.0 / 3.0) * np.pi * ((R + dt_mm) ** 3 - R**3) * cfg.surface_coverage


def K_from_deltaCT_mm(
    deltaCT_mm,
    axial_length,
    iop,
    opa,
    choroidal_thickness_mm=None,
    cfg: FriedenwaldConfig = FRIEDENWALD,
):
    """Friedenwald K (1/µL) from a peak-to-peak thickness change (mm).

    Elementwise, so all arguments may be scalars or pandas Series / arrays.
    Unusable inputs (non-positive volume or pressures) give NaN instead of
    raising, which keeps cohort loops from dying on a single bad eye.

    ``choroidal_thickness_mm`` seats the shell on top of the choroid; see
    :func:`inner_radius_mm`.
    """
    dV = deltaCT_to_deltaV_uL(
        np.asarray(deltaCT_mm, dtype=float) * 1000.0,
        axial_length,
        choroidal_thickness_mm,
        cfg=cfg,
    )
    P_d, P_s = _pressures(iop, opa, cfg)
    P_d = np.asarray(P_d, dtype=float)
    P_s = np.asarray(P_s, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        K = np.where(
            (dV > 0) & (P_d > 0) & (P_s > 0), np.log10(P_s / P_d) / dV, np.nan
        )
    return K[()] if K.ndim == 0 else K


def deltaA_to_deltaCT_mm(deltaA_px2, cfg: FriedenwaldConfig = FRIEDENWALD):
    """Convert a choroidal area change (px²) to a shell thickness change (µm).

    1. mean axial thickness change: ``ΔA / w_px`` (px) → mm via the axial scale
       (the lateral scale and focal length cancel in the area/width ratio).
    2. convert to µm.
    """
    dt_mm = (deltaA_px2 / cfg.w_px) * cfg.s_axial_mm_per_px
    return dt_mm


def friedenwald_K_from_deltaCT(
    deltaCT_um,
    IOP,
    OPA,
    axial_length,
    choroidal_thickness_mm=None,
    cfg: FriedenwaldConfig = FRIEDENWALD,
) -> float:
    """Scalar Friedenwald rigidity K from a single *measured* ΔCT (µm).

    Scalar counterpart to :func:`K_from_deltaCT_mm` (which takes mm and is
    elementwise). Returns ``nan`` for non-physical inputs.
    """
    P_d, P_s = _pressures(IOP, OPA, cfg)
    dV = deltaCT_to_deltaV_uL(deltaCT_um, axial_length, choroidal_thickness_mm, cfg=cfg)
    invalid = (
        not np.isfinite(dV)
        or dV <= 0
        or not np.isfinite(P_d)
        or P_d <= 0
        or P_s <= 0
        or not np.isfinite(OPA)
    )
    if invalid:
        return float("nan")
    return float(np.log10(P_s / P_d) / dV)


def friedenwald_K(
    merged_df,
    cfg: FriedenwaldConfig = FRIEDENWALD,
    from_area=True,
    thickness_col: str | None = None,
):
    """Friedenwald rigidity K per row of ``merged_df``.

    Expects numeric columns ``IOP``, ``OPA``, ``AxialLength`` and ``deltaA``
    (peak-to-peak area change, px²) — or ``deltaCT`` (µm) when ``from_area`` is
    False. ``thickness_col`` names a column of measured absolute choroidal
    thickness (mm) to add to the inner radius. Returns a copy with ``dV_uL`` and
    ``K`` added; invalid rows get ``K = NaN``.
    """
    g = merged_df.copy()

    P_d, P_s = _pressures(g["IOP"], g["OPA"], cfg)
    ct_mm = g[thickness_col] if thickness_col is not None else None
    if from_area:
        g["dV_uL"] = deltaA_to_deltaV_uL(g["deltaA"], g["AxialLength"], ct_mm, cfg=cfg)
    else:
        g["dV_uL"] = deltaCT_to_deltaV_uL(g["deltaCT"], g["AxialLength"], ct_mm, cfg=cfg)

    bad = (g["dV_uL"] <= 0) | P_d.isna() | (P_d <= 0) | (P_s <= 0) | g["OPA"].isna()
    g["K"] = np.where(bad, np.nan, np.log10(P_s / P_d) / g["dV_uL"])

    return g
