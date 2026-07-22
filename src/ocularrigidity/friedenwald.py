import numpy as np

from ocularrigidity.pipeline_config import FRIEDENWALD, FriedenwaldConfig


def cycle_amplitude(delta_a) -> float:
    """Peak-to-peak area change (px²) of one cycle's ``deltaA`` time series.

    ``delta_a`` is one element of ``deltaA_per_cycle`` (shape ``(T,)``); NaNs
    from lost tracking are ignored.
    """
    delta_a = np.asarray(delta_a, dtype=float)
    return float(np.nanmax(delta_a) - np.nanmin(delta_a))


def inner_radius_mm(axial_length, cfg: FriedenwaldConfig = FRIEDENWALD):
    """Vitreous-chamber radius (mm) ≈ choroid-vitreous interface radius."""
    return (axial_length * cfg.vitreous_chamber_frac) / 2.0


def deltaA_to_deltaV_uL(deltaA_px2, axial_length, cfg: FriedenwaldConfig = FRIEDENWALD):
    """Convert a choroidal area change (px²) to a shell volume change (µL).

    1. mean axial thickness change: ``ΔA / w_px`` (px) → mm via the axial scale
       (the lateral scale and focal length cancel in the area/width ratio).
    2. spherical-shell volume between radius ``R`` and ``R + dt`` over the
       covered fraction of the sphere (mm³ == µL).
    """
    dt_mm = (deltaA_px2 / cfg.w_px) * cfg.s_axial_mm_per_px
    R = inner_radius_mm(axial_length, cfg)
    return (4.0 / 3.0) * np.pi * ((R + dt_mm) ** 3 - R**3) * cfg.surface_coverage


def deltaCT_to_deltaV_uL(
    deltaCT_um, axial_length, cfg: FriedenwaldConfig = FRIEDENWALD
):
    """Convert a choroidal thickness change (µm) to a shell volume change (µL).

    1. convert to mm.
    2. spherical-shell volume between radius ``R`` and ``R + dt`` over the
       covered fraction of the sphere (mm³ == µL).
    """
    dt_mm = deltaCT_um / 1000.0
    R = inner_radius_mm(axial_length, cfg)
    return (4.0 / 3.0) * np.pi * ((R + dt_mm) ** 3 - R**3) * cfg.surface_coverage


def deltaA_to_deltaCT_mm(deltaA_px2, cfg: FriedenwaldConfig = FRIEDENWALD):
    """Convert a choroidal area change (px²) to a shell thickness change (µm).

    1. mean axial thickness change: ``ΔA / w_px`` (px) → mm via the axial scale
       (the lateral scale and focal length cancel in the area/width ratio).
    2. convert to µm.
    """
    dt_mm = (deltaA_px2 / cfg.w_px) * cfg.s_axial_mm_per_px
    return dt_mm


def friedenwald_K(merged_df, cfg: FriedenwaldConfig = FRIEDENWALD, from_area=True):
    """Friedenwald rigidity K per row of ``merged_df``.

    Expects numeric columns ``IOP``, ``OPA``, ``AxialLength`` and ``deltaA``
    (peak-to-peak area change, px²). Returns a copy with ``dV_uL`` and ``K``
    added; invalid rows get ``K = NaN``.
    """
    g = merged_df.copy()

    if cfg.pressure_mode == "diastolic":
        P_d = g["IOP"]
        P_s = g["IOP"] + g["OPA"]
    elif cfg.pressure_mode == "mean":
        P_d = g["IOP"] - g["OPA"] / 2.0
        P_s = g["IOP"] + g["OPA"] / 2.0
    else:
        raise ValueError(cfg.pressure_mode)
    if from_area:
        g["dV_uL"] = deltaA_to_deltaV_uL(g["deltaA"], g["AxialLength"], cfg)
    else:
        g["dV_uL"] = deltaCT_to_deltaV_uL(g["deltaCT"], g["AxialLength"], cfg)

    bad = (g["dV_uL"] <= 0) | P_d.isna() | (P_d <= 0) | (P_s <= 0) | g["OPA"].isna()
    g["K"] = np.where(bad, np.nan, np.log10(P_s / P_d) / g["dV_uL"])

    return g
