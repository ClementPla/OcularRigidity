"""Data layer for the Streamlit cohort browser.

Pure (no Streamlit) helpers that turn a cohort experiments folder — the
per-method/phase outputs of ``scripts/pulsation/infer.py`` — into per-case
tables of the pulsatile metrics (ΔA, ΔCT) and the Friedenwald rigidity K,
merged with the clinical measurements.

Layout consumed (one ``<method>`` = ``<algo>_<phase>``, e.g. ``pca_iq``)::

    <root>/measures_<method>/<case>/deltaA_per_cycle.pkl   (ΔA, min area / cycle)
    <root>/deltaY_<method>.pkl                             (ΔCT proxy / cycle)

``<case>`` is ``<patient>/<date>/Rigidity/<eye>`` and matches the cleaned
``MeasureValue`` path in :func:`load_measurements`, which is how the clinical
IOP / OPA / AxialLength are joined.

This mirrors ``notebooks/cohort_analysis/replication_analysis.ipynb`` but with a
single, physically-consistent unit convention (see :func:`build_case_table`).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ocularrigidity.data.measurements.dataframe import load_measurements
from ocularrigidity.friedenwald import (
    cycle_amplitude,
    deltaA_to_deltaCT_mm,
    friedenwald_K,
)
from ocularrigidity.pipeline_config import FRIEDENWALD

# Numeric metric columns offered to the regression explorer.
METRIC_COLUMNS = [
    "deltaA",
    "deltaCT",
    "deltaCT_estimated",
    "minimal_area",
    "K_area",
    "K_thickness",
    "dV_uL_area",
    "dV_uL_thickness",
    "IOP",
    "OPA",
    "AxialLength",
]


def discover_methods(root: str | Path) -> list[str]:
    """Method suffixes with both a ``measures_<suffix>`` dir and ``deltaY_<suffix>.pkl``."""
    root = Path(root)
    methods = []
    for d in sorted(root.glob("measures_*")):
        if not d.is_dir():
            continue
        suffix = d.name[len("measures_") :]
        if (root / f"deltaY_{suffix}.pkl").exists():
            methods.append(suffix)
    return methods


def pretty_method(suffix: str) -> str:
    """``pca_peak_locked`` -> ``PCA · Peak-locked``."""
    algo, _, phase = suffix.partition("_")
    phase_label = {"iq": "IQ", "peak_locked": "Peak-locked"}.get(
        phase, phase.replace("_", " ").title()
    )
    return f"{algo.upper()} · {phase_label}"


def load_deltaA_per_cycle(root: str | Path, suffix: str) -> pd.DataFrame:
    """Per-cycle ΔA table: ``case_id, cycle, deltaA (px²), minimal_area (px²)``."""
    root = Path(root)
    measures_root = root / f"measures_{suffix}"
    rows = {"case_id": [], "cycle": [], "deltaA": [], "minimal_area": []}
    for f in measures_root.rglob("deltaA_per_cycle.pkl"):
        case_id = f.parent.relative_to(measures_root).as_posix()
        with open(f, "rb") as fh:
            data = pickle.load(fh)
        for i, (area, da) in enumerate(
            zip(data["minA_per_cycle"], data["deltaA_per_cycle"])
        ):
            rows["case_id"].append(case_id)
            rows["cycle"].append(i)
            rows["deltaA"].append(cycle_amplitude(da))
            rows["minimal_area"].append(float(area))
    return pd.DataFrame(rows)


def load_deltaCT_per_cycle(root: str | Path, suffix: str) -> pd.DataFrame:
    """Per-cycle ΔCT table: ``case_id, cycle, deltaCT (µm)``.

    ``deltaY`` is the peak-to-peak choroidal-thickness change in **pixels**; we
    convert to µm with the axial scale so it is comparable to the area-derived
    estimate.
    """
    df = pd.read_pickle(Path(root) / f"deltaY_{suffix}.pkl")
    out = df[["video", "cycle", "deltaY"]].rename(
        columns={"video": "case_id", "deltaY": "deltaCT"}
    )
    out["deltaCT"] = out["deltaCT"] * FRIEDENWALD.s_axial_mm_per_px * 1000.0
    return out


def build_case_table(
    root: str | Path, suffix: str, iop_instrument: str = "Pascal IOP"
) -> pd.DataFrame:
    """One row per case with the pulsatile metrics, clinical values and K.

    Per-cycle ΔA / ΔCT are collapsed to the median across cycles (matching the
    notebook). Two rigidity coefficients are computed:

    * ``K_area`` — from ΔA (px²) via the spherical-shell volume (friedenwald's
      documented default path).
    * ``K_thickness`` — from the measured ΔCT (µm).

    Units: ΔA in px², areas in px², ΔCT / ΔCT_estimated in µm, K in 1/µL.
    """
    root = Path(root)

    da = load_deltaA_per_cycle(root, suffix)
    ct = load_deltaCT_per_cycle(root, suffix)

    da_g = (
        da.groupby("case_id")
        .agg(deltaA=("deltaA", "median"), minimal_area=("minimal_area", "median"))
        .reset_index()
    )
    da_g["deltaCT_estimated"] = deltaA_to_deltaCT_mm(da_g["deltaA"]) * 1000.0
    ct_g = ct.groupby("case_id").agg(deltaCT=("deltaCT", "median")).reset_index()

    df = da_g.merge(ct_g, on="case_id", how="outer")

    # Identity columns derived from the case path (always present).
    parts = df["case_id"].str.split("/")
    df["PatientId"] = parts.str[0]
    df["Date"] = parts.str[1]
    df["Eye"] = parts.str[-1]

    # Clinical join (IOP / OPA / AxialLength) on the video path.
    meas = load_measurements(
        include_OPA=True, include_IOP=True, include_axial_length=True,
        iop_instrument=iop_instrument,
    )
    df = df.merge(
        meas[["MeasureValue", "OPA", "IOP", "AxialLength"]],
        left_on="case_id", right_on="MeasureValue", how="left",
    ).drop(columns=["MeasureValue"])

    # Friedenwald K, both ways. friedenwald_K adds dV_uL + K; rename to keep both.
    ka = friedenwald_K(df, from_area=True)
    df["dV_uL_area"], df["K_area"] = ka["dV_uL"], ka["K"]
    kt = friedenwald_K(df, from_area=False)
    df["dV_uL_thickness"], df["K_thickness"] = kt["dV_uL"], kt["K"]

    ordered = [
        "case_id", "PatientId", "Date", "Eye",
        "deltaA", "deltaCT", "deltaCT_estimated", "minimal_area",
        "IOP", "OPA", "AxialLength",
        "dV_uL_area", "K_area", "dV_uL_thickness", "K_thickness",
    ]
    return df[ordered].sort_values("case_id").reset_index(drop=True)


def regression_stats(df: pd.DataFrame, x: str, y: str) -> dict:
    """Pearson / Spearman / OLS line for ``y`` vs ``x`` on the finite rows."""
    d = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < 3:
        return {"n": len(d)}
    r, p = stats.pearsonr(d[x], d[y])
    rho, p_sp = stats.spearmanr(d[x], d[y])
    slope, intercept, _, _, _ = stats.linregress(d[x], d[y])
    return {
        "n": len(d),
        "pearson_r": r, "pearson_p": p,
        "spearman_rho": rho, "spearman_p": p_sp,
        "slope": slope, "intercept": intercept,
    }


def trim_outliers(df: pd.DataFrame, cols: list[str], quantile: float) -> pd.DataFrame:
    """Drop rows outside the ``[1-q, q]`` quantile of each column (notebook-style)."""
    out = df.copy()
    for c in cols:
        s = pd.to_numeric(out[c], errors="coerce")
        lo, hi = s.quantile([1.0 - quantile, quantile])
        out = out[(s >= lo) & (s <= hi)]
    return out
