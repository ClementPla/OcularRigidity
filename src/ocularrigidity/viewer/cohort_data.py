"""Data layer for the Streamlit cohort browser.

Pure (no Streamlit) helpers that turn a cohort experiments folder — the
per-method/phase outputs of ``scripts/pulsation/infer.py`` and
``scripts/cohort_analysis/extract_deltaA.py`` — into per-case tables of the
pulsatile metrics (ΔA, ΔCT, min CT) and the Friedenwald rigidity K, merged with
the clinical measurements.

Layout consumed (one ``<method>`` = ``<algo>_<phase>``, e.g. ``pca_iq``)::

    <root>/measures_<method>/<case>/deltaA_per_cycle.pkl   (ΔA + boundary displacements)
    <root>/measures_<method>/<case>/segmented_cycles.npz   (choroid masks)

``<case>`` is ``<patient>/<date>/Rigidity/<eye>`` and matches the cleaned
``MeasureValue`` path in :func:`load_measurements`, which is how the clinical
IOP / OPA / AxialLength / HR are joined.

This mirrors ``notebooks/cohort_analysis/prospective.ipynb``: ΔCT is *measured*
by tracking the choroid-sclera interface (:func:`measure_delta_ct_from_disp`)
rather than read from the legacy ``deltaY_<method>.pkl`` harmonic fit, which
also yields the absolute thickness ``minCT`` and the unit-free
``RelativeGrowth = ΔCT / minCT``.
"""

from __future__ import annotations

import json
import pickle
import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from ocularrigidity.consts import (
    CLINICAL_VALUES_PATH,
    MEASUREMENTS_PATH,
    QC_ERRORS_PATH,
)
from ocularrigidity.data.io import load_mask
from ocularrigidity.data.measurements.dataframe import load_measurements
from ocularrigidity.data.measurements.studies import Study
from ocularrigidity.friedenwald import (
    cycle_amplitude,
    deltaA_to_deltaCT_mm,
    friedenwald_K,
)
from ocularrigidity.pipeline_config import DELTA_A
from ocularrigidity.segmentation.closing_structures import trim_choroid
from ocularrigidity.thickness.delta import measure_delta_ct_from_disp

# Columns trimmed off each side of the mask before measuring ΔCT (the edges of
# the B-scan are unreliable). Matches the notebooks.
DELTA_CT_TRIM = 100

# Numeric metric columns offered to the regression / longitudinal explorers.
METRIC_COLUMNS = [
    "deltaA",
    "minimal_area",
    "deltaCT_estimated",
    "deltaCT",
    "minCT",
    "RelativeGrowth",
    "K",
    "K_area",
    "dV_uL",
    "dV_uL_area",
    "IOP",
    "OPA",
    "AxialLength",
    "HR",
]

# Metrics that make sense as the "probed" quantity in a longitudinal analysis.
PROBE_COLUMNS = ["K", "RelativeGrowth", "deltaCT", "minCT", "deltaA", "K_area"]

# Clinical measures tracked over time: offered first in the pickers and selected
# by default. The rest of what the two clinical databases hold (sector RNFL, the
# per-quadrant BMO-MRW, VFI, blood pressure…) stays available behind them.
CLINICAL_MEASURES = [
    "Global RNFL Thickness",
    "Visual Acuity",
    "Pascal IOP",
    "MD",
    "PSD",
    "GCL Volume",
    "Steepest RNFL Thickness",
    "Steepest BMO MRW",
]


def discover_methods(root: str | Path) -> list[str]:
    """Method suffixes with a ``measures_<suffix>`` directory under ``root``."""
    root = Path(root)
    return sorted(
        d.name[len("measures_") :] for d in root.glob("measures_*") if d.is_dir()
    )


def pretty_method(suffix: str) -> str:
    """``pca_peak_locked`` -> ``PCA · Peak-locked``."""
    algo, _, phase = suffix.partition("_")
    phase_label = {"iq": "IQ", "peak_locked": "Peak-locked"}.get(
        phase, phase.replace("_", " ").title()
    )
    return f"{algo.upper()} · {phase_label}"


def load_excluded_cases(path: str | Path = QC_ERRORS_PATH) -> set[str]:
    """Case ids QC-rejected in the gif viewer (``errors.json``).

    Entries are gif file names (``<patient>_<date>_Rigidity_<eye>.gif``); they
    map back to the ``<patient>/<date>/Rigidity/<eye>`` case id.
    """
    path = Path(path)
    if not path.exists():
        return set()
    with open(path) as fh:
        return {e.replace("_", "/").replace(".gif", "") for e in json.load(fh)}


def load_deltaA_per_cycle(root: str | Path, suffix: str) -> pd.DataFrame:
    """Per-cycle ΔA table: ``case_id, cycle, deltaA (px²), minimal_area (px²)``."""
    root = Path(root)
    measures_root = root / f"measures_{suffix}"
    rows = {"case_id": [], "cycle": [], "deltaA": [], "minimal_area": []}
    for f in sorted(measures_root.rglob("deltaA_per_cycle.pkl")):
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


def deltaCT_cache_path(root: str | Path, suffix: str, trim: int) -> Path:
    """Where :func:`load_deltaCT_per_cycle` persists its (slow) result."""
    return Path(root) / f"deltaCT_measured_{suffix}_trim{trim}.parquet"


def load_deltaCT_per_cycle(
    root: str | Path,
    suffix: str,
    trim: int = DELTA_CT_TRIM,
    n_cycles: int = DELTA_A.n_cycles,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Per-cycle *measured* ΔCT table.

    For every case, the boundary displacements stored by ``extract_deltaA.py``
    are replayed against the reference-frame choroid mask of each cycle to get
    the peak-to-peak thickness change (see
    :func:`ocularrigidity.thickness.delta.measure_delta_ct_from_disp`).

    Returns ``case_id, cycle, deltaCT (µm), minCT (µm), RelativeGrowth`` — the
    last being ``ΔCT / minCT``, i.e. the pulsatile thickening as a fraction of
    the choroid itself. Cases whose cycle fails to measure are simply absent.

    This walks every mask on disk, so it is slow (minutes for a full cohort);
    the result is cached to a parquet next to the experiment root and reused
    unless ``use_cache`` is False.
    """
    root = Path(root)
    cache = deltaCT_cache_path(root, suffix, trim)
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    measures_root = root / f"measures_{suffix}"
    rows = []
    for f in sorted(measures_root.rglob("deltaA_per_cycle.pkl")):
        case_id = f.parent.relative_to(measures_root).as_posix()
        mask_file = f.parent / "segmented_cycles.npz"
        if not mask_file.exists():
            continue
        with open(f, "rb") as fh:
            data = pickle.load(fh)
        if "displacement_per_cycle" not in data:  # pre-displacement pickle
            continue

        masks = trim_choroid(load_mask(mask_file), trim)
        frames_per_cycle = masks.shape[0] // n_cycles
        for i in range(n_cycles):
            # measure_delta_ct_from_disp only reads the reference frame's mask,
            # so hand it that single frame rather than the whole cycle.
            ref_mask = masks[i * frames_per_cycle : i * frames_per_cycle + 1]
            try:
                res = measure_delta_ct_from_disp(
                    data["displacement_per_cycle"][i],
                    data["reference_coordinates_per_cycle"][i],
                    ref_mask,
                    reference_frame_idx=0,
                )
            except (ValueError, IndexError):
                continue
            rows.append(
                {
                    "case_id": case_id,
                    "cycle": i,
                    "deltaCT": res.deltaCT_um,
                    "minCT": res.min_ct_mm * 1000.0,
                    "RelativeGrowth": (
                        res.deltaCT_mm / res.min_ct_mm if res.min_ct_mm else np.nan
                    ),
                    "rpe_residual_um": res.rpe_residual_um,
                }
            )

    df = pd.DataFrame(
        rows,
        columns=[
            "case_id",
            "cycle",
            "deltaCT",
            "minCT",
            "RelativeGrowth",
            "rpe_residual_um",
        ],
    )
    if use_cache:
        df.to_parquet(cache, index=False)
    return df


def build_case_table(
    root: str | Path,
    suffix: str,
    iop_instrument: str = "Pascal IOP",
    study: Study | None = None,
    excluded_cases: Iterable[str] | None = None,
    trim: int = DELTA_CT_TRIM,
) -> pd.DataFrame:
    """One row per case with the pulsatile metrics, clinical values and K.

    Per-cycle ΔA / ΔCT / minCT are collapsed to the median across cycles.
    Two rigidity coefficients are computed:

    * ``K`` — from the *measured* ΔCT, with the shell sitting on top of the
      measured choroid (inner radius ``R + minCT``). This is the notebook's K.
    * ``K_area`` — from ΔA (px²) via the same spherical-shell volume but the
      bare vitreous-chamber radius, so it stays defined for the cases whose ΔCT
      could not be measured.

    Units: ΔA and areas in px², ΔCT / ΔCT_estimated / minCT in µm, K in 1/µL.
    ``study`` restricts the cohort (e.g. ``Study.PROSPECTIVE``);
    ``excluded_cases`` drops QC-rejected case ids.
    """
    root = Path(root)

    da = load_deltaA_per_cycle(root, suffix)
    ct = load_deltaCT_per_cycle(root, suffix, trim=trim)

    da_g = (
        da.groupby("case_id")
        .agg(deltaA=("deltaA", "median"), minimal_area=("minimal_area", "median"))
        .reset_index()
    )
    da_g["deltaCT_estimated"] = deltaA_to_deltaCT_mm(da_g["deltaA"]) * 1000.0
    ct_g = (
        ct.groupby("case_id")
        .agg(
            deltaCT=("deltaCT", "median"),
            minCT=("minCT", "median"),
            RelativeGrowth=("RelativeGrowth", "median"),
        )
        .reset_index()
    )

    df = da_g.merge(ct_g, on="case_id", how="outer")
    if excluded_cases:
        df = df[~df["case_id"].isin(set(excluded_cases))]

    # Clinical join (IOP / OPA / AxialLength / HR) on the video path. The study
    # filter lives in load_measurements, so an inner join applies it here.
    meas = load_measurements(
        include_OPA=True,
        include_IOP=True,
        include_axial_length=True,
        include_HR=True,
        which_study=study,
        iop_instrument=iop_instrument,
    )
    meas["PatientId"] = meas["PatientId"].astype(str)
    cols = [
        "MeasureValue",
        "PatientId",
        "Date",
        "Eye",
        "OPA",
        "IOP",
        "AxialLength",
        "HR",
    ]
    df = df.merge(
        meas[cols],
        left_on="case_id",
        right_on="MeasureValue",
        how="inner" if study is not None else "left",
    ).drop(columns=["MeasureValue"])

    # Identity: prefer the DB's own columns — the *path* spells a repeated
    # acquisition of the same eye "OD1", which would not join back onto any
    # clinical table. Fall back to the path for cases absent from the DB.
    parts = df["case_id"].str.split("/")
    df["PatientId"] = df["PatientId"].fillna(parts.str[0])
    df["Date"] = df["Date"].fillna(parts.str[1])
    df["Eye"] = df["Eye"].fillna(parts.str[-1])

    # Friedenwald K, both ways. friedenwald_K adds dV_uL + K; rename to keep both.
    df["minCT_mm"] = df["minCT"] / 1000.0
    ka = friedenwald_K(df, from_area=True)
    df["dV_uL_area"], df["K_area"] = ka["dV_uL"], ka["K"]
    kt = friedenwald_K(df, from_area=False, thickness_col="minCT_mm")
    df["dV_uL"], df["K"] = kt["dV_uL"], kt["K"]

    ordered = [
        "case_id",
        "PatientId",
        "Date",
        "Eye",
        "deltaA",
        "minimal_area",
        "deltaCT_estimated",
        "deltaCT",
        "minCT",
        "RelativeGrowth",
        "IOP",
        "OPA",
        "AxialLength",
        "HR",
        "dV_uL_area",
        "K_area",
        "dV_uL",
        "K",
    ]
    return df[ordered].sort_values("case_id").reset_index(drop=True)


# Identity columns of the wide ``ClinicalValues`` table; everything else in it is
# a candidate measure. ``File`` is the MRN that names the video folder, while
# ``PatientId`` is the internal id the Biomechanics tables (and hence the case
# table) key on — that is the one to join with.
_CLINICAL_VALUES_IDS = ["Id", "Cohort", "PatientId", "File", "Eye", "Date"]

# The same quantity, spelled differently in the two databases.
_CLINICAL_VALUES_RENAME = {"G RNFL Thickness": "Global RNFL Thickness"}

_MEASURE_KEYS = ["PatientId", "Eye", "YearMonth", "MeasureName_y"]


def load_clinical_values_long(
    db_path: str | Path = CLINICAL_VALUES_PATH,
) -> pd.DataFrame:
    """The wide ``ClinicalValues`` table, melted to one row per (visit × measure).

    Returns ``PatientId, Eye, YearMonth, MeasureName_y, MeasureValue_y`` — the
    same long shape the Biomechanics ``Measurements`` table already has, so the
    two sources can simply be stacked.

    Every non-identity column that holds at least one number is taken as a
    measure; that keeps the sector RNFL / BMO-MRW / steepest values and drops the
    purely categorical ones (``Sex``, ``Ethnicity``, the ``… Quadrant`` labels),
    which no downstream regression could consume anyway.

    The table repeats some visits verbatim, so values are collapsed to one per
    (patient, eye, month, measure) — the month being the unit every downstream
    analysis works in.
    """
    with sqlite3.connect(db_path) as con:
        wide = pd.read_sql_query("SELECT * FROM ClinicalValues", con)

    measure_cols = [
        c
        for c in wide.columns
        if c not in _CLINICAL_VALUES_IDS
        and pd.to_numeric(wide[c], errors="coerce").notna().any()
    ]
    long = wide.melt(
        id_vars=["PatientId", "Eye", "Date"],
        value_vars=measure_cols,
        var_name="MeasureName_y",
        value_name="MeasureValue_y",
    )
    long["MeasureName_y"] = long["MeasureName_y"].replace(_CLINICAL_VALUES_RENAME)
    long["PatientId"] = long["PatientId"].astype(str)
    long["YearMonth"] = long["Date"].str[:7]
    long["MeasureValue_y"] = pd.to_numeric(long["MeasureValue_y"], errors="coerce")
    long = long.dropna(subset=["MeasureValue_y"])
    return (
        long.groupby(_MEASURE_KEYS, as_index=False)["MeasureValue_y"]
        .mean()
        .reindex(columns=_MEASURE_KEYS + ["MeasureValue_y"])
    )


def load_clinical_long(
    case_table: pd.DataFrame,
    db_path: str | Path = MEASUREMENTS_PATH,
    clinical_values_path: str | Path | None = CLINICAL_VALUES_PATH,
) -> pd.DataFrame:
    """Join the diagnosis and the longitudinal clinical measures onto the cases.

    Each Rigidity visit is tagged with its same-visit ``Diagnosis`` / ``Type``
    and with every clinical measure recorded in the *same calendar month*
    (``YearMonth``) for that eye — clinical exams rarely fall on the exact day
    of the OCT video.

    Measures come from two databases: the Biomechanics ``Measurements`` table and
    the wide ``ClinicalValues`` table (``clinical_values_path``, pass ``None`` to
    skip it). Where both carry the same measure for the same visit, Biomechanics
    wins, so the extra source only ever *adds* measures — never double-counts one.

    The measure name / value land in ``MeasureName_y`` / ``MeasureValue_y``,
    which is what the :mod:`ocularrigidity.stats.temporal` helpers read by
    default. One row per (visit × measure).
    """
    with sqlite3.connect(db_path) as con:
        diagnosis = pd.read_sql_query("SELECT * FROM Diagnosis", con)
        measures = pd.read_sql_query("SELECT * FROM Measurements", con)

    df = case_table.dropna(subset=["Date"]).copy()
    df["YearMonth"] = df["Date"].str[:7]
    measures["YearMonth"] = measures["Date"].str[:7]

    # ``PatientId`` comes back from SQLite as an int for the all-digit ids, but
    # the case table reads it off the case path (and some ids are not numeric).
    for t in (diagnosis, measures):
        t["PatientId"] = t["PatientId"].astype(str)
    df["PatientId"] = df["PatientId"].astype(str)

    df = df.merge(
        diagnosis[["PatientId", "Date", "Diagnosis", "Type", "Eye"]],
        on=["PatientId", "Date", "Eye"],
        how="left",
    )

    measures_long = measures[
        ["PatientId", "YearMonth", "Eye", "MeasureName", "MeasureValue"]
    ].rename(columns={"MeasureName": "MeasureName_y", "MeasureValue": "MeasureValue_y"})
    if clinical_values_path is not None:
        extra = load_clinical_values_long(clinical_values_path)
        # Keep only what Biomechanics does not already say about that visit.
        known = pd.MultiIndex.from_frame(measures_long[_MEASURE_KEYS])
        extra = extra[~pd.MultiIndex.from_frame(extra[_MEASURE_KEYS]).isin(known)]
        measures_long = pd.concat([measures_long, extra], ignore_index=True)

    return df.merge(measures_long, on=["PatientId", "YearMonth", "Eye"], how="left")


def available_measures(clinical_long: pd.DataFrame) -> list[str]:
    """Clinical measures actually present in the joined table, tracked ones first."""
    present = set(clinical_long["MeasureName_y"].dropna().unique())
    known = [m for m in CLINICAL_MEASURES if m in present]
    return known + sorted(present - set(known))


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
        "pearson_r": r,
        "pearson_p": p,
        "spearman_rho": rho,
        "spearman_p": p_sp,
        "slope": slope,
        "intercept": intercept,
    }


def trim_outliers(df: pd.DataFrame, cols: list[str], quantile: float) -> pd.DataFrame:
    """Drop rows outside the ``[1-q, q]`` quantile of each column (notebook-style)."""
    out = df.copy()
    for c in cols:
        s = pd.to_numeric(out[c], errors="coerce")
        lo, hi = s.quantile([1.0 - quantile, quantile])
        out = out[(s >= lo) & (s <= hi)]
    return out
