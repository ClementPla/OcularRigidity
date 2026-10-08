"""One tidy row per rigidity visit, joined from every source."""

from __future__ import annotations

from datetime import datetime
import json
import sqlite3
from pathlib import Path
from typing import Iterable, Sequence

import dateparser
import numpy as np
import pandas as pd

from ocularrigidity.consts import (
    CLINICAL_VALUES_PATH,
    HEYEX_ONH_PATH,
    MEASUREMENTS_PATH,
    QC_ERRORS_PATH,
    ROOT_CARDIAC_PIPELINE,
    ROOT_CROSS_SECTIONAL_EXCEL,
    STUDY_PATH,
)
from ocularrigidity.data.measurements.dataframe import (
    filter_misregistration,
    load_measurements,
)
from ocularrigidity.data.measurements.igri import add_igri_base
from ocularrigidity.data.measurements.pulsation_results import load_pulsation_results
from ocularrigidity.data.measurements.studies import Study

ONH_SECTORS = [
    "G BMO MRW",
    "T BMO MRW",
    "TS BMO MRW",
    "TI BMO MRW",
    "N BMO MRW",
    "NS BMO MRW",
    "NI BMO MRW",
    "G RNFL Thickness",
    "T RNFL Thickness",
    "TS RNFL Thickness",
    "TI RNFL Thickness",
    "N RNFL Thickness",
    "NS RNFL Thickness",
    "NI RNFL Thickness",
]

# : The two ONH families.
BMO_MRW_SECTORS = [c for c in ONH_SECTORS if c.endswith("BMO MRW")]
RNFL_SECTORS = [c for c in ONH_SECTORS if c.endswith("RNFL Thickness")]

# : ``(output column, candidate sectors)`` for the steepest-change search.
_STEEPEST_FAMILIES = {
    "steepest_change_BMO_MRW": [c for c in BMO_MRW_SECTORS if not c.startswith("G ")],
    "steepest_change_RNFL_Thickness": [
        c for c in RNFL_SECTORS if not c.startswith("G ")
    ],
}

# : The per-eye progression columns :func:`_add_steepest_change` derives.
STEEPEST_CHANGE_COLUMNS = list(_STEEPEST_FAMILIES)

# : ``(family, candidate sectors)`` for the across-sector extrema.
_EXTREMA_FAMILIES = {
    "BMO_MRW": [c for c in BMO_MRW_SECTORS if not c.startswith("G ")],
}

# : The per-visit across-sector columns :func:`_add_sector_extrema` derives.
SECTOR_EXTREMA_COLUMNS = [
    f"{stat}_{family}" for family in _EXTREMA_FAMILIES for stat in ("min", "max")
]

# : What the cardiac pipeline measures.
PULSATION_METRICS = [
    "deltaCT",
    "deltaCT_Mask",
    "minCT",
    "RelativeGrowth",
    "K",
    "dV",
    "K_Mask",
    "thickening_um_s",
    "thinning_um_s",
    "rate_asymmetry",
]

COVARIATES = ["IOP", "OPA", "AxialLength", "HR", "predicted_HR", "Age"]

DEFAULT_MEASURES = [
    "igri_base",
    "G BMO MRW",
    "G RNFL Thickness",
    "steepest_change_BMO_MRW",
    "steepest_change_RNFL_Thickness",
    "Global RNFL Thickness",
    "MD",
    "PSD",
    "GCL Volume",
    "Visual Acuity",
]

# : Identity of a row.
IDENTITY = [
    "case_id",
    "caseId",
    "PatientId",
    "Eye",
    "Date",
    "YearMonth",
    "Sex",
    "Study",
    "Diagnosis",
    "Type",
]

# : Measures named by ``load_measurements`` that would arrive twice
_DUPLICATED_BY_COVARIATE = {
    "Pascal OPA",
    "Axial Length",
    "IOLMaster AL",
    "HR",
}

_AUDIT_COLUMNS = {
    "onh_gap_days",
    "onh_slope_n_exams",
    "onh_slope_span_years",
    "diagnosis_gap_days",
}

# : Identity columns of the wide ``ClinicalValues`` table
_CLINICAL_VALUES_IDS = ["Id", "Cohort", "PatientId", "File", "Eye", "Date"]

# : The same quantity, spelled differently in the two clinical databases.
_CLINICAL_VALUES_RENAME = {"G RNFL Thickness": "Global RNFL Thickness"}


def build_cohort(
    root: str | Path = ROOT_CARDIAC_PIPELINE,
    *,
    study: Study | None = None,
    iop_instrument: str = "Pascal IOP",
    onh_tolerance_days: int = 92,
    diagnosis_tolerance_days: int = 365,
    exclude_qc: bool = False,
    heyex_path: str | Path | None = HEYEX_ONH_PATH,
    clinical_values_path: str | Path | None = CLINICAL_VALUES_PATH,
    measurements_path: str | Path = MEASUREMENTS_PATH,
    overwrite: bool = False,
) -> pd.DataFrame:
    """One row per rigidity visit, every source merged onto it."""
    root = Path(root)
    cases, _cycles = load_pulsation_results(root, overwrite=overwrite)
    cases = filter_misregistration(cases, Path(root) / "misregistration_flags.csv")
    _cycles = filter_misregistration(_cycles, Path(root) / "misregistration_flags.csv")
    visits = load_measurements(
        which_study=study,
        include_OPA=True,
        include_IOP=True,
        include_HR=True,
        include_axial_length=True,
        iop_instrument=iop_instrument,
        include_blood_pressure=True
    ).set_index("Id")
    visits = visits.join(cases.drop(columns=["HR", "caseId"]), how="inner")
    visits = visits.rename(columns={"video": "case_id"})
    visits.index.name = "visit_id"
    visits["caseId"] = (
        visits["PatientId"].astype(str)
        + "/"
        + visits["Date"].astype(str)
        + "/"
        + visits["Eye"].astype(str)
    )
    visits["YearMonth"] = visits["Date"].astype(str).str[:7]
    visits["Study"] = _study_membership(visits)
    if exclude_qc:
        visits = visits[~visits["case_id"].isin(load_excluded_cases())]

    patients = _load_patients(measurements_path)
    visits = _add_demographics(visits, patients)
    visits = _add_diagnosis(visits, measurements_path, diagnosis_tolerance_days)
    visits, onh_cols = _add_onh(
        visits, patients, heyex_path, clinical_values_path, onh_tolerance_days
    )
    visits, clinical_cols = _add_clinical(
        visits, measurements_path, clinical_values_path, iop_instrument
    )
    visits, igri_cols = _add_igri(visits)

    groups = {
        "identity": [c for c in IDENTITY if c in visits.columns],
        "pulsation": [c for c in PULSATION_METRICS if c in visits.columns],
        "covariates": [c for c in COVARIATES if c in visits.columns],
        "onh": onh_cols,
        "clinical": clinical_cols + igri_cols,
    }
    ordered = list(dict.fromkeys(sum(groups.values(), [])))
    rest = [c for c in visits.columns if c not in ordered]
    out = visits[ordered + rest].sort_values(["PatientId", "Date", "Eye"])
    out.attrs["column_groups"] = groups
    return out


def column_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """The column blocks of a cohort frame, recomputed if ``attrs`` was lost."""
    stored = df.attrs.get("column_groups")
    if stored:
        return {k: [c for c in v if c in df.columns] for k, v in stored.items()}

    known = {
        "identity": [c for c in IDENTITY if c in df.columns],
        "pulsation": [c for c in PULSATION_METRICS if c in df.columns],
        "covariates": [c for c in COVARIATES if c in df.columns],
        "onh": [c for c in ONH_SECTORS if c in df.columns],
    }
    claimed = set(sum(known.values(), [])) | _AUDIT_COLUMNS | {"onh_source"}
    known["clinical"] = [
        c
        for c in df.columns
        if c not in claimed and pd.api.types.is_numeric_dtype(df[c])
    ]
    return known


def measure_columns(df: pd.DataFrame) -> list[str]:
    """Everything a rigidity metric can be confronted with: ONH + clinical."""
    groups = column_groups(df)
    present = [
        c
        for c in groups["onh"] + groups["clinical"]
        if c in df.columns and df[c].dtype.kind in "fi" and c not in _AUDIT_COLUMNS
    ]
    head = [m for m in DEFAULT_MEASURES if m in present]
    return head + [c for c in present if c not in head]


def cohort_to_long(
    df: pd.DataFrame, measures: Sequence[str] | None = None
) -> pd.DataFrame:
    """Melt the measure columns into ``MeasureName_y`` / ``MeasureValue_y``."""
    measures = list(measures) if measures is not None else measure_columns(df)
    measures = [m for m in measures if m in df.columns]
    id_cols = [c for c in df.columns if c not in measures]
    long = df.reset_index().melt(
        id_vars=[c for c in df.reset_index().columns if c not in measures],
        value_vars=measures,
        var_name="MeasureName_y",
        value_name="MeasureValue_y",
    )
    del id_cols
    return long.dropna(subset=["MeasureValue_y"]).reset_index(drop=True)


def load_excluded_cases(path: str | Path = QC_ERRORS_PATH) -> set[str]:
    """Video paths rejected on visual QC in the gif viewer (``errors.json``)."""
    path = Path(path)
    if not path.exists():
        return set()
    with open(path) as f:
        return {str(e).replace("_", "/").replace(".gif", "") for e in json.load(f)}


def _study_membership(visits: pd.DataFrame, path: str | Path = STUDY_PATH) -> pd.Series:
    """Every study an eye is enrolled in, joined by ``" / "``."""
    with sqlite3.connect(path) as con:
        studies = pd.read_sql_query("SELECT PatientId, Eye, Study FROM Studies", con)
    studies["Eye"] = studies["Eye"].astype(str).str.strip()
    per_eye = (
        studies.drop_duplicates()
        .groupby(["PatientId", "Eye"])["Study"]
        .agg(lambda s: " / ".join(sorted(s)))
    )
    keys = pd.MultiIndex.from_arrays(
        [visits["PatientId"], visits["Eye"].astype(str).str.strip()]
    )
    return pd.Series(per_eye.reindex(keys).to_numpy(), index=visits.index)


def _load_patients(measurements_path: str | Path) -> pd.DataFrame:
    with sqlite3.connect(measurements_path) as con:
        pat = pd.read_sql_query("SELECT PatientId, File, DOB, Sex FROM Patients", con)
    pat["File"] = pat["File"].astype(str).str.strip()
    pat["DOB"] = pd.to_datetime(pat["DOB"], errors="coerce")
    return pat.drop_duplicates("PatientId")


def _add_demographics(visits: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    out = visits.merge(
        patients[["PatientId", "DOB", "Sex"]], on="PatientId", how="left"
    ).set_index(visits.index)
    out["Age"] = (
        pd.to_datetime(out["Date"], errors="coerce") - out["DOB"]
    ).dt.days / 365.25
    return out.drop(columns="DOB")


def _add_diagnosis(
    visits: pd.DataFrame, measurements_path: str | Path, tolerance_days: int
) -> pd.DataFrame:
    """As-of join of the diagnosis register, with the realised gap kept."""
    with sqlite3.connect(measurements_path) as con:
        diagnosis = pd.read_sql_query(
            "SELECT PatientId, Date, Eye, Diagnosis, Type FROM Diagnosis", con
        )
    diagnosis["_date"] = pd.to_datetime(diagnosis["Date"], errors="coerce")
    diagnosis = (
        diagnosis.dropna(subset=["_date"])
        .rename(columns={"Date": "Diagnosis_Date"})
        .sort_values("_date")
    )
    return _asof_merge(
        visits,
        diagnosis[["PatientId", "Eye", "_date", "Diagnosis_Date", "Diagnosis", "Type"]],
        tolerance_days,
        gap_col="diagnosis_gap_days",
        match_date_col="Diagnosis_Date",
    )


def _add_onh(
    visits: pd.DataFrame,
    patients: pd.DataFrame,
    heyex_path: str | Path | None,
    clinical_values_path: str | Path | None,
    tolerance_days: int,
) -> tuple[pd.DataFrame, list[str]]:
    """ONH sectors from the Heyex reports, back-filled from ClinicalValues."""
    out = visits.copy()
    present: list[str] = []
    exams: list[pd.DataFrame] = []

    if heyex_path is not None and Path(heyex_path).exists():
        heyex = pd.read_pickle(heyex_path).copy()
        heyex["File ID"] = heyex["File ID"].astype(str).str.strip()
        heyex = heyex.merge(
            patients[["PatientId", "File"]],
            left_on="File ID",
            right_on="File",
            how="inner",
        )
        sectors = [c for c in ONH_SECTORS if c in heyex.columns]
        heyex[sectors] = heyex[sectors].apply(pd.to_numeric, errors="coerce")
        heyex["_date"] = pd.to_datetime(heyex["Exam Date"], errors="coerce")
        heyex = (
            heyex.dropna(subset=["_date"])
            .rename(columns={"Exam Date": "onh_exam_date"})
            .sort_values("_date")
        )
        # One exam per (eye, day): the report parser emits one row per PDF pair.
        heyex = heyex.drop_duplicates(["PatientId", "Eye", "_date"])
        out = _asof_merge(
            out,
            heyex[["PatientId", "Eye", "_date", "onh_exam_date"] + sectors],
            tolerance_days,
            gap_col="onh_gap_days",
            match_date_col="onh_exam_date",
        )
        out["onh_source"] = np.where(out[sectors].notna().any(axis=1), "heyex", None)
        present = sectors
        exams.append(heyex[["PatientId", "Eye", "_date"] + sectors])

    if clinical_values_path is not None and Path(clinical_values_path).exists():
        extra = _clinical_values_onh(clinical_values_path)
        sectors = [c for c in ONH_SECTORS if c in extra.columns]
        exams.append(
            extra.assign(
                _date=pd.to_datetime(extra["YearMonth"] + "-01", errors="coerce")
            )[["PatientId", "Eye", "_date"] + sectors]
        )
        if not present:
            out = out.merge(
                extra, on=["PatientId", "Eye", "YearMonth"], how="left"
            ).set_index(visits.index)
            out["onh_source"] = np.where(
                out[sectors].notna().any(axis=1), "clinical_values", None
            )
            present = sectors
        else:
            filled = out.merge(
                extra,
                on=["PatientId", "Eye", "YearMonth"],
                how="left",
                suffixes=("", "_cv"),
            ).set_index(out.index)
            missing = out[present].isna().all(axis=1)
            for c in sectors:
                cv = filled.get(f"{c}_cv")
                if cv is not None:
                    out.loc[missing, c] = cv[missing]
            out.loc[missing & out[present].notna().any(axis=1), "onh_source"] = (
                "clinical_values"
            )

    out, extrema_cols = _add_sector_extrema(out, present)
    out, steepest_cols = _add_steepest_change(out, exams)

    cols = (
        present
        + [
            c
            for c in ("onh_exam_date", "onh_gap_days", "onh_source")
            if c in out.columns
        ]
        + extrema_cols
        + steepest_cols
    )
    return out, cols


def _add_igri(visits: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add the I-GRI base score"""
    out, report = add_igri_base(visits)
    out.attrs = dict(visits.attrs)
    out.attrs["igri_report"] = report
    cols = [c for c in ("igri_base",) if c in out.columns and out[c].notna().any()]
    return out, cols


def _add_sector_extrema(
    visits: pd.DataFrame, present: Sequence[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Per visit, the thinnest and the thickest sector of each ONH family."""
    out = visits.copy()
    added: list[str] = []
    for family, candidates in _EXTREMA_FAMILIES.items():
        names = [
            f"min_{family}",
            f"max_{family}",
            f"min_{family}_sector",
            f"max_{family}_sector",
        ]
        added += names
        for c in names:
            out[c] = np.nan
        cols = [c for c in candidates if c in present and c in out.columns]
        if len(cols) < len(candidates):  # a partial family has no comparable min
            continue
        block = out[cols].apply(pd.to_numeric, errors="coerce")
        full = block[block.notna().all(axis=1)]
        if full.empty:
            continue
        out[f"min_{family}"] = full.min(axis=1).reindex(out.index)
        out[f"max_{family}"] = full.max(axis=1).reindex(out.index)
        for stat, idx in (("min", full.idxmin), ("max", full.idxmax)):
            sector = idx(axis=1).str.split(" ").str[0]
            out[f"{stat}_{family}_sector"] = sector.reindex(out.index)
    return out, added


def _add_steepest_change(
    visits: pd.DataFrame, exams: list[pd.DataFrame]
) -> tuple[pd.DataFrame, list[str]]:
    """Per eye, the ONH sector that is thinning fastest, and how fast (µm/year)."""
    added = (
        [c for c in _STEEPEST_FAMILIES]
        + [f"{c}_sector" for c in _STEEPEST_FAMILIES]
        + ["onh_slope_n_exams", "onh_slope_span_years"]
    )
    if not exams:
        for c in added:
            visits[c] = np.nan
        return visits, added

    # One row per (eye, exam month).
    all_exams = pd.concat(exams, ignore_index=True).dropna(subset=["_date"])
    all_exams["Eye"] = all_exams["Eye"].astype(str).str.strip()
    all_exams["_ym"] = all_exams["_date"].dt.strftime("%Y-%m")
    all_exams = all_exams.drop_duplicates(["PatientId", "Eye", "_ym"])

    rows = []
    for (pid, eye), g in all_exams.groupby(["PatientId", "Eye"]):
        g = g.sort_values("_date")
        years = (g["_date"] - g["_date"].iloc[0]).dt.days.to_numpy() / 365.25
        row = {
            "PatientId": pid,
            "Eye": eye,
            "onh_slope_n_exams": len(g),
            "onh_slope_span_years": float(years.max()),
        }
        for out_col, candidates in _STEEPEST_FAMILIES.items():
            slopes = {}
            for sector in candidates:
                if sector not in g.columns:
                    continue
                y = pd.to_numeric(g[sector], errors="coerce").to_numpy()
                ok = np.isfinite(y)
                if ok.sum() < 2 or years[ok].max() == years[ok].min():
                    continue
                slopes[sector] = float(np.polyfit(years[ok], y[ok], 1)[0])
            if slopes:
                worst = min(slopes, key=slopes.get)  # most negative = fastest loss
                row[out_col] = slopes[worst]
                row[f"{out_col}_sector"] = worst.split(" ")[0]
        rows.append(row)

    per_eye = pd.DataFrame(rows)
    out = visits.copy()
    out["Eye"] = out["Eye"].astype(str).str.strip()
    out = out.merge(per_eye, on=["PatientId", "Eye"], how="left").set_index(
        visits.index
    )
    for c in added:
        if c not in out.columns:
            out[c] = np.nan
    return out, added


def _clinical_values_onh(path: str | Path) -> pd.DataFrame:
    """ONH sectors from ClinicalValues, one row per (patient, eye, month)."""
    with sqlite3.connect(path) as con:
        wide = pd.read_sql_query("SELECT * FROM ClinicalValues", con)
    sectors = [c for c in ONH_SECTORS if c in wide.columns]
    wide[sectors] = wide[sectors].apply(pd.to_numeric, errors="coerce")
    wide["YearMonth"] = wide["Date"].astype(str).str[:7]
    return wide.groupby(["PatientId", "Eye", "YearMonth"], as_index=False)[
        sectors
    ].mean()


def _add_clinical(
    visits: pd.DataFrame,
    measurements_path: str | Path,
    clinical_values_path: str | Path | None,
    iop_instrument: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Longitudinal clinical measures, one column per measure."""
    with sqlite3.connect(measurements_path) as con:
        measures = pd.read_sql_query(
            "SELECT PatientId, Date, Eye, MeasureName, MeasureValue FROM Measurements",
            con,
        )
    measures["YearMonth"] = measures["Date"].astype(str).str[:7]
    long = measures.rename(
        columns={"MeasureName": "MeasureName_y", "MeasureValue": "MeasureValue_y"}
    )[["PatientId", "Eye", "YearMonth", "MeasureName_y", "MeasureValue_y"]]

    if clinical_values_path is not None and Path(clinical_values_path).exists():
        extra = _clinical_values_long(clinical_values_path)
        keys = ["PatientId", "Eye", "YearMonth", "MeasureName_y"]
        known = pd.MultiIndex.from_frame(long[keys])
        extra = extra[~pd.MultiIndex.from_frame(extra[keys]).isin(known)]
        long = pd.concat([long, extra], ignore_index=True)

    drop = (
        _DUPLICATED_BY_COVARIATE
        | {iop_instrument}
        | set(ONH_SECTORS)
        | set(visits.columns)
    )
    long = long[~long["MeasureName_y"].isin(drop)]
    long["MeasureValue_y"] = pd.to_numeric(long["MeasureValue_y"], errors="coerce")
    long = long.dropna(subset=["MeasureValue_y"])

    wide = (
        long.groupby(
            ["PatientId", "Eye", "YearMonth", "MeasureName_y"], as_index=False
        )["MeasureValue_y"]
        .median()
        .pivot_table(
            index=["PatientId", "Eye", "YearMonth"],
            columns="MeasureName_y",
            values="MeasureValue_y",
        )
        .reset_index()
    )
    wide.columns.name = None
    cols = [c for c in wide.columns if c not in ("PatientId", "Eye", "YearMonth")]

    out = visits.merge(
        wide, on=["PatientId", "Eye", "YearMonth"], how="left"
    ).set_index(visits.index)
    assert len(out) == len(visits), "the clinical join must not change the visit count"
    cols = [c for c in cols if out[c].notna().any()]
    return out, cols


def _clinical_values_long(path: str | Path) -> pd.DataFrame:
    """ClinicalValues melted to the long measure shape, month-collapsed."""
    with sqlite3.connect(path) as con:
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
    long["YearMonth"] = long["Date"].astype(str).str[:7]
    long["MeasureValue_y"] = pd.to_numeric(long["MeasureValue_y"], errors="coerce")
    return long.dropna(subset=["MeasureValue_y"])[
        ["PatientId", "Eye", "YearMonth", "MeasureName_y", "MeasureValue_y"]
    ]


def _asof_merge(
    visits: pd.DataFrame,
    right: pd.DataFrame,
    tolerance_days: int,
    *,
    gap_col: str,
    match_date_col: str,
) -> pd.DataFrame:
    """Nearest-in-time merge on (PatientId, Eye), keeping the realised gap."""
    left = visits.copy()
    left["_id"] = left.index
    left["_date"] = pd.to_datetime(left["Date"], errors="coerce")
    ok = left.dropna(subset=["_date"]).sort_values("_date")

    merged = pd.merge_asof(
        ok,
        right.sort_values("_date"),
        on="_date",
        by=["PatientId", "Eye"],
        direction="nearest",
        tolerance=pd.Timedelta(days=tolerance_days),
    )
    matched = pd.to_datetime(merged[match_date_col], errors="coerce")
    merged[gap_col] = (matched - merged["_date"]).dt.days
    out = (
        merged.set_index("_id")
        .rename_axis(visits.index.name)
        .drop(columns="_date")
        .reindex(visits.index)
    )
    return out


def coverage(df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """How many visits / eyes / patients carry each column."""
    columns = list(columns) if columns is not None else measure_columns(df)
    eyes = df[["PatientId", "Eye"]].astype(str).agg("/".join, axis=1)
    rows = []
    for c in columns:
        ok = df[c].notna()
        rows.append(
            {
                "column": c,
                "visits": int(ok.sum()),
                "eyes": int(eyes[ok].nunique()),
                "patients": int(df.loc[ok, "PatientId"].nunique()),
                "coverage": round(float(ok.mean()), 3),
            }
        )
    return (
        pd.DataFrame(rows).sort_values("visits", ascending=False).reset_index(drop=True)
    )


def get_cross_sectional_dx(
    path_excel: Path = ROOT_CROSS_SECTIONAL_EXCEL,
) -> pd.DataFrame:
    df = pd.read_excel(path_excel, sheet_name="Cross-sectional")
    def parse_date(value):
        if pd.isna(value):
            return pd.NaT
        try:
            date = datetime.strptime(str(value).split(" ")[0], "%Y-%m-%d")
            return date
        except ValueError:
            pass
        if isinstance(value, str):
            value = value.lower()
            value = value.replace("juilliet", "juillet")
            value = value.replace("jav", "janvier")
            value = value.replace("aug", "août")
            if "décembre" in value:
                pass
            elif "dec" in value:
                value = value.replace("dec", "décembre")
            elif "de" in value:
                value = value.replace("de", "décembre")
        return dateparser.parse(str(value), languages=["fr"])

    # Drop all NaN columns
    df = df.dropna(axis=1, how="all")
    df["Date"] = df["Date R/V"].apply(parse_date)
    df = df  # .dropna(subset=["Date"])
    df[df["Date"].isna()]
    df = df[["Dossier", "Date", "Œil ", "Dx"]]
    df = df.rename(columns={"Dossier": "PatientId", "Œil ": "Eye", "Dx": "Diagnosis"})
        
    DX_CANON = {
        "gpao": "GPAO",
        "gpao precoce": "GPAO",
        "gnt/gpao": "GPAO",
        "hto": "HTO",
        "normal": "Normal",
        "suspect": "Suspect",
        "suspect gl": "Suspect",
    }

    def tidy_cross_sectional(dx):
        """Excel diagnoses, keyed on (File, Eye) with a usable date."""
        out = dx.copy()
        out["File"] = out["PatientId"].astype(str).str.strip()
        out["Eye"] = out["Eye"].astype(str).str.strip().str.upper()
        out["Diagnosis_xs"] = (
            out["Diagnosis"].astype(str).str.strip().str.lower().map(DX_CANON)
        )
        out["xs_date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.dropna(subset=["xs_date"])
        both = out[out["Eye"] == "OU"]
        out = pd.concat(
            [out[out["Eye"].isin(["OD", "OS"])]]
            + [both.assign(Eye=e) for e in ("OD", "OS")],
            ignore_index=True,
        )
        out["File"] = out["PatientId"].astype(str).str.strip().str.upper()

        return out[["File", "Eye", "xs_date", "Diagnosis_xs"]].sort_values("xs_date")
    return tidy_cross_sectional(df)


def add_cross_sectional_dx(df, dx, tolerance_days=92):
    left = df.copy()
    left["File"] = left["case_id"].str.split("/").str[0].str.upper()
    left["_id"] = left.index
    left["_date"] = pd.to_datetime(left["Date"], errors="coerce")
    left = left.dropna(subset=["_date"]).sort_values("_date")

    right = dx.rename(columns={"xs_date": "_date"})
    right["xs_date"] = right["_date"]

    merged = pd.merge_asof(
        left,
        right,
        on="_date",
        by=["File", "Eye"],
        direction="nearest",
        tolerance=pd.Timedelta(days=tolerance_days),
    )
    merged["xs_gap_days"] = (merged["xs_date"] - merged["_date"]).dt.days
    return (
        merged.set_index("_id")
        .rename_axis(df.index.name)
        .drop(columns="_date")
        .reindex(df.index)
    )