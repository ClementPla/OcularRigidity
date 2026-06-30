import sqlite3
from typing import Optional

import pandas as pd

from ocularrigidity.consts import MEASUREMENTS_PATH, STUDY_PATH
from ocularrigidity.data.measurements.studies import Study


def _merge_measure(df, df_aux, name, measure_names, on_keys, numeric=False):
    """Merge a single derived column, selecting from `measure_names` in priority
    order. `measure_names` is an exact-match list, highest priority first; the
    first instrument that has a value for a given key wins (coalesce)."""
    aux = df_aux[df_aux["MeasureName"].isin(measure_names)].copy()
    if aux.empty:
        df[name] = pd.NA
        return df

    # priority rank so the preferred instrument wins on collisions
    rank = {m: i for i, m in enumerate(measure_names)}
    aux["_rank"] = aux["MeasureName"].map(rank)

    # one value per key: lowest rank (preferred instrument) first
    aux = (
        aux.sort_values("_rank")
        .groupby(on_keys, as_index=False)["MeasureValue"]
        .first()
        .rename(columns={"MeasureValue": name})
    )

    out = df.merge(aux, on=on_keys, how="left")
    if numeric:
        out[name] = pd.to_numeric(out[name], errors="coerce")
    return out


def load_measurements(
    include_diagnosis: bool = False,
    include_OPA: bool = False,
    include_IOP: bool = False,
    include_HR: bool = False,
    include_axial_length: bool = False,
    which_study: Optional[Study] = None,
    iop_instrument: str = "Pascal IOP",   # 'Pascal IOP' (diastolic) matches Pascal OPA
    verbose: bool = False,
) -> pd.DataFrame:
    with sqlite3.connect(MEASUREMENTS_PATH) as con:
        df = pd.read_sql_query(
            "SELECT * FROM measurements WHERE MeasureName LIKE '%PLEX Macular Video%'",
            con,
        )
        full_raw_df = None
        if any([include_OPA, include_IOP, include_HR, include_axial_length]):
            full_raw_df = pd.read_sql_query("SELECT * FROM measurements", con)

    # Clean the main video dataframe
    df = df[~df["MeasureValue"].str.startswith("\\\\Usereve")]
    df["MeasureValue"] = df["MeasureValue"].str.replace("\\\\", "/", regex=False)
    df = df.dropna(subset=["MeasureValue"])

    # SQLite LIKE is case-insensitive, so 'PLEX'/'Plex' variants both arrive ->
    # may produce duplicate rows per (PatientId, Eye, Date). Collapse them.
    df = df.drop_duplicates(subset=["PatientId", "Eye", "Date", "MeasureValue"])
    if verbose:
        print(f"Loaded {len(df)} video measurements after cleaning.")

    if include_diagnosis:
        with sqlite3.connect(MEASUREMENTS_PATH) as con:
            diagnosis = pd.read_sql_query(
                "SELECT PatientId, Diagnosis, Eye, Type FROM diagnosis", con
            )
        diagnosis = diagnosis.drop_duplicates(subset=["PatientId", "Eye"])
        df = df.merge(diagnosis, on=["PatientId", "Eye"], how="left")
        df = df.dropna(subset=["Diagnosis", "Type"])

    if include_OPA and full_raw_df is not None:
        df = _merge_measure(
            df, full_raw_df, "OPA",
            measure_names=["Pascal OPA"],
            on_keys=["PatientId", "Eye", "Date"],
            numeric=True,
        )
        if verbose:
            print(f"After merging OPA, dataset has {len(df)} records.")

    if include_IOP and full_raw_df is not None:
        # exact instrument; fall back only to other Pascal-like diastolic sources
        # if you want — but do NOT silently mix Goldman/ORA conventions.
        iop_priority = [iop_instrument]
        df = _merge_measure(
            df, full_raw_df, "IOP",
            measure_names=iop_priority,
            on_keys=["PatientId", "Eye", "Date"],
            numeric=True,
        )
        if verbose:
            n = df["IOP"].notna().sum()
            print(f"After merging IOP ({iop_instrument}), {n}/{len(df)} have a value.")

    if include_axial_length and full_raw_df is not None:
        # AL is static per eye -> merge on (PatientId, Eye) only, ignoring Date.
        # Prefer IOLMaster (optical biometry) over generic 'Axial Length'.
        df = _merge_measure(
            df, full_raw_df, "AxialLength",
            measure_names=["IOLMaster AL", "Axial Length"],
            on_keys=["PatientId", "Eye"],
            numeric=True,
        )
        if verbose:
            print(f"After merging Axial Length, dataset has {len(df)} records.")

    if include_HR and full_raw_df is not None:
        df = _merge_measure(
            df, full_raw_df, "HR",
            measure_names=["HR"],
            on_keys=["PatientId", "Date"],
            numeric=True,
        )
        if verbose:
            print(f"After merging HR, dataset has {len(df)} records.")

    if which_study is not None:
        with sqlite3.connect(STUDY_PATH) as con:
            study_df = pd.read_sql_query(
                "SELECT PatientId, Eye, Study FROM Studies", con
            )
        study_df = study_df.drop_duplicates(subset=["PatientId", "Eye"])
        df = df.merge(study_df, on=["PatientId", "Eye"], how="left")
        df = df[df["Study"] == which_study.value]

    return df