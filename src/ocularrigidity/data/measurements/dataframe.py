import sqlite3

import pandas as pd

from ocularrigidity.consts import MEASUREMENTS_PATH, STUDY_PATH
from ocularrigidity.data.measurements.studies import Study
from typing import Optional


def _merge_measure(df, df_aux, name, on_keys):
    # AGGREGATE HERE: Ensure one value per key combo to prevent row duplication
    # We take the mean if numeric, or first if it's a string/mixed.
    df_aux_unique = (
        df_aux.groupby(on_keys)["MeasureValue"]
        .first()  # Or .mean() if columns are numeric
        .reset_index()
        .rename(columns={"MeasureValue": name})
    )

    return df.merge(df_aux_unique, on=on_keys, how="left")  # .dropna(subset=[name])


def load_measurements(
    include_diagnosis: bool = False,
    include_OPA: bool = False,
    include_IOP: bool = False,
    include_HR: bool = False,
    include_axial_length: bool = False,
    which_study: Optional[Study] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    with sqlite3.connect(MEASUREMENTS_PATH) as con:
        # Optimization: Filter for "PLEX Macular Video" at the SQL level to save RAM
        df = pd.read_sql_query(
            "SELECT * FROM measurements WHERE MeasureName LIKE '%PLEX Macular Video%'",
            con,
        )

        # Load auxiliary data only if needed to save resources
        full_raw_df = None
        if any([include_OPA, include_IOP, include_HR, include_axial_length]):
            full_raw_df = pd.read_sql_query("SELECT * FROM measurements", con)

    # Clean the main video dataframe
    df = df[~df["MeasureValue"].str.startswith("\\\\Usereve")]
    df["MeasureValue"] = df["MeasureValue"].str.replace("\\\\", "/", regex=False)
    df = df.dropna(subset=["MeasureValue"])
    if verbose:
        print(f"Loaded {len(df)} video measurements after cleaning.")
    if include_diagnosis:
        with sqlite3.connect(MEASUREMENTS_PATH) as con:
            diagnosis = pd.read_sql_query(
                "SELECT PatientId, Diagnosis, Eye, Type FROM diagnosis", con
            )
        # Drop duplicates in diagnosis to prevent row bloat
        diagnosis = diagnosis.drop_duplicates(subset=["PatientId", "Eye"])
        df = df.merge(diagnosis, on=["PatientId", "Eye"], how="left")
        df = df.dropna(subset=["Diagnosis", "Type"])

    # Handle auxiliary merges with the fixed helper
    if include_OPA and full_raw_df is not None:
        aux = full_raw_df[
            full_raw_df["MeasureName"].str.contains("OPA", case=False, na=False)
        ]
        df = _merge_measure(df, aux, "OPA", ["PatientId", "Eye", "Date"])
        if verbose:
            print(f"After merging OPA, dataset has {len(df)} records.")
    if include_IOP and full_raw_df is not None:
        aux = full_raw_df[
            full_raw_df["MeasureName"].str.contains("IOP", case=False, na=False)
        ]
        df = _merge_measure(df, aux, "IOP", ["PatientId", "Eye", "Date"])
        if verbose:
            print(f"After merging IOP, dataset has {len(df)} records.")

    if include_axial_length and full_raw_df is not None:
        aux = full_raw_df[
            full_raw_df["MeasureName"].str.contains(
                "Axial Length", case=False, na=False
            )
            | full_raw_df["MeasureName"].str.contains(
                "IOLMaster AL", case=False, na=False
            )
        ]
        # Use only PatientId/Eye as Axial Length is usually static across dates
        df = _merge_measure(df, aux, "AxialLength", ["PatientId", "Eye"])
        df["AxialLength"] = pd.to_numeric(df["AxialLength"], errors="coerce")
        if verbose:
            print(f"After merging Axial Length, dataset has {len(df)} records.")
    if include_HR and full_raw_df is not None:
        aux = full_raw_df[
            full_raw_df["MeasureName"].str.contains("HR", case=False, na=False)
        ]
        df = _merge_measure(df, aux, "HR", ["PatientId", "Date"])
        df["HR"] = pd.to_numeric(df["HR"], errors="coerce")
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
