import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from ocularrigidity.consts import MEASUREMENTS_PATH, STUDY_PATH
from ocularrigidity.data.measurements.studies import Study

# ---------------------------------------------------------------------------
# Join grains
#
# The Measurements table mixes three kinds of measurement, and each one has to
# be joined on its own key. Using a single (PatientId, Eye, Date) key silently
# drops the last two:
#
#   VISIT_EYE  the measurement is of one eye, at one visit (IOP, OPA, ...)
#   EYE        the measurement is static per eye; biometry is acquired once and
#              reused for years, so joining on Date loses ~2/3 of the values
#   VISIT      the measurement is of the patient, not the eye. These rows are
#              stored with Eye == "other" (always for blood pressure, and for
#              HR from 2021 onwards), so any key containing Eye matches nothing
# ---------------------------------------------------------------------------
VISIT_EYE = ["PatientId", "Eye", "Date"]
EYE = ["PatientId", "Eye"]
VISIT = ["PatientId", "Date"]

#: Output column -> (source MeasureName values, join keys). The source list is
#: exact-match and ordered: the first instrument that has a value for a given
#: key wins. The 'ORA <x>' / '<x>' and 'IOLMaster AL' / 'Axial Length' pairs are
#: the same quantity exported under two names by two eras of the database; they
#: never co-occur on the same key, so their relative order is inconsequential.
MEASURE_SPECS: dict[str, tuple[list[str], list[str]]] = {
    "OPA": (["Pascal OPA"], VISIT_EYE),
    "IOPg": (["ORA IOPg", "IOPg"], VISIT_EYE),
    "IOPcc": (["ORA IOPcc", "IOPcc"], VISIT_EYE),
    "AxialLength": (["IOLMaster AL", "Axial Length"], EYE),
    "CCT": (["CCT"], EYE),
    "CH": (["CH"], EYE),
    "CRF": (["ORA CRF", "CRF"], EYE),
    "HR": (["HR"], VISIT),
    "Systolic": (["Systolic"], VISIT),
    "Diastolic": (["Diastolic"], VISIT),
}


def _merge_measure(df, df_aux, name, measure_names, on_keys, numeric=False):
    """Merge a single derived column, selecting from `measure_names` in priority
    order. `measure_names` is an exact-match list, highest priority first; the
    first instrument that has a value for a given key wins (coalesce). Within a
    single instrument the most recent value wins, which only matters for keys
    that span several dates (the EYE grain)."""
    aux = df_aux[df_aux["MeasureName"].isin(measure_names)].copy()
    if aux.empty:
        df[name] = pd.NA
        return df

    # priority rank so the preferred instrument wins on collisions
    rank = {m: i for i, m in enumerate(measure_names)}
    aux["_rank"] = aux["MeasureName"].map(rank)

    # one value per key: preferred instrument first, then most recent date
    aux = (
        aux.sort_values(["_rank", "Date"], ascending=[True, False])
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
    include_pachymetry: bool = False,
    include_ORA: bool = False,
    include_blood_pressure: bool = False,
    which_study: Optional[Study] = None,
    iop_instrument: str = "Pascal IOP",  # 'Pascal IOP' (diastolic) matches Pascal OPA
    verbose: bool = False,
) -> pd.DataFrame:
    """One row per Plex Macular Video, with the requested clinical measures
    merged on their natural grain (see MEASURE_SPECS)."""
    wanted: list[str] = []
    if include_OPA:
        wanted.append("OPA")
    if include_axial_length:
        wanted.append("AxialLength")
    if include_pachymetry:
        wanted.append("CCT")
    if include_ORA:
        wanted += ["CH", "CRF", "IOPg", "IOPcc"]
    if include_HR:
        wanted.append("HR")
    if include_blood_pressure:
        wanted += ["Systolic", "Diastolic"]

    with sqlite3.connect(MEASUREMENTS_PATH) as con:
        df = pd.read_sql_query(
            "SELECT * FROM measurements WHERE MeasureName LIKE '%PLEX Macular Video%'",
            con,
        )
        full_raw_df = None
        if wanted or include_IOP:
            full_raw_df = pd.read_sql_query("SELECT * FROM measurements", con)

    # Clean the main video dataframe
    df = df[~df["MeasureValue"].str.startswith("\\\\Usereve")]
    df["MeasureValue"] = df["MeasureValue"].str.replace("\\", "/", regex=False)
    df = df.dropna(subset=["MeasureValue"])

    # SQLite LIKE is case-insensitive, so 'PLEX'/'Plex' variants both arrive ->
    # may produce duplicate rows per (PatientId, Eye, Date). Collapse them.
    df = df.drop_duplicates(subset=["PatientId", "Eye", "Date", "MeasureValue"])
    df["YearMonth"] = pd.to_datetime(df["Date"]).dt.to_period("M")
    df["Year"] = pd.to_datetime(df["Date"]).dt.to_period("Y")
    if verbose:
        print(f"Loaded {len(df)} video measurements after cleaning.")

    if include_diagnosis:
        with sqlite3.connect(MEASUREMENTS_PATH) as con:
            diagnosis = pd.read_sql_query(
                "SELECT PatientId, Diagnosis, Date, Eye, Type FROM diagnosis", con
            )

        # Convert the 'Date' column in the diagnosis DataFrame to datetime
        diagnosis["Date"] = pd.to_datetime(diagnosis["Date"])
        diagnosis["YearMonth"] = diagnosis["Date"].dt.to_period("M")
        diagnosis["Year"] = diagnosis["Date"].dt.to_period("Y")
        df = df.merge(diagnosis, on=["PatientId", "Eye", "Year"], how="left")
        df = df.dropna(subset=["Diagnosis", "Type"])

    if include_IOP and full_raw_df is not None:
        # exact instrument; fall back only to other Pascal-like diastolic sources
        # if you want — but do NOT silently mix Goldman/ORA conventions.
        df = _merge_measure(
            df,
            full_raw_df,
            "IOP",
            measure_names=[iop_instrument],
            on_keys=VISIT_EYE,
            numeric=True,
        )
        if verbose:
            n = df["IOP"].notna().sum()
            print(f"After merging IOP ({iop_instrument}), {n}/{len(df)} have a value.")

    for name in wanted:
        measure_names, on_keys = MEASURE_SPECS[name]
        df = _merge_measure(df, full_raw_df, name, measure_names, on_keys, numeric=True)
        if verbose:
            n = df[name].notna().sum()
            grain = "/".join(on_keys)
            print(f"After merging {name} on ({grain}), {n}/{len(df)} have a value.")

    if which_study is not None:
        # An eye can be enrolled in several studies at once (59 of them are), so
        # the Studies table holds more than one row per (PatientId, Eye). Select
        # the study *first*, then filter by membership: de-duplicating on
        # (PatientId, Eye) before the selection keeps whichever row came first
        # and silently drops eyes that genuinely belong to the requested study
        # (42 of the 144 prospective eyes, i.e. ~30% of the arm).
        with sqlite3.connect(STUDY_PATH) as con:
            study_df = pd.read_sql_query(
                "SELECT PatientId, Eye FROM Studies WHERE Study = ?",
                con,
                params=(which_study.value,),
            )
        study_df["Eye"] = study_df["Eye"].astype(str).str.strip()
        members = set(
            map(tuple, study_df.drop_duplicates()[["PatientId", "Eye"]].to_numpy())
        )
        keep = [
            (pid, eye) in members
            for pid, eye in zip(df["PatientId"], df["Eye"].astype(str).str.strip())
        ]
        df = df[keep].copy()
        # Constant by construction: the rows that survive are exactly the ones
        # enrolled in `which_study`. Kept so callers can still read df["Study"].
        df["Study"] = which_study.value
        if verbose:
            n_pat = df["PatientId"].nunique()
            print(
                f"After filtering to study '{which_study.value}', {len(df)} rows "
                f"({n_pat} patients) remain."
            )

    df = manual_correction(df)

    return df


def filter_misregistration(df, misregration):
    if isinstance(misregration, str):
        misregration = Path(misregration)
    if isinstance(misregration, Path) and misregration.exists():
        misreg = pd.read_csv(misregration)

    misreg_cases = misreg[misreg.flag].video.unique()
    df = df[~df["video"].isin(misreg_cases)]
    return df


def manual_correction(df: pd.DataFrame) -> pd.DataFrame:
    """Apply manual corrections to the given dataframe"""

    df.loc[
        (df.PatientId == 686) & (df.Date == "2025-02-19") & (df.Eye == "OS"), "OPA"
    ] = 3.62
    return df
