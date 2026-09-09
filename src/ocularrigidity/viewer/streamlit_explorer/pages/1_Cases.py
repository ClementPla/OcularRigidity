"""The cohort table itself: filter it, read it, export it."""

import streamlit as st

from ocularrigidity.data.measurements.cohort import column_groups
from ocularrigidity.viewer.streamlit_explorer._common import (
    cached_cohort,
    require_selection,
)

st.set_page_config(page_title="Cases", layout="wide")

sel = require_selection()
df = cached_cohort(sel)
groups = column_groups(df)

st.title(f"Cases — {sel.cohort_label}")

# --- filters -----------------------------------------------------------------
f1, f2, f3, f4 = st.columns([1, 1, 1, 2])
only_k = f1.checkbox("Only visits with K", value=False)
only_onh = f1.checkbox("Only visits with ONH", value=False)
eyes = sorted(df["Eye"].dropna().unique())
sel_eyes = f2.multiselect("Eye", eyes, default=eyes)
types = sorted(df["Type"].dropna().unique()) if "Type" in df.columns else []
sel_types = f3.multiselect("Diagnosis type", types, default=types)
search = f4.text_input("Filter by case / patient (substring)", "")

view = df
if only_k:
    view = view[view["K"].notna()]
if only_onh and groups["onh"]:
    view = view[view[groups["onh"][0]].notna()]
if sel_eyes:
    view = view[view["Eye"].isin(sel_eyes)]
if sel_types and types:
    view = view[view["Type"].isin(sel_types) | view["Type"].isna()]
if search.strip():
    s = search.strip()
    view = view[
        view["case_id"].str.contains(s, case=False, na=False)
        | view["PatientId"].astype(str).str.contains(s, case=False, na=False)
    ]

# --- which blocks of columns to show -----------------------------------------
blocks = st.multiselect(
    "Column blocks",
    list(groups),
    default=["identity", "pulsation", "covariates"],
    help="The table is wide: identity, the pulsatile metrics, the clinical "
    "covariates, the ONH sectors and the longitudinal clinical measures.",
)
shown = [c for b in blocks for c in groups[b]]
st.caption(f"{len(view)} / {len(df)} visits · {len(shown)} columns")

fmt = {
    "deltaCT": "%.4f",
    "deltaCT_Mask": "%.4f",
    "minCT": "%.3f",
    "RelativeGrowth": "%.4f",
    "K": "%.4f",
    "K_Mask": "%.4f",
    "thickening_um_s": "%.1f",
    "thinning_um_s": "%.1f",
    "rate_asymmetry": "%.2f",
    "IOP": "%.1f",
    "OPA": "%.1f",
    "AxialLength": "%.2f",
    "HR": "%.0f",
    "predicted_HR": "%.0f",
    "Age": "%.1f",
}
col_config = {
    c: st.column_config.NumberColumn(c, format=fmt[c]) for c in fmt if c in shown
}

st.dataframe(
    view[shown] if shown else view,
    width="stretch",
    column_config=col_config,
    height=560,
)

st.download_button(
    "Download CSV (all columns)",
    view.to_csv().encode(),
    file_name=f"cohort_{sel.cohort_label.replace(' ', '_')}.csv",
    mime="text/csv",
)

with st.expander("Summary statistics"):
    numeric = [c for c in shown if c in view.columns and view[c].dtype.kind in "fi"]
    if numeric:
        st.dataframe(view[numeric].describe().T, width="stretch")
