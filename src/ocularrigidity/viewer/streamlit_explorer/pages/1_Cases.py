"""Per-case table of ΔA, ΔCT and Friedenwald K for the selected method."""

import streamlit as st

from ocularrigidity.viewer import cohort_data as C
from ocularrigidity.viewer.streamlit_explorer._common import (
    cached_case_table,
    require_selection,
)

st.set_page_config(page_title="Cases", layout="wide")

root, suffix, iop = require_selection()
df = cached_case_table(root, suffix, iop)

st.title(f"Cases — {C.pretty_method(suffix)}")

# --- filters -----------------------------------------------------------------
f1, f2, f3 = st.columns([1, 1, 2])
only_k = f1.checkbox("Only cases with K", value=False)
eyes = sorted(df["Eye"].dropna().unique())
sel_eyes = f2.multiselect("Eye", eyes, default=eyes)
search = f3.text_input("Filter by case_id / patient (substring)", "")

view = df.copy()
if only_k:
    view = view[view["K_thickness"].notna()]
if sel_eyes:
    view = view[view["Eye"].isin(sel_eyes)]
if search.strip():
    s = search.strip()
    view = view[
        view["case_id"].str.contains(s, case=False, na=False)
        | view["PatientId"].astype(str).str.contains(s, case=False, na=False)
    ]

st.caption(f"{len(view)} / {len(df)} cases")

# Per-column number formatting.
fmt = {
    "deltaA": "%.0f", "minimal_area": "%.0f",
    "deltaCT": "%.2f", "deltaCT_estimated": "%.2f",
    "IOP": "%.1f", "OPA": "%.1f", "AxialLength": "%.2f",
    "dV_uL_area": "%.3f", "dV_uL_thickness": "%.3f",
    "K_area": "%.4f", "K_thickness": "%.4f",
}
col_config = {
    c: st.column_config.NumberColumn(c, format=fmt[c]) for c in fmt if c in view.columns
}

st.dataframe(
    view, width="stretch", hide_index=True, column_config=col_config, height=560
)

st.download_button(
    "Download CSV",
    view.to_csv(index=False).encode(),
    file_name=f"cases_{suffix}.csv",
    mime="text/csv",
)

with st.expander("Summary statistics"):
    st.dataframe(
        view[[c for c in C.METRIC_COLUMNS if c in view.columns]].describe().T,
        width="stretch",
    )
