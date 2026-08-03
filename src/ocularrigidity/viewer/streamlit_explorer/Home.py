import numpy as np
import streamlit as st

from ocularrigidity.viewer import cohort_data as C
from ocularrigidity.viewer.streamlit_explorer._common import (
    cached_case_table,
    require_selection,
)

st.set_page_config(page_title="Ocular Rigidity — cohort browser", layout="wide")

st.title("Ocular Rigidity — cohort browser")
st.markdown(
    "Browse the precomputed cardiac-pipeline cohort outputs. Choose a **method** "
    "and a **cohort** in the sidebar, then open **Cases**, **Regression** or "
    "**Longitudinal** from the page menu."
)

sel = require_selection()
df = cached_case_table(sel)

st.subheader(f"Overview — {sel.method_label} · {sel.cohort_label}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Cases", len(df))
c2.metric("With K", int(df["K"].notna().sum()))
c3.metric("Median ΔCT (µm)", f"{np.nanmedian(df['deltaCT']):.2f}")
c4.metric("Median K (1/µL)", f"{np.nanmedian(df['K']):.4f}")

st.markdown(
    "- **ΔCT** — *measured* pulsatile choroidal-thickness change (µm, median over "
    "cycles), tracked at the choroid-sclera interface; **ΔCT_estimated** is the "
    "area-derived proxy (from ΔA)\n"
    "- **minCT** — absolute choroidal thickness (µm); **RelativeGrowth** = ΔCT / minCT, "
    "the pulsation as a fraction of the choroid itself\n"
    "- **K** — Friedenwald rigidity (1/µL) from the measured ΔCT, shell radius "
    "corrected by minCT; **K_area** is the ΔA-derived counterpart\n"
    "- Clinical **IOP / OPA / AxialLength / HR** are joined from the measurements DB"
)

with st.expander("Distributions", expanded=True):
    cols = st.multiselect(
        "Columns",
        [c for c in C.METRIC_COLUMNS if c in df.columns],
        default=["deltaCT", "minCT", "RelativeGrowth", "K"],
    )
    if cols:
        st.dataframe(df[cols].describe().T, width="stretch")
