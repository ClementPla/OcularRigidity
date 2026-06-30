"""Streamlit cohort browser — home / overview.

Launch with::

    streamlit run src/ocularrigidity/viewer/streamlit_explorer/Home.py

Pick an **experiments root** and a **method** in the sidebar (the choice carries
across pages). Then use the pages:

* **Cases** — per-case table of ΔA, ΔCT and Friedenwald K.
* **Regression** — pick any two metrics and regress them interactively.

Everything is read from precomputed cohort outputs; nothing is recomputed.
"""

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
    "in the sidebar, then open **Cases** or **Regression** from the page menu."
)

root, suffix, iop = require_selection()
df = cached_case_table(root, suffix, iop)

st.subheader(f"Overview — {C.pretty_method(suffix)}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Cases", len(df))
c2.metric("With clinical (K)", int(df["K_thickness"].notna().sum()))
c3.metric("Median ΔA (px²)", f"{np.nanmedian(df['deltaA']):.0f}")
c4.metric("Median K (1/µL)", f"{np.nanmedian(df['K_thickness']):.4f}")

st.markdown(
    "- **ΔA** — pulsatile choroidal area change (px², median over cycles)\n"
    "- **ΔCT** — measured choroidal-thickness change (µm); "
    "**ΔCT_estimated** is derived from ΔA\n"
    "- **K_area / K_thickness** — Friedenwald rigidity (1/µL) from ΔA and from ΔCT\n"
    "- Clinical **IOP / OPA / AxialLength** are joined from the measurements DB"
)

with st.expander("Distributions", expanded=True):
    cols = st.multiselect(
        "Columns",
        [c for c in C.METRIC_COLUMNS if c in df.columns],
        default=["deltaA", "deltaCT", "K_thickness"],
    )
    if cols:
        st.bar_chart(df[cols].describe().T[["mean", "50%", "std"]])
        st.dataframe(df[cols].describe().T, width="stretch")
