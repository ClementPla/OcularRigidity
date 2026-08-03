"""Interactive regression explorer — choose what to regress against what."""

import itertools

import pandas as pd
import streamlit as st

from ocularrigidity.viewer import cohort_data as C
from ocularrigidity.viewer.streamlit_explorer._common import (
    cached_case_table,
    cached_deltaA,
    cached_deltaCT,
    require_selection,
    show_regression,
)

st.set_page_config(page_title="Regression", layout="wide")

sel = require_selection()
st.title(f"Regression — {sel.method_label} · {sel.cohort_label}")

mode = st.radio(
    "Mode", ["Two metrics", "Test–retest (cycle vs cycle)"], horizontal=True
)

# --- mode 1: any X vs any Y --------------------------------------------------
if mode == "Two metrics":
    df = cached_case_table(sel)
    numeric = [c for c in C.METRIC_COLUMNS if c in df.columns]

    c1, c2, c3 = st.columns(3)
    x = c1.selectbox(
        "X",
        numeric,
        index=numeric.index("deltaCT_estimated") if "deltaCT_estimated" in numeric else 0,
    )
    y = c2.selectbox(
        "Y", numeric, index=numeric.index("deltaCT") if "deltaCT" in numeric else 1
    )
    color = c3.selectbox("Colour by", ["(none)", "Eye"])

    o1, o2, o3 = st.columns(3)
    trim = o1.slider("Outlier trim (keep central quantile)", 0.80, 1.0, 1.0, 0.01)
    logx = o2.checkbox("log X")
    logy = o3.checkbox("log Y")

    ids = [c for c in ["case_id", "PatientId", "Date", "Eye"] if c in df.columns]
    data = df[[x, y] + ids].dropna(subset=[x, y])
    if trim < 1.0:
        data = C.trim_outliers(data, [x, y], trim)

    show_regression(
        data, x, y, color=None if color == "(none)" else color, logx=logx, logy=logy
    )

# --- mode 2: cycle c0 vs cycle c1 (test–retest reproducibility) --------------
else:
    metric = st.selectbox("Metric", ["deltaCT", "minCT", "RelativeGrowth", "deltaA"])
    per_cycle = cached_deltaA(sel) if metric == "deltaA" else cached_deltaCT(sel)
    cycles = sorted(per_cycle["cycle"].unique())
    if len(cycles) < 2:
        st.warning("Need at least two cycles for a test–retest comparison.")
        st.stop()

    pairs = list(itertools.combinations(cycles, 2))
    pair = st.selectbox(
        "Cycle pair", pairs, format_func=lambda p: f"cycle {p[0]} vs cycle {p[1]}"
    )
    trim = st.slider("Outlier trim (keep central quantile)", 0.80, 1.0, 0.99, 0.01)

    c0, c1 = pair
    a = per_cycle[per_cycle["cycle"] == c0].set_index("case_id")[metric]
    b = per_cycle[per_cycle["cycle"] == c1].set_index("case_id")[metric]
    merged = (
        pd.concat({f"{metric}_c{c0}": a, f"{metric}_c{c1}": b}, axis=1)
        .dropna()
        .reset_index()
    )
    xcol, ycol = f"{metric}_c{c0}", f"{metric}_c{c1}"
    if trim < 1.0:
        merged = C.trim_outliers(merged, [xcol, ycol], trim)

    show_regression(
        merged,
        xcol,
        ycol,
        x_label=f"Cycle {c0} {metric}",
        y_label=f"Cycle {c1} {metric}",
    )
