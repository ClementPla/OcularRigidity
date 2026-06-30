"""Interactive regression explorer — choose what to regress against what."""

import itertools

import pandas as pd
import plotly.express as px
import streamlit as st

from ocularrigidity.viewer import cohort_data as C
from ocularrigidity.viewer.streamlit_explorer._common import (
    cached_case_table,
    cached_deltaA,
    cached_deltaCT,
    require_selection,
)

st.set_page_config(page_title="Regression", layout="wide")

root, suffix, iop = require_selection()
st.title(f"Regression — {C.pretty_method(suffix)}")


def _show_stats(df: pd.DataFrame, x: str, y: str) -> None:
    s = C.regression_stats(df, x, y)
    if s.get("n", 0) < 3:
        st.warning("Not enough finite points to regress (need ≥ 3).")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("N", s["n"])
    c2.metric("Pearson r", f"{s['pearson_r']:.3f}", help=f"p = {s['pearson_p']:.2e}")
    c3.metric("Spearman ρ", f"{s['spearman_rho']:.3f}", help=f"p = {s['spearman_p']:.2e}")
    sign = "+" if s["intercept"] >= 0 else "−"
    c4.metric("Slope", f"{s['slope']:.4g}", help=f"y = {s['slope']:.4g}·x {sign} {abs(s['intercept']):.4g}")


def _scatter(df, x, y, color, logx, logy):
    fig = px.scatter(
        df, x=x, y=y, color=color if color != "(none)" else None,
        trendline="ols", trendline_scope="overall",
        hover_data=[c for c in ["case_id", "PatientId", "Date", "Eye"] if c in df.columns],
        log_x=logx, log_y=logy, opacity=0.6,
    )
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")


mode = st.radio(
    "Mode", ["Two metrics", "Test–retest (cycle vs cycle)"], horizontal=True
)

# --- mode 1: any X vs any Y --------------------------------------------------
if mode == "Two metrics":
    df = cached_case_table(root, suffix, iop)
    numeric = [c for c in C.METRIC_COLUMNS if c in df.columns]
    categorical = ["(none)", "Eye"]

    c1, c2, c3 = st.columns(3)
    x = c1.selectbox("X", numeric, index=numeric.index("deltaCT_estimated") if "deltaCT_estimated" in numeric else 0)
    y = c2.selectbox("Y", numeric, index=numeric.index("deltaCT") if "deltaCT" in numeric else 1)
    color = c3.selectbox("Colour by", categorical)

    o1, o2, o3 = st.columns(3)
    trim = o1.slider("Outlier trim (keep central quantile)", 0.80, 1.0, 1.0, 0.01)
    logx = o2.checkbox("log X")
    logy = o3.checkbox("log Y")

    ids = [c for c in ["case_id", "PatientId", "Date", "Eye"] if c in df.columns]
    data = df[[x, y] + ids].dropna(subset=[x, y])
    if trim < 1.0:
        data = C.trim_outliers(data, [x, y], trim)

    _show_stats(data, x, y)
    _scatter(data, x, y, color, logx, logy)

# --- mode 2: cycle c0 vs cycle c1 (test–retest reproducibility) --------------
else:
    metric = st.selectbox("Metric", ["deltaA", "deltaCT"])
    per_cycle = cached_deltaA(root, suffix) if metric == "deltaA" else cached_deltaCT(root, suffix)
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

    _show_stats(merged, xcol, ycol)
    _scatter(merged, xcol, ycol, "(none)", False, False)
