"""Interactive regression explorer — any variable against any other.

Two modes: *any X vs any Y* over the whole cohort table (so a pulsatile metric
can be confronted with an ONH sector or a clinical measure directly), and
*test–retest*, which regresses one cardiac cycle against another and is the
honest ceiling on everything the first mode can find.
"""

import itertools

import pandas as pd
import streamlit as st

from ocularrigidity.data.measurements.cohort import column_groups
from ocularrigidity.viewer import cohort_data as C
from ocularrigidity.viewer.streamlit_explorer._common import (
    cached_cohort,
    cached_cycles,
    require_selection,
    show_regression,
)

st.set_page_config(page_title="Regression", layout="wide")

sel = require_selection()
st.title(f"Regression — {sel.cohort_label}")

mode = st.radio(
    "Mode", ["Two variables", "Test–retest (cycle vs cycle)"], horizontal=True
)

# --- mode 1: any X vs any Y --------------------------------------------------
if mode == "Two variables":
    df = cached_cohort(sel)
    groups = column_groups(df)
    numeric = [
        c
        for block in ("pulsation", "covariates", "onh", "clinical")
        for c in groups[block]
        if c in df.columns and df[c].dtype.kind in "fi"
    ]

    def _default(name: str, fallback: int) -> int:
        return numeric.index(name) if name in numeric else fallback

    c1, c2, c3 = st.columns(3)
    x = c1.selectbox("X", numeric, index=_default("K", 0))
    y = c2.selectbox("Y", numeric, index=_default("G BMO MRW", min(1, len(numeric) - 1)))
    color_by = [
        c for c in ("(none)", "Eye", "Type", "Diagnosis", "Sex", "Study") if c in df.columns or c == "(none)"
    ]
    color = c3.selectbox("Colour by", color_by)

    o1, o2, o3 = st.columns(3)
    trim = o1.slider("Outlier trim (keep central quantile)", 0.80, 1.0, 1.0, 0.01)
    logx = o2.checkbox("log X")
    logy = o3.checkbox("log Y")

    ids = [c for c in ("case_id", "caseId", "PatientId", "Date", "Eye") if c in df.columns]
    keep = list(dict.fromkeys([x, y, *ids] + ([color] if color != "(none)" else [])))
    data = df[keep].dropna(subset=[x, y])
    if trim < 1.0:
        data = C.trim_outliers(data, [x, y], trim)

    st.info(
        "Every visit is one point, so an eye seen four times counts four times. "
        "The Pearson / Spearman p below reads them as independent — for the "
        "repeated-measures version, use the **Longitudinal** page.",
        icon="⚠️",
    )
    show_regression(
        data,
        x,
        y,
        color=None if color == "(none)" else color,
        logx=logx,
        logy=logy,
        hover=ids,
    )

# --- mode 2: cycle c0 vs cycle c1 (test–retest reproducibility) --------------
else:
    per_cycle = cached_cycles(sel)
    metrics = [
        c
        for c in (
            "deltaCT",
            "deltaCT_Mask",
            "minCT",
            "RelativeGrowth",
            "thickening_um_s",
            "thinning_um_s",
            "rate_asymmetry",
            "thickening_fraction",
        )
        if c in per_cycle.columns
    ]
    metric = st.selectbox("Metric", metrics)
    cycles = sorted(per_cycle["cycle"].unique())
    if len(cycles) < 2:
        st.warning("Need at least two cycles for a test–retest comparison.")
        st.stop()

    pairs = list(itertools.combinations(cycles, 2))
    pair = st.selectbox(
        "Cycle pair", pairs, format_func=lambda p: f"cycle {p[0]} vs cycle {p[1]}"
    )
    trim = st.slider("Outlier trim (keep central quantile)", 0.80, 1.0, 0.99, 0.01)

    st.caption(
        "The same eye, the same video, two different cardiac cycles: this is the "
        "metric measuring itself. The correlation here is the ceiling on any "
        "association it can have with anything else — √ICC, more precisely."
    )

    c0, c1 = pair
    a = per_cycle[per_cycle["cycle"] == c0].set_index("video")[metric]
    b = per_cycle[per_cycle["cycle"] == c1].set_index("video")[metric]
    merged = (
        pd.concat({f"{metric}_c{c0}": a, f"{metric}_c{c1}": b}, axis=1)
        .dropna()
        .reset_index()
        .rename(columns={"video": "case_id"})
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
        hover=["case_id"],
    )
