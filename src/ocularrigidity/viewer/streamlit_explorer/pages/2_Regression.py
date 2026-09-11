"""Interactive regression explorer — any variable against any other.

Two modes: *any X vs any Y* over the whole cohort table (so a pulsatile metric
can be confronted with an ONH sector or a clinical measure directly), and
*test–retest*, which regresses one cardiac cycle against another and is the
honest ceiling on everything the first mode can find.

The first mode also has a **Δ Y** switch, which turns the scatter from one point
per visit into one point per eye: X at the eye's earliest visit against the
per-year slope of Y from there on. That is the Longitudinal page's
*Present → future* design, except that both variables are picked by hand rather
than being fixed to (rigidity probe × clinical measure).
"""

import itertools

import pandas as pd
import streamlit as st

from ocularrigidity.data.measurements.cohort import column_groups
from ocularrigidity.stats.temporal import baseline_vs_slope_wide
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
    y = c2.selectbox(
        "Y", numeric, index=_default("G BMO MRW", min(1, len(numeric) - 1))
    )
    delta_y = c2.checkbox(
        "Δ Y per year, against X at baseline",
        help="One point per eye instead of one per visit: X is read at the eye's "
        "earliest visit and Y is replaced by its per-year slope from that visit "
        "onward. Same design as the Longitudinal page's *Present → future*, but "
        "with both variables picked by hand.",
    )
    color_by = [
        c
        for c in ("(none)", "Eye", "Type", "Diagnosis", "Sex", "Study")
        if c in df.columns or c == "(none)"
    ]
    color = c3.selectbox("Colour by", color_by)

    o1, o2, o3 = st.columns(3)
    trim = o1.slider("Outlier trim (keep central quantile)", 0.80, 1.0, 1.0, 0.01)
    logx = o2.checkbox("log X")
    # A slope is signed, so log Y is only offered on the per-visit value.
    logy = o3.checkbox("log Y") if not delta_y else False
    min_points = (
        o3.slider(
            "Min visits per eye",
            2,
            6,
            3,
            help="Two visits make a slope that is arithmetic rather than "
            "progression; three is the first value that fits anything.",
        )
        if delta_y
        else 2
    )

    ids = [
        c for c in ("case_id", "caseId", "PatientId", "Date", "Eye") if c in df.columns
    ]

    if delta_y:
        missing = [c for c in ("PatientId", "Eye", "YearMonth") if c not in df.columns]
        if missing:
            st.error(f"Δ Y needs {', '.join(missing)} to group the visits by eye.")
            st.stop()
        data = baseline_vs_slope_wide(df, x, y, min_points=min_points)
        xcol, ycol = f"{x}_baseline", f"{y}_slope"
        if color != "(none)":
            # An eye's colour is whatever it was at the visit the baseline came
            # from; Type and Diagnosis can change between visits.
            first = (
                df.sort_values("Date")
                .groupby(["PatientId", "Eye"], as_index=False)[color]
                .first()
            )
            data = data.merge(first, on=["PatientId", "Eye"], how="left")
        hover = ["PatientId", "Eye", "n_visits", "span_years"]
        labels = (f"{x} at baseline", f"Δ{y} / year")
    else:
        keep = list(
            dict.fromkeys([x, y, *ids] + ([color] if color != "(none)" else []))
        )
        data = df[keep].dropna(subset=[x, y])
        xcol, ycol = x, y
        hover = ids
        labels = (x, y)

    if trim < 1.0:
        data = C.trim_outliers(data, [xcol, ycol], trim)

    if delta_y:
        st.info(
            f"One point per eye: **{x}** at the earliest visit against the per-year "
            f"slope of **{y}** from there on — {len(data)} eyes. The repeated "
            "visits no longer inflate N, but the two eyes of one patient are "
            "still correlated, and an eye counts the same whether its slope came "
            "from two visits or six (`n_visits` on hover).",
            icon="ℹ️",
        )
        if x == y:
            st.warning(
                f"X and Y are both {x}: a baseline regressed on its own subsequent "
                "slope is negatively biased by construction — the noise in the "
                "first visit enters X positively and the slope negatively. Expect "
                "a downward trend whether or not anything real is happening.",
                icon="⚠️",
            )
    else:
        st.info(
            "Every visit is one point, so an eye seen four times counts four times. "
            "The Pearson / Spearman p below reads them as independent — for the "
            "repeated-measures version, use the **Longitudinal** page.",
            icon="⚠️",
        )

    show_regression(
        data,
        xcol,
        ycol,
        x_label=labels[0],
        y_label=labels[1],
        color=None if color == "(none)" else color,
        logx=logx,
        logy=logy,
        hover=hover,
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
