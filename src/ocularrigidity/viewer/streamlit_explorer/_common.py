"""Shared Streamlit helpers: cached loaders and the sidebar method selector.

Imported by every page via the absolute package path so it resolves no matter
which script Streamlit launches.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Sequence

import pandas as pd
import plotly.express as px
import streamlit as st

from ocularrigidity.consts import ROOT_CARDIAC_PIPELINE
from ocularrigidity.data.measurements.studies import Study
from ocularrigidity.viewer import cohort_data as C
from ocularrigidity.viewer import longitudinal as L

HOVER_IDS = ("case_id", "PatientId", "Date", "Eye")


class Selection(NamedTuple):
    """What the sidebar picked. A tuple, so it keys the cached loaders directly."""

    root: str
    suffix: str
    iop: str
    study: Study | None
    exclude_qc: bool

    @property
    def method_label(self) -> str:
        return C.pretty_method(self.suffix)

    @property
    def cohort_label(self) -> str:
        return self.study.value if self.study else "all studies"


@st.cache_data(show_spinner="Building case table…")
def cached_case_table(sel: Selection) -> pd.DataFrame:
    excluded = C.load_excluded_cases() if sel.exclude_qc else None
    return C.build_case_table(
        sel.root, sel.suffix, sel.iop, study=sel.study, excluded_cases=excluded
    )


@st.cache_data(show_spinner=False)
def cached_deltaA(sel: Selection) -> pd.DataFrame:
    return C.load_deltaA_per_cycle(sel.root, sel.suffix)


@st.cache_data(show_spinner="Measuring ΔCT (first run walks every mask)…")
def cached_deltaCT(sel: Selection) -> pd.DataFrame:
    df = C.load_deltaCT_per_cycle(sel.root, sel.suffix)
    if sel.exclude_qc:
        df = df[~df["case_id"].isin(C.load_excluded_cases())]
    return df


@st.cache_data(show_spinner="Joining clinical measures…")
def cached_clinical_long(sel: Selection) -> pd.DataFrame:
    return C.load_clinical_long(cached_case_table(sel))


@st.cache_data(show_spinner=False)
def cached_design(
    sel: Selection,
    probe: str,
    design: str,
    measure: str,
    params: tuple[tuple[str, object], ...],
) -> tuple[pd.DataFrame, str, str]:
    """One design's plotting frame — cached for the same reason as the screening.

    Every tab body reruns on every click, so the ~60 frames behind the plots would
    otherwise be rebuilt each time.
    """
    return L.build(design, cached_clinical_long(sel), probe, measure, dict(params))


@st.cache_data(show_spinner="Screening every clinical measure…")
def cached_screen(
    sel: Selection, probe: str, design: str, params: tuple[tuple[str, object], ...]
) -> pd.DataFrame:
    """Rank every measure under one design — a model per measure, so cached.

    Streamlit runs *every* tab body on *every* rerun, so without this a single
    click would re-fit all six designs over all ~40 measures. ``params`` is the
    design's settings as sorted key/value pairs, which keeps it hashable (and
    means moving one tab's slider only invalidates that tab's ranking).
    """
    long_df = cached_clinical_long(sel)
    return L.screen(design, long_df, probe, C.available_measures(long_df), dict(params))


def sidebar_selector() -> Selection | None:
    """Root / method / cohort picker shared across pages; persists in session."""
    st.sidebar.header("Experiment")
    root = st.sidebar.text_input(
        "Experiments root",
        value=st.session_state.get("root", str(ROOT_CARDIAC_PIPELINE)),
    )
    st.session_state["root"] = root

    methods = C.discover_methods(root) if Path(root).is_dir() else []
    if not methods:
        st.sidebar.error("No `measures_*` method folders under this root.")
        return None

    labels = {C.pretty_method(m): m for m in methods}
    prev = st.session_state.get("method")
    default = next((lbl for lbl, s in labels.items() if s == prev), list(labels)[0])
    label = st.sidebar.selectbox(
        "Method", list(labels), index=list(labels).index(default)
    )
    suffix = labels[label]
    st.session_state["method"] = suffix

    st.sidebar.header("Cohort")
    studies = {"All": None} | {s.value.capitalize(): s for s in Study}
    study = studies[st.sidebar.selectbox("Study", list(studies))]
    exclude_qc = st.sidebar.checkbox(
        "Exclude QC-rejected cases",
        value=True,
        help=f"Drops the {len(C.load_excluded_cases())} cases flagged in the gif viewer's errors.json.",
    )
    iop = st.sidebar.selectbox(
        "IOP instrument", ["Pascal IOP", "Goldman IOP", "ORA IOPcc"], index=0
    )
    return Selection(root, suffix, iop, study, exclude_qc)


def require_selection() -> Selection:
    """Run the selector and stop the page if no valid method is chosen."""
    sel = sidebar_selector()
    if sel is None:
        st.warning("Pick a valid experiments root in the sidebar to continue.")
        st.stop()
    return sel


def show_regression(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    x_label: str | None = None,
    y_label: str | None = None,
    color: str | None = None,
    logx: bool = False,
    logy: bool = False,
    height: int = 620,
    hover: Sequence[str] = HOVER_IDS,
    show_stats: bool = True,
) -> None:
    """Stats row (N / r / ρ / slope) + OLS scatter — the notebook's regression plot.

    ``show_stats=False`` drops the Pearson row, for the designs whose rows repeat
    within an eye and whose inference therefore has to be clustered instead.
    """
    s = C.regression_stats(df, x, y)
    if s.get("n", 0) < 3:
        st.warning(f"Not enough finite points to regress {y} on {x} (need ≥ 3).")
        return

    if show_stats:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("N", s["n"])
        c2.metric(
            "Pearson r", f"{s['pearson_r']:.3f}", help=f"p = {s['pearson_p']:.2e}"
        )
        c3.metric(
            "Spearman ρ", f"{s['spearman_rho']:.3f}", help=f"p = {s['spearman_p']:.2e}"
        )
        sign = "+" if s["intercept"] >= 0 else "−"
        c4.metric(
            "Slope",
            f"{s['slope']:.4g}",
            help=f"y = {s['slope']:.4g}·x {sign} {abs(s['intercept']):.4g}",
        )

    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        trendline="ols",
        trendline_scope="overall",
        hover_data=[c for c in hover if c in df.columns],
        labels={x: x_label or x, y: y_label or y},
        log_x=logx,
        log_y=logy,
        opacity=0.6,
    )
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")


def show_box(
    df: pd.DataFrame,
    value: str,
    group: str,
    *,
    logy: bool = False,
    height: int = 520,
) -> None:
    """Box plot of ``value`` by ``group``, annotated with the per-group N."""
    d = df.dropna(subset=[value, group])
    if d.empty:
        st.warning(f"No rows with both {value} and {group}.")
        return
    counts = d.groupby(group)[value].count()
    fig = px.box(
        d,
        x=group,
        y=value,
        points="outliers",
        color=group,
        log_y=logy,
        hover_data=[c for c in HOVER_IDS if c in d.columns],
    )
    fig.update_layout(
        height=height,
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title=f"{group}  ({', '.join(f'{k}: N={v}' for k, v in counts.items())})",
    )
    st.plotly_chart(fig, width="stretch")
