"""Shared Streamlit helpers: cached loaders and the sidebar method selector.

Imported by every page via the absolute package path so it resolves no matter
which script Streamlit launches.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import NamedTuple, Sequence

import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components
from plotly.offline import get_plotlyjs_version

from ocularrigidity.consts import ROOT_CARDIAC_PIPELINE
from ocularrigidity.data.measurements.cohort import (
    DEFAULT_MEASURES,
    build_cohort,
    cohort_to_long,
    load_excluded_cases,
)
from ocularrigidity.data.measurements.dataframe import filter_misregistration
from ocularrigidity.data.measurements.pulsation_results import load_pulsation_results
from ocularrigidity.data.measurements.studies import Study
from ocularrigidity.viewer import cohort_data as C
from ocularrigidity.viewer import longitudinal as L

HOVER_IDS = ("case_id", "PatientId", "Date", "Eye")

#: Browser-side statistics for :func:`show_regression`. Read once at import.
_LIVE_STATS_JS = (Path(__file__).parent / "_live_stats.js").read_text()
_PLOTLY_JS = f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"


class Selection(NamedTuple):
    """What the sidebar picked. A tuple, so it keys the cached loaders directly."""

    root: str
    iop: str
    study: Study | None
    exclude_qc: bool

    @property
    def cohort_label(self) -> str:
        return self.study.value if self.study else "all studies"


@st.cache_data(show_spinner="Building the cohort table…")
def cached_cohort(sel: Selection) -> pd.DataFrame:
    """The one wide table every cohort page reads — see :func:`build_cohort`.

    The first call for a root has to measure ΔCT over every mask (minutes);
    it is pickled next to the pipeline outputs, so later runs are instant.
    """
    return build_cohort(
        sel.root,
        study=sel.study,
        iop_instrument=sel.iop,
        exclude_qc=sel.exclude_qc,
    )


@st.cache_data(show_spinner=False)
def cached_cycles(sel: Selection) -> pd.DataFrame:
    """Per-(video, cardiac cycle) metrics — the test-retest / reliability input."""
    _cases, cycles = load_pulsation_results(Path(sel.root))
    cycles = filter_misregistration(
        cycles, Path(sel.root) / "misregistration_flags.csv"
    )
    cycles = cycles.rename(columns={"caseId_path": "video"})
    if sel.exclude_qc:
        cycles = cycles[~cycles["video"].isin(load_excluded_cases())]
    return cycles


@st.cache_data(show_spinner="Melting the measures…")
def cached_clinical_long(sel: Selection) -> pd.DataFrame:
    """The cohort table in the long shape the longitudinal designs read."""
    return cohort_to_long(cached_cohort(sel))


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
    return L.screen(design, long_df, probe, available_measures(long_df), dict(params))


def available_measures(long_df: pd.DataFrame) -> list[str]:
    """Measures present in the long frame, the ones worth testing first."""
    present = list(dict.fromkeys(long_df["MeasureName_y"].dropna()))
    head = [m for m in DEFAULT_MEASURES if m in present]
    return head + sorted(m for m in present if m not in head)


def sidebar_selector() -> Selection | None:
    """Root / cohort picker shared across pages; persists in session."""
    st.sidebar.header("Experiment")
    root = st.sidebar.text_input(
        "Experiments root",
        value=st.session_state.get("root", str(ROOT_CARDIAC_PIPELINE)),
    )
    st.session_state["root"] = root

    if not (Path(root).is_dir() and C.has_measures(root)):
        st.sidebar.error("No `measures/` folder under this root.")
        return None

    st.sidebar.header("Cohort")
    studies = {"All": None} | {s.value.capitalize(): s for s in Study}
    study = studies[st.sidebar.selectbox("Study", list(studies))]
    exclude_qc = st.sidebar.checkbox(
        "Exclude QC-rejected cases",
        value=True,
        help=f"Drops the {len(load_excluded_cases())} cases flagged in the gif viewer's errors.json.",
    )
    iop = st.sidebar.selectbox(
        "IOP instrument", ["Pascal IOP", "Goldman IOP", "ORA IOPcc"], index=0
    )
    return Selection(root, iop, study, exclude_qc)


def require_selection() -> Selection:
    """Run the selector and stop the page if no valid root is chosen."""
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
    """OLS scatter whose N / r / ρ / slope describe the points *currently shown*.

    The stats used to be a row of ``st.metric`` above the chart, and they went
    stale the moment anyone clicked a legend entry: Streamlit reruns Python on a
    widget change, and a Plotly legend click is not one, so the row kept
    reporting the whole cohort while the plot showed a subset. They now sit in a
    box inside the plot and are recomputed in the browser on every legend
    toggle, together with the fitted line — which turns "does this association
    survive dropping the OHT eyes?" into one click.

    The cost is that the figure is rendered through ``components.html`` rather
    than ``st.plotly_chart``: it needs its own plotly.js from the CDN, and it
    does not inherit the Streamlit theme (hence the explicit template).
    ``_live_stats.js`` holds the browser side; its formulas are the ones
    :func:`ocularrigidity.viewer.cohort_data.regression_stats` uses, verified
    against scipy.

    ``show_stats=False`` drops the box, for the designs whose rows repeat within
    an eye and whose inference therefore has to be clustered instead.
    """
    if C.regression_stats(df, x, y).get("n", 0) < 3:
        st.warning(f"Not enough finite points to regress {y} on {x} (need ≥ 3).")
        return

    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        hover_data=[c for c in hover if c in df.columns],
        labels={x: x_label or x, y: y_label or y},
        log_x=logx,
        log_y=logy,
        opacity=0.6,
        template="plotly_white",
    )
    # Empty on arrival: the browser fills both from whatever the legend shows.
    fig.add_scatter(
        x=[],
        y=[],
        mode="lines",
        name="OLS",
        meta={"role": "fit"},
        line=dict(color="#2b2b2b", width=2),
        hoverinfo="skip",
        showlegend=False,
    )
    if show_stats:
        fig.add_annotation(
            name="live-stats",
            text="",
            xref="paper",
            yref="paper",
            x=0.012,
            y=0.988,
            xanchor="left",
            yanchor="top",
            align="left",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="rgba(0,0,0,0.22)",
            borderwidth=1,
            borderpad=7,
            font=dict(size=12, family="ui-monospace, SFMono-Regular, Menlo, monospace"),
        )
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10))

    # Stable per configuration, so an unchanged chart is not remounted on rerun.
    div_id = (
        "reg-"
        + hashlib.md5(f"{x}|{y}|{color}|{logx}|{logy}|{len(df)}".encode()).hexdigest()[
            :12
        ]
    )
    opts = json.dumps({"logx": bool(logx), "showStats": bool(show_stats)})
    config = json.dumps({"responsive": True, "displaylogo": False})
    components.html(
        f"""
<div id="{div_id}" style="width:100%;height:{height}px;"></div>
<script src="{_PLOTLY_JS}" charset="utf-8"></script>
<script>{_LIVE_STATS_JS}</script>
<script>
if (window.Plotly) {{
  initLiveRegression("{div_id}", {pio.to_json(fig)}, {config}, {opts});
}} else {{
  document.getElementById("{div_id}").innerHTML =
    "<p style='font:13px system-ui;color:#b00'>plotly.js could not be loaded "
    + "from the CDN — this chart needs network access.</p>";
}}
</script>
""",
        height=height + 12,
        scrolling=False,
    )


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
