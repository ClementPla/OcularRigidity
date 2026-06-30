"""Shared Streamlit helpers: cached loaders and the sidebar method selector.

Imported by every page via the absolute package path so it resolves no matter
which script Streamlit launches.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ocularrigidity.consts import ROOT_CARDIAC_PIPELINE
from ocularrigidity.viewer import cohort_data as C


@st.cache_data(show_spinner="Building case table…")
def cached_case_table(root: str, suffix: str, iop_instrument: str) -> pd.DataFrame:
    return C.build_case_table(root, suffix, iop_instrument)


@st.cache_data(show_spinner=False)
def cached_deltaA(root: str, suffix: str) -> pd.DataFrame:
    return C.load_deltaA_per_cycle(root, suffix)


@st.cache_data(show_spinner=False)
def cached_deltaCT(root: str, suffix: str) -> pd.DataFrame:
    return C.load_deltaCT_per_cycle(root, suffix)


def sidebar_selector() -> tuple[str, str | None, str]:
    """Root + method picker shared across pages; selection persists in session."""
    st.sidebar.header("Experiment")
    root = st.sidebar.text_input(
        "Experiments root",
        value=st.session_state.get("root", str(ROOT_CARDIAC_PIPELINE)),
    )
    st.session_state["root"] = root

    methods = C.discover_methods(root) if Path(root).is_dir() else []
    if not methods:
        st.sidebar.error("No `measures_*` / `deltaY_*.pkl` method pairs under root.")
        return root, None, "Pascal IOP"

    labels = {C.pretty_method(m): m for m in methods}
    prev = st.session_state.get("method")
    default = next((lbl for lbl, s in labels.items() if s == prev), list(labels)[0])
    label = st.sidebar.selectbox(
        "Method", list(labels), index=list(labels).index(default)
    )
    suffix = labels[label]
    st.session_state["method"] = suffix

    iop = st.sidebar.selectbox(
        "IOP instrument", ["Pascal IOP", "Goldman IOP", "ORA IOPcc"], index=0
    )
    return root, suffix, iop


def require_selection() -> tuple[str, str, str]:
    """Run the selector and stop the page if no valid method is chosen."""
    root, suffix, iop = sidebar_selector()
    if suffix is None:
        st.warning("Pick a valid experiments root in the sidebar to continue.")
        st.stop()
    return root, suffix, iop
