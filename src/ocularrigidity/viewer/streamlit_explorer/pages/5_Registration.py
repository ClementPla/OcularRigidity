import hashlib
from pathlib import Path
import numpy as np
import streamlit as st

from ocularrigidity.viewer import cohort_data as C
from ocularrigidity.viewer import render as R
from ocularrigidity.viewer.streamlit_explorer._common import (
    cached_case_table,
    require_selection,
)
from ocularrigidity.registration.rigid import register_masks_by_displacement

st.set_page_config(page_title="Registration", layout="wide")


root, suffix, iop = require_selection()
st.title(f"Registration — {C.pretty_method(suffix)}")

df = cached_case_table(root, suffix, iop)
show_cols = [
    c
    for c in ["case_id", "PatientId", "Date", "Eye", "deltaA", "deltaCT", "K_thickness"]
    if c in df.columns
]
st.caption(
    "Select a case, tune the parameters in the sidebar, then **Run segmentation**."
)
with st.expander("Show case table"):
    event = st.dataframe(
        df[show_cols],
        on_select="rerun",
        selection_mode="single-row",
        hide_index=True,
        width="stretch",
        height=260,
        key="infer_cases",
    )
rows = event.selection.rows
if not rows:
    st.info("Select a case above to segment.")
    st.stop()
case = df.iloc[rows[0]]["case_id"]
# --- parameters --------------------------------------------------------------
sb = st.sidebar
with sb:
    correct_dx = st.checkbox(
        "Correct for lateral displacement", value=True, help="Align masks to cube"
    )
    flatten_rpe = st.checkbox(
        "Flatten RPE", value=True, help="Flatten cube to RPE before registration"
    )
    lateral_method = st.radio(
        "Lateral registration method",
        ["fullframe", "xcorr"],
        index=0,
        help="Method to compute lateral displacement",
    )


def _mkv_raw(root, suffix, case):
    return Path(root) / ".." / "compressed" / case / "cube.mp4"


def mask_raw(root, case):
    return Path(root) / ".." / "masks" / case / "mask.npz"


@st.cache_data(show_spinner="Registering masks…")
def registration(
    _masks_data, _frames_data, case, correct_dx, flatten_rpe, lateral_method
):
    registered_masks, registered_cube = register_masks_by_displacement(
        _masks_data,
        _frames_data,
        correct_dx=correct_dx,
        flatten_rpe=flatten_rpe,
        lateral_method=lateral_method,
    )
    return registered_masks, registered_cube


def _register_and_render():
    mkv = _mkv_raw(root, suffix, case)
    max_frame = 50
    indices = np.arange(10, max_frame)
    with st.spinner("Loading video…"):
        raw_frames = R.read_cube(
            str(mkv), _indices=indices
        )  # (T, H, W) — segment on full frames
    with st.spinner("Loading mask…"):
        raw_masks = R.read_masks(str(mask_raw(root, case)))  # (T, H, W)

    raw_masks = raw_masks[indices]

    registered_masks, registered_frames = registration(
        raw_masks, raw_frames, case, correct_dx, flatten_rpe, lateral_method
    )
    return (raw_masks, registered_masks), (raw_frames, registered_frames)


(raw_masks, registered_masks), (raw_frames, registered_frames) = _register_and_render()

c1, c2 = st.columns(2)
sig = hashlib.md5(
    f"{case}|{correct_dx}|{flatten_rpe}|{lateral_method}".encode()
).hexdigest()[:10]
base = f"ref_{sig}"


with c1:
    st.subheader("Raw frames")
    raw_frames = R.write_mp4(
        raw_frames,
        str(R.WORKDIR / f"{base}_seg.mp4"),
        fps=10,
    )
    st.video(raw_frames, loop=True, autoplay=True, muted=True)
with c2:
    st.subheader("Registered frames")
    registered_frames = R.write_mp4(
        registered_frames.cpu().numpy(),
        str(R.WORKDIR / f"{base}_reg.mp4"),
        fps=10,
    )
    st.video(registered_frames, loop=True, autoplay=True, muted=True)
