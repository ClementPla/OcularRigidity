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
from ocularrigidity.registration.rigid import register_videos
from ocularrigidity.registration.config import RegistrationConfig

st.set_page_config(page_title="Registration", layout="wide")


root, suffix, iop = require_selection()
st.title(f"Registration — {C.pretty_method(suffix)}")

df = cached_case_table(root, suffix, iop)
show_cols = [
    c
    for c in ["case_id", "PatientId", "Date", "Eye", "deltaA", "deltaCT", "K_thickness"]
    if c in df.columns
]
st.caption("Select a case, tune the parameters in the sidebar.")
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
    fovea_correction_enabled = st.checkbox(
        "Fovea correction",
        value=True,
        help="Estimate fovea location from ILM and shift frames/masks accordingly",
    )
    correct_transversal = st.checkbox(
        "Correct for lateral displacement", value=True, help="Align masks to cube"
    )
    correct_axial = st.checkbox(
        "Correct for axial displacement", value=True, help="Align masks to cube"
    )
    flatten_rpe = st.checkbox(
        "Flatten RPE", value=True, help="Flatten cube to RPE before registration"
    )
    lateral_method = st.radio(
        "Lateral registration method",
        ["fullframe", "xcorr", "both"],
        index=0,
        help="Method to compute lateral displacement",
    )
    axial_refinement = st.checkbox(
        "Axial refinement", value=True, help="Refine axial alignment after lateral"
    )
    max_vshift = st.slider(
        "Max vertical shift (pixels)",
        min_value=0,
        max_value=50,
        value=10,
        step=1,
        help="Maximum vertical shift allowed during registration",
    )

    max_dx = st.slider(
        "Max lateral shift (pixels)",
        min_value=0,
        max_value=50,
        value=16,
        step=1,
        help="Maximum lateral shift allowed during registration",
    )
    smooth_transversal = st.checkbox(
        "Smooth lateral shifts",
        value=True,
        help="Smooth lateral shifts over time to reduce noise",
    )

    smooth_transversal_sigma = st.slider(
        "Smooth lateral shifts sigma",
        min_value=0.0,
        max_value=20.0,
        value=4.0,
        step=0.1,
        help="Sigma for Gaussian smoothing of lateral shifts",
    )

    subpixel = st.checkbox(
        "Subpixel registration",
        value=True,
        help="Use subpixel registration for more accurate alignment",
    )


def _mkv_raw(root, suffix, case):
    return Path(root) / ".." / "compressed" / case / "cube.mp4"


def mask_raw(root, case):
    return Path(root) / ".." / "masks" / case / "mask.npz"


@st.cache_data(show_spinner="Registering masks…")
def registration(
    _masks_data,
    _frames_data,
    case,
    correct_transversal,
    correct_axial,
    flatten_rpe,
    lateral_method,
    axial_refinement,
    max_vshift,
    smooth_transversal,
    smooth_transversal_sigma,
    max_lateral_shift,
    subpixel,
    fovea_correction_enabled,
):
    registered_masks, registered_cube = register_videos(
        _masks_data,
        _frames_data,
        RegistrationConfig(
            correct_transversal=correct_transversal,
            correct_axial=correct_axial,
            flatten_rpe=flatten_rpe,
            lateral_method=lateral_method,
            axial_refinement=axial_refinement,
            max_axial_shift=max_vshift,
            smooth_transversal=smooth_transversal,
            smooth_transversal_sigma=smooth_transversal_sigma,
            max_lateral_shift=max_lateral_shift,
            subpixel=subpixel,
            fovea_correction_enabled=fovea_correction_enabled,
        ),
    )
    return registered_masks, registered_cube


def _register_and_render():
    mkv = _mkv_raw(root, suffix, case)
    max_frame = 512
    indices = np.arange(10, max_frame, 4)
    with st.spinner("Loading video…"):
        raw_frames = R.read_cube(str(mkv), _indices=indices)  # (T, H, W)
    with st.spinner("Loading mask…"):
        raw_masks = R.read_masks(str(mask_raw(root, case)))  # (T, H, W)

    raw_masks = raw_masks[indices]

    registered_masks, registered_frames = registration(
        raw_masks,
        raw_frames,
        case,
        correct_transversal=correct_transversal,
        correct_axial=correct_axial,
        flatten_rpe=flatten_rpe,
        lateral_method=lateral_method,
        axial_refinement=axial_refinement,
        max_vshift=max_vshift,
        smooth_transversal=smooth_transversal,
        smooth_transversal_sigma=smooth_transversal_sigma,
        max_lateral_shift=max_dx,
        subpixel=subpixel,
        fovea_correction_enabled=fovea_correction_enabled,
    )
    return (raw_masks, registered_masks), (raw_frames, registered_frames)


(raw_masks, registered_masks), (raw_frames, registered_frames) = _register_and_render()

c1, c2 = st.columns(2)
sig = hashlib.md5(
    f"{case}|{correct_transversal}|{correct_axial}|{flatten_rpe}|{lateral_method}|{axial_refinement}|{max_vshift}|{smooth_transversal}|{smooth_transversal_sigma}|{max_dx}|{subpixel}|{fovea_correction_enabled}".encode()
).hexdigest()[:16]
base = f"ref_{sig}"


with c1:
    st.subheader("Raw frames")
    if (R.WORKDIR / f"{base}_seg.mp4").exists():
        raw_frames = R.WORKDIR / f"{base}_seg.mp4"
    else:
        raw_frames = R.write_mp4(
            raw_frames,
            str(R.WORKDIR / f"{base}_seg.mp4"),
            fps=100,
        )
    st.video(raw_frames, loop=True, autoplay=True, muted=True)
with c2:
    st.subheader("Registered frames")
    if (R.WORKDIR / f"{base}_reg.mp4").exists():
        registered_frames = R.WORKDIR / f"{base}_reg.mp4"
    else:
        registered_frames = R.write_mp4(
            registered_frames.cpu().numpy(),
            str(R.WORKDIR / f"{base}_reg.mp4"),
            fps=100,
        )
    st.video(registered_frames, loop=True, autoplay=True, muted=True)
