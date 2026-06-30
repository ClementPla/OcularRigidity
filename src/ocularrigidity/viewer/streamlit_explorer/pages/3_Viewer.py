"""Per-case video viewer: click a row to load its one-cycle, segmentation and
quiver. Optimised for speed — frames are downscaled hard and encoded with a fast
libx264 preset, since this is for browsing, not publication figures.
"""

import pickle
from pathlib import Path

import streamlit as st

from ocularrigidity.viewer import cohort_data as C
from ocularrigidity.viewer import render as R
from ocularrigidity.viewer.streamlit_explorer._common import (
    cached_case_table,
    require_selection,
)

st.set_page_config(page_title="Viewer", layout="wide")

# Speed-first encode defaults: small frames, fast preset, low (but fine) quality.
QUALITY = 5
PRESET = "ultrafast"
FPS = 10
CROP = 1024  # center-crop each frame to a CROP×CROP square before downscaling


def _mkv(root, suffix, case):
    return Path(root) / f"one_cycle_{suffix}" / case / "one_cycle.mkv"


def _measures(root, suffix, case):
    return Path(root) / f"measures_{suffix}" / case


@st.cache_data(show_spinner="Rendering one-cycle…")
def render_one_cycle(root, suffix, case, factor):
    mkv = _mkv(root, suffix, case)
    if not mkv.exists():
        return None
    cube = R.resize_cube(R.center_crop_square(R.read_cube(str(mkv)), CROP), factor)
    out = R.WORKDIR / f"{suffix}__{case.replace('/', '_')}_oc_sq{CROP}_d{factor}.mp4"
    if not out.exists():
        R.write_mp4(cube, str(out), fps=FPS, quality=QUALITY, preset=PRESET)
    return str(out)


@st.cache_data(show_spinner="Rendering overlay…")
def render_overlay(root, suffix, case, factor, alpha):
    mkv, seg = (
        _mkv(root, suffix, case),
        _measures(root, suffix, case) / "segmented_cycles.npz",
    )
    if not mkv.exists() or not seg.exists():
        return None
    cube_full, masks_full = R.read_cube(str(mkv)), R.read_masks(str(seg))
    if masks_full.shape[0] != cube_full.shape[0]:
        return None
    cube = R.resize_cube(R.center_crop_square(cube_full, CROP), factor)
    masks = R.resize_cube(
        R.center_crop_square(masks_full.astype("uint8"), CROP), factor, nearest=True
    ).astype(bool)
    out = (
        R.WORKDIR
        / f"{suffix}__{case.replace('/', '_')}_mask_sq{CROP}_d{factor}_a{alpha:.2f}.mp4"
    )
    if not out.exists():
        R.write_mp4(
            R.overlay_video(cube, masks, alpha),
            str(out),
            fps=FPS,
            quality=QUALITY,
            preset=PRESET,
        )
    return str(out)


@st.cache_data(show_spinner="Rendering quiver…")
def render_quiver(
    root, suffix, case, factor, cycle, arrow_scale, stride, smooth, only_y
):
    mkv, da = (
        _mkv(root, suffix, case),
        _measures(root, suffix, case) / "deltaA_per_cycle.pkl",
    )
    if not mkv.exists() or not da.exists():
        return None
    with open(da, "rb") as fh:
        data = pickle.load(fh)
    if not data.get("displacement_per_cycle"):
        return None
    raw = R.read_cube(str(mkv))
    x0, y0, _ = R.square_crop_offset(raw.shape[1], raw.shape[2], CROP)
    cube = R.resize_cube(R.center_crop_square(raw, CROP), factor)
    tag = f"{suffix}__{case.replace('/', '_')}_quiver_sq{CROP}_d{factor}_c{cycle}_a{arrow_scale}_s{stride}_w{smooth}_y{int(only_y)}.mp4"
    out = R.WORKDIR / tag
    if not out.exists():
        R.render_quiver(
            cube,
            data["displacement_per_cycle"],
            data["reference_coordinates_per_cycle"],
            str(out),
            cycle=None if cycle == "All" else int(cycle),
            arrow_scale=float(arrow_scale),
            stride=int(stride),
            smooth_window=int(smooth),
            only_y=bool(only_y),
            coord_scale=1.0 / factor,
            crop_offset=(x0, y0),
            quality=QUALITY,
            preset=PRESET,
        )
    return str(out)


# --- page --------------------------------------------------------------------

root, suffix, iop = require_selection()
st.title(f"Viewer — {C.pretty_method(suffix)}")

# Render controls (kept in the sidebar so the main area stays focused).
st.sidebar.header("Render (speed-first)")
factor = st.sidebar.select_slider("Downscale", options=[2, 3, 4, 6, 8], value=4)
alpha = st.sidebar.slider("Overlay opacity", 0.0, 1.0, 0.4, 0.05)
q_cycle = st.sidebar.selectbox("Quiver cycle", ["0", "1", "2", "All"], index=0)
q_arrow = st.sidebar.slider("Arrow scale", 1.0, 60.0, 20.0, 1.0)
q_stride = st.sidebar.slider("Arrow stride", 1, 20, 2, 1)
q_smooth = st.sidebar.slider("Smooth window", 0, 15, 0, 1)
q_only_y = st.sidebar.checkbox("Quiver only_y", value=False)

df = cached_case_table(root, suffix, iop)
show_cols = [
    c
    for c in ["case_id", "PatientId", "Date", "Eye", "deltaA", "deltaCT", "K_thickness"]
    if c in df.columns
]

st.caption(
    "Click a row to load its one-cycle video, segmentation overlay and displacement quiver."
)
event = st.dataframe(
    df[show_cols],
    on_select="rerun",
    selection_mode="single-row",
    hide_index=True,
    width="stretch",
    height=300,
    key="viewer_cases",
)

rows = event.selection.rows
if not rows:
    st.info("Select a case above to render its videos.")
    st.stop()

case = df.iloc[rows[0]]["case_id"]
st.subheader(case)

oc = render_one_cycle(root, suffix, case, factor)
overlay = render_overlay(root, suffix, case, factor, alpha)
quiver = render_quiver(
    root, suffix, case, factor, q_cycle, q_arrow, q_stride, q_smooth, q_only_y
)


def _play(path):
    # loop + muted autoplay so the short clips run continuously without a click.
    st.video(path, loop=True, autoplay=True, muted=True, width=600)


c1, c2, c3 = st.columns(3)
with c1:
    _play(oc)
with c2:
    _play(overlay)
with c3:
    _play(quiver)
