"""Per-case video viewer: click a row to load its one-cycle, segmentation and
quiver. Optimised for speed — frames are downscaled hard and encoded with a fast
libx264 preset, since this is for browsing, not publication figures.
"""

import pickle
from pathlib import Path

import streamlit as st

from ocularrigidity.viewer import render as R
from ocularrigidity.viewer.quiver import QuiverStyle
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
def render_quiver(root, suffix, case, factor, cycle, style: QuiverStyle):
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

    # The CSI options read the interfaces off the masks, which must therefore be
    # cropped and downscaled exactly like the frames they are drawn on.
    masks = None
    if (
        style.only_orthogonal_to_border
        or style.show_csi_summary
        or style.show_only_csi_anchors
    ):
        seg = _measures(root, suffix, case) / "segmented_cycles.npz"
        if not seg.exists():
            return None
        masks_full = R.read_masks(str(seg))
        if masks_full.shape[0] != raw.shape[0]:
            return None
        masks = R.resize_cube(
            R.center_crop_square(masks_full.astype("uint8"), CROP), factor, nearest=True
        ).astype(bool)

    key = "_".join(str(v) for v in style)
    out = (
        R.WORKDIR
        / f"{suffix}__{case.replace('/', '_')}_quiver_sq{CROP}_d{factor}_c{cycle}_{key}.mp4"
    )
    if not out.exists():
        R.render_quiver(
            cube,
            data["displacement_per_cycle"],
            data["reference_coordinates_per_cycle"],
            str(out),
            masks=masks,
            cycle=None if cycle == "All" else int(cycle),
            style=style,
            fps=FPS,
            coord_scale=1.0 / factor,
            crop_offset=(x0, y0),
            quality=QUALITY,
            preset=PRESET,
        )
    return str(out)


# --- page --------------------------------------------------------------------

sel = require_selection()
root, suffix = sel.root, sel.suffix
st.title(f"Viewer — {sel.method_label}")

# Render controls (kept in the sidebar so the main area stays focused).
st.sidebar.header("Render (speed-first)")
factor = st.sidebar.select_slider("Downscale", options=[2, 3, 4, 6, 8], value=4)
alpha = st.sidebar.slider("Overlay opacity", 0.0, 1.0, 0.4, 0.05)

st.sidebar.header("Quiver")
q_cycle = st.sidebar.selectbox("Cycle", ["0", "1", "2", "All"], index=0)

# Every QuiverStyle option, so the viewer and the gif renderer expose the same
# knobs. Defaults stay speed-first (no CSI extraction unless asked).
d = QuiverStyle()
component = st.sidebar.radio(
    "Displacement component",
    ["Full vector", "Axial only", "Across the boundary"],
    index=1,
    help=(
        "Axial only: drop the lateral component, dominated by residual "
        "registration jitter. Across the boundary: project onto the local mask "
        "normal, keeping the motion through the membrane."
    ),
)
style = QuiverStyle(
    stride=st.sidebar.slider("Arrow stride", 1, 20, 2, 1),
    arrow_scale=st.sidebar.slider("Arrow scale", 1.0, 60.0, 20.0, 1.0),
    min_magnitude=st.sidebar.slider(
        "Min magnitude (px)", 0.0, 2.0, d.min_magnitude, 0.01
    ),
    smooth_window=st.sidebar.slider("Smooth window", 0, 15, 0, 1),
    cyclic=st.sidebar.checkbox("Cyclic smoothing", value=d.cyclic),
    only_y=component == "Axial only",
    only_orthogonal_to_border=component == "Across the boundary",
    border_normal_sigma=st.sidebar.slider(
        "Border normal σ (px)", 0.0, 8.0, d.border_normal_sigma, 0.5
    ),
    show_csi_summary=st.sidebar.checkbox("CSI summary arrow", value=False),
    show_only_csi_anchors=st.sidebar.checkbox("CSI anchors only", value=False),
    side_by_side=st.sidebar.checkbox("Side by side", value=d.side_by_side),
    annotate_scale=st.sidebar.checkbox("Annotate scale", value=d.annotate_scale),
    arrow_cmap=st.sidebar.selectbox(
        "Arrow colormap", ["viridis", "turbo", "magma", "plasma", "coolwarm"]
    ),
    arrow_thickness=st.sidebar.slider("Arrow thickness", 1, 4, d.arrow_thickness, 1),
    tip_length=st.sidebar.slider("Tip length", 0.0, 1.0, d.tip_length, 0.05),
)

df = cached_case_table(sel)
show_cols = [
    c
    for c in ["case_id", "PatientId", "Date", "Eye", "deltaA", "deltaCT", "minCT", "K"]
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
quiver = render_quiver(root, suffix, case, factor, q_cycle, style)


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
