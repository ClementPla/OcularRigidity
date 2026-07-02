import hashlib
from pathlib import Path
import numpy as np
import streamlit as st

from ocularrigidity.segmentation.fovea.from_ilm import estimate_fovea_from_ilm
from ocularrigidity.segmentation.postprocess.blob import (
    keep_largest_connected_component,
)
from ocularrigidity.viewer import cohort_data as C
from ocularrigidity.viewer import render as R
from ocularrigidity.viewer.streamlit_explorer._common import (
    cached_case_table,
    require_selection,
)
import plotly.graph_objects as go

from ocularrigidity.segmentation.postprocess.interfaces import (
    clean_boundaries,
    extract_boundaries_fast,
    rebuild_mask,
)

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


def _mkv_raw(root, suffix, case):
    return Path(root) / ".." / "compressed" / case / "cube.mp4"


def mask_raw(root, case):
    return Path(root) / ".." / "masks" / case / "mask.npz"


mkv = _mkv_raw(root, suffix, case)
max_frame = 128
indices = np.arange(10, max_frame, 4)
with st.spinner("Loading video…"):
    raw_frames = R.read_cube(str(mkv), _indices=indices)  # (T, H, W)
with st.spinner("Loading mask…"):
    raw_masks = R.read_masks(str(mask_raw(root, case)), _indices=indices)  # (T, H, W)

index = st.slider(
    "Frame index for BM visualization",
    min_value=0,
    max_value=len(raw_masks) - 1,
    value=0,
    step=1,
)
bm, csi = clean_boundaries(*extract_boundaries_fast(raw_masks))
max_thickness_um = 425
axial_resolution_um = 1.95
upper_retinal_bbox = bm - (
    max_thickness_um / axial_resolution_um
)  # upper retinal boundary
# Draw with Plotly the image, the BM and the CSI


roi_mask = rebuild_mask(upper_retinal_bbox, bm, raw_masks.shape[1])
roi_mask = roi_mask.astype(bool) & (raw_frames > 25)
roi_mask = keep_largest_connected_component(
    roi_mask
)  # keep largest connected component

ilm, bm = clean_boundaries(*extract_boundaries_fast(roi_mask))

fovea_location = estimate_fovea_from_ilm(ilm)
fig = go.Figure()

fig.add_trace(
    go.Heatmap(
        z=raw_frames[index],
        zmin=0,
        zmax=255,
        colorscale="gray",
        showscale=False,
    )
)
# fig.add_trace(
#     go.Heatmap(
#         z=roi_mask[index].astype(np.uint8),
#         zmin=0,
#         zmax=1,
#         zmid=0.5,
#         colorscale=[[0, "#000000"], [1, "#00EEDA"]],
#         opacity=0.25,
#         showscale=False,
#     )
# )

fig.add_trace(
    go.Scatter(
        x=[fovea_location[index][0]],
        y=[fovea_location[index][1]],
        mode="markers",
        marker=dict(color="yellow", size=10),
        name="Estimated Fovea",
    )
)
fig.add_trace(
    go.Scatter(
        x=np.arange(raw_masks.shape[2]),
        y=ilm[index],
        mode="lines",
        line=dict(color="green", width=2),
        name="ILM",
    )
)

fig.add_trace(
    go.Scatter(
        x=np.arange(raw_masks.shape[2]),
        y=bm[index],
        mode="lines",
        line=dict(color="red", width=2),
        name="BM",
    )
)
fig.add_trace(
    go.Scatter(
        x=np.arange(raw_masks.shape[2]),
        y=upper_retinal_bbox[index],
        mode="lines",
        line=dict(color="blue", width=2),
        name="upper retinal boundary",
    )
)

fig.update_yaxes(autorange="reversed")

# --- FIX: Force equal aspect ratio ---
fig.update_layout(
    yaxis=dict(
        scaleanchor="x",
        scaleratio=1,
    ),
    dragmode="pan",
)
st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})
