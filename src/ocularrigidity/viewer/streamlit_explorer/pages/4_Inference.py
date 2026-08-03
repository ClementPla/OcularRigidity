import hashlib
from pathlib import Path
import numpy as np
import streamlit as st

from ocularrigidity.viewer import render as R
from ocularrigidity.viewer.streamlit_explorer._common import (
    cached_case_table,
    require_selection,
)

st.set_page_config(page_title="Inference", layout="wide")

CROP = 1024  # display-only center crop


@st.cache_resource(show_spinner="Loading segmentation model…")
def get_model():
    from ocularrigidity.segmentation.utils import get_choroid_segmentation_model

    return get_choroid_segmentation_model().cuda()


def _mkv_one_cycle(root, suffix, case):
    return Path(root) / f"one_cycle_{suffix}" / case / "one_cycle.mkv"


def _mkv_raw(root, suffix, case):
    return Path(root) / ".." / "compressed" / case / "cube.mp4"


# --- page --------------------------------------------------------------------

sel = require_selection()
root, suffix = sel.root, sel.suffix
st.title(f"Segmentation inference — {sel.method_label}")

df = cached_case_table(sel)
show_cols = [
    c
    for c in ["case_id", "PatientId", "Date", "Eye", "deltaA", "deltaCT", "K_thickness"]
    if c in df.columns
]
st.caption(
    "Select a case, tune the parameters in the sidebar, then **Run segmentation**."
)
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
    which_video = st.radio(
        "Video to display", ["one_cycle", "raw"], horizontal=True, index=0
    )
    max_frame = st.slider(
        "max_frame (for display only)",
        1,
        512,
        512,
        1,
        help="Only the first N frames are shown in the video preview. The model runs on all frames.",
    )
sb.header("Inference")
scale = sb.select_slider("scale_factor", options=[0.25, 0.5, 0.75, 1.0], value=0.5)
batch = sb.slider("batch_size", 1, 32, 16, 1)
use_amp = sb.checkbox("use_amp (fp16)", value=True)
use_gc = sb.checkbox("use_graphcut", value=True)

sb.header("Graph-cut", help="Only used when use_graphcut is on.")
gc_lambda = sb.slider("lambda_smooth", 0.0, 5.0, 0.5, 0.1)
gc_max_step = sb.slider("max_step", 1, 6, 2, 1)
gc_prob = sb.slider("prob_threshold", 0.0, 1.0, 0.3, 0.05)
gc_bm = sb.slider("bm_threshold", 0.0, 1.0, 0.5, 0.05)
gc_temporal = sb.checkbox("temporal_smooth", value=False)
gc_t_iters = sb.slider("temporal_iterations", 1, 10, 4, 1)
gc_t_mu = sb.slider("temporal_mu", 0.0, 5.0, 1.0, 0.1)
gc_t_sigma = sb.slider("temporal_sigma", 0.0, 5.0, 2.0, 0.1)

sb.header("Display")
alpha = sb.slider("Overlay opacity", 0.0, 1.0, 0.4, 0.05)
factor = sb.select_slider("Downscale", options=[1, 2, 3, 4], value=2)

run = st.button("Run segmentation", type="primary")


def _segment_and_render():
    from ocularrigidity.segmentation.inference import infer

    if which_video == "raw":
        mkv = _mkv_raw(root, suffix, case)
    else:
        mkv = _mkv_one_cycle(root, suffix, case)
    if not mkv.exists():
        st.error("This case has no `one_cycle.mkv`.")
        return None

    gc_kwargs = dict(
        lambda_smooth=gc_lambda,
        max_step=int(gc_max_step),
        prob_threshold=gc_prob,
        bm_threshold=gc_bm,
        temporal_smooth=bool(gc_temporal),
        temporal_iterations=int(gc_t_iters),
        temporal_mu=gc_t_mu,
        temporal_sigma=gc_t_sigma,
    )
    with st.spinner("Loading video…"):
        cube_full = R.read_cube(
            str(mkv), _indices=np.arange(max_frame)
        )  # (T, H, W) — segment on full frames
    with st.spinner("Running the choroid model…"):
        masks = infer(
            get_model(),
            cube_full,
            scale_factor=float(scale),
            batch_size=int(batch),
            use_graphcut=bool(use_gc),
            graphcut_kwargs=gc_kwargs if use_gc else None,
            use_amp=bool(use_amp),
            return_logit=False,
            verbose=False,
        )

    # Crop + downscale for display only.
    cube = R.resize_cube(R.center_crop_square(cube_full, CROP), factor)
    mask_d = R.resize_cube(
        R.center_crop_square(masks.astype("uint8"), CROP), factor, nearest=True
    ).astype(bool)

    sig = hashlib.md5(
        f"{case}|{scale}|{batch}|{use_amp}|{use_gc}|{gc_kwargs}|{alpha}|{factor}".encode()
    ).hexdigest()[:10]
    base = f"infer_{sig}"
    oc = R.write_mp4(
        cube, str(R.WORKDIR / f"{base}_oc.mp4"), fps=10, quality=5, preset="ultrafast"
    )
    seg = R.write_mp4(
        R.overlay_video(cube, mask_d, alpha),
        str(R.WORKDIR / f"{base}_seg.mp4"),
        fps=10,
        quality=5,
        preset="ultrafast",
    )
    coverage = float(mask_d.any(axis=1).mean())  # rough fraction of columns covered
    return {
        "case": case,
        "oc": oc,
        "seg": seg,
        "coverage": coverage,
        "params": gc_kwargs,
    }


if run:
    try:
        result = _segment_and_render()
        if result is not None:
            st.session_state["infer_result"] = result
    except Exception as e:  # surface the torchaudio/env error etc. gracefully
        st.error(f"Inference failed: {e}")
        st.exception(e)

# --- results (persist across slider tweaks until the next Run) ----------------
result = st.session_state.get("infer_result")
if result is None:
    st.info("Set parameters and press **Run segmentation**.")
    st.stop()

if result["case"] != case:
    st.warning(
        f"Showing the previous run for `{result['case']}`. Press **Run** to segment `{case}`."
    )

st.caption(f"Mask column coverage ≈ {result['coverage']:.2f}")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**One-cycle**")
    st.video(result["oc"], loop=True, autoplay=True, muted=True)
with c2:
    st.markdown("**Segmentation overlay**")
    st.video(result["seg"], loop=True, autoplay=True, muted=True)
