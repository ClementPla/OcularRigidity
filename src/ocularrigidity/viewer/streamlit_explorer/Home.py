"""Landing page: what the cohort table holds, and how well each source joined."""

import numpy as np
import streamlit as st

from ocularrigidity.data.measurements.cohort import column_groups, coverage
from ocularrigidity.viewer.streamlit_explorer._common import (
    cached_cohort,
    require_selection,
)

st.set_page_config(page_title="Ocular Rigidity — cohort browser", layout="wide")

st.title("Ocular Rigidity — cohort browser")
st.markdown(
    "One row per rigidity visit, every source merged onto it — the cardiac "
    "pipeline's pulsatile metrics, the clinical scalars, the Heyex ONH sectors "
    "and the diagnosis register. Choose a **method** and a **cohort** in the "
    "sidebar, then open **Cases**, **Regression** or **Longitudinal**."
)

sel = require_selection()
df = cached_cohort(sel)
groups = column_groups(df)

st.subheader(f"Overview — {sel.cohort_label}")

eyes = df[["PatientId", "Eye"]].astype(str).agg("/".join, axis=1)
per_eye = eyes.value_counts()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Visits", len(df))
c2.metric("Eyes", int(per_eye.size), help=f"{df['PatientId'].nunique()} patients")
c3.metric("Eyes with ≥ 2 visits", int((per_eye >= 2).sum()))
c4.metric("With K", int(df["K"].notna().sum()))
c5.metric("With ONH", int(df[groups["onh"][0]].notna().sum()) if groups["onh"] else 0)

m1, m2, m3 = st.columns(3)
m1.metric("Median ΔCT (mm)", f"{np.nanmedian(df['deltaCT']):.4f}")
m2.metric("Median minCT (mm)", f"{np.nanmedian(df['minCT']):.3f}")
m3.metric("Median K (1/µL)", f"{np.nanmedian(df['K']):.4f}")

st.markdown(
    "- **ΔCT** — pulsatile choroidal-thickness change (mm, median over the cycles), "
    "tracked at the choroid-sclera interface; **ΔCT_Mask** is the mask-based "
    "estimator of the same quantity\n"
    "- **minCT** — absolute choroidal thickness (mm); **RelativeGrowth** = ΔCT / minCT\n"
    "- **K** — Friedenwald rigidity (1/µL) from the measured ΔCT\n"
    "- **thickening / thinning (µm/s), rate_asymmetry** — how fast the choroid fills "
    "and drains within one cycle, and the ratio of the two limbs\n"
    "- **ONH** — BMO-MRW and sector RNFL, as-of matched to the visit "
    "(`onh_gap_days` is the realised gap; `onh_source` says which export it came from)"
)

with st.expander("Coverage — how many visits each variable actually reaches", expanded=True):
    st.caption(
        "The join audit. A variable that covers few visits cannot support an "
        "association however good it looks: read this before the p-values."
    )
    cov = coverage(df)
    st.dataframe(
        cov,
        width="stretch",
        hide_index=True,
        column_config={
            "coverage": st.column_config.ProgressColumn(
                "coverage", min_value=0.0, max_value=1.0, format="%.2f"
            )
        },
        height=420,
    )

with st.expander("Distributions", expanded=True):
    numeric = groups["pulsation"] + groups["covariates"]
    cols = st.multiselect(
        "Columns", numeric, default=[c for c in ["deltaCT", "minCT", "K"] if c in numeric]
    )
    if cols:
        st.dataframe(df[cols].describe().T, width="stretch")
