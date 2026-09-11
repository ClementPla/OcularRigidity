"""Predict one variable from several, with a split that respects the eye.

The other pages test one association at a time. This one fits a model on a set
of inputs and reports how well it predicts a held-out variable -- which only
means anything if the split is grouped by patient and the test fold is read
once. Both are enforced in :mod:`ocularrigidity.stats.supervised`; this page
only drives it.

The target is either a value at a visit or an eye's per-year rate of change.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from ocularrigidity.data.measurements.cohort import (
    COVARIATES,
    PULSATION_METRICS,
    column_groups,
    measure_columns,
)
from ocularrigidity.stats import supervised as S
from ocularrigidity.viewer.streamlit_explorer._common import (
    cached_cohort,
    require_selection,
    show_regression,
)

st.set_page_config(page_title="Modelling", layout="wide")

sel = require_selection()
st.title(f"Modelling — {sel.cohort_label}")

df = cached_cohort(sel)
groups = column_groups(df)
numeric = [
    c
    for c in dict.fromkeys(
        list(PULSATION_METRICS) + list(COVARIATES) + measure_columns(df)
    )
    if c in df.columns and df[c].dtype.kind in "fi"
]
if len(numeric) < 2:
    st.error("Need at least two numeric columns to model anything.")
    st.stop()

# --- what to predict, from what ---------------------------------------------
c1, c2 = st.columns([1, 2])
target = c1.selectbox(
    "Target", numeric, index=numeric.index("K") if "K" in numeric else 0
)
mode = c1.radio(
    "Predict",
    [S.VISIT, S.RATE],
    format_func=lambda m: (
        "value at a visit" if m == S.VISIT else "per-year rate of change"
    ),
    help="A rate collapses each eye to one row: its inputs at the earliest "
    "visit against the slope of the target from there on.",
)
min_points = c1.slider("Min visits per eye", 2, 6, 3) if mode == S.RATE else 3
# Deliberately not IOP/OPA/AxialLength: those are the ingredients of K, the
# default target, so they would hand a new user a circular model on first load.
default_inputs = [
    c for c in ("Age", "minCT", "G RNFL Thickness", "MD", "PSD") if c in numeric
][:5]
inputs = c2.multiselect(
    "Inputs",
    [c for c in numeric if c != target],
    default=[c for c in default_inputs if c != target],
    help="The target is removed from this list automatically if picked.",
)

# --- how to split ------------------------------------------------------------
s1, s2, s3, s4 = st.columns(4)
test_size = s1.slider("Test share", 0.10, 0.40, 0.20, 0.05)
val_size = s2.slider("Val share", 0.10, 0.40, 0.20, 0.05)
seed = s3.number_input("Split seed", 0, 9999, 0, 1)
MISSING_CHOICES = {
    "drop the row": None,
    "impute: median": "median",
    "impute: mean": "mean",
    "impute: 5-NN": "knn",
}
missing_how = s4.selectbox(
    "Missing inputs",
    list(MISSING_CHOICES),
    help="Imputers are fitted inside the pipeline, on the training fold only — "
    "filling the whole frame first would leak the test fold's median into "
    "training. A missing target is always dropped: y is never invented.",
)
impute = MISSING_CHOICES[missing_how]
add_indicator = s4.checkbox(
    "…and flag what was filled",
    value=True,
    disabled=impute is None,
    help="Adds a was-missing column per imputed feature, so the model can use "
    "the absence itself. Missingness here is not random — an eye too advanced "
    "to field-test is a different eye.",
)

# --- the model ---------------------------------------------------------------
m1, m2 = st.columns([1, 3])
model_name = m1.selectbox("Model", list(S.MODELS))
spec = S.MODELS[model_name]
m1.caption(spec["note"])
params = {"seed": int(seed)}
cols = m2.columns(max(1, len(spec["params"])))
for (pname, pspec), col in zip(spec["params"].items(), cols):
    kind, default = pspec[0], pspec[1]
    if kind == "int":
        params[pname] = col.number_input(pname, pspec[2], pspec[3], default, 1)
    elif kind == "log":
        params[pname] = float(
            10
            ** col.slider(
                f"{pname} (log10)",
                float(np.log10(pspec[2])),
                float(np.log10(pspec[3])),
                float(np.log10(default)),
                0.1,
            )
        )
    else:
        params[pname] = col.text_input(pname, default)

if not inputs:
    st.warning("Pick at least one input variable.")
    st.stop()

circular = S.circular_inputs(target, inputs)
if circular:
    st.error(
        f"**{', '.join(circular)}** "
        + (
            "is an algebraic ingredient of"
            if len(circular) == 1
            else "are algebraic ingredients of"
        )
        + f" **{target}** (or made of it). The model will recover that formula and "
        "report a high R² that says nothing about biology — the same trap the "
        "correlation pages call *circular*. Remove them, or read the score as a "
        "check that the pipeline is self-consistent rather than as a finding.",
        icon="⚠️",
    )

X, y, grp, info = S.build_design(
    df,
    inputs,
    target,
    mode=mode,
    min_points=min_points,
    drop_incomplete=impute is None,
)
if len(X) < 20:
    st.error(
        f"Only {len(X)} usable rows for this combination "
        f"({info.get('dropped', 0)} dropped for missing values). "
        "Pick fewer inputs, or a target with better coverage."
    )
    st.stop()

split = S.split_by_group(grp, test_size=test_size, val_size=val_size, seed=int(seed))
n_rows = split.value_counts().to_dict()
n_pat = {s: int(grp[split == s].nunique()) for s in S.SPLITS}

d1, d2, d3, d4 = st.columns(4)
d1.metric("Rows", f"{len(X)}", delta=f"-{info['dropped']} dropped", delta_color="off")
d2.metric("Patients", f"{info['n_groups']}")
d3.metric(
    "Train / Val / Test rows", " / ".join(str(n_rows.get(s, 0)) for s in S.SPLITS)
)
d4.metric("… patients", " / ".join(str(n_pat[s]) for s in S.SPLITS))

if min(n_pat.values()) == 0:
    st.error(
        "A fold came out empty — lower the test/val shares or pick a wider target."
    )
    st.stop()
st.caption(
    "Rows are split by **patient**, so an eye is wholly inside one fold — a random "
    "row split would put the same eye either side and the test score would measure "
    "memorisation. Choose the model on **val**; **test** is the number you read once. "
    + (
        f"Each row is one eye: its inputs at baseline against the per-year slope of "
        f"{target}."
        if mode == S.RATE
        else "Each row is one visit."
    )
)

if impute is not None and info["n_imputed"]:
    miss = pd.Series(info["missing_per_input"], name="missing")
    miss = miss[miss > 0].sort_values(ascending=False)
    pct = 100 * info["n_imputed"] / max(1, info["n_rows"])
    with st.expander(
        f"⚠️ {info['n_imputed']} of {info['n_rows']} rows ({pct:.0f}%) have a filled "
        f"value — {missing_how}",
        expanded=pct > 25,
    ):
        st.dataframe(
            pd.DataFrame(
                {
                    "missing": miss,
                    "% of rows": (100 * miss / info["n_rows"]).round(1),
                }
            ),
            width="stretch",
        )
        st.caption(
            "These cells are filled from the training fold, not measured. A "
            "feature that is mostly filled is mostly a constant, and its "
            "importance below will read as noise. Compare against **drop the "
            "row** before believing a score that depends on it."
        )

if not st.button("Train", type="primary"):
    st.stop()

try:
    pipe, metrics, pred = S.fit_evaluate(
        S.make_pipeline(model_name, params, impute=impute, add_indicator=add_indicator),
        X,
        y,
        split,
    )
except Exception as exc:  # a bad hyperparameter should not blank the page
    st.error(f"Training failed: {exc}")
    st.stop()

# --- results -----------------------------------------------------------------
st.subheader("Scores")
wide = metrics.pivot(index="split", columns="model", values=["r2", "mae", "rmse", "n"])
tab = pd.DataFrame(
    {
        "n": wide[("n", "fitted")],
        "R² (model)": wide[("r2", "fitted")],
        "R² (predict-the-mean)": wide[("r2", "dummy")],
        "MAE": wide[("mae", "fitted")],
        "RMSE": wide[("rmse", "fitted")],
    }
).reindex([s for s in S.SPLITS if s in wide.index])
st.dataframe(
    tab.style.format(
        {
            "n": "{:.0f}",
            "R² (model)": "{:+.3f}",
            "R² (predict-the-mean)": "{:+.3f}",
            "MAE": "{:.4g}",
            "RMSE": "{:.4g}",
        }
    ),
    width="stretch",
)

test_r2 = tab.loc["test", "R² (model)"] if "test" in tab.index else np.nan
dummy_r2 = tab.loc["test", "R² (predict-the-mean)"] if "test" in tab.index else np.nan
train_r2 = tab.loc["train", "R² (model)"] if "train" in tab.index else np.nan
if np.isfinite(test_r2):
    if test_r2 <= dummy_r2:
        st.error(
            f"Test R² {test_r2:+.3f} does not beat predicting the mean "
            f"({dummy_r2:+.3f}). On this data the inputs carry no usable signal "
            "about the target — a negative R² is worse than a flat line, not a "
            "weak fit.",
            icon="⚠️",
        )
    elif np.isfinite(train_r2) and train_r2 - test_r2 > 0.3:
        st.warning(
            f"Train R² {train_r2:+.3f} against test {test_r2:+.3f}: the model is "
            "memorising. Fewer inputs, or a smaller/more regularised model.",
            icon="⚠️",
        )
    else:
        st.success(f"Test R² {test_r2:+.3f}, against {dummy_r2:+.3f} for the mean.")

st.subheader("Predicted vs actual")
tabs = st.tabs([f"{s} (n={n_rows.get(s, 0)})" for s in S.SPLITS if s in set(split)])
for tab_ui, s in zip(tabs, [s for s in S.SPLITS if s in set(split)]):
    with tab_ui:
        sub = pred[pred["split"] == s]
        show_regression(
            sub,
            "y_true",
            "y_pred",
            x_label=f"actual {target}",
            y_label=f"predicted {target}",
            hover=[],
            height=460,
        )

st.subheader("Permutation importance (test fold)")
imp = S.importances(pipe, X, y, split, on="test", seed=int(seed))
if imp.empty:
    st.info("Too few test rows to permute.")
else:
    fig = px.bar(
        imp,
        x="importance",
        y="feature",
        orientation="h",
        error_x="std",
        template="plotly_white",
        labels={"importance": "drop in test R² when shuffled"},
    )
    fig.update_layout(
        height=60 + 34 * len(imp),
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Measured on held-out rows: permuting a feature on the training set "
        "mostly reports how much the model memorised it. A negative bar means "
        "the model scored better without the feature — noise at this sample size."
    )

with st.expander("Residuals"):
    fig = px.scatter(
        pred,
        x="y_pred",
        y="residual",
        color="split",
        opacity=0.6,
        template="plotly_white",
        labels={"y_pred": f"predicted {target}"},
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#888")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Structure here — a trend, a fan — means the model is missing something."
    )
