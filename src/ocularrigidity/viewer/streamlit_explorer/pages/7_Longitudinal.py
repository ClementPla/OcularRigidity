"""Longitudinal analyses of the prospective cohort.

Mirrors ``notebooks/cohort_analysis/prospective.ipynb``: each Rigidity visit is
tagged with its diagnosis and with the clinical measures of the same calendar
month, then a probed rigidity metric (K, RelativeGrowth, …) is confronted with
those measures five ways — co-progression, present-vs-future, present-vs-next
change, visit-pair rates and plain cross-sectional association — plus a group
comparison. The designs themselves live in
:mod:`ocularrigidity.viewer.longitudinal`; this page only drives them.

Measures are either picked by hand or **screened**: every measure is tested under
every design and the most significant ones are plotted, with BH q-values so the
selection is not read as a discovery.
"""

import streamlit as st

import numpy as np
from scipy.stats import mannwhitneyu

from ocularrigidity.data.measurements.cohort import (
    DEFAULT_MEASURES,
    PULSATION_METRICS,
    STEEPEST_CHANGE_COLUMNS,
    measure_columns,
)
from ocularrigidity.stats.inference import cluster_robust_ols
from ocularrigidity.viewer import longitudinal as L
from ocularrigidity.viewer.streamlit_explorer._common import (
    cached_clinical_long,
    cached_cohort,
    cached_design,
    cached_screen,
    require_selection,
    show_box,
    show_regression,
)

st.set_page_config(page_title="Longitudinal", layout="wide")

sel = require_selection()
st.title(f"Longitudinal — {sel.cohort_label}")

if sel.study is None:
    st.info(
        "These analyses follow eyes across visits — pick **Prospective** in the "
        "sidebar to restrict the cohort to the longitudinal study."
    )

cohort = cached_cohort(sel)
long_df = cached_clinical_long(sel)
measures = [m for m in measure_columns(cohort) if m in set(long_df["MeasureName_y"])]
if not measures:
    st.error("No clinical measures joined onto these cases.")
    st.stop()

c1, c2, c3 = st.columns([1, 2, 1])
probe = c1.selectbox(
    "Probed metric",
    [c for c in PULSATION_METRICS if c in long_df.columns],
    index=0,
    help="The rigidity metric confronted with the clinical measures.",
)
screen_all = c1.checkbox(
    "Screen all measures",
    value=False,
    help=f"Rather than picking by hand, test all {len(measures)} measures under "
    "each tab's design and plot the most significant ones for that design.",
)
if screen_all:
    top_n = c2.slider("Keep the top N", 3, 20, 10)
    selected = None  # each tab ranks its own
    c2.caption(
        f"Each tab screens all {len(measures)} measures under its own design and "
        f"plots its top {top_n}."
    )
else:
    top_n = 10
    selected = c2.multiselect(
        "Clinical measures",
        measures,
        default=[m for m in DEFAULT_MEASURES if m in measures] or measures[:3],
    )
logy = c3.checkbox("log Y (boxes)", value=False)

n_eyes = long_df.groupby(["PatientId", "Eye"]).ngroups
visits = long_df[["PatientId", "Eye", "Date"]].drop_duplicates()
per_eye = visits.groupby(["PatientId", "Eye"]).size()
m1, m2, m3 = st.columns(3)
m1.metric("Eyes", n_eyes)
m2.metric("Visits", len(visits))
m3.metric("Eyes with ≥ 2 visits", int((per_eye >= 2).sum()))

if not screen_all and not selected:
    st.warning("Pick at least one clinical measure.")
    st.stop()


def targets_for(design: str, params: dict) -> list[str]:
    """The measures this tab plots: its own top N when screening, else the picks.

    When screening, the ranking table is rendered first — it is the actual answer,
    the plots below it are just the top rows drawn out.
    """
    if not screen_all:
        return selected

    ranked = cached_screen(sel, probe, design, tuple(sorted(params.items())))
    if ranked.empty:
        st.warning("No measure could be scored under this design.")
        return []

    label = L.EFFECT_LABEL[design]
    n_p = int((ranked["p"] < 0.05).sum())
    n_q = int((ranked["q (BH)"] < 0.05).sum())
    st.caption(
        f"Screened {len(ranked)} measures — {n_p} with p < 0.05, of which "
        f"**{n_q} survive the multiplicity correction** (q < 0.05). Act on the "
        f"q-value: the top rows were *chosen* for a small p out of {len(ranked)} "
        "tries, so about two would look significant by chance alone. Ranking is "
        f"rank-based (Spearman / Mann–Whitney); a **{label} far from the "
        "Spearman ρ means the association is a single leverage point**, not a "
        "finding."
    )
    fmt = {
        "N": st.column_config.NumberColumn(format="%d"),
        label: st.column_config.NumberColumn(format="%.3g"),
        L.SPEARMAN_COL: st.column_config.NumberColumn(format="%.3f"),
        "p": st.column_config.NumberColumn(format="%.4f"),
        "q (BH)": st.column_config.NumberColumn(format="%.3f"),
    }
    st.dataframe(
        ranked.head(top_n),
        hide_index=True,
        width="stretch",
        column_config={k: v for k, v in fmt.items() if k in ranked.columns},
    )
    return ranked["measure"].head(top_n).tolist()


tabs = st.tabs(
    [
        "Co-progression",
        "Present → future",
        "Present → next change",
        "Visit-pair rates",
        "Cross-sectional",
        "Groups",
    ]
)

# --- 1. co-progression: slope of the probe vs slope of the measure, per eye ---
with tabs[0]:
    st.caption(
        "Per eye, fit a per-year linear slope for both the probed metric and the "
        "measure, then correlate the two slopes: do they progress together?"
    )
    params = {"min_points": st.slider("Min visits per eye", 2, 6, 3, key="slope_min")}
    for m in targets_for(L.COPROGRESSION, params):
        st.subheader(m)
        frame, x, y = cached_design(
            sel, probe, L.COPROGRESSION, m, tuple(sorted(params.items()))
        )
        show_regression(frame, x, y, x_label=f"Δ{probe}/Δt", y_label=f"Δ{m}/Δt")

# --- 2. baseline probe vs future progression of the measure ------------------
with tabs[1]:
    st.caption(
        "Does the probed metric *today* predict how the measure moves *after*? "
        "Baseline = the eye's earliest visit; the measure's slope is fit from there on."
    )
    params = {"min_points": st.slider("Min visits per eye", 2, 6, 3, key="future_min")}
    for m in targets_for(L.FUTURE, params):
        st.subheader(m)
        frame, x, y = cached_design(
            sel, probe, L.FUTURE, m, tuple(sorted(params.items()))
        )
        show_regression(
            frame,
            x,
            y,
            x_label=f"Baseline {probe} (present)",
            y_label=f"Δ{m}/Δt (per year, future)",
        )

# --- 3. lagged: probe at a visit vs how the measure moves next ----------------
with tabs[2]:
    st.caption(
        "The predictive question at the interval level: does the probed metric "
        "**at a visit** anticipate how the measure moves **over the interval that "
        "follows**? Keeps the temporal order (unlike the visit-pair rates, which "
        "correlate two simultaneous rates) while using every interval, not just "
        "each eye's first visit (unlike present → future)."
    )
    c1, c2, c3 = st.columns(3)
    params = {
        "consecutive": c1.checkbox(
            "Consecutive visits only", value=True, key="next_consecutive"
        ),
        "min_dt": c2.slider(
            "Min interval (years)",
            0.0,
            2.0,
            0.25,
            0.05,
            help="Rates over a very short interval are mostly noise: a small "
            "measurement error divided by a small Δt.",
        ),
        "adjust": c3.checkbox(
            "Adjust for the measure's baseline",
            value=True,
            help="Progression depends on how far the disease already is, and that "
            "starting point may itself correlate with the probe. Adjusting partials "
            "it out.",
        ),
    }
    st.info(
        "An eye contributes one point per interval, so the rows are **not** "
        "independent. The slope is tested with standard errors clustered by eye — "
        "a plain Pearson p would read N intervals where the study has N eyes.",
        icon="⚠️",
    )
    for m in targets_for(L.NEXT_CHANGE, params):
        st.subheader(m)
        frame, x, y = cached_design(
            sel, probe, L.NEXT_CHANGE, m, tuple(sorted(params.items()))
        )
        covariates = (f"{m}_baseline",) if params["adjust"] else ()
        res = cluster_robust_ols(frame, x, y, covariates=covariates)
        if "slope" not in res:
            st.warning(
                f"Not enough intervals to fit {m} "
                f"({res['n']} over {res.get('n_clusters', 0)} eyes)."
            )
            continue

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Intervals", res["n"], help=f"over {res['n_clusters']} eyes")
        k2.metric(
            "Slope",
            f"{res['slope']:.4g}",
            help=f"95% CI [{res['ci_low']:.4g}, {res['ci_high']:.4g}] — change in "
            f"Δ{m}/Δt per unit of {probe}",
        )
        k3.metric("p (clustered)", f"{res['p']:.3f}", help=f"t = {res['t']:.2f}")
        k4.metric("R²", f"{res['r2']:.3f}")
        show_regression(
            frame,
            x,
            y,
            x_label=f"{probe} at the visit (present)",
            y_label=f"Δ{m}/Δt over the next interval (per year)",
            show_stats=False,
            hover=["PatientId", "Eye", "date1", "date2", "dt_years"],
        )

# --- 4. visit-interval rates: one point per pair of visits --------------------
with tabs[3]:
    st.caption(
        "The unit of analysis is the visit *interval*: the per-year rate of change "
        "of both variables across it. An eye with V visits gives V−1 points."
    )
    params = {
        "consecutive": st.checkbox(
            "Consecutive visits only",
            value=True,
            help="Off: every visit pair — denser, but the intervals overlap and are "
            "not independent.",
        )
    }
    for m in targets_for(L.PAIR_RATES, params):
        st.subheader(m)
        frame, x, y = cached_design(
            sel, probe, L.PAIR_RATES, m, tuple(sorted(params.items()))
        )
        show_regression(
            frame,
            x,
            y,
            x_label=f"Δ{probe}/Δt (per year)",
            y_label=f"Δ{m}/Δt (per year)",
        )

# --- 5. plain cross-sectional association, all visits pooled -----------------
with tabs[4]:
    st.caption("All visits pooled, no temporal structure: probe vs measure.")
    for m in targets_for(L.CROSS_SECTION, {}):
        st.subheader(m)
        frame, x, y = cached_design(
            sel, probe, L.CROSS_SECTION, m, tuple(sorted({}.items()))
        )
        show_regression(frame, x, y, y_label=m)

# --- 6. group comparisons ----------------------------------------------------
with tabs[5]:
    st.caption(
        "Top: the probe by diagnosis. Below: eyes split at the median rate of "
        "change of a measure (progression group), or at the median of its value. "
        "The screening ranks these by a Mann–Whitney test between the two arms."
    )
    diag_cols = [c for c in ["Type", "Diagnosis"] if c in long_df.columns]
    if diag_cols:
        by = st.selectbox("Diagnosis grouping", diag_cols)
        show_box(long_df.drop_duplicates(subset=["case_id"]), probe, by, logy=logy)

    st.divider()
    split = st.radio(
        "Split eyes by",
        ["Rate of change of the measure", "Value of the measure"],
        horizontal=True,
    )
    params = {"split_on_value": split.startswith("Value"), "min_dt": 0.25}
    for m in targets_for(L.GROUPS, params):
        st.subheader(m)
        frame, value, group = cached_design(
            sel, probe, L.GROUPS, m, tuple(sorted(params.items()))
        )
        if len(frame) < 2:
            st.warning(f"Not enough rows to split {m} ({len(frame)}).")
            continue
        show_box(frame, value, group, logy=logy)

    # --- eyes split by how fast their worst ONH sector is thinning ------------
    st.divider()
    st.subheader("Steepest ONH change")
    st.caption(
        "One point per **eye**, not per visit: the steepest-change columns are a "
        "per-eye slope (µm/year over that eye's whole ONH series), so repeating "
        "them once per visit would weight an eye by how often it was scanned. "
        "The probe is the eye's median across its visits."
    )
    available = [c for c in STEEPEST_CHANGE_COLUMNS if c in cohort.columns]
    if not available:
        st.info("No steepest-change columns in this cohort table.")
    else:
        g1, g2, g3, g4 = st.columns([2, 2, 1, 1])
        change_col = g1.selectbox("Progression measure", available)
        split_how = g2.radio(
            "Split", ["Median", "Extreme thirds"], horizontal=True,
            help="Extreme thirds drops the middle third: a cleaner contrast on "
            "fewer eyes.",
        )
        min_exams = g3.number_input("Min exams", 2, 20, 4, help="Slopes fit on two "
            "exams are arithmetic, not progression.")
        min_span = g4.number_input("Min span (yr)", 0.0, 5.0, 1.0, 0.5)

        eyes = (
            cohort[
                (cohort["onh_slope_n_exams"] >= min_exams)
                & (cohort["onh_slope_span_years"] >= min_span)
            ]
            .groupby(["PatientId", "Eye"], as_index=False)
            .agg({probe: "median", change_col: "first"})
            .dropna(subset=[probe, change_col])
        )
        if len(eyes) < 6:
            st.warning(f"Only {len(eyes)} eyes pass the slope-quality filter.")
        else:
            # More negative = losing faster, so the *low* arm is the progressing one.
            if split_how == "Median":
                cut = eyes[change_col].median()
                eyes["Group"] = np.where(
                    eyes[change_col] <= cut, "Fast thinning", "Slow / stable"
                )
            else:
                lo, hi = eyes[change_col].quantile([1 / 3, 2 / 3])
                eyes["Group"] = np.where(
                    eyes[change_col] <= lo,
                    "Fastest third",
                    np.where(eyes[change_col] >= hi, "Slowest third", None),
                )
                eyes = eyes[eyes["Group"].notna()]

            arms = [g[probe].to_numpy() for _, g in eyes.groupby("Group")]
            if len(arms) == 2 and min(len(a) for a in arms) >= 3:
                u, p = mannwhitneyu(arms[0], arms[1], alternative="two-sided")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Eyes", len(eyes))
                k2.metric("AUC", f"{u / (len(arms[0]) * len(arms[1])):.3f}",
                          help="0.5 = the two arms are indistinguishable.")
                k3.metric("p (Mann–Whitney)", f"{p:.3f}")
                k4.metric(
                    "Δ median",
                    f"{np.median(arms[0]) - np.median(arms[1]):.4g}",
                    help=f"{probe}: first arm minus second, in the units of the probe.",
                )
                st.caption(
                    "Uncorrected: this is one more test in the same family as the "
                    "screening above. And a *negative* steepest change is expected "
                    "for almost every eye — it is the minimum of six noisy slopes — "
                    "so the split ranks eyes against each other, it does not "
                    "separate progressors from non-progressors."
                )
            show_box(eyes, probe, "Group", logy=logy)

            with st.expander(f"Distribution of {change_col} itself"):
                st.caption("Where the cut falls, and how heavy the tail is.")
                st.dataframe(
                    eyes.groupby("Group")[change_col]
                    .describe()[["count", "mean", "50%", "min", "max"]],
                    width="stretch",
                )
