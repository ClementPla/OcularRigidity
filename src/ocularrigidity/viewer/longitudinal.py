"""The six longitudinal designs, as data rather than as page code.

Each design answers the same question — *is the probed rigidity metric related to
this clinical measure?* — but pairs the visits differently, so each has its own
frame to plot and its own statistic to rank by. Defining them once here lets the
Streamlit page both *plot* one measure and *screen* all of them through exactly
the same code path (and lets the screening be cached, which matters: it fits a
model per measure per design).

A design is built with :func:`build` (frame + the two columns to regress) and
scored with :func:`score` (N, effect size, p-value). :func:`screen` runs the
score over every measure and ranks it, correcting for the fact that picking the
best of ~40 measures is a multiple comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr, spearmanr

from ocularrigidity.stats.inference import cluster_robust_ols, fdr_qvalues
from ocularrigidity.stats.temporal import (
    baseline_vs_future_slope,
    baseline_vs_next_rate,
    compute_slope,
    pairwise_rate_of_change,
)

COPROGRESSION = "coprogression"
FUTURE = "future"
NEXT_CHANGE = "next_change"
PAIR_RATES = "pair_rates"
CROSS_SECTION = "cross_section"
GROUPS = "groups"

# What the ranking table calls the effect size of each design.
EFFECT_LABEL = {
    COPROGRESSION: "Pearson r",
    FUTURE: "Pearson r",
    NEXT_CHANGE: "Slope (clustered)",
    PAIR_RATES: "Pearson r",
    CROSS_SECTION: "Pearson r",
    GROUPS: "AUC (low vs high)",
}

GROUP_COLS = ["PatientId", "Eye"]

# Ranking is done on a RANK-based p (Spearman / Mann-Whitney), never on Pearson.
# A screen sorted by Pearson p is a leverage-point detector: on this cohort the
# top "hit" of the co-progression design was r = -0.84, p = 2e-20 — and r = +0.03
# once the single most extreme eye was dropped (its Spearman rho was -0.03). The
# Pearson r is still reported next to it: when the two disagree that badly, the
# association is one point, not a finding.
SPEARMAN_COL = "Spearman ρ"


def numeric_subset(long_df: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Rows for one measure, with its value coerced into a column named after it."""
    d = long_df[long_df["MeasureName_y"] == measure].copy()
    d[measure] = pd.to_numeric(d["MeasureValue_y"], errors="coerce")
    return d


def build(
    design: str, long_df: pd.DataFrame, probe: str, measure: str, params: dict
) -> tuple[pd.DataFrame, str, str]:
    """The frame and the two columns this design regresses, for one measure.

    For :data:`GROUPS` the ``y`` column is the categorical ``"Group"``: the frame
    is a box-plot input rather than a scatter.
    """
    if design == COPROGRESSION:
        slopes = (
            numeric_subset(long_df, measure)
            .groupby(GROUP_COLS)
            .apply(
                compute_slope,
                probe,
                measure,
                min_points=params["min_points"],
                include_groups=False,
            )
            .dropna()
            .reset_index()
        )
        return slopes, f"{probe}_slope", f"{measure}_slope"

    if design == FUTURE:
        paired = baseline_vs_future_slope(
            long_df, measure, value_col=probe, min_points=params["min_points"]
        )
        return paired, f"{probe}_baseline", f"{measure}_slope"

    if design == NEXT_CHANGE:
        pairs = baseline_vs_next_rate(
            long_df,
            measure,
            value_col=probe,
            consecutive_only=params["consecutive"],
            min_dt_years=params["min_dt"],
        )
        return pairs, f"{probe}_baseline", f"{measure}_rate"

    if design == PAIR_RATES:
        pairs = pairwise_rate_of_change(
            long_df, measure, value_col=probe, consecutive_only=params["consecutive"]
        )
        return pairs, f"{probe}_rate", f"{measure}_rate"

    if design == CROSS_SECTION:
        return numeric_subset(long_df, measure), probe, measure

    if design == GROUPS:
        if params["split_on_value"]:
            d = numeric_subset(long_df, measure)
            median = d[measure].median()
            d["Group"] = np.where(
                d[measure] <= median, f"Low {measure}", f"High {measure}"
            )
            return d, probe, "Group"
        # Otherwise the lagged pairing of NEXT_CHANGE, dichotomised: the probe at
        # a visit, against how fast the measure moved over the interval after it.
        pairs = baseline_vs_next_rate(
            long_df,
            measure,
            value_col=probe,
            consecutive_only=True,
            min_dt_years=params.get("min_dt", 0.25),
        )
        rate = f"{measure}_rate"
        if not pairs.empty:
            median_rate = pairs[rate].median()
            pairs["Group"] = np.where(
                pairs[rate] <= median_rate,
                f"Low {measure} rate",
                f"High {measure} rate",
            )
        return pairs, f"{probe}_baseline", "Group"

    raise ValueError(f"Unknown design: {design!r}")


def score(
    design: str, long_df: pd.DataFrame, probe: str, measure: str, params: dict
) -> dict | None:
    """One measure under one design: ``{N, effect, Spearman ρ, p}``, or None.

    ``p`` is what the screen ranks on, and it is always **rank-based** — Spearman
    for the correlation designs, Mann-Whitney for the group split (see
    :data:`SPEARMAN_COL`). The design's own effect size is reported beside it:
    Pearson r, or the cluster-robust slope for :data:`NEXT_CHANGE`, where an eye
    contributes several intervals and a naive p would reward whichever measure
    repeats most within an eye.
    """
    frame, x, y = build(design, long_df, probe, measure, params)
    if frame.empty:
        return None

    if design == GROUPS:
        arms = [
            pd.to_numeric(g[x], errors="coerce").dropna()
            for _, g in frame.dropna(subset=[y]).groupby(y)
        ]
        if len(arms) != 2 or min(len(a) for a in arms) < 3:
            return None
        # Rank-based already: the probe is skewed, so a t-test would follow the
        # tail rather than the shift between the groups. Effect = the AUC (0.5 =
        # the two groups are indistinguishable).
        u, p = mannwhitneyu(arms[0], arms[1], alternative="two-sided")
        n1, n2 = len(arms[0]), len(arms[1])
        return {"N": n1 + n2, "effect": float(u) / (n1 * n2), "p": float(p)}

    d = frame[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < 3 or d[x].nunique() < 2 or d[y].nunique() < 2:
        return None
    rho, p_rho = spearmanr(d[x], d[y])
    if not np.isfinite(p_rho):
        return None

    if design == NEXT_CHANGE:
        covariates = (f"{measure}_baseline",) if params["adjust"] else ()
        res = cluster_robust_ols(frame, x, y, covariates=covariates)
        if "slope" not in res:
            return None
        # Here the clustered p *is* the rank-agnostic one we want (it is the
        # repeated-measures problem, not the outlier problem, that dominates).
        return {
            "N": res["n"],
            "effect": res["slope"],
            SPEARMAN_COL: float(rho),
            "p": res["p"],
        }

    r, _ = pearsonr(d[x], d[y])
    return {
        "N": int(len(d)),
        "effect": float(r),
        SPEARMAN_COL: float(rho),
        "p": float(p_rho),
    }


def screen(
    design: str,
    long_df: pd.DataFrame,
    probe: str,
    measures: list[str],
    params: dict,
) -> pd.DataFrame:
    """Score every measure under one design and rank it by p, with BH q-values.

    Returns ``measure, N, <effect>, Spearman ρ, p, q (BH)``, most significant
    first. Two things make the top row readable rather than misleading:

    * ``p`` is rank-based (see :func:`score`), so a single leverage point cannot
      manufacture a hit — sorting on a Pearson p would do exactly that here;
    * ``q (BH)`` is the number to act on. The top rows were *selected* for a small
      p out of ~40 tries, so their raw p is optimistic by construction — about two
      of forty land under 0.05 by chance alone.
    """
    rows = []
    for m in measures:
        try:
            r = score(design, long_df, probe, m, params)
        except (ValueError, KeyError, IndexError, ZeroDivisionError):
            r = None
        if r is not None and np.isfinite(r["p"]):
            rows.append({"measure": m, **r})

    label = EFFECT_LABEL[design]
    if not rows:
        return pd.DataFrame(columns=["measure", "N", label, "p", "q (BH)"])

    out = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    out["q (BH)"] = fdr_qvalues(out["p"])
    return out.rename(columns={"effect": label})
