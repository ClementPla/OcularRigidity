"""Regression with standard errors that survive repeated measures.

The interval-level designs in :mod:`ocularrigidity.stats.temporal` hand one row
per *visit interval*, so one eye contributes several rows. A plain Pearson /
OLS p-value treats those as independent subjects and is therefore
anti-conservative — it reads ~200 points where the study has ~95 eyes. Clustering
the standard errors on the eye fixes the inference (the slope is unchanged; its
uncertainty grows to reflect that the rows repeat).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


def fdr_qvalues(pvalues) -> np.ndarray:
    """Benjamini-Hochberg q-values for a family of p-values.

    Screening every clinical measure and keeping the best few is a multiple
    comparison: with 40 measures, two land under p < 0.05 by chance alone. The
    q-value is the false-discovery rate at which that measure would be called —
    it is the number to read when the measure was *selected* for being small.
    """
    p = np.asarray(pvalues, dtype=float)
    finite = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    if finite.any():
        q[finite] = multipletests(p[finite], method="fdr_bh")[1]
    return q


def cluster_robust_ols(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    group_cols=("PatientId", "Eye"),
    covariates: tuple[str, ...] = (),
) -> dict:
    """OLS of ``y`` on ``x`` with standard errors clustered by ``group_cols``.

    ``covariates`` are adjusted for (partialled out), e.g. the measure's own
    baseline value — progression usually depends on how far the disease already
    is, and that starting point may itself correlate with ``x``.

    Returns a dict with the slope on ``x``, its cluster-robust standard error,
    t / p, the 95% CI, the number of rows and of clusters, and the model R².
    Returns ``{"n": ...}`` alone when there is not enough data to fit.
    """
    cols = [x, y, *covariates]
    d = df[cols + list(group_cols)].copy()
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna()

    groups = d[list(group_cols)].astype(str).agg("/".join, axis=1)
    n_clusters = groups.nunique()
    # Cluster-robust SEs need more clusters than parameters to be meaningful.
    if len(d) < len(cols) + 2 or n_clusters < 3:
        return {"n": len(d), "n_clusters": n_clusters}

    design = sm.add_constant(d[[x, *covariates]], has_constant="add")
    fit = sm.OLS(d[y], design).fit(
        cov_type="cluster", cov_kwds={"groups": groups.to_numpy()}
    )
    lo, hi = fit.conf_int().loc[x]
    return {
        "n": int(len(d)),
        "n_clusters": int(n_clusters),
        "slope": float(fit.params[x]),
        "se": float(fit.bse[x]),
        "t": float(fit.tvalues[x]),
        "p": float(fit.pvalues[x]),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "r2": float(fit.rsquared),
        "covariates": list(covariates),
    }
