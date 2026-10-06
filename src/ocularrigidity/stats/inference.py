"""Regression with standard errors that survive repeated measures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


def fdr_qvalues(pvalues) -> np.ndarray:
    """Benjamini-Hochberg q-values for a family of p-values."""
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
    """OLS of ``y`` on ``x`` with standard errors clustered by ``group_cols``."""
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
