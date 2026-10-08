"""The Integrated Glaucoma Risk Index, base term (I-GRI.base)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

# : Feature order used everywhere below
FEATURES = ("PSD", "MD", "RNFL_S", "RNFL_I", "RNFL_T", "IOP")

# : The importance ratios of Equation (3).
WEIGHTS = {
    "PSD": 0.27,
    "MD": 0.14,
    "RNFL_S": 0.11,
    "RNFL_I": 0.31,
    "RNFL_T": 0.10,
    "IOP": 0.07,
}

# : Min/max of the paper's own dataset (Table 6).
PAPER_RANGES = {
    "PSD": (0.95, 16.9),
    "MD": (-24.1, 6.39),
    "RNFL_S": (6.0, 172.0),
    "RNFL_I": (0.0, 195.0),
    "RNFL_T": (20.0, 110.0),
    "IOP": (5.0, 29.0),
}

# : Features the paper takes as ``1 - normalized``.
REVERSED = ("RNFL_S", "RNFL_I", "RNFL_T", "IOP")

REVERSED_CORRECTED = ("MD", "RNFL_S", "RNFL_I", "RNFL_T")

ONH_FEATURES = ("RNFL_S", "RNFL_I", "RNFL_T")

# : Which ONH source the RNFL sectors must come from.
DEFAULT_ONH_SOURCE = "heyex"

# : How the six features are read off a cohort frame.
DEFAULT_COLUMNS: dict[str, str | tuple[str, ...]] = {
    "PSD": "PSD",
    "MD": "MD",
    "RNFL_S": ("TS RNFL Thickness", "NS RNFL Thickness"),
    "RNFL_I": ("TI RNFL Thickness", "NI RNFL Thickness"),
    "RNFL_T": ("T RNFL Thickness",),
    "IOP": "IOP",
}


def igri_features(
    df: pd.DataFrame,
    columns: Mapping[str, str | tuple[str, ...]] | None = None,
    onh_source: str | None = DEFAULT_ONH_SOURCE,
    source_col: str = "onh_source",
) -> pd.DataFrame:
    """The six raw inputs, one column per feature, indexed like ``df``."""
    columns = dict(columns or DEFAULT_COLUMNS)
    out = pd.DataFrame(index=df.index, columns=list(FEATURES), dtype=float)
    for feature in FEATURES:
        spec = columns.get(feature)
        names = (spec,) if isinstance(spec, str) else tuple(spec or ())
        present = [c for c in names if c in df.columns]
        if not present:
            continue
        # Mean of the sectors making up a quadrant
        out[feature] = df[present].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    if onh_source is not None and source_col in df.columns:
        wrong = df[source_col].ne(onh_source)
        out.loc[wrong, list(ONH_FEATURES)] = np.nan
    return out


def local_ranges(
    df: pd.DataFrame,
    columns: Mapping[str, str | tuple[str, ...]] | None = None,
    onh_source: str | None = DEFAULT_ONH_SOURCE,
) -> dict[str, tuple[float, float]]:
    """Min/max taken from ``df`` itself, as an alternative to :data:`PAPER_RANGES`."""
    feat = igri_features(df, columns, onh_source)
    return {
        f: (float(feat[f].min()), float(feat[f].max()))
        for f in FEATURES
        if feat[f].notna().any()
    }


def normalize_features(
    feat: pd.DataFrame,
    ranges: Mapping[str, tuple[float, float]] = PAPER_RANGES,
    reverse: Sequence[str] = REVERSED,
) -> pd.DataFrame:
    """Min-max each feature into [0, 1], reversing the ones named in ``reverse``."""
    out = pd.DataFrame(index=feat.index, columns=list(FEATURES), dtype=float)
    for f in FEATURES:
        if f not in ranges:
            continue
        lo, hi = ranges[f]
        if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
            continue
        v = ((feat[f] - lo) / (hi - lo)).clip(0.0, 1.0)
        out[f] = 1.0 - v if f in reverse else v
    return out


def igri_base(
    df: pd.DataFrame,
    *,
    columns: Mapping[str, str | tuple[str, ...]] | None = None,
    ranges: Mapping[str, tuple[float, float]] = PAPER_RANGES,
    onh_source: str | None = DEFAULT_ONH_SOURCE,
    faithful: bool = True,
) -> pd.Series:
    """Equation (3) as a Series indexed like ``df``"""
    norm = normalize_features(
        igri_features(df, columns, onh_source),
        ranges,
        REVERSED if faithful else REVERSED_CORRECTED,
    )
    complete = norm[list(FEATURES)].notna().all(axis=1)
    return sum(norm[f] * WEIGHTS[f] for f in FEATURES).where(complete)


def add_igri_base(
    df: pd.DataFrame,
    *,
    columns: Mapping[str, str | tuple[str, ...]] | None = None,
    ranges: Mapping[str, tuple[float, float]] = PAPER_RANGES,
    onh_source: str | None = DEFAULT_ONH_SOURCE,
    faithful: bool = True,
    keep_normalized: bool = False,
    prefix: str = "igri",
) -> tuple[pd.DataFrame, dict]:
    """Add ``igri_base`` to a copy of ``df``."""
    reverse = REVERSED if faithful else REVERSED_CORRECTED
    feat = igri_features(df, columns, onh_source)
    norm = normalize_features(feat, ranges, reverse)
    complete = norm[list(FEATURES)].notna().all(axis=1)
    base = sum(norm[f] * WEIGHTS[f] for f in FEATURES).where(complete)

    out = df.copy()
    out[f"{prefix}_base"] = base
    if keep_normalized:
        for f in FEATURES:
            out[f"{prefix}_{f}_normal"] = norm[f]

    report = {
        "n_rows": len(df),
        "n_scored": int(base.notna().sum()),
        "missing_per_feature": {f: int(norm[f].isna().sum()) for f in FEATURES},
        "clipped_per_feature": {
            f: int(((feat[f] < ranges[f][0]) | (feat[f] > ranges[f][1])).sum())
            for f in FEATURES
            if f in ranges
        },
        "onh_source": onh_source,
        "onh_source_enforced": bool(onh_source is None or "onh_source" in df.columns),
        "onh_source_counts": (
            df["onh_source"].value_counts(dropna=False).to_dict()
            if "onh_source" in df.columns
            else None
        ),
        "faithful": faithful,
        "reversed": tuple(reverse),
    }
    return out, report
