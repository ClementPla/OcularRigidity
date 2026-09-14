"""The Integrated Glaucoma Risk Index, base term (I-GRI.base).

Jung et al., *Development of the Integrated Glaucoma Risk Index*, Diagnostics
12(3):734, 2022 -- https://doi.org/10.3390/diagnostics12030734 -- Equation (3)::

    I-GRI.base = PSD.normal    x 0.27 + MD.normal     x 0.14
               + RNFL_S.normal x 0.11 + RNFL_I.normal x 0.31
               + RNFL_T.normal x 0.10 + IOP.normal    x 0.07

A fixed published formula, not a model fitted here: the weights and the min/max
the features are normalized against both come from the paper.

The paper's full index adds a second term, ``I-GRI = I-GRI.base x 0.8 + NNI x
0.2``, where NNI is the share of an eye's five nearest neighbours labelled
glaucoma. That needs a reference cohort labelled glaucoma/normal, which this one
does not have -- its Normal, OHT and Suspect eyes carry no ONH exam at all, so
none of them can even form the six-feature vector a neighbour search needs.
Only Equation (3) is computed here, and the column is named ``igri_base``
accordingly. The published classification threshold (0.36) and stage means
belong to the full index and are deliberately not reproduced: dropping the NNI
term takes up to 0.2 off every score, and more off a glaucoma eye than a healthy
one, so no published cut-off transfers to this column.

Reading the output
------------------
Higher means greater risk/severity. The value lies in [0, 1] because the weights
sum to 1 and each normalized feature is clipped to [0, 1] against
:data:`PAPER_RANGES` -- the paper's own Table 6. Keeping their ranges rather
than this cohort's is what keeps a value here on the same scale as a value
there; :func:`local_ranges` switches to cohort-relative normalization instead,
which spreads the index over the full range at the cost of that comparability.

Two published quirks, kept on purpose
-------------------------------------
Equation (3) reverses ``RNFL_S, RNFL_I, RNFL_T`` **and IOP** ("Among the four
features [...] the lower the feature value, the greater the risk/severity"), and
does not reverse MD. Taken literally, a *lower* IOP and a *better* visual field
each push the index up, which is backwards on both counts. It is what the paper
specifies, and it does reproduce the group means the paper reports, so
``faithful=True`` (the default) follows it. ``faithful=False`` flips those two
terms to the clinically sensible direction; the result is then no longer the
published index.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

#: Feature order used everywhere below -- also the order of Equation (3).
FEATURES = ("PSD", "MD", "RNFL_S", "RNFL_I", "RNFL_T", "IOP")

#: The importance ratios of Equation (3). They sum to 1, which is what bounds
#: the index once every normalized feature is in [0, 1].
WEIGHTS = {
    "PSD": 0.27,
    "MD": 0.14,
    "RNFL_S": 0.11,
    "RNFL_I": 0.31,
    "RNFL_T": 0.10,
    "IOP": 0.07,
}

#: Min/max of the paper's own dataset (Table 6).
PAPER_RANGES = {
    "PSD": (0.95, 16.9),
    "MD": (-24.1, 6.39),
    "RNFL_S": (6.0, 172.0),
    "RNFL_I": (0.0, 195.0),
    "RNFL_T": (20.0, 110.0),
    "IOP": (5.0, 29.0),
}

#: Features the paper takes as ``1 - normalized``. See the module note: IOP
#: being here, and MD not, is the paper's and is clinically backwards.
REVERSED = ("RNFL_S", "RNFL_I", "RNFL_T", "IOP")

#: The same list under ``faithful=False``: every feature for which lower really
#: is worse, and only those.
REVERSED_CORRECTED = ("MD", "RNFL_S", "RNFL_I", "RNFL_T")

#: The features that come from the ONH exam, and so are the ones the
#: ``onh_source`` filter applies to. PSD/MD come from the visual field and IOP
#: from tonometry; neither is affected by which OCT source won a row.
ONH_FEATURES = ("RNFL_S", "RNFL_I", "RNFL_T")

#: Which ONH source the RNFL sectors must come from. ``build_cohort`` takes them
#: from the Heyex report parse where it has one and falls back to ClinicalValues
#: per row, recording the winner in ``onh_source`` -- so the sector columns are a
#: mix. The two sources overlap on ~470 eye-months and agree exactly on ~70% of
#: values (r = 0.97-0.99), but individual rows differ by as much as 138 um, which
#: is more than enough to move an index whose RNFL terms carry weight 0.52.
#: Heyex is the direct parse of the report PDFs, so it is the one to keep.
DEFAULT_ONH_SOURCE = "heyex"

#: How the six features are read off a cohort frame. A tuple means "the mean of
#: these columns": the paper uses the four-quadrant OCT report (S/I/T/N) while
#: this cohort carries the six Garway-Heath sectors, so the superior and
#: inferior quadrants are each rebuilt from their two sectors. That mapping is
#: an interpretation, not something the paper states -- pass ``columns`` to
#: override it. ``IOP`` follows whichever instrument the cohort was built with;
#: the paper used Goldmann applanation, so ``{"IOP": "Goldman IOP"}`` is the
#: closest match to it.
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
    """The six raw inputs, one column per feature, indexed like ``df``.

    Columns absent from ``df`` come back all-NaN rather than raising, so a frame
    that never had a visual field still produces a well-formed result.

    ``onh_source`` blanks the RNFL features on rows whose ``source_col`` says the
    sectors came from somewhere else -- see :data:`DEFAULT_ONH_SOURCE`. Those
    rows then fail the all-six-present test and go unscored, which is the point:
    a value half from one instrument's parse and half from another's is not on
    one scale. Pass ``None`` to take whatever ``build_cohort`` left in the
    columns. If ``source_col`` is not in ``df`` the filter cannot be applied;
    :func:`add_igri_base` says so in its report rather than failing quietly.
    """
    columns = dict(columns or DEFAULT_COLUMNS)
    out = pd.DataFrame(index=df.index, columns=list(FEATURES), dtype=float)
    for feature in FEATURES:
        spec = columns.get(feature)
        names = (spec,) if isinstance(spec, str) else tuple(spec or ())
        present = [c for c in names if c in df.columns]
        if not present:
            continue
        # Mean of the sectors making up a quadrant; one sector short of the pair
        # still yields a quadrant rather than a hole.
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
    """Min/max taken from ``df`` itself, as an alternative to :data:`PAPER_RANGES`.

    Nothing is clipped and the index uses its full range, but the value becomes
    cohort-relative: it can no longer be compared with one computed elsewhere,
    and it shifts as the cohort grows.
    """
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
    """Min-max each feature into [0, 1], reversing the ones named in ``reverse``.

    Values outside ``ranges`` are clipped, which is the honest reading of a
    published range: an eye thinner than the paper's thinnest is at the top of
    the risk scale, not off it. :func:`add_igri_base` reports how often it bites.
    """
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
    """Equation (3) as a Series indexed like ``df``; NaN where a feature is missing."""
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
    """Add ``igri_base`` to a copy of ``df``. Returns ``(frame, report)``.

    The report carries what the column cannot: how many rows had all six
    features, which feature was the binding constraint, and how many values had
    to be clipped onto the paper's ranges. A feature clipped often means this
    cohort sits outside the range the index was calibrated on.

    A row is scored only when all six features are present. A partial weighted
    sum is not on a comparable scale: the weights of the missing terms simply
    vanish rather than being redistributed, so an eye missing its visual field
    would score 0.41 lower than an identical eye that has one.
    """
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
        # False means the filter was asked for but the frame had no onh_source
        # column, so the RNFL values are whatever build_cohort left there.
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
