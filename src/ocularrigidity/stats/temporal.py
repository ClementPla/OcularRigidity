import pandas as pd
import numpy as np
import seaborn as sns


def compute_slope(group, col_entry1, col_entry2, min_points, col_date="YearMonth"):
    # 1. Clean data locally within the group
    df = group[[col_date, col_entry1, col_entry2]].copy()

    df[col_date] = pd.to_datetime(df[col_date], format="%Y-%m", errors="coerce")
    df[col_entry1] = pd.to_numeric(df[col_entry1], errors="coerce")
    df[col_entry2] = pd.to_numeric(df[col_entry2], errors="coerce")
    df = df.dropna()

    if len(df) < min_points:
        return pd.Series({f"{col_entry1}_slope": np.nan, f"{col_entry2}_slope": np.nan})

    dates = df[col_date].to_numpy()

    delta_days = (dates - dates.min()) / np.timedelta64(1, "D")
    x = delta_days / 365.25  # Accurately account for leap years

    # 4. Zero-variance check
    if np.all(x == x[0]):
        return pd.Series({f"{col_entry1}_slope": np.nan, f"{col_entry2}_slope": np.nan})

    entry1_slope = np.polyfit(x, df[col_entry1].to_numpy(), 1)[0]
    entry2_slope = np.polyfit(x, df[col_entry2].to_numpy(), 1)[0]

    return pd.Series(
        {f"{col_entry1}_slope": entry1_slope, f"{col_entry2}_slope": entry2_slope}
    )


def baseline_vs_future_slope(
    df,
    measure,
    *,
    value_col="K",
    date_col="YearMonth",
    measure_name_col="MeasureName_y",
    measure_value_col="MeasureValue_y",
    group_cols=("PatientId", "Eye"),
    min_points=2,
):
    """Baseline (present) ``value_col`` vs future progression slope of ``measure``.

    Unlike :func:`compute_slope` (which correlates the *rates of change* of two
    variables), this pairs a single **present** value against a **future** trend:
    for each group (e.g. patient-eye), the ``value_col`` at the earliest visit is
    taken as the baseline, and the per-year linear slope of ``measure`` is fit
    over that group's visits from the baseline onward. This tests whether the
    present rigidity predicts subsequent progression of ``measure``.

    ``df`` is long — one row per (group, visit, measure) — and is filtered to
    ``measure`` via ``measure_name_col``. Dates are parsed as ``%Y-%m``.

    Returns one row per group, ``[*group_cols, f"{value_col}_baseline",
    f"{measure}_slope"]``. Groups with fewer than ``min_points`` distinct
    visit-months, or zero time-spread, are dropped.
    """
    group_cols = list(group_cols)
    d = df[df[measure_name_col] == measure].copy()
    d[measure] = pd.to_numeric(d[measure_value_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d["_date"] = pd.to_datetime(d[date_col], format="%Y-%m", errors="coerce")
    d = d.dropna(subset=[value_col, measure, "_date"])

    # Collapse to one value per visit-month per group (multiple rows can share a
    # month after the measure merge).
    d = d.groupby(group_cols + ["_date"])[[value_col, measure]].mean().reset_index()

    rows = []
    for keys, g in d.groupby(group_cols):
        g = g.sort_values("_date")
        if len(g) < min_points:
            continue
        baseline_value = g[value_col].iloc[0]  # present (earliest visit)
        x = (g["_date"] - g["_date"].iloc[0]).dt.days.to_numpy() / 365.25
        if np.all(x == x[0]):
            continue
        slope = np.polyfit(x, g[measure].to_numpy(), 1)[0]  # future progression
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append((*keys, baseline_value, slope))

    return pd.DataFrame(
        rows, columns=group_cols + [f"{value_col}_baseline", f"{measure}_slope"]
    )


def baseline_vs_slope_wide(
    df,
    x,
    y,
    *,
    date_col="YearMonth",
    group_cols=("PatientId", "Eye"),
    min_points=2,
):
    """Baseline ``x`` vs the per-year slope of ``y``, on a **wide** frame.

    The wide-frame twin of :func:`baseline_vs_future_slope`. That one reads the
    long measure table, so it can only ever pair the probe against a melted
    ``MeasureName_y``; here both are ordinary columns, which lets any two
    variables of the cohort table be paired -- a covariate against an ONH sector,
    one clinical measure against another.

    Per group (patient-eye): ``x`` at the earliest visit is the baseline, and
    ``y`` is regressed on time from that visit onward, in units per year. Visits
    are collapsed to one per ``date_col`` first, so an eye seen twice inside one
    month does not weigh double.

    The pairing is what makes this worth a separate design: one point per eye
    rather than one per visit, so the correlation is not inflated by repeated
    measurements -- at the cost of every eye counting the same whether its slope
    was fit over two visits or six. ``n_visits`` and ``span_years`` come back
    alongside so a caller can drop the slopes that are arithmetic rather than
    progression.

    Returns one row per group: ``[*group_cols, f"{x}_baseline", f"{y}_slope",
    "n_visits", "span_years"]``. Groups with fewer than ``min_points`` distinct
    dates, or with no spread in time, are dropped.
    """
    group_cols = list(group_cols)
    # dict.fromkeys, not a set: x may be y (baseline vs own slope is a valid ask)
    # and selecting the same label twice would give a duplicated column.
    d = df[list(dict.fromkeys(group_cols + [date_col, x, y]))].copy()
    d[x] = pd.to_numeric(d[x], errors="coerce")
    d[y] = pd.to_numeric(d[y], errors="coerce")
    d["_date"] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[x, y, "_date"])

    values = list(dict.fromkeys([x, y]))
    d = d.groupby(group_cols + ["_date"], as_index=False)[values].mean()

    rows = []
    for keys, g in d.groupby(group_cols):
        g = g.sort_values("_date")
        if len(g) < min_points:
            continue
        t = (g["_date"] - g["_date"].iloc[0]).dt.days.to_numpy() / 365.25
        if np.all(t == t[0]):
            continue
        slope = np.polyfit(t, g[y].to_numpy(), 1)[0]
        residual = g[y].to_numpy() - (slope * t + g[y].iloc[0])
        lsnr = slope / np.std(residual) if np.std(residual) > 0 else np.nan
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append((*keys, g[x].iloc[0], slope, residual, lsnr, len(g), float(t.max())))

    return pd.DataFrame(
        rows,
        columns=group_cols + [f"{x}_baseline", f"{y}_slope", f"{y}_residual", f"{y}_lsnr", "n_visits", "span_years"],
    )


def baseline_vs_next_rate(
    df,
    measure,
    *,
    value_col="K",
    date_col="YearMonth",
    measure_name_col="MeasureName_y",
    measure_value_col="MeasureValue_y",
    group_cols=("PatientId", "Eye"),
    consecutive_only=True,
    min_dt_years=0.25,
):
    """``value_col`` at a visit vs how ``measure`` moves over the interval that follows.

    The lagged, interval-level design: for every pair of visits of one eye, the
    predictor is ``value_col`` measured at the *first* visit of the pair, and the
    response is the per-year rate of change of ``measure`` across the interval,
    ``(m2 - m1) / (t2 - t1)``. It asks whether the metric *today* anticipates the
    change that happens *next*.

    This sits between the two other designs:

    * :func:`baseline_vs_future_slope` is also predictive but spends a whole eye
      on a single point (its first visit vs one slope fit over all the rest);
    * :func:`pairwise_rate_of_change` also works per interval but correlates two
      *rates*, which is contemporaneous — it cannot separate leading from merely
      co-occurring.

    Here an eye with V visits contributes V-1 points *and* the temporal order is
    kept. The price is that those points are not independent (they share an eye),
    so the association must be tested with standard errors clustered by eye —
    see :func:`ocularrigidity.stats.inference.cluster_robust_ols`. The measure's
    own baseline (``f"{measure}_baseline"``) is returned so it can be adjusted
    for: progression usually depends on how bad the eye already is.

    Returns
    -------
    pandas.DataFrame
        One row per visit interval:
        ``[*group_cols, "date1", "date2", "dt_years", f"{value_col}_baseline",
        f"{measure}_baseline", f"{measure}_delta", f"{measure}_rate"]``.
    """
    import itertools

    group_cols = list(group_cols)
    d = df[df[measure_name_col] == measure].copy()
    d[measure] = pd.to_numeric(d[measure_value_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d["_date"] = pd.to_datetime(d[date_col], format="%Y-%m", errors="coerce")
    d = d.dropna(subset=[value_col, measure, "_date"])

    # One value per visit-month per eye (rows can share a month after the merge).
    d = d.groupby(group_cols + ["_date"])[[value_col, measure]].mean().reset_index()

    rows = []
    for keys, g in d.groupby(group_cols):
        g = g.sort_values("_date").reset_index(drop=True)
        if len(g) < 2:
            continue
        pairs = (
            zip(range(len(g) - 1), range(1, len(g)))
            if consecutive_only
            else itertools.combinations(range(len(g)), 2)
        )
        keys = keys if isinstance(keys, tuple) else (keys,)
        for i, j in pairs:
            dt = (g["_date"][j] - g["_date"][i]).days / 365.25
            if dt < min_dt_years:
                continue
            delta = g[measure][j] - g[measure][i]
            rows.append(
                (
                    *keys,
                    g["_date"][i],
                    g["_date"][j],
                    dt,
                    g[value_col][i],  # the predictor: probe at the interval's start
                    g[measure][i],  # where the measure started, to adjust for
                    delta,
                    delta / dt,
                )
            )

    return pd.DataFrame(
        rows,
        columns=group_cols
        + [
            "date1",
            "date2",
            "dt_years",
            f"{value_col}_baseline",
            f"{measure}_baseline",
            f"{measure}_delta",
            f"{measure}_rate",
        ],
    )


def pairwise_rate_of_change(
    df,
    measure,
    *,
    value_col="K",
    date_col="YearMonth",
    measure_name_col="MeasureName_y",
    measure_value_col="MeasureValue_y",
    group_cols=("PatientId", "Eye"),
    consecutive_only=True,
    min_dt_years=1e-3,
):
    """Per-year rate of change of ``value_col`` and ``measure`` between visit pairs.

    Each returned row is a **pair of visits** within a group: the rate of change
    of both variables across that interval, ``(v2 - v1) / (t2 - t1)`` (per year).
    Correlating the two rate columns asks whether changes in ``value_col`` track
    changes in ``measure`` at the visit-interval level — so a subject with V
    visits contributes V-1 points (consecutive) rather than a single point.

    This differs from :func:`compute_slope` (one fitted slope per subject) and
    from :func:`baseline_vs_future_slope` (one present-vs-future point per
    subject): here the unit of analysis is the visit interval.

    Two arguments carry traps. ``consecutive_only=False`` takes every visit
    pair rather than adjacent ones — denser, but the intervals overlap and are
    not independent. ``min_dt_years`` skips pairs shorter than that interval,
    guarding divide-by-zero for visits collapsed into the same month.

    Returns one row per visit pair, ``[*group_cols, "date1", "date2",
    "dt_years", f"{value_col}_rate", f"{measure}_rate"]``.
    """
    import itertools

    group_cols = list(group_cols)
    d = df[df[measure_name_col] == measure].copy()
    d[measure] = pd.to_numeric(d[measure_value_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d["_date"] = pd.to_datetime(d[date_col], format="%Y-%m", errors="coerce")
    d = d.dropna(subset=[value_col, measure, "_date"])

    # One value per visit-month per group (rows can share a month after merge).
    d = d.groupby(group_cols + ["_date"])[[value_col, measure]].mean().reset_index()

    rows = []
    for keys, g in d.groupby(group_cols):
        g = g.sort_values("_date").reset_index(drop=True)
        if len(g) < 2:
            continue
        pairs = (
            zip(range(len(g) - 1), range(1, len(g)))
            if consecutive_only
            else itertools.combinations(range(len(g)), 2)
        )
        keys = keys if isinstance(keys, tuple) else (keys,)
        for i, j in pairs:
            dt = (g["_date"][j] - g["_date"][i]).days / 365.25
            if dt < min_dt_years:
                continue
            v_rate = (g[value_col][j] - g[value_col][i]) / dt
            m_rate = (g[measure][j] - g[measure][i]) / dt
            rows.append(
                (
                    *keys,
                    g["_date"][i],
                    g["_date"][j],
                    dt,
                    v_rate,
                    m_rate,
                )
            )

    return pd.DataFrame(
        rows,
        columns=group_cols
        + ["date1", "date2", "dt_years", f"{value_col}_rate", f"{measure}_rate"],
    )
