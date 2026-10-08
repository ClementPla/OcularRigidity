import pandas as pd
import numpy as np


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
    """Baseline (present) ``value_col`` vs future progression slope of ``measure``."""
    group_cols = list(group_cols)
    d = df[df[measure_name_col] == measure].copy()
    d[measure] = pd.to_numeric(d[measure_value_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d["_date"] = pd.to_datetime(d[date_col], format="%Y-%m", errors="coerce")
    d = d.dropna(subset=[value_col, measure, "_date"])

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
    """Baseline ``x`` vs the per-year slope of ``y``, on a **wide** frame."""
    group_cols = list(group_cols)
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
    """``value_col`` at a visit vs how ``measure`` moves over the interval that follows."""
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
    """Per-year rate of change of ``value_col`` and ``measure`` between visit pairs."""
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
