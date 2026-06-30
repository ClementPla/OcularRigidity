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
