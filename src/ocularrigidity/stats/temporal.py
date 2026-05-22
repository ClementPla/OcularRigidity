import pandas as pd
import numpy as np
import seaborn as sns


def compute_slope(group, col_entry1, col_entry2, min_points, col_date="YearMonth"):
    # Keep only valid numeric rows
    df = group[[col_date, col_entry1, col_entry2]].copy()
    df[col_date] = pd.to_datetime(df[col_date], format="%Y-%m", errors="coerce")
    df[col_entry1] = pd.to_numeric(df[col_entry1], errors="coerce")
    df[col_entry2] = pd.to_numeric(df[col_entry2], errors="coerce")
    df = df.dropna()

    if len(df) < min_points:
        return pd.Series({f"{col_entry1}_slope": np.nan, f"{col_entry2}_slope": np.nan})

    x = ((df[col_date] - df[col_date].min()).dt.days.to_numpy(dtype=float)) / 365
    # Avoid fitting when all x values are identical
    if np.all(x == x[0]):
        return pd.Series({f"{col_entry1}_slope": np.nan, f"{col_entry2}_slope": np.nan})

    entry1_slope = np.polyfit(x, df[col_entry1].to_numpy(dtype=float), 1)[0]
    entry2_slope = np.polyfit(x, df[col_entry2].to_numpy(dtype=float), 1)[0]

    return pd.Series(
        {f"{col_entry1}_slope": entry1_slope, f"{col_entry2}_slope": entry2_slope}
    )
