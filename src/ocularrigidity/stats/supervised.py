"""Predict one cohort variable from several others"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor  # noqa: F401  (kept for callers)
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DERIVED_FROM = {
    "K": {"deltaCT", "AxialLength", "IOP", "OPA"},
    "K_Mask": {"deltaCT_Mask", "AxialLength", "IOP", "OPA"},
    "RelativeGrowth": {"deltaCT", "minCT"},
}


def circular_inputs(target: str, inputs: Sequence[str]) -> list[str]:
    """Inputs algebraically tied to ``target``"""
    bad = set(DERIVED_FROM.get(target, set())) & set(inputs)
    bad |= {c for c in inputs if target in DERIVED_FROM.get(c, set())}
    return sorted(bad)


VISIT = "visit"
RATE = "rate"
TARGET_MODES = (VISIT, RATE)

SPLITS = ("train", "val", "test")

# : Model registry.
MODELS: dict[str, dict] = {
    "Ridge (linear baseline)": {
        "build": lambda p: Ridge(alpha=p["alpha"]),
        "params": {"alpha": ("log", 1.0, 1e-3, 1e3)},
        "note": "Linear. Here to say whether anything non-linear is earning its keep.",
    },
    "Random forest": {
        "build": lambda p: RandomForestRegressor(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"] or None,
            min_samples_leaf=p["min_samples_leaf"],
            random_state=p["seed"],
            n_jobs=-1,
        ),
        "params": {
            "n_estimators": ("int", 300, 50, 1000),
            "max_depth": ("int", 0, 0, 20),
            "min_samples_leaf": ("int", 3, 1, 20),
        },
        "note": "Non-linear, handles interactions, hard to overfit badly. max_depth 0 = unlimited.",
    },
    "Gradient boosting": {
        "build": lambda p: HistGradientBoostingRegressor(
            max_iter=p["max_iter"],
            learning_rate=p["learning_rate"],
            max_depth=p["max_depth"] or None,
            min_samples_leaf=p["min_samples_leaf"],
            random_state=p["seed"],
        ),
        "params": {
            "max_iter": ("int", 200, 20, 1000),
            "learning_rate": ("log", 0.05, 1e-3, 1.0),
            "max_depth": ("int", 3, 0, 12),
            "min_samples_leaf": ("int", 5, 1, 40),
        },
        "note": "Usually the strongest on tabular data of this size.",
    },
    "MLP (neural net)": {
        "build": lambda p: MLPRegressor(
            hidden_layer_sizes=tuple(
                int(x) for x in str(p["hidden"]).split(",") if x.strip()
            )
            or (64,),
            alpha=p["alpha"],
            learning_rate_init=p["learning_rate_init"],
            max_iter=p["max_iter"],
            early_stopping=True,
            n_iter_no_change=20,
            random_state=p["seed"],
        ),
        "params": {
            "hidden": ("text", "64,32", None, None),
            "alpha": ("log", 1e-3, 1e-6, 1e2),
            "learning_rate_init": ("log", 1e-3, 1e-5, 1e-1),
            "max_iter": ("int", 800, 100, 5000),
        },
        "note": (
            "Scaling is mandatory and the pipeline does it. Carves its own early-"
            "stopping slice out of train, so the val fold below stays clean."
        ),
    },
}


# : Ways to fill a missing input.
IMPUTERS = {
    "median": lambda ind: SimpleImputer(strategy="median", add_indicator=ind),
    "mean": lambda ind: SimpleImputer(strategy="mean", add_indicator=ind),
    "knn": lambda ind: KNNImputer(n_neighbors=5, add_indicator=ind),
}


def make_pipeline(
    model_name: str,
    params: dict,
    *,
    impute: str | None = "median",
    add_indicator: bool = False,
) -> Pipeline:
    """Impute -> scale -> estimate."""
    steps = []
    if impute is not None:
        steps.append(("impute", IMPUTERS[impute](add_indicator)))
    steps.append(("scale", StandardScaler()))
    steps.append(("model", MODELS[model_name]["build"](params)))
    return Pipeline(steps)


def build_design(
    df: pd.DataFrame,
    inputs: Sequence[str],
    target: str,
    *,
    mode: str = VISIT,
    min_points: int = 3,
    group_col: str = "PatientId",
    eye_cols: Sequence[str] = ("PatientId", "Eye"),
    date_col: str = "YearMonth",
    drop_incomplete: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    """``(X, y, groups, info)`` for one target mode."""
    inputs = [c for c in dict.fromkeys(inputs) if c != target]
    info: dict = {"mode": mode, "inputs": list(inputs), "target": target}
    if not inputs or target not in df.columns:
        empty = pd.DataFrame(columns=list(inputs))
        return empty, pd.Series(dtype=float), pd.Series(dtype=object), info

    if mode == VISIT:
        cols = list(dict.fromkeys([*inputs, target, group_col]))
        d = df[[c for c in cols if c in df.columns]].copy()
        for c in [*inputs, target]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        info["n_before"] = len(d)
        d = d.dropna(subset=[target])
        if drop_incomplete:
            d = d.dropna(subset=list(inputs))
        X, y, groups = d[inputs], d[target], d[group_col]
    elif mode == RATE:
        rows = _rate_rows(df, inputs, target, list(eye_cols), date_col, min_points)
        info["n_before"] = df.groupby(list(eye_cols)).ngroups
        if len(rows):
            rows = rows.dropna(subset=["_y"])
            if drop_incomplete:
                rows = rows.dropna(subset=list(inputs))
        X = rows[inputs] if len(rows) else pd.DataFrame(columns=list(inputs))
        y = rows["_y"] if len(rows) else pd.Series(dtype=float)
        groups = rows[group_col] if len(rows) else pd.Series(dtype=object)
    else:
        raise ValueError(f"unknown mode {mode!r}; expected one of {TARGET_MODES}")

    info["n_rows"] = len(X)
    info["n_groups"] = int(groups.nunique()) if len(groups) else 0
    info["dropped"] = int(info["n_before"] - len(X))
    info["missing_per_input"] = {c: int(X[c].isna().sum()) for c in X.columns}
    info["n_imputed"] = int(X.isna().any(axis=1).sum())
    return X, y, groups, info


def _rate_rows(
    df: pd.DataFrame,
    inputs: Sequence[str],
    target: str,
    eye_cols: list[str],
    date_col: str,
    min_points: int,
) -> pd.DataFrame:
    """One row per eye: inputs at baseline, ``_y`` the target's per-year slope."""
    cols = list(dict.fromkeys([*eye_cols, date_col, *inputs, target]))
    d = df[[c for c in cols if c in df.columns]].copy()
    for c in [*inputs, target]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["_date"] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=["_date", target])
    if d.empty:
        return pd.DataFrame(
            columns=[*eye_cols, *inputs, "_y", "n_visits", "span_years"]
        )

    value_cols = [c for c in dict.fromkeys([*inputs, target]) if c in d.columns]
    d = d.groupby([*eye_cols, "_date"], as_index=False)[value_cols].mean()

    out = []
    for keys, g in d.groupby(eye_cols):
        g = g.sort_values("_date")
        if len(g) < min_points:
            continue
        t = (g["_date"] - g["_date"].iloc[0]).dt.days.to_numpy() / 365.25
        if np.all(t == t[0]):
            continue
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(eye_cols, keys))
        row["_y"] = float(np.polyfit(t, g[target].to_numpy(), 1)[0])
        for c in inputs:
            row[c] = g[c].iloc[0] if c in g.columns else np.nan
        row["n_visits"] = len(g)
        row["span_years"] = float(t.max())
        out.append(row)
    return pd.DataFrame(out)


def split_by_group(
    groups: pd.Series,
    *,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 0,
) -> pd.Series:
    """Label every row ``train`` / ``val`` / ``test``, splitting whole groups."""
    out = pd.Series("train", index=groups.index, dtype=object)
    if groups.empty:
        return out
    n_groups = groups.nunique()
    if n_groups < 3 or test_size <= 0:
        return out

    idx = np.arange(len(groups))
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    rest_i, test_i = next(gss.split(idx, groups=groups))
    out.iloc[test_i] = "test"

    if val_size > 0 and len(rest_i):
        rest_groups = groups.iloc[rest_i]
        if rest_groups.nunique() >= 2:
            # rescale: val_size is a share of everything, taken out of the rest
            frac = min(max(val_size / max(1e-9, 1.0 - test_size), 1e-9), 0.9)
            gss2 = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
            _, val_local = next(gss2.split(np.arange(len(rest_i)), groups=rest_groups))
            out.iloc[rest_i[val_local]] = "val"
    return out


def _scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) < 2:
        return {"n": int(len(y_true)), "r2": np.nan, "mae": np.nan, "rmse": np.nan}
    return {
        "n": int(len(y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
    }


def fit_evaluate(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    split: pd.Series,
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame]:
    """Fit on ``train``, score every fold, and return the per-row predictions."""
    tr = split == "train"
    if tr.sum() < 3:
        raise ValueError(f"only {int(tr.sum())} training rows; need at least 3")

    pipe.fit(X[tr], y[tr])
    dummy = DummyRegressor(strategy="mean").fit(X[tr], y[tr])

    pred = pd.DataFrame(
        {
            "y_true": y,
            "y_pred": pipe.predict(X),
            "y_dummy": dummy.predict(X),
            "split": split,
        },
        index=X.index,
    )
    pred["residual"] = pred["y_true"] - pred["y_pred"]

    rows = []
    for name in SPLITS:
        m = split == name
        if not m.any():
            continue
        rows.append(
            {"split": name, "model": "fitted", **_scores(y[m], pred.loc[m, "y_pred"])}
        )
        rows.append(
            {"split": name, "model": "dummy", **_scores(y[m], pred.loc[m, "y_dummy"])}
        )
    return pipe, pd.DataFrame(rows), pred


def importances(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    split: pd.Series,
    *,
    on: str = "test",
    n_repeats: int = 20,
    seed: int = 0,
) -> pd.DataFrame:
    """Permutation importance on a held-out fold, as a drop in R^2."""
    m = split == on
    if m.sum() < 5:
        return pd.DataFrame(columns=["feature", "importance", "std"])
    r = permutation_importance(
        pipe, X[m], y[m], n_repeats=n_repeats, random_state=seed, scoring="r2"
    )
    return (
        pd.DataFrame(
            {
                "feature": list(X.columns),
                "importance": r.importances_mean,
                "std": r.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
