"""Boundaries → the model's input map, shared by the training cache and inference.

One function, so the cache built from ``measure.pkl`` and the live
:class:`LearnedTraceSource` cannot drift apart.

The registered BM/CSI positions are integer pixels: a single A-scan moves by
about ±0.7 px, mostly quantisation. Averaging ``pool`` neighbouring columns
recovers sub-pixel motion and shrinks the map at the same time.
"""

import numpy as np
from scipy.interpolate import interp1d

N_COLS = 256


def pool_columns(boundaries: np.ndarray, n_cols: int = N_COLS) -> np.ndarray:
    """``(T, 2, W)`` → ``(T, 2, n_cols)`` by NaN-aware averaging of column blocks."""
    T, C, W = boundaries.shape
    pool = W // n_cols
    b = boundaries[..., : pool * n_cols].reshape(T, C, n_cols, pool)
    with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        return np.nanmean(b, axis=-1)


def boundaries_to_uniform(
    boundaries: np.ndarray,
    timestamps_seconds: np.ndarray,
    uniform_time: np.ndarray,
    gap_mask: np.ndarray,
    n_cols: int = N_COLS,
) -> np.ndarray:
    """Pooled BM/CSI deviations on the uniform grid, ``(T_uniform, 2, n_cols)``.

    Each column's temporal median is removed, which leaves small numbers that
    survive float16 storage (raw positions ~500 px would lose the sub-pixel
    part). Gap samples and holes are NaN.
    """
    pooled = pool_columns(np.asarray(boundaries, dtype=np.float32), n_cols)
    with np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        pooled = pooled - np.nanmedian(pooled, axis=0, keepdims=True)

    valid = ~np.isnan(pooled).all(axis=(1, 2))
    out = interp1d(
        timestamps_seconds[valid],
        pooled[valid],
        axis=0,
        kind="linear",
        fill_value=np.nan,
        bounds_error=False,
    )(uniform_time)
    out[gap_mask] = np.nan
    return out.astype(np.float32)
