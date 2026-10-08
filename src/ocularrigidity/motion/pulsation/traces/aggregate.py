"""Collapse a source's traces to a single trace, as a trace-source *wrapper*."""

from dataclasses import dataclass, replace
from typing import Literal, Optional

import numpy as np
from scipy.stats import trim_mean

from ocularrigidity.motion.pulsation.traces.base import AbstractTraceSource, Traces

AggregateMethod = Literal["mean", "median", "trimmed_mean", "max", "min", "range"]


@dataclass
class AggregateConfig:
    """How the ``(T_kept, K)`` traces are collapsed to ``(T_kept, 1)``."""

    method: AggregateMethod = "mean"
    standardize: bool = True
    # Fraction cut from *each* tail when ``method="trimmed_mean"``.
    trim: float = 0.1
    verbose: bool = True


def _standardized(values: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance per trace, so no trace dominates by scale."""
    mu = values.mean(axis=0, keepdims=True)
    sd = values.std(axis=0, keepdims=True)
    return (values - mu) / np.where(sd > 0, sd, 1.0)


def _reduce(values: np.ndarray, cfg: AggregateConfig) -> np.ndarray:
    """``(T_kept, K)`` -> ``(T_kept,)`` under ``cfg.method``."""
    if cfg.method == "mean":
        return values.mean(axis=1)
    if cfg.method == "median":
        return np.median(values, axis=1)
    if cfg.method == "trimmed_mean":
        if not 0.0 <= cfg.trim < 0.5:
            raise ValueError(f"trim must be in [0, 0.5); got {cfg.trim}.")
        return trim_mean(values, cfg.trim, axis=1)
    if cfg.method == "max":
        return values.max(axis=1)
    if cfg.method == "min":
        return values.min(axis=1)
    if cfg.method == "range":
        return values.max(axis=1) - values.min(axis=1)
    raise ValueError(
        f"Unknown aggregation method {cfg.method!r}; expected one of "
        f"mean, median, trimmed_mean, max, min, range."
    )


class AggregateTraceSource(AbstractTraceSource):
    """Wraps a source and returns its traces collapsed into one."""

    def __init__(
        self,
        source: AbstractTraceSource,
        config: Optional[AggregateConfig] = None,
    ):
        super().__init__()
        self.source = source
        self.config = config or AggregateConfig()

    def compute(self) -> Traces:
        cfg = self.config
        base = self.source.traces
        values = _standardized(base.values) if cfg.standardize else base.values

        if cfg.verbose:
            detail = f" (trim={cfg.trim})" if cfg.method == "trimmed_mean" else ""
            print(
                f"AggregateTraceSource: {base.n_traces} traces -> 1 "
                f"by {cfg.method}{detail}"
            )

        return replace(
            base,
            values=_reduce(values, cfg)[:, None],
            mixing=None,
            source_map=None,
        )
