"""Aggregators: collapse ``(T_kept, K)`` candidate traces into one signal.

Kept as its own axis rather than folded into the phase estimators: otherwise
every (phase method × aggregation) pair would need its own class. A phase
estimator holds an aggregator and stays agnostic about how the traces were
reduced.

All aggregators return the aggregated trace **on the full uniform grid**, NaN
where the sample was not kept, so phase code can reason about gaps directly.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ocularrigidity.motion.pulsation.rate import RateEstimate
from ocularrigidity.motion.pulsation.traces import Traces


def _standardized(values: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance per trace, so no trace dominates by scale."""
    mu = values.mean(axis=0, keepdims=True)
    sd = values.std(axis=0, keepdims=True)
    return (values - mu) / np.where(sd > 0, sd, 1.0)


class AbstractTraceAggregator(ABC):
    """Reduces the candidate traces to the single signal phase is read from."""

    @abstractmethod
    def aggregate(self, traces: Traces, rate: Optional[RateEstimate]) -> np.ndarray:
        """Return a ``(T_uniform,)`` trace, NaN outside ``traces.kept_mask``."""


class SelectBestComponent(AbstractTraceAggregator):
    """Keep only the trace the rate estimator judged most cardiac.

    This is what the original pipeline did. Requires a rate estimate that
    ranks traces (``best_index``).
    """

    def aggregate(self, traces: Traces, rate: Optional[RateEstimate]) -> np.ndarray:
        if rate is None or rate.best_index is None:
            raise ValueError(
                "SelectBestComponent needs a RateEstimate with a best_index. "
                "Use a ranking rate estimator (e.g. LombScargleRateEstimator), "
                "or switch to MeanTrace/SingleTrace aggregation."
            )
        return traces.full(rate.best_index)


class SingleTrace(AbstractTraceAggregator):
    """Keep one fixed trace, by index. Useful when K == 1 or for debugging."""

    def __init__(self, index: int = 0):
        self.index = index

    def aggregate(self, traces: Traces, rate: Optional[RateEstimate]) -> np.ndarray:
        return traces.full(self.index)


class MeanTrace(AbstractTraceAggregator):
    """Plain average across traces. Needs no rate estimate."""

    def __init__(self, standardize: bool = True):
        self.standardize = standardize

    def aggregate(self, traces: Traces, rate: Optional[RateEstimate]) -> np.ndarray:
        values = _standardized(traces.values) if self.standardize else traces.values
        return traces.embed(values.mean(axis=1))


class PowerWeightedMean(AbstractTraceAggregator):
    """Average across traces, weighted by the rate estimator's quality scores.

    A softer :class:`SelectBestComponent`: instead of betting everything on the
    top-scoring trace, it keeps the runners-up in proportion to their score.
    ``power`` sharpens (``> 1``) or flattens (``< 1``) the weighting.
    """

    def __init__(self, standardize: bool = True, power: float = 1.0):
        self.standardize = standardize
        self.power = power

    def aggregate(self, traces: Traces, rate: Optional[RateEstimate]) -> np.ndarray:
        if rate is None or rate.weights is None:
            raise ValueError(
                "PowerWeightedMean needs a RateEstimate carrying per-trace weights."
            )
        w = np.asarray(rate.weights, dtype=float) ** self.power
        total = w.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Rate estimate weights are all zero; cannot aggregate.")
        w = w / total
        values = _standardized(traces.values) if self.standardize else traces.values
        return traces.embed(values @ w)
