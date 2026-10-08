"""Rate estimation: pick the cardiac frequency (and the most cardiac trace)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ocularrigidity.motion.pulsation.traces import Traces


@dataclass
class RateEstimate:
    freq: float  # Hz
    # Index of the trace judged most cardiac, if the estimator ranks them.
    best_index: Optional[int] = None
    # Per-trace quality, ≥ 0, for weighted aggregation.
    weights: Optional[np.ndarray] = None
    # Free-form estimator output (LS power spectrum, FAPs, …) for diagnostics.
    diagnostics: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    confidence: str = "unknown"

    @property
    def bpm(self) -> float:
        return self.freq * 60.0


class AbstractRateEstimator(ABC):
    """Turns candidate traces into a cardiac frequency."""

    @abstractmethod
    def estimate(self, traces: Traces) -> RateEstimate:
        """Score ``traces`` and return the cardiac frequency estimate."""
