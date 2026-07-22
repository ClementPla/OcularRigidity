"""Rate estimation: pick the cardiac frequency (and the most cardiac trace).

A :class:`AbstractRateEstimator` scores the candidate traces and returns a
:class:`RateEstimate` holding the frequency, which trace carried it, and
per-trace weights that aggregators can reuse.

The rate stage is **optional**. Phase estimators that recover frequency on their
own (Hilbert, peak locking) run happily with ``rate=None``; those that need a
carrier (IQ demodulation) declare ``requires_rate = True`` and the extractor
fails fast with a clear message rather than deep inside the maths.

**Adding an estimator:** subclass :class:`AbstractRateEstimator` and implement
``estimate`` (see ``lomb_scargle.py`` for a full one, ``fixed.py`` for the
minimal one).
"""

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
    # Per-trace quality, ≥ 0, for weighted aggregation. None if unranked.
    weights: Optional[np.ndarray] = None
    # Free-form estimator output (LS power spectrum, FAPs, …) for diagnostics.
    diagnostics: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    confidence: str = "unknown"

    @property
    def bpm(self) -> float:
        return self.freq * 60.0


class AbstractRateEstimator(ABC):
    """Turns candidate traces into a cardiac frequency.

    Implement :meth:`estimate`. Ranking the traces is optional but recommended:
    filling ``best_index``/``weights`` is what lets the
    ``SelectBestComponent`` and ``PowerWeightedMean`` aggregators work.
    """

    @abstractmethod
    def estimate(self, traces: Traces) -> RateEstimate:
        """Score ``traces`` and return the cardiac frequency estimate."""
