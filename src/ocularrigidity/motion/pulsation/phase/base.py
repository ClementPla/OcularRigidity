"""Phase estimation: from an aggregated trace to a cardiac phase per frame.

An :class:`AbstractPhaseEstimator` owns two decisions:

1. *How* to reduce the candidate traces — delegated to an
   :class:`AbstractTraceAggregator` (aggregate-before, the default), or, if
   ``per_trace=True``, phase is computed on every trace and the resulting phases
   are combined circularly (aggregate-after).
2. How to read phase out of a trace — the one abstract method,
   :meth:`AbstractPhaseEstimator.phase_from_trace`.

Everything else (embedding on the uniform grid, resampling onto the original
frame timestamps, deriving the rate from the phase when no ``RateEstimate`` was
supplied) is shared here.

**Adding an estimator:** subclass :class:`AbstractPhaseEstimator`, implement
``phase_from_trace``, and set ``requires_rate`` if you need a carrier frequency.
See ``demodulation.py``, ``peak_locking.py``, ``hilbert.py``.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Optional

import numpy as np

from ocularrigidity.motion.pulsation.phase.aggregation import (
    AbstractTraceAggregator,
    MeanTrace,
)
from ocularrigidity.motion.pulsation.rate import RateEstimate
from ocularrigidity.motion.pulsation.traces import Traces


@dataclass
class PhaseTrack:
    """Cardiac phase on both time bases, plus the rate it implies."""

    phase_uniform: np.ndarray  # radians, [0, 2π), NaN where not estimated
    good_uniform: np.ndarray  # bool
    phase_per_frame: np.ndarray  # fraction, [0, 1)
    good_per_frame: np.ndarray  # bool
    freq: float  # Hz
    uniform_time: np.ndarray

    @property
    def bpm(self) -> float:
        return self.freq * 60.0

    @property
    def inst_bpm(self) -> np.ndarray:
        """Instantaneous BPM trace on the uniform grid (diagnostic)."""
        out = np.full(len(self.uniform_time), np.nan)
        in_run = self.good_uniform.astype(int)
        starts = np.where(np.diff(np.r_[0, in_run]) == 1)[0]
        ends = np.where(np.diff(np.r_[in_run, 0]) == -1)[0] + 1
        for s, e in zip(starts, ends):
            if e - s > 2:
                unw = np.unwrap(self.phase_uniform[s:e])
                out[s:e] = np.gradient(unw, self.uniform_time[s:e]) / (2 * np.pi) * 60.0
        return out


class AbstractPhaseEstimator(ABC):
    """Turns candidate traces into a per-frame cardiac phase.

    Subclasses implement :meth:`phase_from_trace` and set ``requires_rate`` if
    they need a carrier frequency to work at all.
    """

    #: Whether a RateEstimate is mandatory (IQ demodulation) or merely helpful.
    requires_rate: ClassVar[bool] = False

    def __init__(
        self,
        aggregator: Optional[AbstractTraceAggregator] = None,
        per_trace: bool = False,
    ):
        self.aggregator = aggregator or MeanTrace()
        self.per_trace = per_trace
        self.notes: list[str] = []

    # -- subclass hook --------------------------------------------------
    @abstractmethod
    def phase_from_trace(
        self,
        trace: np.ndarray,
        traces: Traces,
        rate: Optional[RateEstimate],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(phase_uniform, good_uniform)`` for one full-grid trace.

        ``trace`` is on the full uniform grid with NaN outside the kept samples;
        ``phase_uniform`` is in radians and need only be meaningful where the
        returned ``good_uniform`` is True.
        """

    # -- template -------------------------------------------------------
    def estimate(
        self, traces: Traces, rate: Optional[RateEstimate] = None
    ) -> PhaseTrack:
        if self.requires_rate and rate is None:
            raise ValueError(
                f"{type(self).__name__} requires a RateEstimate (it demodulates at "
                "a carrier frequency). Supply a rate estimator, or use a "
                "self-contained estimator such as HilbertPhaseEstimator."
            )

        if self.per_trace:
            phases, goods = [], []
            for k in range(traces.n_traces):
                p, g = self.phase_from_trace(traces.full(k), traces, rate)
                phases.append(p)
                goods.append(g)
            weights = None if rate is None else rate.weights
            phase_u, good_u = self.aggregate_phases(phases, goods, weights)
        else:
            trace = self.aggregator.aggregate(traces, rate)
            phase_u, good_u = self.phase_from_trace(trace, traces, rate)

        return self.build_track(phase_u, good_u, traces, rate)

    # -- shared helpers -------------------------------------------------
    def aggregate_phases(
        self,
        phases: list[np.ndarray],
        goods: list[np.ndarray],
        weights: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Circular (vector) mean of per-trace phases — the aggregate-after hook.

        Phases live on a circle, so they are averaged as unit vectors; a plain
        arithmetic mean would break across the 0/2π wrap. A sample is good if
        any contributing trace was good there.
        """
        P = np.stack(phases)  # (K, T)
        G = np.stack(goods)  # (K, T)
        w = np.ones(len(phases)) if weights is None else np.asarray(weights, float)
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)[:, None]
        if w.sum() == 0:
            w = np.ones_like(w)
        z = np.where(G, np.exp(1j * np.nan_to_num(P)), 0.0) * w
        acc = z.sum(axis=0)
        good = G.any(axis=0) & (np.abs(acc) > 0)
        phase = np.where(good, np.mod(np.angle(acc), 2 * np.pi), np.nan)
        return phase, good

    def build_track(
        self,
        phase_u: np.ndarray,
        good_u: np.ndarray,
        traces: Traces,
        rate: Optional[RateEstimate] = None,
    ) -> PhaseTrack:
        """Package a uniform-grid phase as a :class:`PhaseTrack`.

        Public because a subclass whose aggregation does not fit the
        ``estimate`` template is expected to override ``estimate`` outright and
        call this to get the per-frame resampling and rate derivation for free.
        """
        t = traces.uniform_time
        ts = traces.timestamps_seconds
        freq = rate.freq if rate is not None else self.derive_freq(phase_u, good_u, t)

        gi = np.where(good_u)[0]
        if len(gi) == 0:
            self.notes.append("No reliable phase could be estimated.")
            return PhaseTrack(
                phase_uniform=phase_u,
                good_uniform=good_u,
                phase_per_frame=np.full(len(ts), np.nan),
                good_per_frame=np.zeros(len(ts), dtype=bool),
                freq=freq,
                uniform_time=t,
            )

        # Resample on the unit circle, so interpolation does not cut across the
        # 0/2π wrap and invent a mid-cycle phase.
        z = np.exp(1j * phase_u[gi])
        zr = np.interp(ts, t[gi], z.real)
        zi = np.interp(ts, t[gi], z.imag)
        phase_per_frame = (np.angle(zr + 1j * zi) / (2 * np.pi)) % 1.0
        good_per_frame = np.interp(ts, t, good_u.astype(float)) > 0.5

        return PhaseTrack(
            phase_uniform=phase_u,
            good_uniform=good_u,
            phase_per_frame=phase_per_frame,
            good_per_frame=good_per_frame,
            freq=freq,
            uniform_time=t,
        )

    @staticmethod
    def derive_freq(
        phase_u: np.ndarray, good_u: np.ndarray, uniform_time: np.ndarray
    ) -> float:
        """Rate implied by the phase itself: median dφ/dt over good runs.

        This is what makes the rate stage optional — an estimator that locks
        onto the beat without being told the frequency still reports one.
        """
        in_run = good_u.astype(int)
        starts = np.where(np.diff(np.r_[0, in_run]) == 1)[0]
        ends = np.where(np.diff(np.r_[in_run, 0]) == -1)[0] + 1
        rates = []
        for s, e in zip(starts, ends):
            if e - s > 2:
                unw = np.unwrap(phase_u[s:e])
                rates.append(np.gradient(unw, uniform_time[s:e]) / (2 * np.pi))
        if not rates:
            return float("nan")
        return float(np.median(np.concatenate(rates)))
