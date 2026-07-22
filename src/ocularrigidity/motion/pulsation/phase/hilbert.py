"""Analytic-signal (Hilbert) phase — needs no carrier frequency."""

from dataclasses import dataclass
from typing import ClassVar, Literal, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert, medfilt

from ocularrigidity.motion.pulsation.phase.aggregation import AbstractTraceAggregator
from ocularrigidity.motion.pulsation.phase.base import AbstractPhaseEstimator


@dataclass
class HilbertPhaseConfig:
    # Fraction of a nominal cycle used to smooth the analytic phase. 0 disables.
    smoother_cycles: float = 0.0


@dataclass
class AmplitudeWeightedHilbertConfig:
    # How the per-trace analytic phases become one phase. "integrated_frequency"
    # rebuilds phase from the weighted-median instantaneous frequency (immune to
    # genuine phase lags across the scan); "circular_mean" averages the phases
    # directly (keeps beat shape, smears any lag).
    combine: Literal["integrated_frequency", "circular_mean"] = "integrated_frequency"
    # Median filter applied to the instantaneous frequency, in nominal cycles.
    freq_median_cycles: float = 1.0
    # Fraction of the record trimmed at each end, where filtfilt and the
    # analytic signal have edge transients.
    edge_frac: float = 0.05


class HilbertPhaseEstimator(AbstractPhaseEstimator):
    """Analytic-signal phase — no carrier frequency needed.

    The traces are already bandpassed to the cardiac band by the trace source,
    so the Hilbert transform's narrowband assumption holds and the instantaneous
    phase is meaningful. Gaps are bridged by linear interpolation before the
    transform (which is global and cannot tolerate NaNs) and masked out again
    afterwards.
    """

    requires_rate: ClassVar[bool] = False

    def __init__(
        self,
        config: Optional[HilbertPhaseConfig] = None,
        aggregator: Optional[AbstractTraceAggregator] = None,
        per_trace: bool = False,
    ):
        super().__init__(aggregator, per_trace)
        self.config = config or HilbertPhaseConfig()

    def phase_from_trace(self, trace, traces, rate):
        cfg = self.config
        n = len(traces.uniform_time)
        x = np.asarray(trace, dtype=float)
        valid = ~np.isnan(x)
        if not valid.any():
            return np.full(n, np.nan), np.zeros(n, dtype=bool)

        idx = np.arange(n)
        filled = x.copy()
        filled[~valid] = np.interp(idx[~valid], idx[valid], x[valid])
        filled = filled - filled.mean()

        analytic = hilbert(filled)
        phase = np.unwrap(np.angle(analytic))

        if cfg.smoother_cycles > 0:
            f0 = rate.freq if rate is not None else None
            if f0 is None or not np.isfinite(f0) or f0 <= 0:
                # Fall back on the mean slope of the unwrapped phase.
                f0 = max(
                    np.gradient(phase, traces.uniform_time).mean() / (2 * np.pi), 1e-6
                )
            sigma_t = (cfg.smoother_cycles / f0) / traces.dt
            phase = gaussian_filter1d(phase, sigma=sigma_t, mode="nearest")

        good = valid & ~traces.gap_mask
        return np.mod(phase, 2 * np.pi), good


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Value at which the cumulative weight crosses half the total."""
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not ok.any():
        return np.nan
    v, w = values[ok], weights[ok]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cdf = np.cumsum(w) / w.sum()
    return float(v[np.searchsorted(cdf, 0.5)])


class AmplitudeWeightedHilbertPhaseEstimator(AbstractPhaseEstimator):
    """Per-trace Hilbert, combined by envelope-weighted instantaneous frequency.

    Each trace gets its own analytic signal; the instantaneous frequencies are
    then merged across traces with a *time-varying* weight — each trace counts
    in proportion to its envelope amplitude at that instant, so A-scans where
    the pulsation is momentarily weak stop dragging the estimate around.

    This deliberately overrides :meth:`estimate` rather than using the
    aggregate-before / aggregate-after hooks: the weights vary along time as
    well as across traces, and the merge happens in *frequency* space, neither
    of which the ``AbstractTraceAggregator`` contract expresses. That is the
    escape hatch — implement ``estimate`` and call ``build_track``.

    ``inst_freq`` holds the merged frequency trace after the run, for plotting.
    """

    requires_rate: ClassVar[bool] = False

    def __init__(
        self,
        config: Optional[AmplitudeWeightedHilbertConfig] = None,
        aggregator: Optional[AbstractTraceAggregator] = None,
    ):
        super().__init__(aggregator, per_trace=True)
        self.config = config or AmplitudeWeightedHilbertConfig()
        #: Envelope-weighted instantaneous frequency (Hz) on the uniform grid.
        self.inst_freq: Optional[np.ndarray] = None
        #: Per-trace envelope amplitude, shape (T_uniform, K).
        self.envelope: Optional[np.ndarray] = None

    def phase_from_trace(self, trace, traces, rate):
        """Unwrapped analytic phase of one trace (kept for the base contract)."""
        phase, _, good = self._analytic(trace, traces)
        return np.mod(phase, 2 * np.pi), good

    @staticmethod
    def _analytic(trace, traces):
        """(unwrapped phase, envelope, good) of one full-grid trace."""
        n = len(traces.uniform_time)
        x = np.asarray(trace, dtype=float)
        valid = ~np.isnan(x)
        if not valid.any():
            return np.full(n, np.nan), np.zeros(n), np.zeros(n, dtype=bool)
        idx = np.arange(n)
        filled = x.copy()
        filled[~valid] = np.interp(idx[~valid], idx[valid], x[valid])
        z = hilbert(filled - filled.mean())
        return np.unwrap(np.angle(z)), np.abs(z), valid & ~traces.gap_mask

    def estimate(self, traces, rate=None):
        cfg = self.config
        t = traces.uniform_time
        n, K = len(t), traces.n_traces

        PHI = np.empty((n, K))
        A = np.empty((n, K))
        G = np.empty((n, K), dtype=bool)
        for k in range(K):
            PHI[:, k], A[:, k], G[:, k] = self._analytic(traces.full(k), traces)

        F = np.gradient(PHI, t, axis=0) / (2 * np.pi)
        weights = np.where(G, A, 0.0)
        f_inst = np.array([_weighted_median(F[i], weights[i]) for i in range(n)])

        # Bridge and smooth the frequency trace before it is trusted.
        finite = np.isfinite(f_inst)
        if finite.sum() < 2:
            self.notes.append("Instantaneous frequency could not be estimated.")
            return self.build_track(np.full(n, np.nan), np.zeros(n, bool), traces, rate)
        f_inst = np.interp(np.arange(n), np.flatnonzero(finite), f_inst[finite])

        f_nominal = rate.freq if rate is not None else float(np.median(f_inst))
        if cfg.freq_median_cycles > 0 and f_nominal > 0:
            win = int(round(cfg.freq_median_cycles * traces.fs / f_nominal)) | 1
            if 1 < win < n:
                f_inst = medfilt(f_inst, win)

        good = G.any(axis=1) & ~traces.gap_mask
        m = int(cfg.edge_frac * n)
        if m > 0:
            good[:m] = False
            good[-m:] = False

        if cfg.combine == "integrated_frequency":
            # φ(t) = 2π ∫ f dt — consistent with f_inst by construction, and
            # unaffected by real phase lags between A-scans.
            phase = (
                2
                * np.pi
                * np.concatenate(
                    [[0.0], np.cumsum(0.5 * (f_inst[1:] + f_inst[:-1]) * np.diff(t))]
                )
            )
        else:
            acc = (np.where(G, np.exp(1j * PHI), 0.0) * weights).sum(axis=1)
            good &= np.abs(acc) > 0
            phase = np.angle(acc)

        self.inst_freq = f_inst
        self.envelope = A
        return self.build_track(np.mod(phase, 2 * np.pi), good, traces, rate)
