"""Beat-by-beat phase, locked to detected systolic peaks."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import skew

from ocularrigidity.motion.pulsation.band import CardiacBand
from ocularrigidity.motion.pulsation.phase.aggregation import AbstractTraceAggregator
from ocularrigidity.motion.pulsation.phase.base import AbstractPhaseEstimator
from ocularrigidity.motion.pulsation.rate import RateEstimate


@dataclass
class PeakLockConfig:
    # Reject peaks closer together than this fraction of the nominal period.
    min_period_frac: float = 0.7
    prominence_frac: float = 0.4
    # Keep only inter-peak intervals within this fraction of the median.
    cycle_min_frac: float = 0.7
    cycle_max_frac: float = 1.4
    # Drop a cycle entirely if more than this fraction of it falls in a gap.
    max_gap_frac: float = 0.5


class PeakLockedPhaseEstimator(AbstractPhaseEstimator):
    """Phase ramps linearly from 0 at each detected systolic peak.

    Unlike demodulation this makes no constant-rate assumption: every beat gets
    its own 0→2π ramp, so beat-to-beat variability is preserved rather than
    smoothed away. A ``RateEstimate`` is used only to size the peak-rejection
    window; without one, the top of ``band`` is used (the most permissive
    choice).
    """

    def __init__(
        self,
        config: PeakLockConfig | None = None,
        aggregator: AbstractTraceAggregator | None = None,
        band: CardiacBand | None = None,
    ):
        super().__init__(aggregator, per_trace=False)
        self.config = config or PeakLockConfig()
        self.band = band or CardiacBand()

    def _nominal_freq(self, rate: RateEstimate | None) -> float:
        if rate is not None and np.isfinite(rate.freq):
            return rate.freq
        return self.band.effective_hz_range[1]

    def phase_from_trace(self, trace, traces, rate):
        cfg = self.config
        f0 = self._nominal_freq(rate)
        fs = traces.fs
        gap_mask = traces.gap_mask
        n = len(traces.uniform_time)

        x = np.asarray(trace, dtype=float).copy()
        valid = ~np.isnan(x)
        if not valid.any():
            return np.full(n, np.nan), np.zeros(n, dtype=bool)

        # Systole should be the sharp excursion: if the trace is left-skewed the
        # sharp feature points down, so flip it before peak finding.
        if skew(x[valid]) < 0:
            x = -x

        min_dist = max(1, int(cfg.min_period_frac / f0 * fs))
        prom = cfg.prominence_frac * np.nanstd(x)
        peaks, _ = find_peaks(np.nan_to_num(x), distance=min_dist, prominence=prom)

        phase = np.full(n, np.nan)
        good = np.zeros(n, dtype=bool)
        if len(peaks) < 2:
            self.notes.append("Fewer than two peaks detected; no phase locked.")
            return phase, good

        durations = np.diff(peaks)
        med = np.median(durations)
        ok_cycle = (durations > cfg.cycle_min_frac * med) & (
            durations < cfg.cycle_max_frac * med
        )

        for i, ok in enumerate(ok_cycle):
            if not ok:
                continue
            p0, p1 = peaks[i], peaks[i + 1]
            if gap_mask[p0:p1].mean() > cfg.max_gap_frac:
                continue  # cycle is mostly gap — drop entirely
            phase[p0:p1] = np.linspace(0.0, 2 * np.pi, p1 - p0, endpoint=False)
            good[p0:p1] = ~gap_mask[p0:p1]  # keep non-gap samples only

        return phase, good
