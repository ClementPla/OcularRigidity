import numpy as np

from ocularrigidity.motion.pulsation.phase.base import AbstractPhaseEstimator


class LinearPhaseEstimator(AbstractPhaseEstimator):
    """Estimates phase as a linear function of time."""

    def phase_from_trace(self, trace, traces, rate):
        t = traces.uniform_time
        f0 = rate.freq

        phase = np.mod(2 * np.pi * f0 * t, 2 * np.pi)
        good = np.ones_like(phase, dtype=bool)
        return phase, good
