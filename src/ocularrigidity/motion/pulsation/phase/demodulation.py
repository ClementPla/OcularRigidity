"""IQ (quadrature) demodulation at the cardiac carrier frequency."""

from dataclasses import dataclass
from typing import ClassVar, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ocularrigidity.motion.pulsation.phase.aggregation import AbstractTraceAggregator
from ocularrigidity.motion.pulsation.phase.base import AbstractPhaseEstimator


@dataclass
class IQPhaseConfig:
    smoother_cycles: float = 2.0
    density_threshold: float = 0.5
    freq_tolerance: float = 0.2


class IQDemodPhaseEstimator(AbstractPhaseEstimator):
    """Quadrature demodulation at the cardiac frequency.

    Mixing the trace down by ``exp(-2iπ f0 t)`` puts the cardiac component at DC,
    where a low-pass of a couple of cycles isolates it; the residual envelope
    angle is the slow phase drift around the nominal beat. Needs ``f0``, hence
    ``requires_rate``.
    """

    requires_rate: ClassVar[bool] = True

    def __init__(
        self,
        config: IQPhaseConfig | None = None,
        aggregator: AbstractTraceAggregator | None = None,
        per_trace: bool = False,
    ):
        super().__init__(aggregator, per_trace)
        self.config = config or IQPhaseConfig()

    def phase_from_trace(self, trace, traces, rate):
        cfg = self.config
        f0 = rate.freq
        t = traces.uniform_time
        dt = traces.dt

        phasor = np.exp(-1j * 2 * np.pi * f0 * t)
        z = trace * phasor  # NaNs propagate

        sigma_t = (cfg.smoother_cycles / f0) / dt
        nan = np.isnan(trace)
        valid = (~nan).astype(np.float32)
        I_filled = np.where(nan, 0.0, z.real)
        Q_filled = np.where(nan, 0.0, z.imag)

        # Normalized (NaN-aware) convolution: divide the smoothed signal by the
        # smoothed validity so gaps dilute rather than bias the estimate.
        I_num = gaussian_filter1d(I_filled, sigma=sigma_t, mode="nearest")
        Q_num = gaussian_filter1d(Q_filled, sigma=sigma_t, mode="nearest")
        den = gaussian_filter1d(valid, sigma=sigma_t, mode="nearest")
        good = den > cfg.density_threshold
        safe_den = np.where(good, den, 1.0)
        I_lp = np.where(good, I_num / safe_den, 0.0)
        Q_lp = np.where(good, Q_num / safe_den, 0.0)

        env_phase = np.arctan2(Q_lp, I_lp)
        phase = np.mod(2 * np.pi * f0 * t + env_phase, 2 * np.pi)

        # Estimate the instantaneous frequency from the unwrapped phase
        unwrapped_phase = np.unwrap(phase)
        inst_freq = np.gradient(unwrapped_phase, dt) / (2 * np.pi)
        # Remove from good the points where the instantaneous frequency is not within a reasonable range of the expected frequency
        freq_tolerance = cfg.freq_tolerance * f0  # 20% tolerance
        good = good & (np.abs(inst_freq - f0) < freq_tolerance)
        return phase, good
