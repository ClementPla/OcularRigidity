from typing import ClassVar, Optional

import numpy as np

from ocularrigidity.motion.pulsation.phase.base import AbstractPhaseEstimator
from ocularrigidity.motion.pulsation.traces.base import Traces


class LinearPhaseEstimator(AbstractPhaseEstimator):
    """Estimates phase as a linear function of time.

    The ramp starts at the first uniform sample; one of three mutually
    exclusive anchors shifts it:

    ``initial_phase``
        Phase of that first sample, in cycles. The only anchor that does not
        involve ``f0``, so it can be fixed *before* the rate is estimated.
    ``offset_time``
        Phase 0 falls this many seconds after the first sample.
    ``offset_in_frame``
        The same, counted in frames (converted with the sampling rate).
    """

    requires_rate: ClassVar[bool] = True

    #: (which anchor is set, its value), or None for a bare ramp.
    _anchor: Optional[tuple[str, float]] = None

    def phase_from_trace(self, trace, traces: Traces, rate):
        t = traces.uniform_time
        f0 = rate.freq

        phase = np.mod(
            2 * np.pi * f0 * (t - t[0]) - self.offset(f0, traces.fs), 2 * np.pi
        )
        good = np.ones_like(phase, dtype=bool)
        return phase, good

    def offset(self, frequency, sampling_rate) -> float:
        """The constant term subtracted from the ramp, in radians."""
        if self._anchor is None:
            return 0.0
        kind, value = self._anchor
        if kind == "initial_phase":
            # A phase is already an angle: scaling by 2π is the whole
            # conversion, and no f0 enters. That is what lets this anchor be
            # set before the rate is known.
            return -2 * np.pi * value
        if kind == "offset_time":
            return 2 * np.pi * value * frequency
        return 2 * np.pi * value * frequency / sampling_rate

    # -- anchors --------------------------------------------------------
    def _get_anchor(self, kind: str) -> Optional[float]:
        if self._anchor is not None and self._anchor[0] == kind:
            return self._anchor[1]
        return None

    def _set_anchor(self, kind: str, value: Optional[float]) -> None:
        if value is None:
            if self._anchor is not None and self._anchor[0] == kind:
                self._anchor = None
            return
        if self._anchor is not None and self._anchor[0] != kind:
            raise ValueError(
                f"Cannot set {kind}: {self._anchor[0]} is already set to "
                f"{self._anchor[1]!r}. Set that one to None first."
            )
        self._anchor = (kind, float(value))

    @property
    def initial_phase(self) -> Optional[float]:
        """Phase of the first sample, in cycles."""
        return self._get_anchor("initial_phase")

    @initial_phase.setter
    def initial_phase(self, value: Optional[float]) -> None:
        self._set_anchor("initial_phase", value)

    @property
    def offset_time(self) -> Optional[float]:
        """Seconds after the first sample at which phase 0 falls."""
        return self._get_anchor("offset_time")

    @offset_time.setter
    def offset_time(self, value: Optional[float]) -> None:
        self._set_anchor("offset_time", value)

    @property
    def offset_in_frame(self) -> Optional[float]:
        """Frames after the first sample at which phase 0 falls."""
        return self._get_anchor("offset_in_frame")

    @offset_in_frame.setter
    def offset_in_frame(self, value: Optional[float]) -> None:
        self._set_anchor("offset_in_frame", value)
