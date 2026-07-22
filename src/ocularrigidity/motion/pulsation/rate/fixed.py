from ocularrigidity.motion.pulsation.rate.base import (
    AbstractRateEstimator,
    RateEstimate,
)
from ocularrigidity.motion.pulsation.traces import Traces


class FixedRateEstimator(AbstractRateEstimator):
    def __init__(self, bpm: float, index: int = 0):
        self.bpm = float(bpm)
        self.index = index

    def estimate(self, traces: Traces) -> RateEstimate:
        return RateEstimate(
            freq=self.bpm / 60.0,
            best_index=self.index,
            notes=[f"Cardiac rate fixed at {self.bpm:.1f} bpm."],
            confidence="unknown",
        )
