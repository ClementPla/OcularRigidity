"""Stage 2"""

from ocularrigidity.motion.pulsation.rate.base import (
    AbstractRateEstimator,
    RateEstimate,
)
from ocularrigidity.motion.pulsation.rate.fixed import FixedRateEstimator
from ocularrigidity.motion.pulsation.rate.lomb_scargle import (
    LombScargleConfig,
    LombScargleRateEstimator,
)

__all__ = [
    "RateEstimate",
    "AbstractRateEstimator",
    "LombScargleRateEstimator",
    "LombScargleConfig",
    "FixedRateEstimator",
]
