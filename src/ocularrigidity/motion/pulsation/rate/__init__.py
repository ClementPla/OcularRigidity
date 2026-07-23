"""Stage 2 — rate estimators (optional).

``base.py`` holds the contract (:class:`RateEstimate`,
:class:`AbstractRateEstimator`); one module per method. Add a new estimator as a
new module here and export it below.
"""

from ocularrigidity.motion.pulsation.rate.base import (
    AbstractRateEstimator,
    RateEstimate,
)
from ocularrigidity.motion.pulsation.rate.fixed import FixedRateEstimator
from ocularrigidity.motion.pulsation.rate.lomb_scargle import (
    LombScargleConfig,
    LombScargleRateEstimator,
    lomb_scargle_power,
)

__all__ = [
    "RateEstimate",
    "AbstractRateEstimator",
    "LombScargleRateEstimator",
    "LombScargleConfig",
    "lomb_scargle_power",
    "FixedRateEstimator",
]
