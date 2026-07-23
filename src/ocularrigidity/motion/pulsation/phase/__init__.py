"""Stage 3 — phase estimators, and the aggregators they reduce traces with.

``base.py`` holds the contract (:class:`PhaseTrack`,
:class:`AbstractPhaseEstimator`) and ``aggregation.py`` the orthogonal
"how do K traces become one" axis. One module per phase method. Add a new
estimator (or a new aggregator, in ``aggregation.py``) and export it below.
"""

from ocularrigidity.motion.pulsation.phase.aggregation import (
    AbstractTraceAggregator,
    MeanTrace,
    OptimizedSpectralCombination,
    PowerWeightedMean,
    SelectBestComponent,
    SingleTrace,
    SpectralCombinationConfig,
    SpectralCombinationResult,
)
from ocularrigidity.motion.pulsation.phase.base import (
    AbstractPhaseEstimator,
    PhaseTrack,
)
from ocularrigidity.motion.pulsation.phase.demodulation import (
    IQDemodPhaseEstimator,
    IQPhaseConfig,
)
from ocularrigidity.motion.pulsation.phase.hilbert import (
    AmplitudeWeightedHilbertConfig,
    AmplitudeWeightedHilbertPhaseEstimator,
    HilbertPhaseConfig,
    HilbertPhaseEstimator,
)
from ocularrigidity.motion.pulsation.phase.peak_locking import (
    PeakLockConfig,
    PeakLockedPhaseEstimator,
)

__all__ = [
    "PhaseTrack",
    "AbstractPhaseEstimator",
    "IQDemodPhaseEstimator",
    "IQPhaseConfig",
    "PeakLockedPhaseEstimator",
    "PeakLockConfig",
    "HilbertPhaseEstimator",
    "HilbertPhaseConfig",
    "AmplitudeWeightedHilbertPhaseEstimator",
    "AmplitudeWeightedHilbertConfig",
    "AbstractTraceAggregator",
    "SelectBestComponent",
    "SingleTrace",
    "MeanTrace",
    "PowerWeightedMean",
    "OptimizedSpectralCombination",
    "SpectralCombinationConfig",
    "SpectralCombinationResult",
]
