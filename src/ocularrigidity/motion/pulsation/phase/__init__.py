"""Stage 3"""

from ocularrigidity.motion.pulsation.phase.aggregation import (
    AbstractTraceAggregator,
    MeanTrace,
    PowerWeightedMean,
    SelectBestComponent,
    SingleTrace,
)
from ocularrigidity.motion.pulsation.phase.anchoring import (
    ThicknessAnchorConfig,
    ThicknessAnchoredPhaseEstimator,
    thickness_minimum_phase,
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
    "ThicknessAnchoredPhaseEstimator",
    "ThicknessAnchorConfig",
    "thickness_minimum_phase",
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
]
