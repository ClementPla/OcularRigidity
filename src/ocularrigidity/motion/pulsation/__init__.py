"""Cardiac pulse extraction, composed from three swappable stages.

    traces/  ->  rate/ (optional)  ->  phase/

One subpackage per stage, each with a ``base.py`` holding the ABC and the type
it passes on, then one module per concrete method with that method's config
dataclass beside it — so adding a method means adding a single file.

``band.py`` holds the physiological prior, shared by the trace bandpass and the
rate estimators; ``extractor.py`` the orchestrator; ``n_cycle_reconstructor.py``
the folding; ``pipeline.py`` the end-to-end wiring.
"""

from ocularrigidity.motion.pulsation.band import CardiacBand
from ocularrigidity.motion.pulsation.extractor import PulseExtractor
from ocularrigidity.motion.pulsation.n_cycle_reconstructor import (
    NCycleConfig,
    NCycleReconstructor,
)
from ocularrigidity.motion.pulsation.pipeline import run_composed_pipeline
from ocularrigidity.motion.pulsation.phase import (
    AbstractPhaseEstimator,
    AmplitudeWeightedHilbertConfig,
    AmplitudeWeightedHilbertPhaseEstimator,
    AbstractTraceAggregator,
    HilbertPhaseConfig,
    HilbertPhaseEstimator,
    IQDemodPhaseEstimator,
    IQPhaseConfig,
    MeanTrace,
    PeakLockConfig,
    PeakLockedPhaseEstimator,
    PhaseTrack,
    PowerWeightedMean,
    SelectBestComponent,
    SingleTrace,
)
from ocularrigidity.motion.pulsation.rate import (
    AbstractRateEstimator,
    FixedRateEstimator,
    LombScargleConfig,
    LombScargleRateEstimator,
    RateEstimate,
)
from ocularrigidity.motion.pulsation.traces import (
    AbstractTraceSource,
    AbstractUniformTraceSource,
    ArrayTraceSource,
    BandPassFilterTraceConfig,
    BandPassFilterTraceSource,
    DecompositionConfig,
    DecomposedTraceSource,
    MaskThicknessTraceSource,
    MaskTraceConfig,
    Traces,
    UniformTraceConfig,
)
from ocularrigidity.motion.pulsation.traces.coherence import (
    CoherenceConfig,
    CoherentTraceSource,
)

__all__ = [
    # Orchestration
    "PulseExtractor",
    "NCycleReconstructor",
    "NCycleConfig",
    "run_composed_pipeline",
    "CardiacBand",
    # Stage 1: traces
    "AbstractTraceSource",
    "AbstractUniformTraceSource",
    "ArrayTraceSource",
    "UniformTraceConfig",
    "MaskThicknessTraceSource",
    "MaskTraceConfig",
    "BandPassFilterTraceSource",
    "BandPassFilterTraceConfig",
    "CoherentTraceSource",
    "CoherenceConfig",
    "DecomposedTraceSource",
    "DecompositionConfig",
    "Traces",
    # Stage 2: rate
    "AbstractRateEstimator",
    "RateEstimate",
    "LombScargleRateEstimator",
    "LombScargleConfig",
    "FixedRateEstimator",
    # Stage 3: phase (+ aggregation)
    "AbstractPhaseEstimator",
    "PhaseTrack",
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
]
