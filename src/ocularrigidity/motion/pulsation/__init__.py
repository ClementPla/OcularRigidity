"""Cardiac pulse extraction, composed from three swappable stages.

    traces/  →  rate/ (optional)  →  phase/

One subpackage per stage. Each has a ``base.py`` holding the ABC and the data
type it passes on, then one module per concrete method — with that method's
config dataclass next to it, so adding a method means adding a single file:

    traces/   base.py  mask.py  decomposition.py
    rate/     base.py  lomb_scargle.py  fixed.py
    phase/    base.py  aggregation.py  demodulation.py  peak_locking.py
              hilbert.py

At the package root: ``band.py`` (the physiological prior, shared by the trace
bandpass and the rate estimators), ``extractor.py`` (:class:`PulseExtractor`,
the orchestrator), ``n_cycle_reconstructor.py`` (folding), ``pipeline.py``
(end-to-end wiring) and ``legacy.py`` (the pre-refactor
:class:`MaskPulseExtractor` facade — do not extend).

Typical use::

    PulseExtractor(
        trace_source=DecomposedTraceSource(MaskThicknessTraceSource(reg, aligner)),
        rate_estimator=LombScargleRateEstimator(),
        phase_estimator=IQDemodPhaseEstimator(aggregator=SelectBestComponent()),
    )
"""

from ocularrigidity.motion.pulsation.band import CardiacBand
from ocularrigidity.motion.pulsation.extractor import PulseExtractor
from ocularrigidity.motion.pulsation.legacy import (
    MaskPulseExtractor,
    PulseExtractionConfig,
)
from ocularrigidity.motion.pulsation.n_cycle_reconstructor import (
    NCycleConfig,
    NCycleReconstructor,
)
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
from ocularrigidity.motion.pulsation.pipeline import run_cardiac_pipeline
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

#: Deprecated alias — the monolithic base class is gone, ``PulseExtractor`` is
#: now concrete and composed. Kept so old imports and type hints resolve.
AbstractPulseExtractor = PulseExtractor

__all__ = [
    # Orchestration
    "PulseExtractor",
    "NCycleReconstructor",
    "NCycleConfig",
    "run_cardiac_pipeline",
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
    # Legacy
    "MaskPulseExtractor",
    "PulseExtractionConfig",
    "AbstractPulseExtractor",
]
