"""Deprecated import shim — configs now live next to the code they configure.

    CardiacBand           → ocularrigidity.motion.pulsation.band
    UniformTraceConfig    → ...pulsation.traces.base
    MaskTraceConfig       → ...pulsation.traces.mask
    DecompositionConfig   → ...pulsation.traces.decomposition
    LombScargleConfig     → ...pulsation.rate.lomb_scargle
    IQPhaseConfig         → ...pulsation.phase.demodulation
    PeakLockConfig        → ...pulsation.phase.peak_locking
    HilbertPhaseConfig    → ...pulsation.phase.hilbert
    PulseExtractionConfig → ...pulsation.legacy
    NCycleConfig          → ...pulsation.n_cycle_reconstructor

All of them are re-exported from the package root, so
``from ocularrigidity.motion.pulsation import X`` is the import to use.

This module must keep existing: ``CardiacPipelineResults`` pickles embed the
config classes by their original module path, and unpickling an old result file
resolves them here.
"""

from ocularrigidity.motion.pulsation.band import CardiacBand
from ocularrigidity.motion.pulsation.legacy import PulseExtractionConfig
from ocularrigidity.motion.pulsation.n_cycle_reconstructor import NCycleConfig
from ocularrigidity.motion.pulsation.phase import (
    HilbertPhaseConfig,
    IQPhaseConfig,
    PeakLockConfig,
)
from ocularrigidity.motion.pulsation.rate import LombScargleConfig
from ocularrigidity.motion.pulsation.traces import (
    DecompositionConfig,
    MaskTraceConfig,
    UniformTraceConfig,
)

__all__ = [
    "CardiacBand",
    "UniformTraceConfig",
    "MaskTraceConfig",
    "DecompositionConfig",
    "LombScargleConfig",
    "IQPhaseConfig",
    "PeakLockConfig",
    "HilbertPhaseConfig",
    "PulseExtractionConfig",
    "NCycleConfig",
]
