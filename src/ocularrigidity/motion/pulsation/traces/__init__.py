"""Stage 1 — trace sources.

``base.py`` holds the contract (:class:`Traces`, :class:`AbstractTraceSource`)
and the shared uniform-grid plumbing. One module per source: ``mask.py`` for
segmented thickness, ``decomposition.py`` for the ICA/PCA wrapper. Add a new
source as a new module here and export it below.
"""

from ocularrigidity.motion.pulsation.traces.array import ArrayTraceSource
from ocularrigidity.motion.pulsation.traces.base import (
    AbstractTraceSource,
    AbstractUniformTraceSource,
    Traces,
    UniformTraceConfig,
)
from ocularrigidity.motion.pulsation.traces.decomposition import (
    DecompositionConfig,
    DecomposedTraceSource,
)
from ocularrigidity.motion.pulsation.traces.mask import (
    MaskThicknessTraceSource,
    MaskTraceConfig,
)

__all__ = [
    "Traces",
    "AbstractTraceSource",
    "AbstractUniformTraceSource",
    "ArrayTraceSource",
    "UniformTraceConfig",
    "MaskThicknessTraceSource",
    "MaskTraceConfig",
    "DecomposedTraceSource",
    "DecompositionConfig",
]
