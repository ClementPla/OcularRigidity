"""Self-supervised pulse extraction (SiNC, Speth et al."""

from ocularrigidity.motion.pulsation.sinc.losses import (
    SiNCLoss,
    SiNCLossConfig,
    peak_frequency,
)
from ocularrigidity.motion.pulsation.sinc.model import PulseNet
from ocularrigidity.motion.pulsation.sinc.module import (
    SiNCModule,
    SiNCTrainConfig,
    rate_metrics,
)
from ocularrigidity.motion.pulsation.sinc.trace_source import LearnedTraceSource

__all__ = [
    "LearnedTraceSource",
    "PulseNet",
    "SiNCLoss",
    "SiNCLossConfig",
    "SiNCModule",
    "SiNCTrainConfig",
    "peak_frequency",
    "rate_metrics",
]
