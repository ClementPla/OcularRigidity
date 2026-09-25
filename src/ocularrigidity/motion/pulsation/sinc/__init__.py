"""Self-supervised pulse extraction (SiNC, Speth et al. CVPR 2023), Stage 0.

A small network turns the registered BM/CSI boundary map into one pulse
waveform, trained without labels or an expected BPM: bandwidth, sparsity and
batch-variance losses on its power spectrum. Spectra are computed at each
sample's own timestamp, so gaps need no interpolation of the output.

``preprocess`` builds the network input (shared by training and inference),
``data`` caches the cohort and draws clips, ``losses``/``model``/``module``
train, ``evaluate`` compares against the chain run without an expected BPM,
and ``trace_source`` plugs a trained network into :class:`PulseExtractor`.
"""

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
