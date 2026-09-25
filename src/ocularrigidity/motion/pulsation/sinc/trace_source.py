"""A trained SiNC network as a trace source for :class:`PulseExtractor`.

Wraps a :class:`MaskThicknessTraceSource` for its timeline and gap mask (the
same bad-frame rule the training cache was built with), feeds the registered
BM/CSI lines through the shared preprocessing, and returns the network output
as a single trace. Downstream stages are unchanged::

    mask = MaskThicknessTraceSource(registrator, aligner, stages["trace"])
    PulseExtractor(
        trace_source=LearnedTraceSource(
            BandPassFilterTraceSource(mask, stages["bandpass"]), module
        ),
        rate_estimator=LombScargleRateEstimator(stages["rate"]),  # expected_bpm=None
        phase_estimator=ThicknessAnchoredPhaseEstimator(
            IQDemodPhaseEstimator(stages["phase"]), reference=mask
        ),
    )

The network's waveform has an arbitrary sign and lag, so IQ phase 0 means
nothing physical; the anchoring wrapper moves it onto minimal choroid thickness,
which is what makes the folded cycles start at the same point of every beat.
"""

import numpy as np
import torch

from ocularrigidity.motion.pulsation.sinc.module import SiNCModule
from ocularrigidity.motion.pulsation.sinc.preprocess import boundaries_to_uniform
from ocularrigidity.motion.pulsation.traces import AbstractTraceSource, Traces


def _find(source, attr):
    """Walk a chain of trace-source decorators for the first one with ``attr``."""
    seen = set()
    while source is not None and id(source) not in seen:
        seen.add(id(source))
        if hasattr(source, attr):
            return getattr(source, attr)
        source = getattr(source, "source", None)
    return None


class LearnedTraceSource(AbstractTraceSource):
    """``source`` is a :class:`MaskThicknessTraceSource` or any decorator over
    one (e.g. its bandpass). Wrapping the bandpass keeps ``filtered_signal`` in
    the chain, so :class:`CardiacPipelineResults` records the same fields as
    the classical chain and its consumers (amplitude, viewers) keep working;
    the network itself reads the registered boundaries, not that signal.
    """

    def __init__(self, source: AbstractTraceSource, module: SiNCModule):
        super().__init__()
        self.source = source
        self.module = module
        self.registered_video = _find(source, "registered_video")
        self.aligner = _find(source, "aligner")
        if self.registered_video is None or self.aligner is None:
            raise ValueError("source chain has no registered_video/aligner")

    def compute(self) -> Traces:
        lines = self.registered_video.registered_lines
        if isinstance(lines, torch.Tensor):
            lines = lines.cpu().numpy()
        gap = self.source.gap_mask
        x = boundaries_to_uniform(
            lines, self.aligner.timestamps_seconds, self.aligner.uniform_time, gap
        )
        self.module.eval()
        y, mask, _ = self.module.predict_video(x, self.aligner.dt)
        y = y.float().cpu().numpy()

        kept = mask.cpu().numpy() & ~gap
        full = np.where(kept, y, np.nan)[:, None]
        return Traces(
            values=full[kept],
            uniform_time=self.aligner.uniform_time,
            kept_mask=kept,
            gap_mask=gap,
            timestamps_seconds=self.aligner.timestamps_seconds,
            mixing=None,
            source_map=full,
        )

    def reset(self) -> None:
        super().reset()
        self.source.reset()


_MODULES: dict = {}


def load_sinc_module(checkpoint, device: str = "cuda") -> SiNCModule:
    """A trained module, loaded once per process and checkpoint."""
    key = (str(checkpoint), device)
    if key not in _MODULES:
        module = SiNCModule.load_from_checkpoint(checkpoint, map_location=device)
        _MODULES[key] = module.eval()
    return _MODULES[key]
