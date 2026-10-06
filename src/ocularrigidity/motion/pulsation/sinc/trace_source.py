"""A trained SiNC network as a trace source for :class:`PulseExtractor`."""

from functools import lru_cache

import numpy as np
import torch

from ocularrigidity.consts import SINC_REVISION
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
    """``source`` is a :class:`MaskThicknessTraceSource` or any decorator over one (e.g. its bandpass)."""

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


def load_sinc_module(revision: str = SINC_REVISION, device: str = "cuda") -> SiNCModule:
    """The trained module from the Hub, loaded once per process."""
    return _load(revision, device)


@lru_cache(maxsize=None)
def _load(revision: str, device: str) -> SiNCModule:
    module = SiNCModule.from_pretrained(
        "ClementP/OCTVideoPulsationMeasure", revision=revision
    )
    return module.to(device).eval()
