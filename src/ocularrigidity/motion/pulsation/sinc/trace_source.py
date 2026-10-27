"""A trained SiNC network as a trace source for :class:`PulseExtractor`."""

from functools import lru_cache

from typing import Optional

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

from ocularrigidity.consts import SINC_REVISION
from ocularrigidity.motion.pulsation.band import CardiacBand
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


def bandpass_input(x: np.ndarray, dt: float, band: CardiacBand) -> np.ndarray:
    """The model's input map ``(T, 2, n_cols)`` band-passed along time to
    ``band`` (zero-phase, 2nd-order Butterworth).

    NaN samples (gaps) are bridged by linear interpolation for the filter and
    put back afterwards; a series with fewer than two valid samples is left as
    it is.
    """
    lo, hi = band.effective_hz_range
    nyq = 0.5 / dt
    sos = butter(2, [lo / nyq, min(hi / nyq, 0.99)], btype="bandpass", output="sos")
    T = x.shape[0]
    flat = x.reshape(T, -1).astype(np.float64)
    out = flat.copy()
    t = np.arange(T)
    for j in range(flat.shape[1]):
        ok = np.isfinite(flat[:, j])
        if ok.sum() < 2:
            continue
        filled = flat[:, j] if ok.all() else np.interp(t, t[ok], flat[ok, j])
        out[:, j] = sosfiltfilt(sos, filled)
        out[~ok, j] = np.nan
    return out.reshape(x.shape).astype(np.float32)


class LearnedTraceSource(AbstractTraceSource):
    """``source`` is a :class:`MaskThicknessTraceSource` or any decorator over one (e.g. its bandpass).

    ``input_band``, if given, band-passes the model's input to it first. The
    model was trained on unfiltered boundaries and takes no expected rate; this
    is the way to hand it one, by leaving it only that band to find a pulse in.
    """

    def __init__(
        self,
        source: AbstractTraceSource,
        module: SiNCModule,
        input_band: Optional[CardiacBand] = None,
    ):
        super().__init__()
        self.source = source
        self.module = module
        self.input_band = input_band
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
        if self.input_band is not None:
            x = bandpass_input(x, self.aligner.dt, self.input_band)
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
