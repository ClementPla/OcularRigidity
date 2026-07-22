"""Trace sources: everything that turns a video into candidate 1-D traces.

A :class:`AbstractTraceSource` promises one thing — a ``(T_kept, K)`` bundle of
candidate temporal traces on the uniform time grid, plus the masks needed to put
them back on the original frame timeline. *What* the traces are is up to the
implementation.

**Adding a source:** if you have a per-frame ``(T, W)`` map, subclass
:class:`AbstractUniformTraceSource` and implement ``raw_signal`` — resampling,
gap marking and the cardiac bandpass are done for you (see ``mask.py``). If you
transform another source's traces, subclass :class:`AbstractTraceSource` and
implement ``compute`` (see ``decomposition.py``).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.interpolate import interp1d

from ocularrigidity.motion.filters._1d import spatio_temporal_filter
from ocularrigidity.motion.pulsation.band import CardiacBand
from ocularrigidity.motion.video_timeline_aligner import VideoTimelineAligner


@dataclass
class UniformTraceConfig:
    """Resampling + filtering shared by every ``AbstractUniformTraceSource``."""

    band: CardiacBand = field(default_factory=CardiacBand)
    sigma_col: float = 5.0
    verbose: bool = True


@dataclass
class Traces:
    """Candidate temporal traces on the uniform grid.

    ``values`` is ``(T_kept, K)`` — only the samples where every trace is finite
    are kept, so decomposition and periodogram code can assume no NaNs.
    ``kept_mask`` and ``gap_mask`` are on the *full* uniform grid, so anything
    needing gap-awareness (peak locking, folding) can recover the alignment.
    """

    values: np.ndarray  # (T_kept, K)
    uniform_time: np.ndarray  # (T_uniform,)
    kept_mask: np.ndarray  # (T_uniform,) bool
    gap_mask: np.ndarray  # (T_uniform,) bool
    timestamps_seconds: np.ndarray  # (T_frames,) original frame times
    mixing: Optional[np.ndarray] = None  # (W, K) spatial pattern, if meaningful
    source_map: Optional[np.ndarray] = None  # (T_uniform, W) signal traces came from

    @property
    def time(self) -> np.ndarray:
        """Uniform timestamps of the kept samples."""
        return self.uniform_time[self.kept_mask]

    @property
    def n_traces(self) -> int:
        return self.values.shape[1]

    @property
    def dt(self) -> float:
        return float(self.uniform_time[1] - self.uniform_time[0])

    @property
    def fs(self) -> float:
        return 1.0 / self.dt

    def full(self, k: int) -> np.ndarray:
        """Trace ``k`` embedded on the full uniform grid, NaN where not kept."""
        out = np.full(len(self.uniform_time), np.nan)
        out[self.kept_mask] = self.values[:, k]
        return out

    def embed(self, values_kept: np.ndarray) -> np.ndarray:
        """Embed an arbitrary kept-length vector on the full uniform grid."""
        out = np.full(len(self.uniform_time), np.nan)
        out[self.kept_mask] = values_kept
        return out


class AbstractTraceSource(ABC):
    """Produces the candidate traces a pulse extractor will work from.

    Implement :meth:`compute` and you are done; ``traces`` caches it. If your
    source is a per-frame 2-D map (something × time), prefer subclassing
    :class:`AbstractUniformTraceSource`, which already handles resampling onto
    the uniform grid, gap marking and the cardiac bandpass.
    """

    def __init__(self):
        self._traces: Optional[Traces] = None
        self.notes: list[str] = []

    @abstractmethod
    def compute(self) -> Traces:
        """Build the traces. Called once; the result is cached in ``traces``."""

    @property
    def traces(self) -> Traces:
        if self._traces is None:
            self._traces = self.compute()
        return self._traces

    def reset(self) -> None:
        self._traces = None
        self.notes = []


class AbstractUniformTraceSource(AbstractTraceSource):
    """Base for sources backed by a per-frame ``(T, W)`` map.

    Subclasses supply :meth:`raw_signal` (holes marked NaN) and optionally
    override :meth:`bad_frame`. Resampling onto the uniform grid, gap marking,
    NaN-aware spatial smoothing and the cardiac bandpass are shared here — they
    are properties of *making a trace*, not of any particular signal domain.
    """

    def __init__(
        self,
        aligner: VideoTimelineAligner,
        config: Optional[UniformTraceConfig] = None,
    ):
        super().__init__()
        self.aligner = aligner
        self.config = config or UniformTraceConfig()

        self._signal = None
        self._gap_mask = None
        self._interpolated_signal = None
        self._interpolated_validity = None
        self._filtered_signal = None

    # -- subclass hooks -------------------------------------------------
    @abstractmethod
    def raw_signal(self) -> np.ndarray:
        """Per-frame signal, shape ``(T, W)``, holes marked as NaN."""

    def bad_frame(self) -> np.ndarray:
        """Per-original-frame invalidity flag. Default: fully-NaN rows."""
        return np.isnan(self.signal).all(axis=1)

    # -- shared plumbing ------------------------------------------------
    @property
    def verbose(self) -> bool:
        return self.config.verbose

    @property
    def signal(self) -> np.ndarray:
        if self._signal is None:
            self._signal = self.raw_signal()
        return self._signal

    @property
    def timestamps_seconds(self):
        return self.aligner.timestamps_seconds

    @property
    def uniform_time(self):
        return self.aligner.uniform_time

    @property
    def fs(self) -> float:
        return self.aligner.fs

    @property
    def gap_mask(self):
        """Uniform-grid gap mask: time gaps combined with this domain's bad frames."""
        if self._gap_mask is None:
            self._gap_mask = self.aligner.gap_mask(self.bad_frame())
        return self._gap_mask

    @property
    def interpolated_signal(self):
        """``signal`` interpolated onto the uniform grid; gaps set to NaN."""
        if self._interpolated_signal is None:
            signal = self.signal
            valid = ~np.isnan(signal).any(axis=1)
            if not valid.any():
                out = np.full(
                    (len(self.uniform_time), signal.shape[1]),
                    np.nan,
                    dtype=signal.dtype,
                )
            else:
                out = interp1d(
                    self.timestamps_seconds[valid],
                    signal[valid],
                    axis=0,
                    kind="linear",
                    fill_value=np.nan,
                    bounds_error=False,
                )(self.uniform_time)
            out[self.gap_mask] = np.nan
            self._interpolated_signal = out
        return self._interpolated_signal

    @property
    def interpolated_validity(self):
        """Per-sample validity fraction on the uniform grid; gaps set to 0."""
        if self._interpolated_validity is None:
            valid = (~np.isnan(self.signal)).astype(np.float32)
            out = interp1d(
                self.timestamps_seconds,
                valid,
                axis=0,
                kind="linear",
                fill_value=0.0,
                bounds_error=False,
            )(self.uniform_time)
            out[self.gap_mask] = 0.0
            self._interpolated_validity = out
        return self._interpolated_validity

    @property
    def filtered_signal(self):
        """Spatially smoothed (NaN-aware) and temporally bandpassed signal map."""
        if self._filtered_signal is not None:
            return self._filtered_signal

        nyq = 0.5 * self.fs
        lo_bpm, hi_bpm = self.config.band.effective_bpm_range
        low = (lo_bpm / 60.0) / nyq
        high = min((hi_bpm / 60.0) / nyq, 0.99)
        not_gap = (~self.gap_mask).astype(np.float32)[:, None]
        data_masked = np.nan_to_num(self.interpolated_signal, nan=0.0) * not_gap
        valid_masked = self.interpolated_validity * not_gap
        filtered = spatio_temporal_filter(
            data_masked,
            spatial_sigma=self.config.sigma_col,
            temporal_low_freq=low,
            temporal_high_freq=high,
            fs=self.fs,
            validity_mask=valid_masked,
        )
        filtered[self.gap_mask] = np.nan
        self._filtered_signal = filtered
        return self._filtered_signal

    def compute(self) -> Traces:
        filtered = self.filtered_signal
        kept = ~np.isnan(filtered).any(axis=1)
        return Traces(
            values=filtered[kept],
            uniform_time=self.uniform_time,
            kept_mask=kept,
            gap_mask=self.gap_mask,
            timestamps_seconds=self.timestamps_seconds,
            mixing=None,
            source_map=filtered,
        )

    def reset(self) -> None:
        super().reset()
        self._signal = None
        self._gap_mask = None
        self._interpolated_signal = None
        self._interpolated_validity = None
        self._filtered_signal = None
