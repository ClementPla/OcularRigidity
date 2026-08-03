"""Cardiac bandpass + NaN-aware spatial smoothing of a uniform-grid source.

This used to live inside ``AbstractUniformTraceSource`` (the old
``filtered_signal`` property). It is a stage of its own so a source can be fed
to the decomposition raw, and so the band can be changed without rebuilding the
signal map.

Filtering runs on the *full* uniform grid rather than on ``source.traces``:
``filtfilt`` needs contiguous, evenly-spaced samples, and the kept-sample view
has the gaps collapsed out of it.
"""

from dataclasses import dataclass, field

import numpy as np

from ocularrigidity.motion.filters._1d import spatio_temporal_filter
from ocularrigidity.motion.pulsation.band import CardiacBand
from ocularrigidity.motion.pulsation.traces.base import (
    AbstractTraceSource,
    AbstractUniformTraceSource,
    Traces,
)


@dataclass
class BandPassFilterTraceConfig:
    band: CardiacBand = field(default_factory=CardiacBand)
    sigma_col: float = 5.0
    verbose: bool = True


class BandPassFilterTraceSource(AbstractTraceSource):
    """Wraps a uniform source and returns its bandpassed traces.

    One trace per column of the source's signal map, restricted to the cardiac
    band and smoothed across columns with ``sigma_col`` (validity-weighted, so
    holes do not bleed into their neighbours).
    """

    def __init__(
        self,
        source: AbstractUniformTraceSource,
        config: BandPassFilterTraceConfig | None = None,
    ):
        super().__init__()
        self.source = source
        self.config = config or BandPassFilterTraceConfig()
        self._filtered_signal = None

    # -- the source's uniform grid, delegated ---------------------------
    @property
    def uniform_time(self) -> np.ndarray:
        return self.source.uniform_time

    @property
    def timestamps_seconds(self) -> np.ndarray:
        return self.source.timestamps_seconds

    @property
    def gap_mask(self) -> np.ndarray:
        return self.source.gap_mask

    @property
    def fs(self) -> float:
        return self.source.fs

    @property
    def interpolated_validity(self) -> np.ndarray:
        """Source validity with gap samples zeroed — the filter's weights."""
        not_gap = (~self.gap_mask).astype(np.float32)[:, None]
        return self.source.interpolated_validity * not_gap

    @property
    def filtered_signal(self) -> np.ndarray:
        """``(T_uniform, W)`` filtered map, NaN where the sample is unusable."""
        if self._filtered_signal is None:
            self._filtered_signal = self._filter()
        return self._filtered_signal

    # -- contract -------------------------------------------------------
    def _filter(self) -> np.ndarray:
        cfg = self.config
        gap = self.gap_mask

        # Gap samples are zero-filled and given zero weight: filtfilt needs a
        # continuous series, and the weights keep those samples from pulling
        # the smoother. They are re-marked NaN once filtering is done.
        not_gap = (~gap).astype(np.float32)[:, None]
        data = np.nan_to_num(self.source.interpolated_signal, nan=0.0) * not_gap

        nyq = 0.5 * self.fs
        lo_bpm, hi_bpm = cfg.band.effective_bpm_range
        low = (lo_bpm / 60.0) / nyq
        high = min((hi_bpm / 60.0) / nyq, 0.99)

        filtered = spatio_temporal_filter(
            data,
            spatial_sigma=cfg.sigma_col,
            temporal_low_freq=low,
            temporal_high_freq=high,
            fs=self.fs,
            validity_mask=self.interpolated_validity,
        )
        filtered[gap] = np.nan
        return filtered

    def compute(self) -> Traces:
        filtered = self.filtered_signal
        # Same kept-sample rule as the uniform sources: a sample survives only
        # if every trace is finite there, so the decomposition and the
        # periodogram can assume no NaNs.
        kept = ~np.isnan(filtered).any(axis=1)

        if not kept.any():
            msg = (
                "Bandpass left no usable sample: every uniform sample has at "
                "least one NaN trace (a column with no valid data anywhere "
                "will do it)."
            )
            self.notes.append(msg)
            if self.config.verbose:
                print(msg)

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
        self.source.reset()
        self._filtered_signal = None

    @property
    def notes_all(self) -> list[str]:
        return list(self.source.notes) + list(self.notes)
