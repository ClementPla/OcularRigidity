"""Timeline alignment: maps irregular frame timestamps onto a uniform grid.

This is deliberately signal-agnostic. It knows *when* frames were captured, not
*what* is in them. The mask/frame-domain notion of a "bad frame" is injected by
the extractor via :meth:`gap_mask`, so the aligner can be shared by any
``PulseExtractor``.
"""

from enum import Enum
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ocularrigidity.registration.registration_engine import VideoRegistrator


class TimeUnits(Enum):
    MICROSECONDS = "us"
    MILLISECONDS = "ms"
    SECONDS = "s"


class VideoTimelineAligner:
    def __init__(
        self,
        registered_video: VideoRegistrator,
        timestamps: Union[Path, str, pd.Series, np.ndarray, list],
        units_in_timestamps: TimeUnits = TimeUnits.MICROSECONDS,
    ):
        """``timestamps`` may be a path to a headerless single-column CSV of
        timestamps, or the timestamps themselves as a sequence/Series/array."""
        self.registered_video = registered_video
        self.timestamps = timestamps
        self.units_in_timestamps = units_in_timestamps

        self._timestamps_seconds = None
        self._uniform_time = None
        self._neighbor_query = None

    @property
    def _frame_slice(self) -> slice:
        """Frame trimming — kept in sync with the registrator (single source)."""
        reg = self.registered_video
        end = None if reg.drop_last_n_frames == 0 else -reg.drop_last_n_frames
        return slice(reg.skip_first_n_frames, end)

    @property
    def timestamps_seconds(self):
        if self._timestamps_seconds is None:
            if isinstance(self.timestamps, (str, Path)):
                ts_series = pd.read_csv(
                    self.timestamps, header=None, names=["timestamp"]
                )["timestamp"]
            else:
                ts_series = pd.Series(np.asarray(self.timestamps).ravel())
            ts_series = ts_series.sort_values().reset_index(drop=True)
            ts = ts_series[self._frame_slice].to_numpy()
            if self.units_in_timestamps == TimeUnits.MICROSECONDS:
                self._timestamps_seconds = (ts - ts[0]) / 1e6
            elif self.units_in_timestamps == TimeUnits.MILLISECONDS:
                self._timestamps_seconds = (ts - ts[0]) / 1e3
            else:
                self._timestamps_seconds = (ts - ts[0]).astype(float)
        return self._timestamps_seconds

    @property
    def uniform_time(self):
        if self._uniform_time is None:
            ts = self.timestamps_seconds
            dt = self.dt
            n = int(np.floor((ts[-1] - ts[0]) / dt)) + 1
            self._uniform_time = ts[0] + np.arange(n) * dt
        return self._uniform_time

    @property
    def dt(self) -> float:
        return float(np.median(np.diff(self.timestamps_seconds)))

    @property
    def fs(self) -> float:
        return 1.0 / self.dt

    @property
    def _neighbor(self):
        """(far_from_any_frame, nearest_frame_idx) for each uniform sample.

        Purely time-based: ``far_from_any_frame`` marks uniform samples with no
        real timestamp within 2×dt_p95; ``nearest_frame_idx`` maps each uniform
        sample to its closest original frame (used to propagate bad-frame flags).
        """
        if self._neighbor_query is None:
            ts = self.timestamps_seconds
            dt_p95 = float(np.percentile(np.diff(ts), 95))
            dists, nearest_idx = cKDTree(ts[:, None]).query(
                self.uniform_time[:, None], k=1
            )
            far_from_any_frame = dists > 2 * dt_p95
            self._neighbor_query = (far_from_any_frame, nearest_idx)
        return self._neighbor_query

    @property
    def far_from_any_frame(self):
        return self._neighbor[0]

    @property
    def nearest_frame_idx(self):
        return self._neighbor[1]

    def gap_mask(self, bad_frame: np.ndarray) -> np.ndarray:
        """Uniform-grid gap mask, given a per-original-frame ``bad_frame`` flag.

        True where the grid is far from any real frame, or where its nearest
        real frame is flagged bad by the (domain-specific) extractor.
        """
        far_from_any_frame, nearest_idx = self._neighbor
        return far_from_any_frame | bad_frame[nearest_idx]
