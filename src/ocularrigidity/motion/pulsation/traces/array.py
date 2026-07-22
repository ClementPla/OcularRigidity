"""A per-frame ``(T, W)`` array you already have, as a trace source.

For signals the pipeline does not compute itself — boundary displacement
``dY``, an intensity profile, anything measured elsewhere. Resampling onto the
uniform grid, gap marking and the cardiac bandpass come from
:class:`AbstractUniformTraceSource`, so the array only has to be per-frame and
NaN-marked where it has holes.
"""

from typing import Optional

import numpy as np

from ocularrigidity.motion.pulsation.traces.base import (
    AbstractUniformTraceSource,
    UniformTraceConfig,
)
from ocularrigidity.motion.video_timeline_aligner import VideoTimelineAligner


class ArrayTraceSource(AbstractUniformTraceSource):
    """Wrap an existing ``(T, W)`` array, one trace per column.

    ``signal`` must have one row per frame on the aligner's timeline. Holes are
    marked NaN; a row that is entirely NaN counts as a bad frame.
    """

    def __init__(
        self,
        signal: np.ndarray,
        aligner: VideoTimelineAligner,
        config: Optional[UniformTraceConfig] = None,
        registered_video=None,
    ):
        super().__init__(aligner, config)
        self._array = np.asarray(signal, dtype=float)
        if self._array.ndim != 2:
            raise ValueError(
                f"signal must be 2-D (T, W); got shape {self._array.shape}."
            )
        n_frames = len(aligner.timestamps_seconds)
        if len(self._array) != n_frames:
            raise ValueError(
                f"signal has {len(self._array)} rows but the aligner has "
                f"{n_frames} frames; they must share a timeline."
            )
        # Only needed if a consumer (folding, a viewer) wants the frames back.
        if registered_video is not None:
            self.registered_video = registered_video

    def raw_signal(self) -> np.ndarray:
        return self._array
