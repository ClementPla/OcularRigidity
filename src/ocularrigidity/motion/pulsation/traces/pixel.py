import numpy as np

from ocularrigidity.motion.pulsation.traces.base import (
    AbstractUniformTraceSource,
)
from ocularrigidity.motion.video_timeline_aligner import VideoTimelineAligner


class PixelTraceSource(AbstractUniformTraceSource):
    def __init__(
        self,
        registered_video,
        aligner: VideoTimelineAligner,
        config=None,
    ):
        super().__init__(aligner, config)
        self.registered_video = registered_video

    def raw_signal(self) -> np.ndarray:
        if self._signal is None:
            frames = self.registered_video.registered_frames
            masks = self.registered_video.registered_masks.astype(bool)
            # Adjust mask -> trim

            # Return the trace as T x N where N is the number of pixels in the mask
            self._signal = frames[masks].reshape(frames.shape[0], -1)
        return self._signal

    @property
    def filtered_signal(self):
        return self.raw_signal()
