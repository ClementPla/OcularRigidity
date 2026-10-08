import numpy as np

from ocularrigidity.motion.pulsation.traces.base import (
    AbstractUniformTraceSource,
)
from ocularrigidity.motion.video_timeline_aligner import VideoTimelineAligner


def _normalize_per_frame(values: np.ndarray, axis) -> np.ndarray:
    """Divide each frame by its own mean, leaving blank frames at 0.

    Registration can blank a frame entirely (a dropped acquisition, or every
    A-scan zeroed by the bad-A-scan filter). Its mean is then 0, and the plain
    division used to fill that frame with ``inf``/``NaN`` — enough, on its own,
    to make the whole trace matrix non-finite and to abort the condition
    downstream (``sklearn`` refuses non-finite input to the SVD). A blank frame
    carries no intensity to normalise, so it is simply left at 0: the frame stays
    in place, keeping the time base intact, and contributes nothing.
    """
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values, axis=axis, keepdims=True)
    usable = np.isfinite(mean) & (mean != 0)
    return np.divide(values, mean, out=np.zeros_like(values), where=usable)


class PixelTraceSource(AbstractUniformTraceSource):
    def __init__(
        self,
        registered_video,
        aligner: VideoTimelineAligner,
        config=None,
    ):
        super().__init__(aligner, config)
        self.registered_video = registered_video
        self._normalized_trace = None

    def raw_signal(self) -> np.ndarray:
        if self._signal is None:
            frames = self.registered_video.registered_frames
            masks = self.registered_video.registered_masks.astype(bool)
            # Adjust mask -> trim

            # Return the trace as T x N where N is the number of pixels in the mask
            self._signal = frames[masks].reshape(frames.shape[0], -1)
        return self._signal

    def normalized_trace(self) -> np.ndarray:
        """``raw_signal()`` with each frame divided by its mean over the traces.

        Shape ``(T, N)``, like ``raw_signal()``. This removes the gain or
        illumination drift COMMON to every pixel of the ROI at a given frame —
        and with it the part of the choroidal pulse that beats in phase across
        the ROI, which is largely the case inside the choroid. By construction
        the spatial mean of the result is 1 at every frame, so it carries no
        pulse and cannot serve as a sign reference (see
        ``notebook/test_norm_int.ipynb``).

        Same operation as the former ``normalized_signal``: that version also
        divided each whole frame by its mean before extracting the ROI, but a
        per-frame scalar cancels out in this division, so the result is equal.
        """
        if self._normalized_trace is None:
            self._normalized_trace = _normalize_per_frame(self.raw_signal(), axis=1)
        return self._normalized_trace

    @property
    def filtered_signal(self):
        return self.raw_signal()

    def reset(self) -> None:
        super().reset()
        self._normalized_trace = None
