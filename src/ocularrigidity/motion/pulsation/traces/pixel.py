"""Raw registered-video pixels as one trace per pixel (or per super-pixel).

Unlike :class:`~ocularrigidity.motion.pulsation.traces.mask.MaskThicknessTraceSource`,
which derives one value per A-scan from the segmented thickness, this source
reads intensities directly off ``registered_video.registered_frames`` — the
segmentation mask is only used to pick *which* pixels are worth tracking.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ocularrigidity.motion.pulsation.traces.base import (
    AbstractUniformTraceSource,
    UniformTraceConfig,
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


@dataclass
class PixelTraceConfig(UniformTraceConfig):
    """Extra knobs specific to the raw-pixel source."""

    # Restrict the (already time-intersected) choroid ROI, as a fraction of
    # its OWN extent — not the full frame. `col_frac` is applied once,
    # globally, to the ROI's column bounding box: (1/3, 2/3) keeps only the
    # central third of A-scans. `row_frac` is applied per kept A-scan, to
    # that column's own row extent within the ROI: (0.0, 1/3) keeps only the
    # top/innermost third of the choroid band at each A-scan (row 0 = top of
    # frame). Set either to None to skip that restriction.
    col_frac: Optional[tuple[float, float]] = (1 / 4, 3 / 4)
    row_frac: Optional[tuple[float, float]] = (0.0, 1.0)
    # Spatial median-pooling scales, in pixels. 1 keeps the raw per-pixel
    # traces; each b > 1 adds one trace per non-overlapping b x b block,
    # imitating a multi-resolution pyramid ahead of the SVD/ICA step.
    block_sizes: tuple[int, ...] = (1, 3, 5)
    # Flattening a 2-D pixel ROI collapses spatial adjacency, and traces at
    # different block sizes sit side by side in the same axis besides — so the
    # spatial smoothing that makes sense for per-A-scan traces (mask.py) is
    # off by default here.
    sigma_col: float = 0.0
    normalize_intensity: bool = True
    # A frame whose mask covers less than this fraction of the temporal MEDIAN
    # mask area is left out of the ROI intersection (see
    # :attr:`PixelTraceSource.base_roi`). Registration can blank or badly
    # truncate a frame, and since the ROI is an intersection over time, one such
    # frame vetoes pixels every other frame agrees on. Measured on this cohort,
    # the smallest per-frame mask of a healthy condition sits at 0.50 of its
    # median (median across conditions 0.84), while degenerate frames are at or
    # near 0 — so 0.3 separates the two without touching healthy recordings.
    # 0.0 disables the area test and only drops completely empty masks.
    min_mask_area_frac: float = 0.3


class PixelTraceSource(AbstractUniformTraceSource):
    """One trace per pixel/super-pixel: raw registered-video intensity over time.

    ``registered_masks`` shifts extent frame to frame (the choroid itself
    pulses), so there is no single per-frame pixel set to index with — doing
    ``frames[masks]`` breaks because the count of ``True`` pixels isn't
    constant across frames. Instead, the trace grid is built from
    :attr:`base_roi`, the *intersection* of the mask over time: pixels inside
    the choroid in every frame whose mask is large enough to be trusted (frames
    blanked or truncated by registration are skipped, see :attr:`base_roi`).
    That fixes the pixel set once, which is what lets ``(T, N)`` indexing work
    at all, and as a side effect every trace is valid in every frame — no holes
    to mark, unlike the thickness source.
    ``col_frac``/``row_frac`` (see :class:`PixelTraceConfig`) then trim that
    ROI to a sub-region of its own extent — by default the central third of
    A-scans and, within each of those A-scans, the top third of the choroid
    band.

    For ``block_sizes`` beyond 1, :attr:`base_roi` is downsampled by
    non-overlapping ``b x b`` blocks, keeping only blocks *fully* inside it —
    same mask, coarser grid — and the frame is spatially (not temporally)
    median-pooled over each kept block. Traces from every configured scale are
    concatenated into one ``(T, N_total)`` signal; :attr:`scale_of_trace`
    records which block size each column came from.
    """

    def __init__(
        self,
        registered_video,
        aligner: VideoTimelineAligner,
        config: Optional[PixelTraceConfig] = None,
    ):
        super().__init__(aligner, config or PixelTraceConfig())
        self.registered_video = registered_video
        self._base_roi = None
        self._scale_of_trace = None

    @property
    def base_roi(self) -> np.ndarray:
        """Full-resolution ``(H, W)`` boolean ROI: pixels inside the mask in
        every frame whose mask is big enough to be trusted (see
        :attr:`PixelTraceConfig.min_mask_area_frac`), further restricted per
        :attr:`PixelTraceConfig.col_frac` / :attr:`PixelTraceConfig.row_frac`."""
        if self._base_roi is None:
            cfg: PixelTraceConfig = self.config
            masks = self.registered_video.registered_masks.astype(bool)
            # The ROI is an intersection over time, so a frame whose mask is
            # blank or badly truncated vetoes, on its own, pixels that every
            # other frame agrees on — "no trace survived at any block size".
            # Such a frame says nothing reliable about which pixels are choroid,
            # so it is dropped from the intersection rather than allowed to
            # decide it. Its samples stay in the traces (at 0, see
            # `_normalize_per_frame`); only its vote on the pixel set is lost.
            areas = masks.sum(axis=(1, 2))
            # Reference taken over NON-EMPTY frames: a plain median would
            # collapse to 0, and disable the test entirely, on a video where
            # more than half the frames came out blank.
            non_vides = areas[areas > 0]
            if non_vides.size == 0:
                raise ValueError(
                    "Every frame has an empty mask: no ROI can be built."
                )
            # The `max(..., 1)` keeps empty masks out even when the area test is
            # disabled with `min_mask_area_frac = 0`.
            seuil = max(cfg.min_mask_area_frac * float(np.median(non_vides)), 1.0)
            keep = areas >= seuil
            n_drop = int((~keep).sum())
            if n_drop:
                msg = (f"{n_drop}/{keep.size} frame(s) left out of the ROI "
                       f"intersection: mask smaller than "
                       f"{cfg.min_mask_area_frac:.0%} of the median area.")
                self.notes.append(msg)
                if self.verbose:
                    print(msg)
            if not keep.any():
                raise ValueError(
                    "No frame has a mask above "
                    f"{cfg.min_mask_area_frac:.0%} of the median area."
                )
            roi = masks.all(axis=0, where=keep[:, None, None])
            if cfg.col_frac is not None:
                roi = self._restrict_col_frac(roi, cfg.col_frac)
            if cfg.row_frac is not None:
                roi = self._restrict_row_frac_per_col(roi, cfg.row_frac)
            self._base_roi = roi
        return self._base_roi

    @staticmethod
    def _restrict_col_frac(roi: np.ndarray, frac: tuple[float, float]) -> np.ndarray:
        """Keep only the columns inside ``frac`` of the ROI's own column
        extent (its bounding box in x), e.g. ``(1/3, 2/3)`` for the central
        third of the A-scans the mask actually spans."""
        cols = np.where(roi.any(axis=0))[0]
        if cols.size == 0:
            return roi
        col_min, col_max = int(cols.min()), int(cols.max())
        span = col_max - col_min + 1
        lo = col_min + int(np.floor(frac[0] * span))
        hi = col_min + int(np.ceil(frac[1] * span))
        keep = np.zeros(roi.shape[1], dtype=bool)
        keep[lo:hi] = True
        return roi & keep[None, :]

    @staticmethod
    def _restrict_row_frac_per_col(
        roi: np.ndarray, frac: tuple[float, float]
    ) -> np.ndarray:
        """Per column, keep only the rows inside ``frac`` of THAT column's own
        row extent within the ROI — not a single global row band — since the
        choroid band's row range can shift from one A-scan to the next."""
        out = np.zeros_like(roi)
        for col in np.where(roi.any(axis=0))[0]:
            rows = np.where(roi[:, col])[0]
            row_min, row_max = int(rows.min()), int(rows.max())
            span = row_max - row_min + 1
            lo = row_min + int(np.floor(frac[0] * span))
            hi = row_min + int(np.ceil(frac[1] * span))
            out[lo:hi, col] = roi[lo:hi, col]
        return out

    @staticmethod
    def _block_all(roi: np.ndarray, block: int) -> np.ndarray:
        """Downsample a boolean ``(H, W)`` map by ``block``: a coarse cell is
        True iff every fine pixel in its block is True. Trailing rows/columns
        that don't fill a whole block are dropped."""
        if block == 1:
            return roi
        h, w = roi.shape
        hc, wc = h - h % block, w - w % block
        cropped = roi[:hc, :wc]
        return cropped.reshape(hc // block, block, wc // block, block).all(
            axis=(1, 3)
        )

    @staticmethod
    def _block_median(frames: np.ndarray, block: int) -> np.ndarray:
        """Spatial median-pool each frame independently: ``(T, H, W) ->
        (T, H//block, W//block)``. Purely spatial — every frame is pooled on
        its own, no mixing across time."""
        if block == 1:
            return frames
        t, h, w = frames.shape
        hc, wc = h - h % block, w - w % block
        cropped = frames[:, :hc, :wc]
        reshaped = cropped.reshape(t, hc // block, block, wc // block, block)
        return np.median(reshaped, axis=(2, 4))

    def raw_signal(self) -> np.ndarray:
        """Concatenated multi-resolution pixel intensities: one column per
        kept pixel/super-pixel at each configured block size, shape
        ``(T, N_total)``."""
        frames = self.registered_video.registered_frames
        base_roi = self.base_roi
        cfg: PixelTraceConfig = self.config

        if cfg.normalize_intensity:
            frames = _normalize_per_frame(frames, axis=(1, 2))

        columns = []
        scale_of_trace = []
        for block in self.config.block_sizes:
            roi_b = self._block_all(base_roi, block)
            n_b = int(roi_b.sum())
            if n_b == 0:
                msg = (
                    f"Block size {block}x{block}: no super-pixel fully inside "
                    "the mask in every frame; skipped."
                )
                self.notes.append(msg)
                if self.verbose:
                    print(msg)
                continue
            pooled = self._block_median(frames, block)
            columns.append(pooled[:, roi_b].astype(float))
            scale_of_trace.append(np.full(n_b, block, dtype=int))

        if not columns:
            raise ValueError(
                "No trace survived at any configured block size "
                f"({self.config.block_sizes})."
            )
        self._scale_of_trace = np.concatenate(scale_of_trace)
        return np.concatenate(columns, axis=1)

    @property
    def scale_of_trace(self) -> np.ndarray:
        """Block size each column of ``signal``/``traces.values`` came from,
        shape ``(N_total,)``."""
        if self._scale_of_trace is None:
            _ = self.signal  # populates it as a side effect of raw_signal()
        return self._scale_of_trace

    def reset(self) -> None:
        super().reset()
        self._base_roi = None
        self._scale_of_trace = None

    def normalized_signal(self) -> None:
        """Normalize each trace to zero mean, unit variance, in-place."""
        if not self.config.normalize_intensity:
            return
        sig = self.raw_signal()
        # Same per-frame division as in `raw_signal`, across traces this time,
        # and the same blank-frame guard: a row of zeros has a zero mean.
        return _normalize_per_frame(sig, axis=1)

    @property
    def filtered_signal(self) -> np.ndarray:
        return self.raw_signal()