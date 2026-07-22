"""Segmented retinal thickness as one trace per A-scan."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ocularrigidity.motion.pulsation.traces.base import (
    AbstractUniformTraceSource,
    UniformTraceConfig,
)
from ocularrigidity.motion.video_timeline_aligner import VideoTimelineAligner


@dataclass
class MaskTraceConfig(UniformTraceConfig):
    """Extra knobs specific to the segmented-thickness source."""

    col_slice: Optional[slice] = None
    # A frame whose mean thickness deviates by more than this fraction of the
    # video median is treated as a bad frame.
    outlier_thickness_frac: float = 0.25


class MaskThicknessTraceSource(AbstractUniformTraceSource):
    """One trace per A-scan: segmented retinal thickness over time."""

    def __init__(
        self,
        registered_video,
        aligner: VideoTimelineAligner,
        config: Optional[MaskTraceConfig] = None,
    ):
        super().__init__(aligner, config or MaskTraceConfig())
        self.registered_video = registered_video

    def raw_signal(self) -> np.ndarray:
        """Thickness restricted to ``col_slice``, with holes (0/NaN) and outlier
        frames → NaN."""
        cfg: MaskTraceConfig = self.config
        col_slice = cfg.col_slice
        src = self.registered_video.thickness
        thickness = (src[:, col_slice] if col_slice is not None else src).copy()

        # Trim fully-invalid border columns (a hole — 0 or NaN — in every frame),
        # then unify hole-marking on NaN so all downstream validity checks (which
        # key on isnan) catch degenerate boundaries.
        x_valid = np.where(~np.isnan(thickness) & (thickness != 0))[1]
        slices = slice(np.min(x_valid), np.max(x_valid) + 1)
        thickness = thickness[:, slices]
        thickness[thickness == 0] = np.nan

        has_holes = np.isnan(thickness).any(axis=1)
        clean = thickness[~has_holes]
        if clean.size == 0:
            msg = "All frames contain holes; thickness fully masked."
            self.notes.append(msg)
            if self.verbose:
                print(msg)
            thickness[:] = np.nan
            return thickness

        med = np.nanmedian(clean)
        frame_mean = np.nanmean(thickness, axis=1)
        tol = cfg.outlier_thickness_frac
        bad_frames = (frame_mean < (1 - tol) * med) | (frame_mean > (1 + tol) * med)

        n_bad = int(bad_frames.sum())
        if n_bad and self.verbose:
            print(
                f"Marking {n_bad}/{len(bad_frames)} frames as bad based on "
                f"outlier thickness (median={med:.1f})"
            )
        thickness[bad_frames] = np.nan
        return thickness
