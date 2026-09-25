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

    col_slice: slice | None = None
    # A frame whose mean thickness deviates by more than this fraction of the
    # video median is treated as a bad frame.
    outlier_thickness_frac: float = 0.25
    # Columns that are a hole (0/NaN) in at least this fraction of frames are
    # dropped, wherever they sit — e.g. the optic nerve head, which leaves a
    # hole in the middle of every frame. 1.0 drops only always-invalid columns;
    # lower it if the ONH boundary jitters from frame to frame.
    max_column_hole_frac: float = 1.0


class MaskThicknessTraceSource(AbstractUniformTraceSource):
    """One trace per A-scan: segmented retinal thickness over time."""

    def __init__(
        self,
        registered_video,
        aligner: VideoTimelineAligner,
        config: MaskTraceConfig | None = None,
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

        # Unify hole-marking on NaN so all downstream validity checks (which key
        # on isnan) catch degenerate boundaries, then drop persistently-invalid
        # columns — border trim and interior holes (ONH) alike.
        thickness[thickness == 0] = np.nan
        hole_frac = np.isnan(thickness).mean(axis=0)
        keep_cols = hole_frac < cfg.max_column_hole_frac
        # Indices into the (col_sliced) source, to map traces back to A-scans.
        self.kept_columns = np.flatnonzero(keep_cols)
        n_dropped = int((~keep_cols).sum())
        if n_dropped and self.verbose:
            print(
                f"Dropping {n_dropped}/{len(keep_cols)} columns with holes in "
                f">= {cfg.max_column_hole_frac:.0%} of frames"
            )
        thickness = thickness[:, keep_cols]

        has_holes = np.isnan(thickness).any(axis=1)
        clean = thickness[~has_holes]
        if clean.size == 0:
            msg = (
                "All frames contain holes; thickness fully masked. If a hole "
                "is expected (e.g. ONH), lower `max_column_hole_frac`."
            )
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
