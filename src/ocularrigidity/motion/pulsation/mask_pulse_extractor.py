"""Mask-based pulse extraction: the signal is retinal thickness per A-scan."""

import numpy as np

from ocularrigidity.motion.pulsation.abstract_pulse_extractor import (
    AbstractPulseExtractor,
)


class MaskPulseExtractor(AbstractPulseExtractor):
    @property
    def signal(self) -> np.ndarray:
        """Thickness restricted to ``col_slice``, with holes (0/NaN) and
        outlier frames → NaN.

        This is the mask-domain realisation of ``AbstractPulseExtractor.signal``;
        everything downstream (interpolation, filtering, ICA, phase) is shared.
        """
        if self._signal is not None:
            return self._signal

        col_slice = self.config.col_slice
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
            if self.verbose:
                print("All frames contain holes; thickness fully masked.")
            thickness[:] = np.nan
            self._signal = thickness
            return self._signal

        med = np.nanmedian(clean)
        frame_mean = np.nanmean(thickness, axis=1)
        bad_frames = (frame_mean < 0.75 * med) | (frame_mean > 1.25 * med)

        n_bad = int(bad_frames.sum())
        if n_bad and self.verbose:
            print(
                f"Marking {n_bad}/{len(bad_frames)} frames as bad based on "
                f"outlier thickness (median={med:.1f})"
            )
        thickness[bad_frames] = np.nan
        self._signal = thickness
        return self._signal
