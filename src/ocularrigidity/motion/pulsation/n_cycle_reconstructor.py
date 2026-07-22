"""N-cycle reconstruction: fold registered frames into averaged cardiac cycles.

Deliberately decoupled from rate/phase estimation. It consumes any
``PulseExtractor`` for phase, cardiac frequency
and registered frames, and owns its own folding results — it does not write
back onto the extractor.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from ocularrigidity.motion.one_cycle import (
    _auto_n_bins,
    fold_video_numba_mean,
    fold_video_numba_median,
)
from ocularrigidity.motion.pulsation.extractor import PulseExtractor


@dataclass
class NCycleConfig:
    n_cycle: int = 1
    n_bins: Optional[int] = None
    target_frames_per_bin: int = 25
    fold_method: str = "mean"
    # Only honoured by the legacy ``MaskPulseExtractor``, which exposes both
    # phases at once; a composed ``PulseExtractor`` already is one phase method.
    phase_method: Literal["iq", "peak_locked"] = "peak_locked"
    verbose: bool = True


class NCycleReconstructor:
    def __init__(
        self,
        extractor: PulseExtractor,
        config: Optional[NCycleConfig] = None,
    ):
        self.extractor = extractor
        self.config = config or NCycleConfig()

        self.cycles: Optional[np.ndarray] = None
        self.counts: Optional[np.ndarray] = None
        self.n_bins: Optional[int] = None
        self.n_cycle: Optional[int] = None
        self.notes: list[str] = []

    def _default_phase(self):
        """Phase to fold by.

        A composed ``PulseExtractor`` already *is* one phase method, so its
        ``phase_per_frame`` is the answer. ``config.phase_method`` only applies
        to the legacy ``MaskPulseExtractor``, which exposes both phases at once.
        """
        ex = self.extractor
        if self.config.phase_method == "peak_locked" and hasattr(
            ex, "phase_per_frame_peak_locked"
        ):
            return ex.phase_per_frame_peak_locked, ex.good_per_frame_peak_locked
        return ex.phase_per_frame, ex.good_per_frame

    def compute(
        self,
        *,
        phase_per_frame=None,
        good_per_frame=None,
        cardiac_freq=None,
        n_bins: Optional[int] = None,
        n_cycle: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fold ``registered_frames`` into ``n_cycle`` averaged cardiac cycles.

        Defaults to the extractor's cached phase / good mask / cardiac_freq and
        the config's folding parameters, so re-running with a different
        ``n_bins`` does not retrigger the upstream pipeline.
        """
        ex = self.extractor
        cfg = self.config
        verbose = cfg.verbose

        ph_default, gd_default = self._default_phase()
        if phase_per_frame is None:
            phase_per_frame = ph_default
        if good_per_frame is None:
            good_per_frame = gd_default
        if cardiac_freq is None:
            cardiac_freq = ex.cardiac_freq
        if n_cycle is None:
            n_cycle = cfg.n_cycle
        if n_bins is None:
            n_bins = cfg.n_bins

        timestamps = ex.timestamps_seconds
        n_good = int(good_per_frame.sum())

        if n_bins is None:
            n_bins = _auto_n_bins(
                n_good // max(1, n_cycle),
                fs=ex.fs,
                cardiac_freq=cardiac_freq,
                target_per_bin=cfg.target_frames_per_bin,
            )
            if verbose:
                print(
                    f"Auto-selected n_bins = {n_bins} "
                    f"(per-chunk budget ~{n_good // max(1, n_cycle)} frames)"
                )

        fold_fn = (
            fold_video_numba_mean
            if cfg.fold_method == "mean"
            else fold_video_numba_median
        )

        t0 = timestamps[0]
        chunk_duration = (timestamps[-1] - t0) / n_cycle

        cycles_per_chunk: list[np.ndarray | None] = []
        counts_per_chunk: list[np.ndarray | None] = []

        for i in range(n_cycle):
            t_lo = t0 + i * chunk_duration
            t_hi = t0 + (i + 1) * chunk_duration
            if i == n_cycle - 1:
                chunk_mask = (timestamps >= t_lo) & (timestamps <= t_hi)
            else:
                chunk_mask = (timestamps >= t_lo) & (timestamps < t_hi)

            n_good_chunk = int(good_per_frame[chunk_mask].sum())
            if n_good_chunk < n_bins:
                msg = (
                    f"Chunk {i + 1}/{n_cycle} (t={t_lo:.1f}-{t_hi:.1f}s): "
                    f"{n_good_chunk} good frames < n_bins={n_bins}; skipping."
                )
                self.notes.append(msg)
                if verbose:
                    print(f"  WARNING: {msg}")
                cycles_per_chunk.append(None)
                counts_per_chunk.append(None)
                continue

            chunk_cycle, chunk_counts = fold_fn(
                ex.registered_frames[chunk_mask],
                phase_per_frame[chunk_mask],
                good_per_frame[chunk_mask],
                n_bins=n_bins,
                verbose=verbose,
            )
            cycles_per_chunk.append(chunk_cycle)
            counts_per_chunk.append(chunk_counts)

        first_ok = next((c for c in cycles_per_chunk if c is not None), None)
        if first_ok is None:
            raise RuntimeError(
                f"All {n_cycle} chunks were skipped; no chunk had ≥{n_bins} good frames."
            )
        fill_value = np.nan if np.issubdtype(first_ok.dtype, np.floating) else 0
        placeholder_cycle = np.full_like(first_ok, fill_value)
        first_counts = next(c for c in counts_per_chunk if c is not None)
        placeholder_counts = np.zeros_like(first_counts)

        cycles_per_chunk = [
            c if c is not None else placeholder_cycle for c in cycles_per_chunk
        ]
        counts_per_chunk = [
            c if c is not None else placeholder_counts for c in counts_per_chunk
        ]

        cycles = np.concatenate(cycles_per_chunk, axis=0)
        counts = np.concatenate(counts_per_chunk, axis=0)

        self.cycles = cycles
        self.counts = counts
        self.n_bins = n_bins
        self.n_cycle = n_cycle

        return cycles, counts
