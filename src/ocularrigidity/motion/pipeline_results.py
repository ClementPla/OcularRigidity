import dataclasses
from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from ocularrigidity.motion.pulsation import NCycleConfig, PulseExtractionConfig

if TYPE_CHECKING:
    from ocularrigidity.motion.pulsation import (
        MaskPulseExtractor,
        NCycleReconstructor,
    )


@dataclass
class CardiacPipelineResults:
    # --- Provenance ------------------------------------------------------
    video: Path
    config: PulseExtractionConfig
    fold_config: Optional[NCycleConfig]
    # Registration provenance (the registrator is not pickled with the result)
    skip_first_n_frames: int
    drop_last_n_frames: int
    flatten_rpe: bool
    correct_transversal: bool
    # Effective search band after ``expected_bpm`` anchoring, and the frequency
    # override actually in force (if any).
    bpm_range: tuple[float, float]
    override_cardiac_freq: Optional[float]

    # --- Results (signal-generic naming) --------------------------------
    registered_boundaries: np.ndarray
    timestamps_seconds: np.ndarray
    uniform_time: np.ndarray
    gap_mask: np.ndarray
    signal: np.ndarray  # 2D (was `thickness`)
    interpolated_signal: np.ndarray  # 2D (was `interpolated_thickness`)
    filtered_signal: np.ndarray  # 2D
    separable_components: np.ndarray  # 2D
    ica_mixing: np.ndarray  # 2D
    lomb_scargle_results: dict
    best_component_idx: int
    cardiac_freq: float
    phase_uniform: np.ndarray
    good_uniform: np.ndarray
    phase_per_frame: np.ndarray
    good_per_frame: np.ndarray
    phase_uniform_peak_locked: np.ndarray
    good_uniform_peak_locked: np.ndarray
    phase_per_frame_peak_locked: np.ndarray
    good_per_frame_peak_locked: np.ndarray
    confidence: str
    notes: list[str]

    # --- Folding (from NCycleReconstructor) -----------------------------
    cycles: Optional[np.ndarray] = None
    counts: Optional[np.ndarray] = None
    n_bins: Optional[int] = None
    n_cycle: Optional[int] = None

    # Convenience accessors so viewers written against the extractor work
    # against results too.
    @property
    def expected_bpm(self):
        return self.config.expected_bpm

    @property
    def component_kept_mask(self):
        return ~np.isnan(self.filtered_signal).any(axis=1)

    @classmethod
    def from_objects(
        cls,
        extractor: "MaskPulseExtractor",
        reconstructor: "Optional[NCycleReconstructor]" = None,
    ) -> "CardiacPipelineResults":
        ex = extractor
        reg = ex.registered_video
        rec = reconstructor
        notes = list(ex.notes) + (list(rec.notes) if rec is not None else [])
        return cls(
            video=reg.video,
            config=ex.config,
            fold_config=rec.config if rec is not None else None,
            skip_first_n_frames=reg.skip_first_n_frames,
            drop_last_n_frames=reg.drop_last_n_frames,
            flatten_rpe=reg.flatten_rpe,
            correct_transversal=reg.correct_transversal,
            bpm_range=ex.bpm_range,
            override_cardiac_freq=ex.cardiac_freq if ex._is_freq_overridden else None,
            registered_boundaries=reg.registered_lines,
            timestamps_seconds=ex.timestamps_seconds,
            uniform_time=ex.uniform_time,
            gap_mask=ex.gap_mask,
            signal=ex.signal,
            interpolated_signal=ex.interpolated_signal,
            filtered_signal=ex.filtered_signal,
            separable_components=ex.separable_components,
            ica_mixing=ex.ica_mixing,
            lomb_scargle_results=ex.lomb_scargle_results,
            best_component_idx=ex.best_component_idx,
            cardiac_freq=ex.cardiac_freq,
            phase_uniform=ex.phase_uniform,
            good_uniform=ex.good_uniform,
            phase_per_frame=ex.phase_per_frame,
            good_per_frame=ex.good_per_frame,
            phase_uniform_peak_locked=ex.phase_uniform_peak_locked,
            good_uniform_peak_locked=ex.good_uniform_peak_locked,
            phase_per_frame_peak_locked=ex.phase_per_frame_peak_locked,
            good_per_frame_peak_locked=ex.good_per_frame_peak_locked,
            confidence=ex.confidence,
            notes=notes,
            cycles=rec.cycles if rec is not None else None,
            counts=rec.counts if rec is not None else None,
            n_bins=rec.n_bins if rec is not None else None,
            n_cycle=rec.n_cycle if rec is not None else None,
        )

    def __post_init__(self):
        if isinstance(self.video, str):
            self.video = Path(self.video)

        if isinstance(self.registered_boundaries, torch.Tensor):
            self.registered_boundaries = self.registered_boundaries.cpu().numpy()

    def save(self, path: Path, include_cycles: bool = True) -> None:
        if include_cycles:
            obj = self
        else:
            obj = dataclasses.replace(self, cycles=None, counts=None)
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "CardiacPipelineResults":
        with open(path, "rb") as f:
            return pickle.load(f)
