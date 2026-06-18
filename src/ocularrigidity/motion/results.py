from dataclasses import dataclass
import dataclasses
from pathlib import Path
import pickle
from typing import Optional
import numpy as np

from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from ocularrigidity.motion.pulsation import CardiacCycleExtractor


@dataclass
class CardiacPipelineResults:
    # Parameters for reproducibility and traceability
    video: Path
    skip_first_n_frames: int
    drop_last_n_frames: int
    flatten: bool
    horizontal_alignment: bool
    bpm_range: tuple[float, float]
    override_cardiac_freq: Optional[float]
    expected_bpm: Optional[float]
    butter_order: int
    n_separable_components: int
    sigma_col: float
    col_slice: slice
    ls_freq_oversample: float
    ls_concentration_band_hz: float
    phase_smoother_cycles: float
    phase_density_threshold: float
    ICA_or_PCA: str
    harmonic_correction: bool
    harmonic_tolerance_bpm: float
    harmonic_min_power_ratio: float

    # Results
    registered_boundaries: np.ndarray
    timestamps_seconds: np.ndarray
    uniform_time: np.ndarray
    gap_mask: np.ndarray
    thickness: np.ndarray  # 2D
    interpolated_thickness: np.ndarray  # 2D
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

    cycles: Optional[np.ndarray] = None
    counts: Optional[np.ndarray] = None
    n_bins: Optional[int] = None
    n_cycle: Optional[int] = None

    @classmethod
    def from_extractor(cls, ex: "CardiacCycleExtractor") -> "CardiacPipelineResults":
        return cls(
            video=ex.registrator.video,
            skip_first_n_frames=ex.skip_first_n_frames,
            drop_last_n_frames=ex.drop_last_n_frames,
            flatten=ex.registrator.flatten,
            horizontal_alignment=ex.registrator.horizontal_alignment,
            bpm_range=ex.bpm_range,
            override_cardiac_freq=ex.cardiac_freq,
            expected_bpm=ex.expected_bpm,
            butter_order=ex.butter_order,
            n_separable_components=ex.n_separable_components,
            sigma_col=ex.sigma_col,
            col_slice=ex.col_slice,
            ls_freq_oversample=ex.ls_freq_oversample,
            ls_concentration_band_hz=ex.ls_concentration_band_hz,
            phase_smoother_cycles=ex.phase_smoother_cycles,
            phase_density_threshold=ex.phase_density_threshold,
            ICA_or_PCA=ex.ICA_or_PCA,
            harmonic_correction=ex.harmonic_correction,
            harmonic_tolerance_bpm=ex.harmonic_tolerance_bpm,
            harmonic_min_power_ratio=ex.harmonic_min_power_ratio,
            registered_boundaries=ex.registrator.registered_lines,
            timestamps_seconds=ex.timestamps_seconds,
            uniform_time=ex.uniform_time,
            gap_mask=ex.gap_mask,
            thickness=ex.thickness,
            interpolated_thickness=ex.interpolated_thickness,
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
            notes=ex.notes,
            cycles=ex.cycles,
            counts=ex.counts,
            n_bins=ex.n_bins,
            n_cycle=ex.n_cycle,
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
