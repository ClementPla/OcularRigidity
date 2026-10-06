from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
import pickle
import pickletools
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from ocularrigidity.data.compression import cube_to_mkv_lossless
from ocularrigidity.registration.config import RegistrationConfig

if TYPE_CHECKING:
    from ocularrigidity.motion.pulsation import (
        NCycleConfig,
        NCycleReconstructor,
        PulseExtractor,
    )
    from ocularrigidity.motion.pulsation.phase import PhaseTrack


def _find_stage(source, attr: str):
    """Walk a wrapped trace-source chain for the stage carrying ``attr``."""
    seen = set()
    while source is not None and id(source) not in seen:
        seen.add(id(source))
        if hasattr(source, attr):
            return source
        source = getattr(source, "source", None)
    return None


class _SkipLargePayloads:
    """File wrapper that seeks past big pickle payloads instead of reading them."""

    def __init__(self, f, threshold: int = 1 << 16):
        self._f = f
        self._threshold = threshold

    def read(self, n: int = -1) -> bytes:
        if n is not None and n > self._threshold:
            self._f.seek(n, 1)
            return bytes(n)
        return self._f.read(n)

    def readline(self) -> bytes:
        return self._f.readline()

    def tell(self) -> int:
        return self._f.tell()


def peek_cardiac_freq(path: Path) -> Optional[float]:
    """Read only ``cardiac_freq`` out of a pickled CardiacPipelineResults."""
    try:
        with open(path, "rb") as fh:
            found_key = False
            for op, arg, _pos in pickletools.genops(_SkipLargePayloads(fh)):
                if found_key:
                    # The key string is memoised before its value is
                    # emitted, and protocol 4+ may open a frame in
                    # between.
                    if op.name in ("MEMOIZE", "FRAME", "PUT", "BINPUT", "LONG_BINPUT"):
                        continue
                    if op.name in ("BINFLOAT", "FLOAT"):
                        return arg
                    return None
                # 'override_cardiac_freq' is a different key and does not match.
                if arg == "cardiac_freq":
                    found_key = True
    except Exception:
        return None
    return None


@dataclass
class CardiacPipelineResults:
    # --- Provenance ------------------------------------------------------
    video: Path
    config: dict  # the per-stage config dict; provenance for this chain
    fold_config: Optional[NCycleConfig]
    skip_first_n_frames: int
    drop_last_n_frames: int
    flatten_rpe: bool
    correct_transversal: bool
    bpm_range: tuple[float, float]
    override_cardiac_freq: Optional[float]

    # --- Results (signal-generic naming) --------------------------------
    registered_boundaries: np.ndarray
    timestamps_seconds: np.ndarray
    uniform_time: np.ndarray
    gap_mask: np.ndarray
    signal: np.ndarray
    interpolated_signal: np.ndarray
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

    # --- Registration provenance, in full ---------------------------------
    registration: Optional[RegistrationConfig] = None

    # Convenience accessors so viewers written against the extractor work against results too.
    @property
    def expected_bpm(self):
        """The HR the band was anchored on, whichever chain produced this."""
        if isinstance(self.config, dict):  # composed chain: per-stage configs
            rate = self.config.get("rate")
            return rate.band.expected_bpm if rate is not None else None
        return self.config.expected_bpm

    @property
    def component_kept_mask(self):
        return ~np.isnan(self.filtered_signal).any(axis=1)

    @property
    def phase_track(self) -> "PhaseTrack":
        """The stored phase, rebuilt as the object the estimators hand back."""
        from ocularrigidity.motion.pulsation.phase import PhaseTrack

        return PhaseTrack(
            phase_uniform=self.phase_uniform,
            good_uniform=self.good_uniform,
            phase_per_frame=self.phase_per_frame,
            good_per_frame=self.good_per_frame,
            freq=self.cardiac_freq,
            uniform_time=self.uniform_time,
        )

    @property
    def inst_bpm(self) -> np.ndarray:
        """Instantaneous BPM on the uniform grid, NaN outside good runs."""
        return self.phase_track.inst_bpm

    @property
    def cardiac_bpm(self) -> float:
        return self.cardiac_freq * 60.0

    @classmethod
    def from_composed(
        cls,
        extractor: "PulseExtractor",
        reconstructor: "Optional[NCycleReconstructor]" = None,
        *,
        stage_configs: Optional[dict] = None,
    ) -> "CardiacPipelineResults":
        """Package a composed :class:`PulseExtractor` (the test.ipynb chain)."""
        ex = extractor
        rec = reconstructor
        reg = ex.registered_video

        filter_stage = _find_stage(ex.trace_source, "filtered_signal")
        uniform_stage = _find_stage(ex.trace_source, "interpolated_signal")
        rate = ex.rate

        n_uniform = len(ex.uniform_time)
        n_frames = len(ex.timestamps_seconds)
        nan_uniform = np.full(n_uniform, np.nan)
        nan_frames = np.full(n_frames, np.nan)

        notes = list(ex.notes) + (list(rec.notes) if rec is not None else [])
        return cls(
            video=reg.video,
            config=stage_configs,
            fold_config=rec.config if rec is not None else None,
            skip_first_n_frames=reg.skip_first_n_frames,
            drop_last_n_frames=reg.drop_last_n_frames,
            flatten_rpe=reg.flatten_rpe,
            correct_transversal=reg.correct_transversal,
            registration=getattr(reg, "config", None),
            bpm_range=(
                stage_configs["rate"].band.effective_bpm_range
                if stage_configs and "rate" in stage_configs
                else (np.nan, np.nan)
            ),
            override_cardiac_freq=ex._freq_override,
            registered_boundaries=reg.registered_lines,
            timestamps_seconds=ex.timestamps_seconds,
            uniform_time=ex.uniform_time,
            gap_mask=ex.gap_mask,
            signal=uniform_stage.signal if uniform_stage is not None else None,
            interpolated_signal=(
                uniform_stage.interpolated_signal if uniform_stage is not None else None
            ),
            filtered_signal=(
                filter_stage.filtered_signal if filter_stage is not None else None
            ),
            separable_components=ex.traces.values,
            ica_mixing=ex.traces.mixing,
            lomb_scargle_results=(rate.diagnostics if rate is not None else {}),
            best_component_idx=rate.best_index if rate is not None else None,
            cardiac_freq=ex.cardiac_freq,
            phase_uniform=ex.phase_uniform,
            good_uniform=ex.good_uniform,
            phase_per_frame=ex.phase_per_frame,
            good_per_frame=ex.good_per_frame,
            phase_uniform_peak_locked=nan_uniform,
            good_uniform_peak_locked=np.zeros(n_uniform, dtype=bool),
            phase_per_frame_peak_locked=nan_frames,
            good_per_frame_peak_locked=np.zeros(n_frames, dtype=bool),
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


def write_fold_outputs(
    result: CardiacPipelineResults,
    one_cycle_path: Path,
    measure_path: Path,
    fps: int,
) -> None:
    """Lossless cycle video + measures pickle."""
    one_cycle_path.parent.mkdir(parents=True, exist_ok=True)
    cube_to_mkv_lossless(result.cycles, str(one_cycle_path), fps=fps)
    measure_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(measure_path, include_cycles=False)
