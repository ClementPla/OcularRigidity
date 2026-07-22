"""Study-level configuration: what settings *this* cohort is processed with.

The distinction against the per-component configs (``RegistrationConfig``,
``PulseExtractionConfig``, ``NCycleConfig``, …) is deliberate:

- A **component** config is the argument list of one class. It lives next to
  that class, and its defaults say what the algorithm does by default.
- A **study** config is what this pipeline decided to run. It lives here, is
  frozen, and is exposed as a singleton the batch scripts import.

Study configs therefore *hold* component configs rather than mirroring their
fields — a mirrored field is a default that can silently drift away from the
one the library actually uses. What legitimately belongs here on its own is
whatever the components know nothing about: cross-stage invariants
(``N_CYCLES``), rig calibration (``AXIAL_PIXEL_SIZE_MM``), sweep axes and
output-file metadata.
"""

from dataclasses import dataclass, field, replace
from typing import Literal, Optional

from ocularrigidity.consts import AXIAL_PIXEL_SIZE_MM
from ocularrigidity.motion.pulsation import NCycleConfig, PulseExtractionConfig
from ocularrigidity.registration.config import RegistrationConfig

# Number of cardiac cycles that are folded, segmented and measured. Shared by
# the pulsation fold, the per-cycle deltaY fit and the deltaA tracking — they
# must agree, so they all derive from this single value.
N_CYCLES = 3

# ``AXIAL_PIXEL_SIZE_MM`` is re-exported from consts.py, where library code can
# reach it without importing this (study-level) module.

__all__ = [
    "N_CYCLES",
    "AXIAL_PIXEL_SIZE_MM",
    "RegistrationConfig",
    "PulsationConfig",
    "DeltaYConfig",
    "SegmentationConfig",
    "DeltaAConfig",
    "FriedenwaldConfig",
    "MisregistrationConfig",
    "REGISTRATION",
    "PULSATION",
    "DELTA_Y",
    "SEGMENTATION",
    "DELTA_A",
    "FRIEDENWALD",
    "MISREGISTRATION",
]


@dataclass(frozen=True)
class PulsationConfig:
    """Cardiac-cycle extraction + folding (pulsation/infer.py).

    ``extraction`` and ``fold`` are the component configs handed straight to
    :func:`run_cardiac_pipeline`; use :meth:`for_video` to stamp in the
    per-video and per-sweep values. The remaining fields are the ones no
    component owns: which variants to sweep, and the output video's fps.

    ``extraction`` is the flat :class:`PulseExtractionConfig` because that is
    what the pipeline entry point still takes. When ``run_cardiac_pipeline``
    moves off the legacy ``MaskPulseExtractor`` facade, this field becomes the
    per-stage configs (``MaskTraceConfig``, ``DecompositionConfig``, …) and
    nothing else here has to change.
    """

    # Values equal to the library defaults are still spelled out: a study
    # config should pin what it ran, so a later change to a library default
    # cannot silently change this cohort's settings.
    extraction: PulseExtractionConfig = field(
        default_factory=lambda: PulseExtractionConfig(
            sigma_col=5.0,
            expected_bpm_band_frac=0.3,
            col_slice=slice(100, 924),
        )
    )
    fold: NCycleConfig = field(
        default_factory=lambda: NCycleConfig(
            n_bins=30,
            n_cycle=N_CYCLES,
            fold_method="median",
        )
    )

    # --- Sweep axes: one run per combination (pulsation/infer.py) ---------
    methods: tuple[str, ...] = ("pca", "ica")
    phase_methods: tuple[str, ...] = ("peak_locked", "iq")
    # fps written into the lossless one_cycle.mkv (display metadata only).
    output_fps: int = 30

    def for_video(
        self,
        *,
        method: Optional[str] = None,
        phase_method: Optional[str] = None,
        expected_bpm: Optional[float] = None,
        verbose: bool = True,
    ) -> tuple[PulseExtractionConfig, NCycleConfig]:
        """The study settings, specialised for one video / one sweep point.

        ``expected_bpm`` is the measured heart rate, which is per-video and so
        cannot live in a cohort-wide config.
        """
        extraction = replace(
            self.extraction,
            expected_bpm=expected_bpm,
            verbose=verbose,
            **({"ICA_or_PCA": method} if method is not None else {}),
        )
        fold = replace(
            self.fold,
            verbose=verbose,
            **({"phase_method": phase_method} if phase_method is not None else {}),
        )
        return extraction, fold


@dataclass(frozen=True)
class DeltaYConfig:
    """Choroid segmentation + cardiac-amplitude (deltaY) fit on one_cycle.mkv."""

    batch_size: int = 32
    n_cycles: int = N_CYCLES
    n_harmonics: int = 1
    residual_threshold_percentile: float = 75.0
    amplitude_threshold_percentile: float = 50.0
    graphcut_kwargs: dict = field(
        default_factory=lambda: dict(
            temporal_smooth=False,
            temporal_iterations=4,
            temporal_mu=1.0,
            temporal_sigma=2.0,
            lambda_smooth=1.0,
        )
    )


@dataclass(frozen=True)
class SegmentationConfig:
    """Per-cycle segmentation pass (cohort_analysis/segment_n_cycles.py)."""

    batch_size: int = 16


@dataclass(frozen=True)
class DeltaAConfig:
    """Boundary displacement / area-change extraction (extract_deltaA.py)."""

    n_cycles: int = N_CYCLES
    method: Literal["optical_flow", "demons"] = "optical_flow"
    smooth_window: int = 11
    lk_window: int = 35


@dataclass(frozen=True)
class FriedenwaldConfig:
    """Spherical-shell geometry + pressures for the Friedenwald rigidity (K).

    Bridges the per-cycle area change produced by extract_deltaA.py
    (``deltaA``, in px²) to a pulsatile choroidal volume change (µL) and the
    Friedenwald coefficient K. These are rig-specific calibration values — set
    them for the OCT used to acquire the cohort.
    """

    # OCT axial scale (mm per pixel). Axial-length independent.
    s_axial_mm_per_px: float = AXIAL_PIXEL_SIZE_MM
    # Lateral width (px) the choroid area is integrated over, i.e. the trimmed
    # mask width fed to the shoelace area. Must match the real segmentation
    # geometry for dV (and hence K) to be unbiased.
    w_px: float = 1024 - (75 * 2)
    # Fraction of the sphere the choroid spans (1.0=full; ~0.7 posterior pole).
    surface_coverage: float = 2 / 3
    # Vitreous-chamber fraction of axial length (choroid-vitreous interface).
    vitreous_chamber_frac: float = 0.83
    # Pressure convention: 'diastolic' (DCT) or 'mean' (Goldmann).
    pressure_mode: Literal["diastolic", "mean"] = "diastolic"


@dataclass(frozen=True)
class MisregistrationConfig:
    """QC thresholds for flagging mis-registered cycles (flag_misregistration.py)."""

    bm_jitter_p95_px: float = 3.0
    jump_px: float = 5.0
    max_jump_frac: float = 0.05
    min_frame_coverage: float = 0.60
    max_frac_empty_columns: float = 0.20
    max_empty_frames: int = 0


# Singletons imported by the pipeline scripts.
REGISTRATION = RegistrationConfig()
PULSATION = PulsationConfig()
DELTA_Y = DeltaYConfig()
SEGMENTATION = SegmentationConfig()
DELTA_A = DeltaAConfig()
FRIEDENWALD = FriedenwaldConfig()
MISREGISTRATION = MisregistrationConfig()
