from dataclasses import dataclass, field
from typing import Literal

# Number of cardiac cycles that are folded, segmented and measured. Shared by
# the pulsation fold, the per-cycle deltaY fit and the deltaA tracking — they
# must agree, so they all derive from this single value.
N_CYCLES = 3
AXIAL_PIXEL_SIZE_MM = 1.95e-3  # mm per pixel, axial scale of the OCT


@dataclass(frozen=True)
class RegistrationConfig:
    skip_first_n_frames: int = 0
    drop_last_n_frames: int = 0
    use_encoded_video: bool = False

    # What to correct.
    correct_transversal: bool = False
    correct_axial: bool = True
    flatten_rpe: bool = False
    axial_refinement: bool = True
    fovea_correction_enabled: bool = True

    # Transversal (x) parameters.
    lateral_method: Literal["xcorr", "fullframe", "both"] = "fullframe"
    max_lateral_shift: int = 2
    smooth_transversal: bool = False
    smooth_transversal_sigma: float = 0.1
    crop_factor: float = (
        0.05  # fraction of the frame width to keep for lateral registration
    )
    scale_factor: float = 0.1  # downscale factor for lateral registration
    transversal_bandpass: tuple[float, float] = (0.03, 0.3)
    axial_bandpass: tuple[float, float] = (0.03, 0.2)
    # Axial (y) parameters. ``max_axial_shift`` is the RPE-refinement pass's
    # maximal tested vertical shift (px).
    max_axial_shift: int = 2

    # General.
    subpixel: bool = False
    batch_size: int = 1


@dataclass(frozen=True)
class PulsationConfig:
    """Cardiac-cycle extraction + folding (pulsation/infer.py)."""

    sigma_col: float = 5.0
    expected_bpm_band_frac: float = 0.3
    n_bins: int = 30
    col_slice: slice = field(default_factory=lambda: slice(100, 924))
    one_cycle_fold_method: Literal["mean", "median"] = "median"
    n_cycle: int = N_CYCLES
    methods: tuple[str, ...] = ("pca", "ica")
    phase_methods: tuple[str, ...] = ("peak_locked", "iq")
    # fps written into the lossless one_cycle.mkv (display metadata only).
    output_fps: int = 30


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
