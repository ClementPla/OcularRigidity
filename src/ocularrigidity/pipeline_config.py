from dataclasses import dataclass, field
from typing import Literal

# Number of cardiac cycles that are folded, segmented and measured. Shared by
# the pulsation fold, the per-cycle deltaY fit and the deltaA tracking — they
# must agree, so they all derive from this single value.
N_CYCLES = 1


@dataclass(frozen=True)
class RegistrationConfig:
    """Parameters that form the registration cache key.

    Consumed by ``RegisteredVideo`` directly (registration/infer.py) and, via
    ``run_cardiac_pipeline``, by pulsation/infer.py. Every field except
    ``batch_size`` participates in the cache key (see
    ``RegisteredVideo._cache_meta``); changing any of them invalidates the cache
    for ALL downstream stages, so change deliberately and regenerate.
    """

    skip_first_n_frames: int = 0
    drop_last_n_frames: int = 0
    flatten: bool = False
    horizontal_alignment: bool = True
    lateral_method: Literal["xcorr", "fullframe", "both"] = "fullframe"
    subpixel: bool = False
    use_encoded_video: bool = True
    # Fraction centrale de la LARGEUR conservee avant la FFT dans le recalage
    # lateral fullframe (estimate_lateral_shift_fullframe), pour eviter les
    # artefacts de bord ; la hauteur n'est pas rognee. Sans effet si
    # lateral_method == "xcorr".
    crop_w_x: float = 0.66
    # Bornes basse/haute (fraction de la freq. de Nyquist) du passe-bande
    # spectral applique dans le recalage lateral fullframe
    # (estimate_lateral_shift_fullframe). Sans effet si lateral_method == "xcorr".
    bp_lo: float = 0.03
    bp_hi: float = 0.2
    # 2e passe (RPE) : recalage axial de chaque A-scan sur la mediane du volume
    # deja recale (compensation d'ombres + LoG + correlation de phase par colonne).
    # Desactive par defaut ; ne modifie ni le cache ni le comportement existant
    # tant qu'il n'est pas active. Voir registration/axial/median_registration.py.
    median_registration: bool = False
    median_max_vshift: int = 30
    # Compensation d'ombres et LoG sont decouples : activables independamment.
    median_use_shadow: bool = False
    median_use_log: bool = False
    median_shadow_n: float = 4.0
    median_shadow_a: float = 0.8
    median_log_kernel_size: int = 9
    median_log_sigma: float = 3.0
    # Not part of the cache key — only affects throughput on a cache miss.
    batch_size: int = 128


@dataclass(frozen=True)
class PulsationConfig:
    """Cardiac-cycle extraction + folding (pulsation/infer.py)."""

    sigma_col: float = 5.0
    expected_bpm_band_frac: float = 0.3
    n_bins: int = 30
    col_slice: slice = field(default_factory=lambda: slice(100, 924))
    one_cycle_fold_method: Literal["mean", "median"] = "median"
    n_cycle: int = N_CYCLES
    methods: tuple[str, ...] = ("pca", "ica", "svd")
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
    s_axial_mm_per_px: float = 1.95e-3
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
