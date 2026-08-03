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

from ocularrigidity.consts import (
    AXIAL_PIXEL_SIZE_MM,
    SEGMENTATION_BATCH_SIZE,
    TRANVERSAL_PIXEL_SIZE_MM,
)
from ocularrigidity.motion.pulsation import (
    BandPassFilterTraceConfig,
    CardiacBand,
    DecompositionConfig,
    IQPhaseConfig,
    LombScargleConfig,
    MaskTraceConfig,
    NCycleConfig,
    PulseExtractionConfig,
)
from ocularrigidity.motion.pulsation.traces.coherence import CoherenceConfig
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
    "ChainConfig",
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
class ChainConfig:
    """The composed trace → rate → phase chain, as run in notebooks/pipeline/test.ipynb.

    This is the recipe that replaced the flat :class:`PulseExtractionConfig`
    recipe. The differences that matter, established by comparing against the
    July-2026 cohort run:

    * ``SelectBestComponent`` for phase aggregation — averaging the PCA
      components instead buries the cardiac one under the rest;
    * ``sigma_col=5.0`` and ``col_slice`` trimming the noisy B-scan edges;
    * an explicit bandpass stage, which the mask source no longer does itself.

    ``band`` is per-video (it is anchored on the measured HR), so it lives in
    :meth:`for_video` rather than in the frozen fields below.
    """

    trace: MaskTraceConfig = field(
        default_factory=lambda: MaskTraceConfig(col_slice=slice(100, 924))
    )
    bandpass: BandPassFilterTraceConfig = field(
        default_factory=lambda: BandPassFilterTraceConfig(sigma_col=5.0)
    )
    coherence: CoherenceConfig = field(
        default_factory=lambda: CoherenceConfig(
            selection="quantile", keep_quantile=0.5
        )
    )
    decomposition: DecompositionConfig = field(
        default_factory=lambda: DecompositionConfig(
            method="PCA", n_components=64, random_state=0, whiten=False
        )
    )
    rate: LombScargleConfig = field(
        default_factory=lambda: LombScargleConfig(concentration_band_hz=0.1)
    )
    phase: IQPhaseConfig = field(
        default_factory=lambda: IQPhaseConfig(
            smoother_cycles=2.0,
            density_threshold=0.5,
            # The instantaneous-frequency gate is disabled: it trims good frames
            # out of the fold, and the July-2026 reference run predates it.
            freq_tolerance=1e9,
        )
    )

    def for_video(
        self, *, expected_bpm: Optional[float] = None, verbose: bool = True
    ) -> dict:
        """The stage configs, with the measured HR stamped into the band.

        One band object, shared by the trace bandpass and the periodogram, so
        the two stages cannot disagree about what counts as cardiac.
        """
        band = CardiacBand(expected_bpm=expected_bpm)
        return dict(
            trace=replace(self.trace, verbose=verbose),
            bandpass=replace(self.bandpass, band=band, verbose=verbose),
            coherence=replace(self.coherence, verbose=verbose),
            decomposition=self.decomposition,
            rate=replace(self.rate, band=band, verbose=verbose),
            phase=self.phase,
        )


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
    # The composed chain, which is what the cohort scripts now run.
    # ``extraction`` above is the superseded flat recipe, kept because old
    # ``measure.pkl`` files carry it as provenance.
    chain: ChainConfig = field(default_factory=ChainConfig)
    fold: NCycleConfig = field(
        default_factory=lambda: NCycleConfig(
            n_bins=30,
            n_cycle=N_CYCLES,
            fold_method="median",
            # The composed chain already *is* one phase method; this only tells
            # the reconstructor not to go looking for the legacy peak-locked one.
            phase_method="iq",
        )
    )

    # --- Sweep axes: one run per combination (pulsation/infer.py) ---------
    methods: tuple[str, ...] = ("pca",)
    phase_methods: tuple[str, ...] = ("iq",)
    # fps written into the lossless one_cycle.mkv (display metadata only).
    output_fps: int = 30

    def chain_for_video(
        self, *, expected_bpm: Optional[float] = None, verbose: bool = True
    ) -> tuple[dict, NCycleConfig]:
        """The composed chain's stage configs plus the fold config, for one video."""
        return (
            self.chain.for_video(expected_bpm=expected_bpm, verbose=verbose),
            replace(self.fold, verbose=verbose),
        )

    def for_video(
        self,
        *,
        method: Optional[str] = None,
        phase_method: Optional[str] = None,
        expected_bpm: Optional[float] = None,
        verbose: bool = True,
    ) -> tuple[PulseExtractionConfig, NCycleConfig]:
        """The *legacy* flat settings, specialised for one video / sweep point.

        ``expected_bpm`` is the measured heart rate, which is per-video and so
        cannot live in a cohort-wide config. New code wants
        :meth:`chain_for_video`.
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

    batch_size: int = SEGMENTATION_BATCH_SIZE
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
    """Segmentation passes (raw videos and folded cycles).

    ``batch_size`` is a memory/throughput knob, not a study decision — override
    it with OCULARRIGIDITY_SEGMENTATION_BATCH rather than editing this.
    """

    batch_size: int = SEGMENTATION_BATCH_SIZE


@dataclass(frozen=True)
class DeltaAConfig:
    """Boundary displacement / area-change extraction (extract_deltaA.py)."""

    n_cycles: int = N_CYCLES
    method: Literal["optical_flow", "demons"] = "optical_flow"
    smooth_window: int = 15
    lk_window: int = 35
    csi_normal_smooth_sigma: float = 0.0


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
    pressure_mode: Literal["diastolic", "mean"] = "mean"


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
