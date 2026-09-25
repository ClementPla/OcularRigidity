"""Study-level configuration: what settings *this* cohort is processed with.

The distinction against the per-component configs (``RegistrationConfig``,
``RegistrationConfig``, ``NCycleConfig``, …) is deliberate:

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
from pathlib import Path
from typing import Literal, Optional

from ocularrigidity.consts import (
    ANCHOR_PHASE,
    AXIAL_PIXEL_SIZE_MM,
    SINC_CHECKPOINT,
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
)
from ocularrigidity.motion.pulsation.phase.anchoring import ThicknessAnchorConfig
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

    The differences that matter, established by comparing against the
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
        default_factory=lambda: CoherenceConfig(selection="quantile", keep_quantile=0.5)
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
            freq_tolerance=1e9,
        )
    )

    # Set, the rate and phase come from this SiNC checkpoint's waveform rather
    # than from the thickness traces (see consts.SINC_CHECKPOINT).
    sinc_checkpoint: Optional[Path] = None
    # Set, phase 0 is moved onto minimal choroid thickness after demodulation.
    anchor: Optional[ThicknessAnchorConfig] = None

    def for_video(
        self, *, expected_bpm: Optional[float] = None, verbose: bool = True
    ) -> dict:
        """The stage configs, with the measured HR stamped into the band.

        One band object, shared by the trace bandpass and the periodogram, so
        the two stages cannot disagree about what counts as cardiac. With SiNC
        the band stays open: the point is to find the rate without an expected
        BPM, and anchoring the band on HR would hand it back.
        """
        if self.sinc_checkpoint is not None:
            expected_bpm = None
        band = CardiacBand(expected_bpm=expected_bpm)
        return dict(
            sinc=self.sinc_checkpoint,
            anchor=self.anchor,
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

    ``chain`` and ``fold`` are the component configs; use
    :meth:`chain_for_video` to stamp in the per-video values. The remaining
    field is the one no component owns: the output video's fps.

    There is no method/phase sweep: the composed ``chain`` *is* the recipe, and
    the decomposition and demodulation are fields of the stage configs it
    holds, saved with every ``measure.pkl``. So the outputs carry no
    ``<method>_<phase>`` suffix — what was ``one_cycle_pca_iq`` is just
    ``one_cycle``.
    """

    # Values equal to the library defaults are still spelled out: a study
    # config should pin what it ran, so a later change to a library default
    # cannot silently change this cohort's settings.
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
    smooth_window: int = 25
    lk_window: int = 35
    csi_normal_smooth_sigma: float = 0.0
    csi_normal_slope_window: int = 51


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
# The cohort runs the *learned* registrator: one encoder pass per frame feeds
# both the segmentation decoder and the (dx, dy) heads, which is what the fused
# stage exists to exploit. The dataclass default stays "classical" so a direct
# caller of `register_videos` is unaffected.
#
# `use_encoded_video=False`: this run reads the raw cube.bin off the share, not
# the compressed mp4 beside it. The mp4 is lossy — mean |difference| 11.7 grey
# levels against the raw cube — so it is not an equivalent input to a
# segmentation, and a cohort measured from a mix of the two would not be
# comparable within itself.
REGISTRATION = RegistrationConfig(method="learned", use_encoded_video=False)
PULSATION = PulsationConfig(
    chain=ChainConfig(
        sinc_checkpoint=SINC_CHECKPOINT,
        # One anchor per folded cycle, on the fold's own segment boundaries,
        # so each cycle starts at its own thickness minimum.
        anchor=ThicknessAnchorConfig(n_segments=N_CYCLES) if ANCHOR_PHASE else None,
    )
)
DELTA_Y = DeltaYConfig()
SEGMENTATION = SegmentationConfig()
DELTA_A = DeltaAConfig()
FRIEDENWALD = FriedenwaldConfig()
MISREGISTRATION = MisregistrationConfig()
