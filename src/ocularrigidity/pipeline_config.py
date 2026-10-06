"""Settings the cohort pipeline runs with."""

from dataclasses import dataclass, field, replace
from typing import Literal, Optional

from ocularrigidity.consts import (  # noqa: F401  (AXIAL_PIXEL_SIZE_MM re-exported)
    AXIAL_PIXEL_SIZE_MM,
    BSCAN_WIDTH_PX,
    CHOROID_TRIM_PX,
    SINC_REVISION,
)
from ocularrigidity.friedenwald import FRIEDENWALD, FriedenwaldConfig  # noqa: F401
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

# Cardiac cycles folded, segmented and measured. Shared by every stage.
N_CYCLES = 3


@dataclass(frozen=True)
class PulsationConfig:
    """Trace -> rate -> phase chain, and the fold."""

    trace: MaskTraceConfig = field(
        default_factory=lambda: MaskTraceConfig(
            col_slice=slice(100, BSCAN_WIDTH_PX - 100)
        )
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
            smoother_cycles=2.0, density_threshold=0.5, freq_tolerance=None
        )
    )
    # If set, rate and phase come from this SiNC checkpoint's waveform.
    sinc_revision: Optional[str] = None
    # If set, phase 0 is moved onto minimal choroid thickness.
    anchor: Optional[ThicknessAnchorConfig] = None

    fold: NCycleConfig = field(
        default_factory=lambda: NCycleConfig(
            n_bins=30, n_cycle=N_CYCLES, fold_method="median", phase_method="iq"
        )
    )
    # Display metadata of one_cycle.mkv.
    output_fps: int = 30

    def chain_for_video(
        self, *, expected_bpm: Optional[float] = None, verbose: bool = True
    ) -> tuple[dict, NCycleConfig]:
        """Stage configs and fold config for one video."""
        # SiNC finds the rate on its own: the band stays open.
        if self.sinc_revision is not None:
            expected_bpm = None
        # One band for the bandpass and the periodogram.
        band = CardiacBand(expected_bpm=expected_bpm)
        stages = dict(
            sinc=self.sinc_revision,
            anchor=self.anchor,
            trace=replace(self.trace, verbose=verbose),
            bandpass=replace(self.bandpass, band=band, verbose=verbose),
            coherence=replace(self.coherence, verbose=verbose),
            decomposition=self.decomposition,
            rate=replace(self.rate, band=band, verbose=verbose),
            phase=self.phase,
        )
        return stages, replace(self.fold, verbose=verbose)


@dataclass(frozen=True)
class DeltaYConfig:
    """Harmonic fit of the thickness amplitude on the folded cycles."""

    n_harmonics: int = 1
    residual_threshold_percentile: float = 75.0
    amplitude_threshold_percentile: float = 50.0


@dataclass(frozen=True)
class DeltaAConfig:
    """Boundary displacement / area change."""

    method: Literal["optical_flow", "demons"] = "optical_flow"
    smooth_window: int = 25
    lk_window: int = 35
    csi_normal_smooth_sigma: float = 0.0
    csi_normal_slope_window: int = 51
    trim: int = CHOROID_TRIM_PX


# Raw cube.bin, not the lossy mp4 next to it.
REGISTRATION = RegistrationConfig(method="learned", use_encoded_video=False)
# SiNC waveform, phase anchored on minimal thickness in each folded cycle.
# For the thickness-trace chain instead: PulsationConfig()
PULSATION = PulsationConfig(
    sinc_revision=SINC_REVISION,
    anchor=ThicknessAnchorConfig(n_segments=N_CYCLES),
)
DELTA_Y = DeltaYConfig()
DELTA_A = DeltaAConfig()
