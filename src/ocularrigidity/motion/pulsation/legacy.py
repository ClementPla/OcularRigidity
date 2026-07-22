"""Backwards-compatible facade over the composed pulse-extraction stack.

``MaskPulseExtractor`` used to be a subclass of a monolithic
``AbstractPulseExtractor``. It is now a thin preset over :class:`PulseExtractor`
that assembles the historical recipe from the flat
:class:`PulseExtractionConfig`:

    MaskThicknessTraceSource → DecomposedTraceSource(ICA)
                             → LombScargleRateEstimator
                             → IQDemod / PeakLocked phase

It also keeps the legacy read-only surface (``signal``, ``filtered_signal``,
``separable_components``, ``lomb_scargle_results``, ``*_peak_locked``, …) alive
for ``CardiacPipelineResults`` and the viewers.

Nothing new should be added here. New code builds a :class:`PulseExtractor` and
picks its own components; new knobs go on the per-stage configs.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ocularrigidity.motion.pulsation.band import CardiacBand
from ocularrigidity.motion.pulsation.extractor import PulseExtractor
from ocularrigidity.motion.pulsation.phase import (
    IQDemodPhaseEstimator,
    IQPhaseConfig,
    PeakLockedPhaseEstimator,
    PhaseTrack,
    SelectBestComponent,
)
from ocularrigidity.motion.pulsation.rate import (
    LombScargleConfig,
    LombScargleRateEstimator,
)
from ocularrigidity.motion.pulsation.traces import (
    DecompositionConfig,
    DecomposedTraceSource,
    MaskThicknessTraceSource,
    MaskTraceConfig,
)
from ocularrigidity.motion.video_timeline_aligner import VideoTimelineAligner
from ocularrigidity.registration.registration_engine import VideoRegistrator


@dataclass
class PulseExtractionConfig:
    """Legacy flat config. Prefer the per-component dataclasses above.

    Retained for backwards compatibility: ``MaskPulseExtractor`` and
    ``run_cardiac_pipeline`` still accept it, and it is stored verbatim on
    ``CardiacPipelineResults`` for provenance.
    """

    # --- Physiological prior --------------------------------------------
    bpm_range: tuple[float, float] = (30.0, 180.0)
    # Fixes the cardiac frequency; disables LS-based frequency search.
    override_bpm: Optional[float] = None
    # When set, the search band is narrowed to
    # [(1-frac), (1+frac)] * expected_bpm, overriding ``bpm_range``.
    expected_bpm: Optional[float] = None
    expected_bpm_band_frac: float = 0.3
    # Retained for traceability; the Butterworth path is currently disabled
    # in favour of the FIR bandpass in ``filtered_signal``.
    butter_order: int = 4

    # --- Spatial smoother -----------------------------------------------
    sigma_col: float = 5.0
    col_slice: Optional[slice] = None

    # --- Decomposition --------------------------------------------------
    n_separable_components: int = 16
    ICA_or_PCA: str = "ICA"
    ica_random_state: int = 0

    # --- Lomb-Scargle scoring -------------------------------------------
    ls_freq_oversample: float = 5.0
    ls_concentration_band_hz: float = 0.1

    # --- Harmonic correction --------------------------------------------
    harmonic_correction: bool = True
    harmonic_tolerance_bpm: float = 12.0
    harmonic_min_power_ratio: float = 0.2
    # Width (bpm) of the Gaussian LS prior around ``expected_bpm``. Distinct
    # from ``harmonic_tolerance_bpm`` (harmonic snapping).
    bpm_prior_sigma_bpm: float = 12.0

    # --- IQ demodulation / phase ----------------------------------------
    phase_smoother_cycles: float = 2.0
    phase_density_threshold: float = 0.5

    # --- Misc -----------------------------------------------------------
    verbose: bool = True

    # ------------------------------------------------------------------
    @property
    def band(self) -> CardiacBand:
        return CardiacBand(
            bpm_range=self.bpm_range,
            expected_bpm=self.expected_bpm,
            expected_bpm_band_frac=self.expected_bpm_band_frac,
            prior_sigma_bpm=self.bpm_prior_sigma_bpm,
        )

    def split(
        self,
    ) -> tuple[MaskTraceConfig, DecompositionConfig, LombScargleConfig, IQPhaseConfig]:
        """Explode this bundle into the per-component configs."""
        band = self.band
        return (
            MaskTraceConfig(
                band=band,
                sigma_col=self.sigma_col,
                verbose=self.verbose,
                col_slice=self.col_slice,
            ),
            DecompositionConfig(
                method=self.ICA_or_PCA,
                n_components=self.n_separable_components,
                random_state=self.ica_random_state,
            ),
            LombScargleConfig(
                band=band,
                freq_oversample=self.ls_freq_oversample,
                concentration_band_hz=self.ls_concentration_band_hz,
                harmonic_correction=self.harmonic_correction,
                harmonic_tolerance_bpm=self.harmonic_tolerance_bpm,
                harmonic_min_power_ratio=self.harmonic_min_power_ratio,
                verbose=self.verbose,
            ),
            IQPhaseConfig(
                smoother_cycles=self.phase_smoother_cycles,
                density_threshold=self.phase_density_threshold,
            ),
        )


class MaskPulseExtractor(PulseExtractor):
    def __init__(
        self,
        registered_video: VideoRegistrator,
        video_timeline_aligner: VideoTimelineAligner,
        config: Optional[PulseExtractionConfig] = None,
    ):
        self.config = config or PulseExtractionConfig()
        trace_cfg, decomp_cfg, ls_cfg, iq_cfg = self.config.split()

        source = DecomposedTraceSource(
            MaskThicknessTraceSource(
                registered_video, video_timeline_aligner, trace_cfg
            ),
            decomp_cfg,
        )
        super().__init__(
            trace_source=source,
            phase_estimator=IQDemodPhaseEstimator(
                iq_cfg, aggregator=SelectBestComponent()
            ),
            rate_estimator=LombScargleRateEstimator(
                ls_cfg, override_bpm=self.config.override_bpm
            ),
            registered_video=registered_video,
            aligner=video_timeline_aligner,
        )

        self._peak_locked: Optional[PhaseTrack] = None
        self._is_freq_overridden = self.config.override_bpm is not None

        if self.config.expected_bpm is not None and self.config.verbose:
            lo, hi = self.bpm_range
            print(
                f"Anchoring bpm_range to expected {self.config.expected_bpm:.1f} bpm: "
                f"({lo:.1f}, {hi:.1f}) "
                f"(±{self.config.expected_bpm_band_frac:.0%}); "
                f"requested {self.config.bpm_range} ignored."
            )

    # ------------------------------------------------------------------
    # Legacy surface
    # ------------------------------------------------------------------
    @property
    def verbose(self) -> bool:
        return self.config.verbose

    @property
    def expected_bpm(self):
        return self.config.expected_bpm

    @property
    def bpm_range(self):
        """Effective search band (after any ``expected_bpm`` anchoring)."""
        return self.config.band.effective_bpm_range

    @property
    def _mask_source(self) -> MaskThicknessTraceSource:
        return self.trace_source.source

    @property
    def signal(self) -> np.ndarray:
        return self._mask_source.signal

    @property
    def bad_frame(self) -> np.ndarray:
        return self._mask_source.bad_frame()

    @property
    def interpolated_signal(self):
        return self._mask_source.interpolated_signal

    @property
    def interpolated_validity(self):
        return self._mask_source.interpolated_validity

    @property
    def filtered_signal(self):
        return self._mask_source.filtered_signal

    @property
    def component_kept_mask(self):
        return self.traces.kept_mask

    @property
    def separable_components(self):
        return self.traces.values

    @property
    def ica_mixing(self):
        return self.traces.mixing

    @property
    def lomb_scargle_results(self):
        return self.rate.diagnostics

    @property
    def best_component_idx(self) -> int:
        return self.rate.best_index

    @PulseExtractor.cardiac_freq.setter
    def cardiac_freq(self, value: float):
        PulseExtractor.cardiac_freq.fset(self, value)
        self._is_freq_overridden = True
        self._peak_locked = None

    # -- peak-locked phase, exposed alongside the IQ phase ---------------
    @property
    def peak_locked(self) -> PhaseTrack:
        if self._peak_locked is None:
            estimator = PeakLockedPhaseEstimator(
                aggregator=SelectBestComponent(), band=self.config.band
            )
            self._peak_locked = estimator.estimate(self.traces, self.rate)
        return self._peak_locked

    @property
    def phase_uniform_peak_locked(self):
        return self.peak_locked.phase_uniform

    @property
    def good_uniform_peak_locked(self):
        return self.peak_locked.good_uniform

    @property
    def phase_per_frame_peak_locked(self):
        return self.peak_locked.phase_per_frame

    @property
    def good_per_frame_peak_locked(self):
        return self.peak_locked.good_per_frame
