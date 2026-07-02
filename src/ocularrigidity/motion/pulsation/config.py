"""Configuration objects for the pulsation pipeline.

These group what used to be long flat argument lists into cohesive dataclasses:

- :class:`PulseExtractionConfig` — everything an ``AbstractPulseExtractor``
  (mask- or frame-based) needs to estimate the cardiac rate and phase.
- :class:`NCycleConfig` — folding parameters for the ``NCycleReconstructor``.

The collaborators (``VideoRegistrator``, ``VideoTimelineAligner``) are *not*
config — they are dependencies passed to the extractor directly.
"""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class PulseExtractionConfig:
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


@dataclass
class NCycleConfig:
    n_cycle: int = 1
    n_bins: Optional[int] = None
    target_frames_per_bin: int = 25
    fold_method: str = "mean"
    phase_method: Literal["iq", "peak_locked"] = "peak_locked"
    verbose: bool = True
