"""The physiological prior: where in frequency to look for the heartbeat.

Shared by the trace bandpass (:mod:`...traces`) and the rate estimators
(:mod:`...rate`), which is why it lives at the package root rather than inside
either stage.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CardiacBand:
    """Where to look for the heartbeat, and what rate we expect to find."""

    bpm_range: tuple[float, float] = (30.0, 180.0)
    # When set, the search band is narrowed to
    # [(1-frac), (1+frac)] * expected_bpm, overriding ``bpm_range``.
    expected_bpm: Optional[float] = None
    expected_bpm_band_frac: float = 0.3
    # Width (bpm) of the Gaussian prior around ``expected_bpm`` used to score
    # candidate peaks. Distinct from ``harmonic_tolerance_bpm``.
    prior_sigma_bpm: float = 12.0

    @property
    def effective_bpm_range(self) -> tuple[float, float]:
        """Search band after any ``expected_bpm`` anchoring."""
        if self.expected_bpm is None:
            return self.bpm_range
        return (
            (1.0 - self.expected_bpm_band_frac) * self.expected_bpm,
            (1.0 + self.expected_bpm_band_frac) * self.expected_bpm,
        )

    @property
    def effective_hz_range(self) -> tuple[float, float]:
        lo, hi = self.effective_bpm_range
        return lo / 60.0, hi / 60.0
