"""The physiological prior: where in frequency to look for the heartbeat."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CardiacBand:
    """Where to look for the heartbeat, and what rate we expect to find."""

    bpm_range: tuple[float, float] = (30.0, 180.0)
    expected_bpm: float | None = None
    expected_bpm_band_frac: float = 0.3
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
