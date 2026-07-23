"""Gap-aware periodogram scoring of the candidate traces."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from astropy.timeseries import LombScargle

from ocularrigidity.motion.pulsation.band import CardiacBand
from ocularrigidity.motion.pulsation.rate.base import (
    AbstractRateEstimator,
    RateEstimate,
)
from ocularrigidity.motion.pulsation.traces import Traces


def lomb_scargle_power(t: np.ndarray, y: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Lomb-Scargle power of a single trace on a given frequency grid.

    Factored out so callers that need a periodogram on an arbitrary
    combination of traces (e.g. an optimizer's objective, evaluated once per
    iteration) share the same normalization as :meth:`LombScargleRateEstimator.score`
    instead of re-deriving it.
    """
    return LombScargle(t, y, normalization="standard").power(freqs)


@dataclass
class LombScargleConfig:
    band: CardiacBand = field(default_factory=CardiacBand)
    freq_oversample: float = 5.0
    concentration_band_hz: float = 0.1

    # --- Harmonic correction --------------------------------------------
    harmonic_correction: bool = True
    harmonic_tolerance_bpm: float = 12.0
    harmonic_min_power_ratio: float = 0.2

    verbose: bool = True


class LombScargleRateEstimator(AbstractRateEstimator):
    """Gap-aware periodogram scoring, with optional harmonic correction.

    Lomb-Scargle rather than an FFT because the kept samples are irregular once
    gaps are dropped, and zero-padding them would invent power at the wrong
    frequencies. Each trace is scored by peak power × spectral concentration ×
    significance, optionally reweighted by a Gaussian prior around
    ``band.expected_bpm``.
    """

    def __init__(
        self,
        config: Optional[LombScargleConfig] = None,
        override_bpm: Optional[float] = None,
    ):
        self.config = config or LombScargleConfig()
        self.override_bpm = override_bpm

    # ------------------------------------------------------------------
    def score(self, traces: Traces) -> dict:
        """Per-trace periodogram and quality scores (no selection yet)."""
        cfg = self.config
        S = traces.values
        t = traces.time

        f_min, f_max = cfg.band.effective_hz_range
        T_total = t[-1] - t[0]
        df = 1.0 / (cfg.freq_oversample * T_total)
        freqs = np.arange(f_min, f_max + df, df)

        n = S.shape[1]
        power = np.empty((len(freqs), n))
        fap = np.empty(n)
        peak_freq = np.empty(n)
        for i in range(n):
            ls = LombScargle(t, S[:, i], normalization="standard")
            power[:, i] = ls.power(freqs)
            j = power[:, i].argmax()
            peak_freq[i] = freqs[j]
            fap[i] = ls.false_alarm_probability(power[j, i])

        delta = cfg.concentration_band_hz
        concentration = np.array(
            [
                power[np.abs(freqs - peak_freq[i]) < delta, i].sum()
                / (power[:, i].sum() + 1e-9)
                for i in range(n)
            ]
        )
        # Combined score: log-odds of significance × spectral concentration
        peak_idx = power.argmax(axis=0)
        peak_power = power[peak_idx, np.arange(power.shape[1])]
        quality = (
            np.log10(peak_power + 1e-12)
            + concentration
            + 0.05 * -np.log10(fap + 1e-300)
        )
        if cfg.band.expected_bpm is not None:
            f_exp = cfg.band.expected_bpm / 60.0
            sigma = cfg.band.prior_sigma_bpm / 60.0
            prior = np.exp(-0.5 * ((peak_freq - f_exp) / sigma) ** 2)
            quality = quality * (0.1 + prior)  # 0.1 floor so we don't fully veto

        return {
            "freqs": freqs,
            "power": power,
            "fap": fap,
            "peak_freq": peak_freq,
            "concentration": concentration,
            "quality": quality,
        }

    # ------------------------------------------------------------------
    def estimate(self, traces: Traces) -> RateEstimate:
        cfg = self.config
        ls = self.score(traces)
        notes: list[str] = []

        if self.override_bpm is not None:
            f0 = float(self.override_bpm) / 60.0
            j = int(np.argmin(np.abs(ls["freqs"] - f0)))
            best = int(np.argmax(ls["power"][j, :]))
            if cfg.verbose:
                print(
                    f"BPM fixed to {f0 * 60:.1f}; selected trace {best} "
                    f"(LS power at f0 = {ls['power'][j, best]:.3f}, "
                    f"FAP = {ls['fap'][best]:.2e})"
                )
            return self._build(f0, best, ls, notes)

        # 1. Initial pick: best trace by quality, at its own peak
        best = int(np.argmax(ls["quality"]))
        f_picked = float(ls["peak_freq"][best])
        f_corrected = f_picked
        best_corrected = best

        # 2. Optional harmonic correction
        if cfg.harmonic_correction and cfg.band.expected_bpm is not None:
            bpm_picked = f_picked * 60.0
            bpm_exp = cfg.band.expected_bpm
            tol_bpm = cfg.harmonic_tolerance_bpm

            candidate = None
            if abs(bpm_picked - 2 * bpm_exp) < tol_bpm:
                candidate = f_picked / 2.0
            elif abs(bpm_picked - 0.5 * bpm_exp) < tol_bpm:
                candidate = f_picked * 2.0

            if candidate is not None:
                f_lo, f_hi = cfg.band.effective_hz_range
                if f_lo <= candidate <= f_hi:
                    j_pick = int(np.argmin(np.abs(ls["freqs"] - f_picked)))
                    j_cand = int(np.argmin(np.abs(ls["freqs"] - candidate)))
                    power_at_cand = ls["power"][j_cand, :]
                    best_at_cand = int(np.argmax(power_at_cand))
                    power_at_pick = ls["power"][j_pick, best]
                    ratio_power = power_at_cand[best_at_cand] / power_at_pick

                    if ratio_power > cfg.harmonic_min_power_ratio:
                        f_corrected = candidate
                        best_corrected = best_at_cand
                        msg = (
                            f"Harmonic correction: snapped from {f_picked * 60:.1f} bpm "
                            f"to {candidate * 60:.1f} bpm "
                            f"(power ratio = {ratio_power:.2f}, "
                            f"trace re-picked: {best} → {best_corrected})"
                        )
                    else:
                        msg = (
                            f"Harmonic candidate {candidate * 60:.1f} bpm rejected: "
                            f"max power across traces = {ratio_power:.2f}× of picked "
                            f"peak (threshold {cfg.harmonic_min_power_ratio})."
                        )
                else:
                    msg = (
                        f"Harmonic candidate {candidate * 60:.1f} bpm rejected: "
                        f"out of band {cfg.band.effective_bpm_range}."
                    )
                notes.append(msg)
                if cfg.verbose:
                    print(f"  {msg}")

        if cfg.verbose:
            print(
                f"Selected trace {best_corrected}: BPM = {f_corrected * 60:.1f}, "
                f"FAP = {ls['fap'][best_corrected]:.2e}, "
                f"concentration = {ls['concentration'][best_corrected]:.2f}"
            )

        # 3. Sanity check vs. expected
        if cfg.band.expected_bpm is not None:
            exp = cfg.band.expected_bpm
            err = abs(f_corrected * 60 - exp) / exp
            if err > 0.15:
                msg = (
                    f"Estimated {f_corrected * 60:.1f} bpm differs from "
                    f"expected {exp:.1f} bpm by {err:.1%}."
                )
                notes.append(msg)
                if cfg.verbose:
                    print(f"  WARNING: {msg}")

        return self._build(f_corrected, best_corrected, ls, notes)

    # ------------------------------------------------------------------
    @staticmethod
    def _confidence(fap: float, concentration: float) -> str:
        if fap < 1e-10 and concentration > 0.5:
            return "high"
        if fap < 1e-3 and concentration > 0.3:
            return "medium"
        return "low"

    def _build(
        self, freq: float, best: int, ls: dict, notes: list[str]
    ) -> RateEstimate:
        weights = np.clip(ls["quality"] - ls["quality"].min(), 0.0, None)
        return RateEstimate(
            freq=float(freq),
            best_index=int(best),
            weights=weights,
            diagnostics=ls,
            notes=notes,
            confidence=self._confidence(ls["fap"][best], ls["concentration"][best]),
        )
