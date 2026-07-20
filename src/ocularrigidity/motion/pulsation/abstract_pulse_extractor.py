"""Shared cardiac rate/phase estimation, independent of the signal source.

An ``AbstractPulseExtractor`` turns a 1-D-per-frame ``signal`` (thickness for
the mask variant, intensity for the future frame variant) into a cardiac
frequency and a per-frame phase. Everything downstream of ``signal`` — spatial
smoothing, ICA/PCA, Lomb-Scargle scoring, frequency/IC selection, IQ and
peak-locked phase — is factored here. Subclasses provide only the signal:

    class MaskPulseExtractor(AbstractPulseExtractor):
        @property
        def signal(self): ...

N-cycle folding is intentionally *not* here; see ``NCycleReconstructor``.
"""

from abc import ABC, abstractmethod

import numpy as np
from astropy.timeseries import LombScargle
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import skew

from ocularrigidity.motion.filters._1d import spatio_temporal_filter
from ocularrigidity.motion.projection._1d import project_into_separable_components
from ocularrigidity.motion.pulsation.config import PulseExtractionConfig
from ocularrigidity.motion.video_timeline_aligner import VideoTimelineAligner
from ocularrigidity.registration.registration_engine import VideoRegistrator


class AbstractPulseExtractor(ABC):
    def __init__(
        self,
        registered_video: VideoRegistrator,
        video_timeline_aligner: VideoTimelineAligner,
        config: PulseExtractionConfig | None = None,
    ):
        self.registered_video = registered_video
        self.aligner = video_timeline_aligner
        self.config = config or PulseExtractionConfig()

        cfg = self.config

        # Anchor the search band to the expected rate when known. Derived here
        # (rather than mutating the config) so the config stays a faithful record
        # of what the caller requested.
        if cfg.expected_bpm is not None:
            self._bpm_range = (
                (1.0 - cfg.expected_bpm_band_frac) * cfg.expected_bpm,
                (1.0 + cfg.expected_bpm_band_frac) * cfg.expected_bpm,
            )
            if cfg.verbose:
                print(
                    f"Anchoring bpm_range to expected {cfg.expected_bpm:.1f} bpm: "
                    f"({self._bpm_range[0]:.1f}, {self._bpm_range[1]:.1f}) "
                    f"(±{cfg.expected_bpm_band_frac:.0%}); "
                    f"requested {cfg.bpm_range} ignored."
                )
        else:
            self._bpm_range = cfg.bpm_range

        # ---- caches --------------------------------------------------
        self._signal = None
        self._gap_mask = None
        self._interpolated_signal = None
        self._interpolated_validity = None
        self._filtered_signal = None

        self._component_kept_mask = None
        self._separable_components = None
        self._ica_mixing = None

        self._ls_results = None
        self._best_component_idx = None
        self._cardiac_freq = (
            float(cfg.override_bpm) / 60.0 if cfg.override_bpm is not None else None
        )
        self._is_freq_overridden = cfg.override_bpm is not None

        self._phase_uniform = None
        self._good_uniform = None
        self._phase_per_frame = None
        self._good_per_frame = None
        self._phase_uniform_pl = None
        self._good_uniform_pl = None
        self._phase_per_frame_pl = None
        self._good_per_frame_pl = None
        self._amplitude_uniform = None
        self._amplitude_per_frame = None

        self.notes: list[str] = []

    # ------------------------------------------------------------------
    # Convenience delegations to the timeline aligner
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
        return self._bpm_range

    @property
    def timestamps_seconds(self):
        return self.aligner.timestamps_seconds

    @property
    def uniform_time(self):
        return self.aligner.uniform_time

    @property
    def dt(self) -> float:
        return self.aligner.dt

    @property
    def fs(self) -> float:
        return self.aligner.fs

    @property
    def gap_mask(self):
        """Uniform-grid gap mask, combining time gaps with this domain's bad frames."""
        if self._gap_mask is None:
            self._gap_mask = self.aligner.gap_mask(self.bad_frame)
        return self._gap_mask

    @property
    def gap_fraction(self) -> float:
        return float(self.gap_mask.mean())

    @property
    def registered_frames(self):
        return self.registered_video.registered_frames

    @property
    def registered_masks(self):
        return self.registered_video.registered_masks

    # ------------------------------------------------------------------
    # Signal source (the only thing subclasses must provide)
    # ------------------------------------------------------------------
    @property
    @abstractmethod
    def signal(self) -> np.ndarray:
        """Raw per-frame signal, shape (T, W), holes marked as NaN.

        Thickness for the mask variant, intensity profile for the frame variant.
        """

    @property
    def bad_frame(self) -> np.ndarray:
        """Per-original-frame invalidity flag. Default: fully-NaN rows.

        Overridable — the frame variant defines bad frames differently.
        """
        return np.isnan(self.signal).all(axis=1)

    @property
    def interpolated_signal(self):
        """``signal`` interpolated onto the uniform grid; gaps set to NaN."""
        if self._interpolated_signal is None:
            signal = self.signal
            valid = ~np.isnan(signal).any(axis=1)
            if not valid.any():
                out = np.full(
                    (len(self.uniform_time), signal.shape[1]),
                    np.nan,
                    dtype=signal.dtype,
                )
            else:
                out = interp1d(
                    self.timestamps_seconds[valid],
                    signal[valid],
                    axis=0,
                    kind="linear",
                    fill_value=np.nan,
                    bounds_error=False,
                )(self.uniform_time)
            out[self.gap_mask] = np.nan
            self._interpolated_signal = out
        return self._interpolated_signal

    @property
    def interpolated_validity(self):
        """Per-sample validity fraction on the uniform grid; gaps set to 0."""
        if self._interpolated_validity is None:
            valid = (~np.isnan(self.signal)).astype(np.float32)
            out = interp1d(
                self.timestamps_seconds,
                valid,
                axis=0,
                kind="linear",
                fill_value=0.0,
                bounds_error=False,
            )(self.uniform_time)
            out[self.gap_mask] = 0.0
            self._interpolated_validity = out
        return self._interpolated_validity

    @property
    def filtered_signal(self):
        """Spatially smoothed (NaN-aware) and temporally bandpassed signal map."""
        if self._filtered_signal is not None:
            return self._filtered_signal

        nyq = 0.5 * self.fs
        low = (self._bpm_range[0] / 60.0) / nyq
        high = min((self._bpm_range[1] / 60.0) / nyq, 0.99)
        not_gap = (~self.gap_mask).astype(np.float32)[:, None]
        data_masked = np.nan_to_num(self.interpolated_signal, nan=0.0) * not_gap
        valid_masked = self.interpolated_validity * not_gap
        filtered = spatio_temporal_filter(
            data_masked,
            spatial_sigma=self.config.sigma_col,
            temporal_low_freq=low,
            temporal_high_freq=high,
            fs=self.fs,
            validity_mask=valid_masked,
        )
        filtered[self.gap_mask] = np.nan
        self._filtered_signal = filtered
        return self._filtered_signal

    @property
    def component_kept_mask(self):
        """Boolean mask on uniform_time: True where ``filtered_signal`` has no NaN."""
        if self._component_kept_mask is None:
            self._component_kept_mask = ~np.isnan(self.filtered_signal).any(axis=1)
        return self._component_kept_mask

    # ------------------------------------------------------------------
    # ICA / PCA decomposition
    # ------------------------------------------------------------------
    def _compute_separable_components(self):
        keep = self.component_kept_mask
        X = self.filtered_signal[keep]
        self._separable_components, self._ica_mixing = (
            project_into_separable_components(
                X,
                method=self.config.ICA_or_PCA.lower(),
                n_components=self.config.n_separable_components,
                random_state=self.config.ica_random_state,
                max_iter=5000,
                whiten="unit-variance",
                tol=0.001,
                fun="cube",
            )
        )

    def _standardize_ic_sign(self):
        """Flip best IC so its sign matches physical signal pulsation."""
        best = self._best_component_idx
        ic = self._separable_components[:, best]
        keep = self.component_kept_mask
        ref = np.nanmean(self.filtered_signal[keep], axis=1)
        sign = np.sign(np.corrcoef(ic, ref)[0, 1])
        if sign < 0:
            self._separable_components[:, best] *= -1
            self._ica_mixing[:, best] *= -1

    @property
    def separable_components(self):
        """ICA temporal components, shape (T_kept, n_ic)."""
        if self._separable_components is None:
            self._compute_separable_components()
        return self._separable_components

    @property
    def ica_mixing(self):
        """ICA spatial mixing matrix, shape (W, n_ic)."""
        if self._ica_mixing is None:
            self._compute_separable_components()
        return self._ica_mixing

    # ------------------------------------------------------------------
    # Lomb-Scargle scoring (gap-aware, no implicit zero-padding)
    # ------------------------------------------------------------------
    def _compute_lomb_scargle(self):
        S = self.separable_components
        t_kept = self.uniform_time[self.component_kept_mask]

        f_min = self._bpm_range[0] / 60.0
        f_max = self._bpm_range[1] / 60.0
        T_total = t_kept[-1] - t_kept[0]
        df = 1.0 / (self.config.ls_freq_oversample * T_total)
        freqs = np.arange(f_min, f_max + df, df)

        n_ic = S.shape[1]
        power = np.empty((len(freqs), n_ic))
        fap = np.empty(n_ic)
        peak_freq = np.empty(n_ic)
        for i in range(n_ic):
            ls = LombScargle(t_kept, S[:, i], normalization="standard")
            power[:, i] = ls.power(freqs)
            j = power[:, i].argmax()
            peak_freq[i] = freqs[j]
            fap[i] = ls.false_alarm_probability(power[j, i])

        delta = self.config.ls_concentration_band_hz
        concentration = np.array(
            [
                power[np.abs(freqs - peak_freq[i]) < delta, i].sum()
                / (power[:, i].sum() + 1e-9)
                for i in range(n_ic)
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
        if self.config.expected_bpm is not None:
            f_exp = self.config.expected_bpm / 60.0
            sigma = self.config.bpm_prior_sigma_bpm / 60.0
            prior = np.exp(-0.5 * ((peak_freq - f_exp) / sigma) ** 2)
            quality = quality * (0.1 + prior)  # 0.1 floor so we don't fully veto

        self._ls_results = {
            "freqs": freqs,
            "power": power,
            "fap": fap,
            "peak_freq": peak_freq,
            "concentration": concentration,
            "quality": quality,
        }

    @property
    def lomb_scargle_results(self):
        if self._ls_results is None:
            self._compute_lomb_scargle()
        return self._ls_results

    # ------------------------------------------------------------------
    # Cardiac IC + frequency selection
    # ------------------------------------------------------------------
    def _ensure_cardiac_selection(self):
        if self._best_component_idx is not None:
            return

        ls = self.lomb_scargle_results
        cfg = self.config

        if self._cardiac_freq is not None:
            # BPM was overridden — pick IC with most LS power at f0
            j = int(np.argmin(np.abs(ls["freqs"] - self._cardiac_freq)))
            self._best_component_idx = int(np.argmax(ls["power"][j, :]))
            self._standardize_ic_sign()
            if cfg.verbose:
                print(
                    f"BPM fixed to {self._cardiac_freq * 60:.1f}; "
                    f"selected IC {self._best_component_idx} "
                    f"(LS power at f0 = {ls['power'][j, self._best_component_idx]:.3f}, "
                    f"FAP = {ls['fap'][self._best_component_idx]:.2e})"
                )
            return

        # 1. Initial pick: best IC by quality (FAP × concentration), at its peak
        best = int(np.argmax(ls["quality"]))
        f_picked = float(ls["peak_freq"][best])
        f_corrected = f_picked
        best_corrected = best

        # 2. Optional harmonic correction
        if cfg.harmonic_correction and cfg.expected_bpm is not None:
            bpm_picked = f_picked * 60.0
            bpm_exp = cfg.expected_bpm
            tol_bpm = cfg.harmonic_tolerance_bpm

            candidate = None
            if abs(bpm_picked - 2 * bpm_exp) < tol_bpm:
                candidate = f_picked / 2.0
            elif abs(bpm_picked - 0.5 * bpm_exp) < tol_bpm:
                candidate = f_picked * 2.0

            if candidate is not None:
                in_band = (
                    self._bpm_range[0] / 60.0
                    <= candidate
                    <= self._bpm_range[1] / 60.0
                )
                if in_band:
                    j_pick = int(np.argmin(np.abs(ls["freqs"] - f_picked)))
                    j_cand = int(np.argmin(np.abs(ls["freqs"] - candidate)))
                    power_at_cand = ls["power"][j_cand, :]  # (n_ic,)
                    ic_best_at_cand = int(np.argmax(power_at_cand))
                    power_at_pick = ls["power"][j_pick, best]
                    ratio_power = power_at_cand[ic_best_at_cand] / power_at_pick

                    if ratio_power > cfg.harmonic_min_power_ratio:
                        f_corrected = candidate
                        best_corrected = ic_best_at_cand
                        msg = (
                            f"Harmonic correction: snapped from {f_picked * 60:.1f} bpm "
                            f"to {candidate * 60:.1f} bpm "
                            f"(power ratio = {ratio_power:.2f}, "
                            f"IC re-picked: {best} → {best_corrected})"
                        )
                    else:
                        msg = (
                            f"Harmonic candidate {candidate * 60:.1f} bpm rejected: "
                            f"max power across ICs = {ratio_power:.2f}× of picked peak "
                            f"(threshold {cfg.harmonic_min_power_ratio})."
                        )
                else:
                    msg = (
                        f"Harmonic candidate {candidate * 60:.1f} bpm rejected: "
                        f"out of band {self._bpm_range}."
                    )
                self.notes.append(msg)
                if cfg.verbose:
                    print(f"  {msg}")

        # 3. Commit selection
        self._best_component_idx = best_corrected
        self._cardiac_freq = f_corrected
        self._standardize_ic_sign()
        if cfg.verbose:
            print(
                f"Selected IC {self._best_component_idx}: "
                f"BPM = {self._cardiac_freq * 60:.1f}, "
                f"FAP = {ls['fap'][self._best_component_idx]:.2e}, "
                f"concentration = {ls['concentration'][self._best_component_idx]:.2f}"
            )

        # 4. Sanity check vs. expected
        if cfg.expected_bpm is not None:
            err = abs(self._cardiac_freq * 60 - cfg.expected_bpm) / cfg.expected_bpm
            if err > 0.15:
                msg = (
                    f"Estimated {self._cardiac_freq * 60:.1f} bpm differs from "
                    f"expected {cfg.expected_bpm:.1f} bpm by {err:.1%}."
                )
                self.notes.append(msg)
                if cfg.verbose:
                    print(f"  WARNING: {msg}")

    @property
    def best_component_idx(self) -> int:
        self._ensure_cardiac_selection()
        return self._best_component_idx

    @property
    def cardiac_freq(self) -> float:
        self._ensure_cardiac_selection()
        return self._cardiac_freq

    @cardiac_freq.setter
    def cardiac_freq(self, value: float):
        """Manual override — re-selects the best IC at the new frequency."""
        self._cardiac_freq = float(value)
        self._is_freq_overridden = True
        self._best_component_idx = None
        self._phase_uniform = None
        self._good_uniform = None
        self._phase_per_frame = None
        self._good_per_frame = None
        self._phase_uniform_pl = None
        self._good_uniform_pl = None
        self._phase_per_frame_pl = None
        self._good_per_frame_pl = None

    @property
    def cardiac_bpm(self) -> float:
        return self.cardiac_freq * 60.0

    # ------------------------------------------------------------------
    # IQ demodulation: per-frame instantaneous phase
    # ------------------------------------------------------------------
    def _compute_phase(self):
        self._ensure_cardiac_selection()
        f0 = self._cardiac_freq
        best = self._best_component_idx

        ic_full = np.full(len(self.uniform_time), np.nan)
        ic_full[self.component_kept_mask] = self.separable_components[:, best]

        t = self.uniform_time
        dt = self.dt

        # Demodulate at f0: brings the cardiac component down to DC
        phasor = np.exp(-1j * 2 * np.pi * f0 * t)
        z = ic_full * phasor  # NaNs propagate

        sigma_t = (self.config.phase_smoother_cycles / f0) / dt
        nan = np.isnan(ic_full)
        valid = (~nan).astype(np.float32)
        I_filled = np.where(nan, 0.0, z.real)
        Q_filled = np.where(nan, 0.0, z.imag)

        I_num = gaussian_filter1d(I_filled, sigma=sigma_t, mode="nearest")
        Q_num = gaussian_filter1d(Q_filled, sigma=sigma_t, mode="nearest")
        den = gaussian_filter1d(valid, sigma=sigma_t, mode="nearest")
        good = den > self.config.phase_density_threshold
        safe_den = np.where(good, den, 1.0)
        I_lp = np.where(good, I_num / safe_den, 0.0)
        Q_lp = np.where(good, Q_num / safe_den, 0.0)

        env_phase = np.arctan2(Q_lp, I_lp)
        amplitude_uniform = np.sqrt(I_lp**2 + Q_lp**2)
        phase_total = 2 * np.pi * f0 * t + env_phase
        phase_uniform = np.mod(phase_total, 2 * np.pi)

        self._phase_uniform = phase_uniform
        self._good_uniform = good
        self._amplitude_uniform = amplitude_uniform

        good_idx = np.where(good)[0]
        if len(good_idx) == 0:
            T_orig = len(self.timestamps_seconds)
            self._phase_per_frame = np.full(T_orig, np.nan)
            self._good_per_frame = np.zeros(T_orig, dtype=bool)
            return

        z_unit = np.exp(1j * phase_uniform[good_idx])
        zr = np.interp(self.timestamps_seconds, t[good_idx], z_unit.real)
        zi = np.interp(self.timestamps_seconds, t[good_idx], z_unit.imag)
        self._phase_per_frame = (np.angle(zr + 1j * zi) / (2 * np.pi)) % 1.0
        self._amplitude_per_frame = np.interp(self.timestamps_seconds, t[good_idx], amplitude_uniform)

        good_float = np.interp(self.timestamps_seconds, t, good.astype(float))
        self._good_per_frame = good_float > 0.5

    @property
    def phase_uniform(self):
        """Cardiac phase on uniform_time, in radians, range [0, 2π).

        Reliable only where ``good_uniform`` is True.
        """
        if self._phase_uniform is None:
            self._compute_phase()
        return self._phase_uniform

    @property
    def good_uniform(self):
        """Reliable-phase mask on uniform_time."""
        if self._good_uniform is None:
            self._compute_phase()
        return self._good_uniform

    @property
    def phase_per_frame(self):
        """Cardiac phase fraction in [0, 1) on original frame timestamps."""
        if self._phase_per_frame is None:
            self._compute_phase()
        return self._phase_per_frame

    @property
    def good_per_frame(self):
        if self._good_per_frame is None:
            self._compute_phase()
        return self._good_per_frame
    
    @property
    def amplitude_uniform(self):
        if self._amplitude_uniform is None:
            self._compute_phase()
        return self._amplitude_uniform
    
    @property
    def amplitude_per_frame(self):
        if self._amplitude_per_frame is None:
            self._compute_phase()
        return self._amplitude_per_frame

    # ------------------------------------------------------------------
    # Peak-locked phase
    # ------------------------------------------------------------------
    def _compute_phase_peak_locked(self):
        self._ensure_cardiac_selection()  # gives us f0 and best_ic_idx
        f0 = self._cardiac_freq
        best = self._best_component_idx
        fs = self.fs

        ic_full = np.full(len(self.uniform_time), np.nan)
        ic_full[self.component_kept_mask] = self.separable_components[:, best]

        valid = ~np.isnan(ic_full)
        if skew(ic_full[valid]) < 0:
            ic_full = -ic_full

        min_dist = int(0.7 / f0 * fs)  # reject peaks closer than 70% of the period
        prom = 0.4 * np.nanstd(ic_full)
        peaks, _ = find_peaks(
            np.nan_to_num(ic_full), distance=min_dist, prominence=prom
        )

        durations = np.diff(peaks)
        med = np.median(durations)
        good_cycle = (durations > 0.7 * med) & (durations < 1.4 * med)

        phase_u = np.full(len(self.uniform_time), np.nan)
        good_u = np.zeros(len(self.uniform_time), dtype=bool)
        for i, ok in enumerate(good_cycle):
            if not ok:
                continue
            p0, p1 = peaks[i], peaks[i + 1]
            gap_frac = self.gap_mask[p0:p1].mean()
            if gap_frac > 0.5:  # cycle is mostly gap — drop entirely
                continue
            phase_u[p0:p1] = np.linspace(0.0, 2 * np.pi, p1 - p0, endpoint=False)
            good_u[p0:p1] = ~self.gap_mask[p0:p1]  # keep non-gap frames only

        gi = np.where(good_u)[0]
        if len(gi) == 0:
            T = len(self.timestamps_seconds)
            self._phase_per_frame_pl = np.full(T, np.nan)
            self._good_per_frame_pl = np.zeros(T, dtype=bool)
            return
        z = np.exp(1j * phase_u[gi])
        zr = np.interp(self.timestamps_seconds, self.uniform_time[gi], z.real)
        zi = np.interp(self.timestamps_seconds, self.uniform_time[gi], z.imag)
        self._phase_per_frame_pl = (np.angle(zr + 1j * zi) / (2 * np.pi)) % 1.0
        good_float = np.interp(
            self.timestamps_seconds, self.uniform_time, good_u.astype(float)
        )
        self._good_per_frame_pl = good_float > 0.5
        self._phase_uniform_pl = phase_u
        self._good_uniform_pl = good_u

    @property
    def phase_per_frame_peak_locked(self):
        """Cycle-locked phase: 0 at each detected systolic peak, linear between."""
        if self._phase_per_frame_pl is None:
            self._compute_phase_peak_locked()
        return self._phase_per_frame_pl

    @property
    def good_per_frame_peak_locked(self):
        """Reliable-phase mask for peak-locked phase, on original frame timestamps."""
        if self._good_per_frame_pl is None:
            self._compute_phase_peak_locked()
        return self._good_per_frame_pl

    @property
    def phase_uniform_peak_locked(self):
        if self._phase_uniform_pl is None:
            self._compute_phase_peak_locked()
        return self._phase_uniform_pl

    @property
    def good_uniform_peak_locked(self):
        if self._good_uniform_pl is None:
            self._compute_phase_peak_locked()
        return self._good_uniform_pl

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @property
    def inst_bpm(self):
        """Instantaneous BPM trace on uniform_time (diagnostic)."""
        good = self.good_uniform
        phase = self.phase_uniform
        out = np.full(len(self.uniform_time), np.nan)
        in_run = good.astype(int)
        starts = np.where(np.diff(np.r_[0, in_run]) == 1)[0]
        ends = np.where(np.diff(np.r_[in_run, 0]) == -1)[0] + 1
        for s, e in zip(starts, ends):
            if e - s > 2:
                unw = np.unwrap(phase[s:e])
                out[s:e] = np.gradient(unw, self.uniform_time[s:e]) / (2 * np.pi) * 60.0
        return out

    @property
    def confidence(self) -> str:
        """Quick tri-level quality label for batch filtering."""
        ls = self.lomb_scargle_results
        best = self.best_component_idx
        fap = ls["fap"][best]
        conc = ls["concentration"][best]
        if fap < 1e-10 and conc > 0.5:
            return "high"
        if fap < 1e-3 and conc > 0.3:
            return "medium"
        return "low"
