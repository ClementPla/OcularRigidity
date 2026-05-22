"""Cardiac cycle extraction from registered video.
The CardiacCycleExtractor class encapsulates the entire pipeline for extracting a clean cardiac pulsation signal from the registered video. It takes care of:
- Loading timestamps and thickness data
- Interpolating thickness onto a uniform time grid, while marking gaps
- Spatially smoothing the thickness map with a NaN-aware Gaussian filter
- Temporally bandpass filtering the thickness signal to isolate cardiac frequencies
- Performing ICA or PCA to find separable components
- Scoring components with a Lomb-Scargle periodogram to find the most cardiac-like one
- Optionally standardizing the sign of the selected component for interpretability
The results are cached as properties, so each step is computed lazily and only once. The final cardiac component can be accessed via the separable_components and ica_mixing properties, and the selected cardiac frequency is available in lomb_scargle_results.
"""

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from scipy.signal import find_peaks
from scipy.signal import (
    firwin,
    filtfilt,
)  # Firwin instead of Butterworth to avoid phase distortion

from ocularrigidity.motion.one_cycle import (
    _auto_n_bins,
    fold_video_numba_mean,
    fold_video_numba_median,
)
from ocularrigidity.motion.registered_video import RegisteredVideo

from astropy.timeseries import LombScargle
from sklearn.decomposition import FastICA
from scipy.stats import skew

from ocularrigidity.motion.results import CardiacPipelineResults


class CardiacCycleExtractor:
    def __init__(
        self,
        registrator: RegisteredVideo,
        timestamps_path: Path,
        *,
        # Physiological prior
        bpm_range: tuple[float, float] = (30.0, 180.0),
        butter_order: int = 4,
        override_bpm: Optional[float] = None,
        expected_bpm: Optional[float] = None,
        # Frame trimming
        skip_first_n_frames: int = 3,
        drop_last_n_frames: int = 0,
        # Spatial smoother
        sigma_col: float = 5.0,
        col_slice: slice = None,
        n_separable_components: int = 16,
        ica_random_state: int = 0,
        # Lomb-Scargle scoring
        ls_freq_oversample: float = 5.0,
        ls_concentration_band_hz: float = 0.1,
        # IQ demodulation
        phase_smoother_cycles: float = 2.0,
        phase_density_threshold: float = 0.5,
        # Misc
        verbose: bool = True,
        ICA_or_PCA: str = "ICA",
        harmonic_correction: bool = True,
        harmonic_tolerance_bpm: float = 12.0,
        harmonic_min_power_ratio: float = 0.2,
    ):
        self.registrator = registrator
        self.timestamps_path = timestamps_path

        self.bpm_range = bpm_range
        self.butter_order = butter_order
        self.expected_bpm = expected_bpm

        self.skip_first_n_frames = skip_first_n_frames
        self.drop_last_n_frames = drop_last_n_frames

        self.sigma_col = sigma_col
        self.col_slice = col_slice

        self.n_separable_components = n_separable_components
        self.ica_random_state = ica_random_state

        self.ls_freq_oversample = ls_freq_oversample
        self.ls_concentration_band_hz = ls_concentration_band_hz

        self.phase_smoother_cycles = phase_smoother_cycles
        self.phase_density_threshold = phase_density_threshold

        self.verbose = verbose

        self.ICA_or_PCA = ICA_or_PCA
        self.harmonic_correction = harmonic_correction
        self.harmonic_tolerance_bpm = harmonic_tolerance_bpm
        self.harmonic_min_power_ratio = harmonic_min_power_ratio
        # ---- caches --------------------------------------------------
        self._timestamps_seconds = None
        self._uniform_time = None
        self._gap_mask = None
        self._interpolated_thickness = None
        self._interpolated_validity = None
        self._filtered_signal = None

        self._component_kept_mask = None
        self._separable_components = None
        self._ica_mixing = None

        self._ls_results = None
        self._best_component_idx = None
        # _cardiac_freq is populated either by override_bpm in __init__ or
        # lazily by _ensure_cardiac_selection.
        self._cardiac_freq = (
            float(override_bpm) / 60.0 if override_bpm is not None else None
        )
        self._is_freq_overridden = override_bpm is not None

        self._phase_uniform = None
        self._good_uniform = None
        self._phase_per_frame = None
        self._good_per_frame = None

        # Folding result (populated by compute_n_cycle_video)
        self.cycles: Optional[np.ndarray] = None
        self.counts: Optional[np.ndarray] = None
        self.n_bins: Optional[int] = None
        self.n_cycle: Optional[int] = None

        self.notes: list[str] = []

        self._phase_uniform_pl = None
        self._good_uniform_pl = None
        self._phase_per_frame_pl = None
        self._good_per_frame_pl = None
        self._thickness = None

    @property
    def thickness(self):
        if self._thickness is None:
            """Thickness restricted to ``col_slice``, with outlier frames → NaN."""
            src = self.registrator.thickness
            thickness = (
                src[:, self.col_slice] if self.col_slice is not None else src
            ).copy()

            has_holes = np.isnan(thickness).any(axis=1) | (thickness == 0).any(axis=1)
            # If the holes are on the boundary
            # Find the closest non-hole pixel to the left and right of the hole, and check if either is close enough to be a reliable proxy for thickness.
            # Find the smallest and largest x index that is not a hole for each frame
            x_valid = np.where(~np.isnan(thickness) & (thickness != 0))[1]
            slices = slice(np.min(x_valid), np.max(x_valid) + 1)
            thickness = thickness[:, slices]

            clean = thickness[~has_holes]
            if clean.size == 0:
                if self.verbose:
                    print("All frames contain holes; thickness fully masked.")
                thickness[:] = np.nan
                return thickness

            med = np.nanmedian(clean)
            frame_mean = np.nanmean(thickness, axis=1)
            bad_frames = (
                (frame_mean < 0.75 * med) | (frame_mean > 1.75 * med) | has_holes
            )

            n_bad = int(bad_frames.sum())
            if n_bad and self.verbose:
                print(
                    f"Marking {n_bad}/{len(bad_frames)} frames as bad based on "
                    f"outlier thickness (median={med:.1f})"
                )
            thickness[bad_frames] = np.nan
            self._thickness = thickness
        return self._thickness

    @property
    def _frame_slice(self) -> slice:
        end = None if self.drop_last_n_frames == 0 else -self.drop_last_n_frames
        return slice(self.skip_first_n_frames, end)

    @property
    def timestamps_seconds(self):
        if self._timestamps_seconds is None:
            ts_df = (
                pd.read_csv(self.timestamps_path, header=None, names=["timestamp"])
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
            ts_df = ts_df[self._frame_slice]
            ts_us = ts_df["timestamp"].to_numpy()
            self._timestamps_seconds = (ts_us - ts_us[0]) / 1e6
        return self._timestamps_seconds

    @property
    def uniform_time(self):
        if self._uniform_time is None:
            ts = self.timestamps_seconds
            dt = self.dt
            n = int(np.floor((ts[-1] - ts[0]) / dt)) + 1
            self._uniform_time = ts[0] + np.arange(n) * dt
        return self._uniform_time

    @property
    def dt(self) -> float:
        return float(np.median(np.diff(self.timestamps_seconds)))

    @property
    def fs(self) -> float:
        return 1.0 / self.dt

    @property
    def gap_mask(self):
        """True where the uniform grid is far from any real timestamp."""
        if self._gap_mask is None:
            dt_p95 = float(np.percentile(np.diff(self.timestamps_seconds), 95))
            dists, nearest_idx = cKDTree(self.timestamps_seconds[:, None]).query(
                self.uniform_time[:, None], k=1
            )
            far_from_any_frame = dists > 2 * dt_p95
            bad_frame = np.isnan(self.thickness).all(axis=1)
            nearest_is_bad = bad_frame[nearest_idx]
            self._gap_mask = far_from_any_frame | nearest_is_bad
        return self._gap_mask

    @property
    def gap_fraction(self) -> float:
        return float(self.gap_mask.mean())

    @property
    def interpolated_thickness(self):
        if self._interpolated_thickness is None:
            thickness = self.thickness
            valid = ~np.isnan(thickness).any(axis=1)
            if not valid.any():
                self._interpolated_thickness = np.full(
                    (len(self.uniform_time), thickness.shape[1]),
                    np.nan,
                    dtype=thickness.dtype,
                )
            else:
                self._interpolated_thickness = interp1d(
                    self.timestamps_seconds[valid],
                    thickness[valid],
                    axis=0,
                    kind="linear",
                    fill_value=np.nan,
                    bounds_error=False,
                )(self.uniform_time)
            self._interpolated_thickness[self.gap_mask] = np.nan
        return self._interpolated_thickness

    @property
    def interpolated_validity(self):
        if self._interpolated_validity is None:
            valid = (~np.isnan(self.thickness)).astype(np.float32)
            self._interpolated_validity = interp1d(
                self.timestamps_seconds,
                valid,
                axis=0,
                kind="linear",
                fill_value=0.0,
                bounds_error=False,
            )(self.uniform_time)
            self._interpolated_validity[self.gap_mask] = 0.0
        return self._interpolated_validity

    @property
    def filtered_signal(self):
        """Spatially smoothed (NaN-aware) and temporally bandpassed thickness map."""
        if self._filtered_signal is not None:
            return self._filtered_signal

        # 1. Spatial Gaussian along W (NaN-aware) ---------------------
        not_gap = (~self.gap_mask).astype(np.float32)[:, None]
        data_masked = np.nan_to_num(self.interpolated_thickness, nan=0.0) * not_gap
        valid_masked = self.interpolated_validity * not_gap
        num = gaussian_filter1d(
            data_masked, sigma=self.sigma_col, axis=1, mode="nearest"
        )
        den = gaussian_filter1d(
            valid_masked, sigma=self.sigma_col, axis=1, mode="nearest"
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            spatial = np.where(den > 0.1, num / den, np.nan)

        # 2. Temporal Butterworth bandpass along T --------------------
        nyq = 0.5 * self.fs
        low = (self.bpm_range[0] / 60.0) / nyq
        high = (self.bpm_range[1] / 60.0) / nyq

        nan_mask = np.isnan(spatial)
        filled = spatial.copy()
        t_idx = np.arange(filled.shape[0])
        for w in range(filled.shape[1]):
            m = nan_mask[:, w]
            if m.all():
                filled[:, w] = 0.0
            elif m.any():
                filled[m, w] = np.interp(t_idx[m], t_idx[~m], filled[~m, w])

        num_taps = 101  # Must be odd for bandpass
        taps = firwin(num_taps, [low, high], pass_zero=False)
        filtered = filtfilt(taps, 1.0, filled, axis=0)
        # sos = butter(self.butter_order, [low, high], btype="bandpass", output="sos")
        # filtered = sosfiltfilt(sos, filled, axis=0)
        filtered[nan_mask] = np.nan
        filtered[self.gap_mask] = np.nan

        self._filtered_signal = filtered
        return self._filtered_signal

    @property
    def component_kept_mask(self):
        """Boolean mask on uniform_time: True where ``filtered_signal`` has no NaN."""
        if self._component_kept_mask is None:
            self._component_kept_mask = ~np.isnan(self.filtered_signal).any(axis=1)
        return self._component_kept_mask

    def _compute_separable_components(self):
        keep = self.component_kept_mask
        X = self.filtered_signal[keep]
        X = X - X.mean(axis=0, keepdims=True)

        if self.ICA_or_PCA.lower() == "pca":
            pca = PCA(n_components=self.n_separable_components)
            self._separable_components = pca.fit_transform(X)  # (T_kept, n_ic)
            # PCA convention: components_ has shape (n_ic, W). Transpose to match
            # FastICA's mixing_ shape (W, n_ic) so downstream code (projection
            # back to thickness, ica_mixing[:, best]) doesn't need to branch.
            self._ica_mixing = pca.components_.T  # (W, n_ic)
        elif self.ICA_or_PCA.lower() == "ica":
            ica = FastICA(
                n_components=self.n_separable_components,
                random_state=self.ica_random_state,
                whiten="unit-variance",
                max_iter=5000,
                tol=0.001,
                fun="cube",
            )
            self._separable_components = ica.fit_transform(X)  # (T_kept, n_ic)
            self._ica_mixing = ica.mixing_  # (W, n_ic)
        else:
            raise ValueError(
                f"ICA_or_PCA must be 'ICA' or 'PCA', got {self.ICA_or_PCA!r}"
            )

    def _standardize_ic_sign(self):
        """Flip best IC so its sign matches physical thickness pulsation."""
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

        f_min = self.bpm_range[0] / 60.0
        f_max = self.bpm_range[1] / 60.0
        T_total = t_kept[-1] - t_kept[0]
        df = 1.0 / (self.ls_freq_oversample * T_total)
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

        delta = self.ls_concentration_band_hz
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
        # log so a 10× FAP gap doesn't drown the peak-power term
        quality = (
            np.log10(peak_power + 1e-12)
            + concentration
            + 0.05 * -np.log10(fap + 1e-300)
        )
        if self.expected_bpm is not None:
            f_exp = self.expected_bpm / 60.0
            sigma = self.harmonic_tolerance_bpm / 60.0
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

        if self._cardiac_freq is not None:
            # BPM was overridden — pick IC with most LS power at f0
            j = int(np.argmin(np.abs(ls["freqs"] - self._cardiac_freq)))
            self._best_component_idx = int(np.argmax(ls["power"][j, :]))
            if self.verbose:
                print(
                    f"BPM fixed to {self._cardiac_freq * 60:.1f}; "
                    f"selected IC {self._best_component_idx} "
                    f"(LS power at f0 = {ls['power'][j, self._best_component_idx]:.3f}, "
                    f"FAP = {ls['fap'][self._best_component_idx]:.2e})"
                )
            return

        #
        # 1. Initial pick: best IC by quality (FAP × concentration), at its peak
        best = int(np.argmax(ls["quality"]))
        f_picked = float(ls["peak_freq"][best])
        f_corrected = f_picked
        best_corrected = best

        # 2. Optional harmonic correction
        if self.harmonic_correction and self.expected_bpm is not None:
            bpm_picked = f_picked * 60.0
            bpm_exp = self.expected_bpm
            tol_bpm = self.harmonic_tolerance_bpm

            # Is the picked frequency near a harmonic of the expected rate?
            candidate = None
            if abs(bpm_picked - 2 * bpm_exp) < tol_bpm:
                candidate = f_picked / 2.0
            elif abs(bpm_picked - 0.5 * bpm_exp) < tol_bpm:
                candidate = f_picked * 2.0

            if candidate is not None:
                in_band = (
                    self.bpm_range[0] / 60.0 <= candidate <= self.bpm_range[1] / 60.0
                )

                if in_band:
                    # Search across ALL ICs for the one with most power at the
                    # candidate frequency. The IC that won at f_picked may not
                    # be the IC carrying the true cardiac signal at the harmonic.
                    j_pick = int(np.argmin(np.abs(ls["freqs"] - f_picked)))
                    j_cand = int(np.argmin(np.abs(ls["freqs"] - candidate)))

                    power_at_cand = ls["power"][j_cand, :]  # (n_ic,)
                    ic_best_at_cand = int(np.argmax(power_at_cand))
                    # Compare the strongest evidence at candidate to the
                    # strongest evidence at picked: this is a fair "is there
                    # actually a signal at the harmonic?" test that doesn't
                    # punish us for IC 4 having weak power at 2× its peak.
                    power_at_pick = ls["power"][j_pick, best]
                    ratio_power = power_at_cand[ic_best_at_cand] / power_at_pick

                    if ratio_power > self.harmonic_min_power_ratio:
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
                            f"(threshold {self.harmonic_min_power_ratio})."
                        )
                else:
                    msg = (
                        f"Harmonic candidate {candidate * 60:.1f} bpm rejected: "
                        f"out of band {self.bpm_range}."
                    )

                self.notes.append(msg)
                if self.verbose:
                    print(f"  {msg}")

        # 3. Commit selection
        self._best_component_idx = best_corrected
        self._cardiac_freq = f_corrected
        self._standardize_ic_sign()
        if self.verbose:
            print(
                f"Selected IC {self._best_component_idx}: "
                f"BPM = {self._cardiac_freq * 60:.1f}, "
                f"FAP = {ls['fap'][self._best_component_idx]:.2e}, "
                f"concentration = {ls['concentration'][self._best_component_idx]:.2f}"
            )

        # 4. Sanity check vs. expected
        if self.expected_bpm is not None:
            err = abs(self._cardiac_freq * 60 - self.expected_bpm) / self.expected_bpm
            if err > 0.15:
                msg = (
                    f"Estimated {self._cardiac_freq * 60:.1f} bpm differs from "
                    f"expected {self.expected_bpm:.1f} bpm by {err:.1%}."
                )
                self.notes.append(msg)
                if self.verbose:
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

        # Lift IC back onto the full uniform_time grid (NaN where we had no data)
        ic_full = np.full(len(self.uniform_time), np.nan)
        ic_full[self.component_kept_mask] = self.separable_components[:, best]

        t = self.uniform_time
        dt = self.dt

        # Demodulate at f0: brings the cardiac component down to DC
        phasor = np.exp(-1j * 2 * np.pi * f0 * t)
        z = ic_full * phasor  # NaNs propagate

        sigma_t = (self.phase_smoother_cycles / f0) / dt
        nan = np.isnan(ic_full)
        valid = (~nan).astype(np.float32)
        I_filled = np.where(nan, 0.0, z.real)
        Q_filled = np.where(nan, 0.0, z.imag)

        I_num = gaussian_filter1d(I_filled, sigma=sigma_t, mode="nearest")
        Q_num = gaussian_filter1d(Q_filled, sigma=sigma_t, mode="nearest")
        den = gaussian_filter1d(valid, sigma=sigma_t, mode="nearest")
        good = den > self.phase_density_threshold
        safe_den = np.where(good, den, 1.0)
        I_lp = np.where(good, I_num / safe_den, 0.0)
        Q_lp = np.where(good, Q_num / safe_den, 0.0)

        env_phase = np.arctan2(Q_lp, I_lp)
        # Total instantaneous phase: linear at 2π f0 t plus the slow envelope phase
        phase_total = 2 * np.pi * f0 * t + env_phase
        phase_uniform = np.mod(phase_total, 2 * np.pi)

        self._phase_uniform = phase_uniform
        self._good_uniform = good

        # Map phase to original (non-uniform) frame times via the unit circle.
        # Use only the "good" uniform samples so missing regions don't drag the
        # interpolation toward zero.
        good_idx = np.where(good)[0]
        if len(good_idx) == 0:
            T_orig = len(self.timestamps_seconds)
            self._phase_per_frame = np.full(T_orig, np.nan)
            self._good_per_frame = np.zeros(T_orig, dtype=bool)
            return

        z_unit = np.exp(1j * phase_uniform[good_idx])
        zr = np.interp(self.timestamps_seconds, t[good_idx], z_unit.real)
        zi = np.interp(self.timestamps_seconds, t[good_idx], z_unit.imag)
        # Output is a fraction in [0, 1) to match what the folding kernels expect.
        self._phase_per_frame = (np.angle(zr + 1j * zi) / (2 * np.pi)) % 1.0

        # A frame is "good" iff its uniform-grid neighbourhood is good.
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

        min_dist = int(0.7 / f0 * fs)  # reject peaks closer than 70% of expected period
        prom = 0.4 * np.nanstd(ic_full)
        peaks, _ = find_peaks(
            np.nan_to_num(ic_full), distance=min_dist, prominence=prom
        )

        # Reject anomalous-duration cycles (saccades, missed beats)
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

        # Map to per-frame via unit circle (same trick as _compute_phase)
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
    def inst_bpm(self):
        """Instantaneous BPM trace on uniform_time (diagnostic).

        Computed per contiguous good-region; NaN elsewhere. Useful for
        sanity-checking that the chosen f0 + envelope phase tracks a
        physiologically plausible heart rate.
        """
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

    # ------------------------------------------------------------------
    # Confidence label
    # ------------------------------------------------------------------
    @property
    def confidence(self) -> str:
        """Quick tri-level quality label for batch filtering.

        Based on Lomb-Scargle false-alarm probability and spectral
        concentration of the chosen IC. Note: when ``override_bpm`` is in
        effect the FAP / concentration are still measured at the chosen
        IC's *natural* peak, which may differ from the override — interpret
        accordingly.
        """
        ls = self.lomb_scargle_results
        best = self.best_component_idx
        fap = ls["fap"][best]
        conc = ls["concentration"][best]
        if fap < 1e-10 and conc > 0.5:
            return "high"
        if fap < 1e-3 and conc > 0.3:
            return "medium"
        return "low"

    @property
    def registered_frames(self):
        return self.registrator.registered_frames

    @property
    def registered_masks(self):
        return self.registrator.registered_masks

    def compute_n_cycle_video(
        self,
        *,
        phase_per_frame=None,
        good_per_frame=None,
        phase_method: Literal["iq", "peak_locked"] = "peak_locked",
        cardiac_freq=None,
        n_bins: Optional[int] = None,
        n_cycle: int = 1,
        target_frames_per_bin: int = 25,
        fold_method: str = "mean",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fold ``registered_frames`` into ``n_cycle`` averaged cardiac cycles.

        Defaults to the cached phase / good mask / cardiac_freq, so re-running
        with a different ``n_bins`` does not retrigger the upstream pipeline.
        """
        if phase_per_frame is None or good_per_frame is None:
            if phase_method == "peak_locked":
                ph_default = self.phase_per_frame_peak_locked
                gd_default = self.good_per_frame_peak_locked
            else:
                ph_default = self.phase_per_frame
                gd_default = self.good_per_frame
            if phase_per_frame is None:
                phase_per_frame = ph_default
            if good_per_frame is None:
                good_per_frame = gd_default
        if cardiac_freq is None:
            cardiac_freq = self.cardiac_freq

        n_good = int(good_per_frame.sum())

        if n_bins is None:
            n_bins = _auto_n_bins(
                n_good // max(1, n_cycle),
                fs=self.fs,
                cardiac_freq=cardiac_freq,
                target_per_bin=target_frames_per_bin,
            )
            if self.verbose:
                print(
                    f"Auto-selected n_bins = {n_bins} "
                    f"(per-chunk budget ~{n_good // max(1, n_cycle)} frames)"
                )

        fold_fn = (
            fold_video_numba_mean if fold_method == "mean" else fold_video_numba_median
        )

        t0 = self.timestamps_seconds[0]
        chunk_duration = (self.timestamps_seconds[-1] - t0) / n_cycle

        cycles_per_chunk: list[np.ndarray | None] = []
        counts_per_chunk: list[np.ndarray | None] = []

        for i in range(n_cycle):
            t_lo = t0 + i * chunk_duration
            t_hi = t0 + (i + 1) * chunk_duration
            if i == n_cycle - 1:
                chunk_mask = (self.timestamps_seconds >= t_lo) & (
                    self.timestamps_seconds <= t_hi
                )
            else:
                chunk_mask = (self.timestamps_seconds >= t_lo) & (
                    self.timestamps_seconds < t_hi
                )

            n_good_chunk = int(good_per_frame[chunk_mask].sum())
            if n_good_chunk < n_bins:
                msg = (
                    f"Chunk {i + 1}/{n_cycle} (t={t_lo:.1f}-{t_hi:.1f}s): "
                    f"{n_good_chunk} good frames < n_bins={n_bins}; skipping."
                )
                self.notes.append(msg)
                if self.verbose:
                    print(f"  WARNING: {msg}")
                cycles_per_chunk.append(None)
                counts_per_chunk.append(None)
                continue

            chunk_cycle, chunk_counts = fold_fn(
                self.registered_frames[chunk_mask],
                phase_per_frame[chunk_mask],
                good_per_frame[chunk_mask],
                n_bins=n_bins,
                verbose=self.verbose,
            )
            cycles_per_chunk.append(chunk_cycle)
            counts_per_chunk.append(chunk_counts)

        first_ok = next((c for c in cycles_per_chunk if c is not None), None)
        if first_ok is None:
            raise RuntimeError(
                f"All {n_cycle} chunks were skipped; no chunk had ≥{n_bins} good frames."
            )
        fill_value = np.nan if np.issubdtype(first_ok.dtype, np.floating) else 0
        placeholder_cycle = np.full_like(first_ok, fill_value)
        first_counts = next(c for c in counts_per_chunk if c is not None)
        placeholder_counts = np.zeros_like(first_counts)

        cycles_per_chunk = [
            c if c is not None else placeholder_cycle for c in cycles_per_chunk
        ]
        counts_per_chunk = [
            c if c is not None else placeholder_counts for c in counts_per_chunk
        ]

        cycles = np.concatenate(cycles_per_chunk, axis=0)
        counts = np.concatenate(counts_per_chunk, axis=0)

        self.cycles = cycles
        self.counts = counts
        self.n_bins = n_bins
        self.n_cycle = n_cycle

        return cycles, counts


def run_cardiac_pipeline(
    video_relpath: str,
    *,
    root_masks: str,
    root_data: str,
    timestamps_path: str,
    # Physiological prior
    bpm_range: tuple[float, float] = (30.0, 180.0),
    override_bpm: Optional[float] = None,
    expected_bpm: Optional[float] = None,
    butter_order: int = 4,
    # Frame trimming
    skip_first_n_frames: int = 3,
    drop_last_n_frames: int = 0,
    # Registration
    refine_iters: int = 2,
    min_pts: int = 10,
    transform: str = "tilt",
    flatten: bool = True,
    horizontal_scaling: bool = False,
    horizontal_alignment: bool = True,
    # Spatial smoother
    sigma_col: float = 5.0,
    col_slice: slice = None,
    # ICA + LS + phase
    n_separable_components: int = 16,
    phase_smoother_cycles: float = 2.0,
    harmonic_correction: bool = True,
    # Optional fold
    compute_n_cycle_video: bool = False,
    n_cycle: int = 1,
    n_bins: Optional[int] = 75,
    target_frames_per_bin: int = 25,
    one_cycle_fold_method: str = "mean",
    ICA_or_PCA: str = "ICA",
    # Misc
    verbose: bool = True,
    use_encoded_video: bool = True,
    phase_method_for_fold: Literal["iq", "peak_locked"] = "peak_locked",
) -> CardiacPipelineResults:
    """Run the cardiac pipeline and return the populated extractor."""
    registrator = RegisteredVideo(
        video=video_relpath,
        root_data=Path(root_data),
        root_masks=Path(root_masks),
        skip_first_n_frames=skip_first_n_frames,
        drop_last_n_frames=drop_last_n_frames,
        refine_iters=refine_iters,
        min_pts=min_pts,
        transform=transform,
        flatten=flatten,
        horizontal_scaling=horizontal_scaling,
        horizontal_alignment=horizontal_alignment,
        verbose=verbose,
        use_encoded_video=use_encoded_video,
    )
    extractor = CardiacCycleExtractor(
        registrator=registrator,
        timestamps_path=timestamps_path,
        bpm_range=bpm_range,
        override_bpm=override_bpm,
        expected_bpm=expected_bpm,
        butter_order=butter_order,
        skip_first_n_frames=skip_first_n_frames,
        drop_last_n_frames=drop_last_n_frames,
        sigma_col=sigma_col,
        col_slice=col_slice,
        n_separable_components=n_separable_components,
        phase_smoother_cycles=phase_smoother_cycles,
        verbose=verbose,
        ICA_or_PCA=ICA_or_PCA,
        harmonic_correction=harmonic_correction,
    )

    if verbose:
        ts = extractor.timestamps_seconds
        print(f"fs = {extractor.fs:.2f} Hz, duration = {ts[-1]:.1f}s, T = {len(ts)}")
        print(f"Gap fraction on uniform grid: {extractor.gap_fraction:.2%}")

    _ = extractor.phase_per_frame

    if verbose:
        T = len(extractor.timestamps_seconds)
        print(f"Good frames for folding: {int(extractor.good_per_frame.sum())} / {T}")
        print(
            f"Cardiac rate: {extractor.cardiac_bpm:.1f} bpm "
            f"(confidence={extractor.confidence})"
        )

    if compute_n_cycle_video:
        extractor.compute_n_cycle_video(
            n_bins=n_bins,
            n_cycle=n_cycle,
            target_frames_per_bin=target_frames_per_bin,
            fold_method=one_cycle_fold_method,
            phase_method=phase_method_for_fold,
        )

    return CardiacPipelineResults.from_extractor(extractor)
