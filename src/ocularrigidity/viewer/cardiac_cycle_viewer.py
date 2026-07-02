import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.timeseries import LombScargle
import numpy as np
from ocularrigidity.motion.pulsation import AbstractPulseExtractor
from ocularrigidity.motion.pipeline_results import CardiacPipelineResults


def plot_cardiac_signals(
    CC_ex: AbstractPulseExtractor | CardiacPipelineResults,
    include_2d_thickness: bool = True,
    include_2d_thickness_filtered: bool = True,
    include_1d_mean_thickness: bool = True,
    include_reconstructed_cycle: bool = True,
    include_instantaneous_bpm: bool = True,
    include_best_ic: bool = True,
    include_power_spectrum: bool = True,
    include_phase: bool = True,
    include_spectrum_mean_thickness: bool = True,
    include_spectrum_mean_thickness_filtered: bool = True,
    include_expected_cardiac_freq: bool = True,
    dimensions: tuple = (12, 3),
):
    N_axes = (
        include_2d_thickness
        + include_2d_thickness_filtered
        + include_reconstructed_cycle
        + include_best_ic
        + include_power_spectrum
        + include_instantaneous_bpm
    )
    fig, axes = plt.subplots(
        N_axes, 1, figsize=(dimensions[0], dimensions[1] * N_axes), squeeze=False
    )
    axes = axes.flatten()
    ax_idx = 0
    uniform_time = CC_ex.uniform_time
    raw_t = CC_ex.timestamps_seconds

    best = CC_ex.best_component_idx
    ic_full = np.full(len(CC_ex.component_kept_mask), np.nan)
    ic_full[CC_ex.component_kept_mask] = CC_ex.separable_components[:, best]

    if include_2d_thickness:
        ax = axes[ax_idx]
        t_uniform = np.linspace(0, raw_t.max(), len(raw_t))
        thick_uniform = interp1d(raw_t, CC_ex.signal, axis=0, bounds_error=False)(
            t_uniform
        )
        ax.imshow(
            thick_uniform.T,
            aspect="auto",
            cmap="viridis",
            origin="lower",
            extent=[0, raw_t.max(), 0, CC_ex.signal.shape[1]],
        )

        ax.set_title("2D Thickness Map (Time x Col)")
        ax.set_ylabel("Column Index")
        ax.set_xlabel("Time (s)")
        if include_1d_mean_thickness:
            mean_thickness = np.nanmean(CC_ex.signal, axis=1)
            # Drop point outside 1st and 99th percentile to avoid outliers dominating the plot
            valid_idx = ~np.isnan(mean_thickness)
            mean_thickness = np.nan_to_num(
                mean_thickness, nan=np.nanmean(mean_thickness)
            )

            # mean_thickness = mean_thickness[valid_idx]
            p1, p99 = np.percentile(mean_thickness, [1, 99])
            mean_thickness = np.clip(mean_thickness, p1, p99)
            # Normalize to one
            mean_thickness = (mean_thickness - mean_thickness.min()) / (
                mean_thickness.max() - mean_thickness.min()
            )
            # Multiply it to cover half the height of the plot
            mean_thickness = mean_thickness * (CC_ex.filtered_signal.T.shape[0] / 2) + (
                CC_ex.filtered_signal.T.shape[0] / 4
            )
            mean_thickness[~valid_idx] = np.nan
            ax.plot(
                raw_t,
                mean_thickness,
                color="red",
                label="Mean Thickness",
                linewidth=0.5,
                linestyle="--",
            )
            ax.legend()
        ax_idx += 1
    if include_2d_thickness_filtered:
        ax = axes[ax_idx]
        ax.imshow(
            CC_ex.filtered_signal.T,
            aspect="auto",
            cmap="viridis",
            origin="lower",
            extent=[0, uniform_time.max(), 0, CC_ex.filtered_signal.T.shape[0]],
        )
        ax.set_title("Smoothed 2D Thickness Map (Time x Col)")
        ax.set_ylabel("Column Index")
        ax.set_xlabel("Time (s)")
        if include_1d_mean_thickness:
            mean_thickness = np.nanmean(CC_ex.filtered_signal, axis=1)
            # Drop point outside 1st and 99th percentile to avoid outliers dominating the plot
            valid_idx = ~np.isnan(mean_thickness)
            # mean_thickness = mean_thickness[valid_idx]
            mean_thickness = np.nan_to_num(
                mean_thickness, nan=np.nanmean(mean_thickness)
            )
            p1, p99 = np.percentile(mean_thickness, [1, 99])
            mean_thickness = np.clip(mean_thickness, p1, p99)
            # Normalize to one
            mean_thickness = (mean_thickness - mean_thickness.min()) / (
                mean_thickness.max() - mean_thickness.min()
            )
            # Multiply it to cover half the height of the plot
            mean_thickness = mean_thickness * (CC_ex.filtered_signal.T.shape[0] / 2) + (
                CC_ex.filtered_signal.T.shape[0] / 4
            )
            # Put back the non_valid points to nan for better visualization
            mean_thickness[~valid_idx] = np.nan
            ax.plot(
                uniform_time,
                mean_thickness,
                color="red",
                label="Mean Thickness",
                linewidth=2,
                linestyle="-",
            )
            ax.legend()
        ax_idx += 1
    if include_best_ic:
        ax = axes[ax_idx]
        ax.plot(
            uniform_time,
            ic_full,
            label=f"Best IC (idx {best})",
            color="green",
            linewidth=1,
        )
        if include_phase:
            ax.plot(
                uniform_time,
                CC_ex.phase_uniform / (2 * np.pi),
                label="Phase (0-1) from IQ demodulation",
                color="orange",
                linewidth=0.5,
                linestyle="--",
            )
            ax.plot(
                uniform_time,
                CC_ex.phase_uniform_peak_locked / (2 * np.pi),
                label="Phase (0-1) from peak-locking",
                color="blue",
                linewidth=0.5,
            )

        if include_1d_mean_thickness:
            mean_thickness = np.nanmean(CC_ex.filtered_signal, axis=1)
            # Make sure it has roughly the same scale as the IC for better visualization
            ic_scale = np.nanmax(np.abs(ic_full))
            mean_thickness = mean_thickness / np.nanmax(mean_thickness) * ic_scale
            valid_idx = ~np.isnan(mean_thickness)
            ax.plot(
                uniform_time,
                mean_thickness,
                color="red",
                label="Mean Thickness",
                linewidth=0.5,
                linestyle="-",
            )
        ax.legend()
        ax.set_title("Best Independent Component")
        ax.set_ylabel("IC Amplitude (a.u.)")

        ax_idx += 1
    if include_power_spectrum:
        ax = axes[ax_idx]
        ls_results = CC_ex.lomb_scargle_results
        freqs = ls_results["freqs"]
        power = ls_results["power"][:, best]
        ax.plot(freqs, power, label=f"IC {best} Power Spectrum", color="green")
        ax.set_title("Power Spectrum")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power Spectral Density")
        cardiac_freq = CC_ex.cardiac_freq
        ax.axvline(
            cardiac_freq,
            color="gray",
            linestyle="--",
            label=f"Detected Cardiac Frequency ({cardiac_freq * 60:.1f} BPM)",
        )
        if include_expected_cardiac_freq and CC_ex.expected_bpm is not None:
            expected_freq = CC_ex.expected_bpm / 60
            ax.axvline(
                expected_freq,
                color="blue",
                linestyle="--",
                label=f"Expected Cardiac Frequency ({expected_freq * 60:.1f} BPM)",
            )
        if include_spectrum_mean_thickness:
            mean_thickness = np.nanmean(CC_ex.signal, axis=1)
            # Use lomb-scargle to get the spectrum of the mean thickness
            valid_idx = ~np.isnan(mean_thickness)
            bpm_range = CC_ex.bpm_range
            freqs = np.linspace(bpm_range[0] / 60, bpm_range[1] / 60, 1000)

            ls = LombScargle(raw_t[valid_idx], mean_thickness[valid_idx])
            power_mt = ls.power(freqs)
            ax.plot(
                freqs,
                power_mt,
                label="Mean Thickness Spectrum",
                color="red",
            )
        if include_spectrum_mean_thickness_filtered:
            mean_thickness = np.nanmean(CC_ex.filtered_signal, axis=1)
            valid_idx = ~np.isnan(mean_thickness)
            bpm_range = CC_ex.bpm_range
            freqs = np.linspace(bpm_range[0] / 60, bpm_range[1] / 60, 1000)

            ls = LombScargle(uniform_time[valid_idx], mean_thickness[valid_idx])
            power_mt = ls.power(freqs)
            ax.plot(
                freqs,
                power_mt,
                label="Smoothed Mean Thickness Spectrum",
                color="orange",
            )
        # Set the legend as horizontal above the plot
        ax.legend(loc="upper center", bbox_to_anchor=(0.2, 1), ncol=1)
        ax_idx += 1

    if include_reconstructed_cycle:
        ax = axes[ax_idx]

        from scipy.signal import find_peaks

        fs = 1 / np.median(np.diff(uniform_time))
        ic = ic_full.copy()
        if np.nanmean(ic[find_peaks(ic, distance=int(0.4 * fs))[0]]) < -np.nanmean(
            ic[find_peaks(-ic, distance=int(0.4 * fs))[0]]
        ):
            ic = -ic

        peaks, _ = find_peaks(
            ic, distance=int(0.5 * fs), prominence=0.5 * np.nanstd(ic)
        )

        phase_locked = np.full_like(ic, np.nan)
        for p0, p1 in zip(peaks[:-1], peaks[1:]):
            phase_locked[p0:p1] = np.linspace(0, 1, p1 - p0, endpoint=False)

        bins_x, template, err = get_folded_template(
            ic_full[CC_ex.component_kept_mask], phase_locked[CC_ex.component_kept_mask]
        )
        ax.fill_between(
            bins_x, template - err, template + err, alpha=0.2, color="green"
        )
        ax.plot(bins_x, template, label="Reconstructed Signal")
        ax.set_title("Reconstructed Cardiac Signal")
        ax.set_ylabel("Amplitude (a.u.)")
        ax.legend()
        ax_idx += 1
    if include_instantaneous_bpm:
        ax = axes[ax_idx]
        ax.plot(uniform_time, CC_ex.inst_bpm, label="Instantaneous BPM")
        ax.set_title("Instantaneous BPM")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("BPM")
        ax.legend()
        ax_idx += 1

    return fig, axes


def get_folded_template(signal, phase, num_bins=100):
    """
    signal: The Best IC (green line in your Plot 3)
    phase: Your self._phase_per_frame (values in [0, 1))
    num_bins: Number of bins for the template (resolution)
    """
    # 1. Discard NaNs
    mask = ~np.isnan(signal) & ~np.isnan(phase)
    clean_sig = signal[mask]
    clean_phase = phase[mask]

    # 2. Bin the data
    bins = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(clean_phase, bins) - 1

    template = np.zeros(num_bins)
    std_error = np.zeros(num_bins)

    for i in range(num_bins):
        group = clean_sig[bin_indices == i]
        if len(group) > 0:
            template[i] = np.mean(group)
            std_error[i] = np.std(group) / np.sqrt(len(group))  # Reliability indicator
        else:
            template[i] = np.nan

    # 3. Smooth the template (optional but recommended)
    # This handles any empty bins or high-frequency jitter
    template = np.nan_to_num(template, nan=np.nanmean(template))

    return bins[:-1], template, std_error
