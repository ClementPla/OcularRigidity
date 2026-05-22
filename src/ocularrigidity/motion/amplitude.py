import numpy as np


from ocularrigidity.motion.results import CardiacPipelineResults


def systolic_diastolic_amplitude_from_thickness(
    r: CardiacPipelineResults,
    *,
    n_bins: int = 50,
    min_frames_per_bin: int = 5,
    phase_method: str = "peak_locked",
    min_snr: float = 3.0,
    min_bin_coverage: float = 0.6,
    aggregate: str = "median",
) -> float:
    thk = r.filtered_signal  # (T, W)
    if thk.ndim != 2:
        return float("nan")
    T, W = thk.shape

    if phase_method == "peak_locked":
        phase_full = r.phase_uniform_peak_locked
        good_full = r.good_uniform_peak_locked
    else:
        phase_full = r.phase_uniform
        good_full = r.good_uniform

    # Frames usable at all: finite phase, flagged good, and present in
    # the thickness map (gap_mask marks missing samples).
    finite_thk = ~np.isnan(thk).all(axis=1)
    usable = good_full & np.isfinite(phase_full) & finite_thk
    if usable.sum() < n_bins * min_frames_per_bin:
        return float("nan")

    phase = phase_full[usable]
    Y = thk[usable].astype(np.float64)  # (T_u, W)

    # --- phase binning ----------------------------------------------
    bins = np.minimum((phase / (2 * np.pi) * n_bins).astype(int), n_bins - 1)
    template = np.full((n_bins, W), np.nan)  # bin-mean per column
    noise = np.full((n_bins, W), np.nan)  # std-error per bin
    for k in range(n_bins):
        sel = bins == k
        if sel.sum() < min_frames_per_bin:
            continue
        seg = Y[sel]  # (n_k, W)
        cnt = np.sum(~np.isnan(seg), axis=0)
        with np.errstate(invalid="ignore"):
            mean_k = np.nanmean(seg, axis=0)
            std_k = np.nanstd(seg, axis=0)
        enough = cnt >= min_frames_per_bin
        template[k, enough] = mean_k[enough]
        noise[k, enough] = std_k[enough] / np.sqrt(
            cnt[enough]
        )  # Noise estimate on the bin means, per column

    filled_bins = np.sum(~np.isnan(template), axis=0)  # per column
    coverage = filled_bins / n_bins

    with np.errstate(invalid="ignore"):
        amp_per_col = np.nanmax(template, axis=0) - np.nanmin(template, axis=0)
    # noise floor on the peak-to-trough statistic: the extremum picks two
    # bins, so the relevant uncertainty combines their standard errors.
    noise_floor = np.sqrt(2.0) * np.nanmedian(noise, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = amp_per_col / noise_floor

    keep = (
        (coverage >= min_bin_coverage)
        & np.isfinite(amp_per_col)
        & np.isfinite(snr)
        & (snr >= min_snr)
    )
    if not keep.any():
        return float("nan")

    selected = amp_per_col[keep]
    return float(np.median(selected) if aggregate == "median" else np.mean(selected))
