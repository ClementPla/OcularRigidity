from typing import Optional
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import (
    firwin,
    filtfilt,
)


def spatio_temporal_filter(
    signal: np.ndarray,
    spatial_sigma: float,
    temporal_low_freq: float,
    temporal_high_freq: float,
    fs: float,
    validity_mask: Optional[np.ndarray] = None,
):
    """
    Applies a spatial low pass filter and and temporal pass-band filer

    Args:
        signal (np.ndarray): TxW array of values to filter. The signal is assumed to be interpolated over time, with a uniform sampling.
        On invalid frames, validity_mask can be used to ignore those pixels in the filtering process.
        spatial_sigma (float): Standard deviation of the Gaussian kernel for spatial filtering, in pixels.
        temporal_low_freq (float): Low frequency cutoff for the temporal bandpass filter, relative to the sampling frequency fs
        temporal_high_freq (float): High frequency cutoff for the temporal bandpass filter, relative to the sampling frequency fs
        fs (float): Sampling frequency of the signal, in Hz
        validity_mask (Optional[np.ndarray], optional): TxW boolean array indicating which pixels are valid (True) or invalid (False). If provided, the filter will ignore invalid pixels. Defaults to None.
    """

    num = gaussian_filter1d(signal, sigma=spatial_sigma, axis=1, mode="nearest")
    den = gaussian_filter1d(
        validity_mask.astype(float)
        if validity_mask is not None
        else np.ones_like(signal),
        sigma=spatial_sigma,
        axis=1,
        mode="nearest",
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        spatial = np.where(den > 0.1, num / den, np.nan)

    nan_mask = np.isnan(spatial)
    filled = spatial.copy()
    n_t = filled.shape[0]
    t_idx = np.arange(n_t)
    for w in range(filled.shape[1]):
        m = nan_mask[:, w]
        if m.all():
            filled[:, w] = 0.0
        elif m.any():
            filled[m, w] = np.interp(t_idx[m], t_idx[~m], filled[~m, w])

    num_taps = int(round(4.0 * fs / temporal_low_freq))
    max_taps = max(3, (n_t // 3) | 1)  # keep 3*num_taps < n_t for default padlen
    num_taps = min(max(num_taps, 31), max_taps)
    if num_taps % 2 == 0:
        num_taps += 1
    if num_taps > max_taps:
        num_taps -= 2
    taps = firwin(num_taps, [temporal_low_freq, temporal_high_freq], pass_zero=False)
    padlen = min(3 * num_taps, n_t - 1)
    filtered = filtfilt(taps, 1.0, filled, axis=0, padlen=padlen)

    filtered[nan_mask] = np.nan
    return filtered
