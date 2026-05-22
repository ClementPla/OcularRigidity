import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def _median_kernel(frames, bin_offsets, frame_indices, n_bins):
    T, H, W = frames.shape
    cycle = np.zeros((n_bins, H, W), dtype=np.float32)

    max_bin = 0
    for b in range(n_bins):
        sz = bin_offsets[b + 1] - bin_offsets[b]
        if sz > max_bin:
            max_bin = sz

    for h in prange(H):
        scratch = np.empty(max_bin, dtype=np.float32)
        for w in range(W):
            for b in range(n_bins):
                start = bin_offsets[b]
                n = bin_offsets[b + 1] - start
                if n == 0:
                    continue
                for i in range(n):
                    scratch[i] = frames[frame_indices[start + i], h, w]
                view = scratch[:n]
                view.sort()
                if n & 1:
                    cycle[b, h, w] = view[n >> 1]
                else:
                    mid = n >> 1
                    cycle[b, h, w] = 0.5 * (view[mid - 1] + view[mid])
    return cycle


def fold_video_numba_median(frames, phase, good_mask, n_bins, verbose=False):
    bin_idx = (phase * n_bins).astype(np.int32) % n_bins
    good = np.flatnonzero(good_mask).astype(np.int32)
    bins = bin_idx[good]
    order = np.argsort(bins, kind="stable")
    frame_indices = good[order]
    counts = np.bincount(bins, minlength=n_bins).astype(np.int32)
    bin_offsets = np.zeros(n_bins + 1, dtype=np.int32)
    bin_offsets[1:] = np.cumsum(counts)

    cycle = _median_kernel(frames, bin_offsets, frame_indices, n_bins)
    if verbose:
        print(
            f"Fold: counts min/mean/max = {counts.min()}/{counts.mean():.1f}/{counts.max()}"
        )
    return cycle, counts


@njit(parallel=True)
def _numba_mean_kernel(frames, bin_idx, good_mask, n_bins):
    T, H, W = frames.shape
    # Accumulator for sums and counts
    sums = np.zeros(
        (n_bins, H, W), dtype=np.float64
    )  # Use float64 for precision during accumulation
    counts = np.zeros(n_bins, dtype=np.int32)

    # Pass 1: Accumulate sums
    # We loop over T and let Numba handle the parallel distribution
    for t in range(T):
        if good_mask[t]:
            b = bin_idx[t]
            counts[b] += 1
            # We add the frame to the specific bin's sum
            for h in range(H):
                for w in range(W):
                    sums[b, h, w] += frames[t, h, w]

    # Pass 2: Divide by counts to get mean
    cycle = np.zeros((n_bins, H, W), dtype=np.float32)
    for b in prange(n_bins):
        if counts[b] > 0:
            for h in range(H):
                for w in range(W):
                    cycle[b, h, w] = sums[b, h, w] / counts[b]

    return cycle, counts


def fold_video_numba_mean(frames, phase, good_mask, n_bins, verbose=False):
    bin_idx = (phase * n_bins).astype(np.int32) % n_bins

    cycle, counts = _numba_mean_kernel(frames, bin_idx, good_mask, n_bins)

    if verbose:
        print(
            f"Fold: counts min/mean/max = {counts.min()}/{counts.mean():.1f}/{counts.max()}"
        )
    return cycle, counts


def amplify_one_cycle(cycle, amplification_factor=3.0, n_components=3):
    mean = cycle.mean(axis=0, keepdims=True)
    cycle_centered = cycle - mean
    U, s, Vt = np.linalg.svd(
        cycle_centered.reshape(cycle.shape[0], -1), full_matrices=False
    )
    s[:n_components] *= amplification_factor

    reconstructed = ((U * s) @ Vt).reshape(cycle.shape) + mean
    return np.clip(reconstructed, 0, 255).astype(np.uint8)


def _auto_n_bins(
    n_good_frames: int,
    fs: float,
    cardiac_freq: float,
    target_per_bin: int = 25,
) -> int:
    """Choose n_bins from frame budget, capped by per-cycle sampling resolution."""
    from_count = max(8, n_good_frames // target_per_bin)
    from_resolution = max(1, int(0.5 * fs / cardiac_freq))
    return min(from_count, from_resolution)


def fit_cardiac_amplitude(y: np.ndarray, n_harmonics: int = 1) -> float:
    """Peak-to-peak of the harmonic reconstruction of one folded cycle."""
    N = len(y)
    t = np.arange(N) / N  # one full period in [0, 1)
    cols = [np.ones(N)]
    for k in range(1, n_harmonics + 1):
        cols.append(np.cos(2 * np.pi * k * t))
        cols.append(np.sin(2 * np.pi * k * t))
    X = np.stack(cols, axis=1)
    coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    fit = X @ coeffs
    return fit, residuals


def estimate_cardiac_amplitude(
    one_cycle_thickness,
    residual_threshold_percentile=50,
    amplitude_threshold_percentile=50,
    n_harmonics=1,
):
    """Estimate the cardiac amplitude from the one-cycle thickness data.

    Args:
        one_cycle_thickness (np.ndarray): TxW array of thickness values for one cycle.
        residual_threshold_percentile (int): Percentile for filtering fits with high residuals.
        amplitude_threshold_percentile (int): Percentile for filtering fits with low amplitude.
    """
    fits = [
        fit_cardiac_amplitude(one_cycle_thickness[:, w], n_harmonics=n_harmonics)
        for w in range(one_cycle_thickness.shape[1])
    ]

    # Filter out the fits with high residuals (bad fit) and plot the rest (i.e below the 75th percentile of residuals).
    residuals = np.array([res.item() if res.size else np.nan for _, res in fits])
    threshold = np.percentile(residuals, residual_threshold_percentile)
    # Filter out the fits with low amplitude (flat line) by keeping only those with amplitude above the 25th percentile.
    amplitudes = np.array([fit.max() - fit.min() for fit, res in fits])
    amplitude_threshold = np.percentile(amplitudes, amplitude_threshold_percentile)
    keep = (residuals <= threshold) & (amplitudes >= amplitude_threshold)

    fits = np.array([fit for fit, res in fits]).T  # TxW
    fits = fits[:, keep]
    return fits, keep
