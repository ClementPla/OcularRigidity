
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from tqdm.auto import tqdm
from numpy.lib.stride_tricks import sliding_window_view


@dataclass
class PeriodicityResult:
    frequencies: np.ndarray  # (R,)
    amplitudes: np.ndarray  # (R, N)
    phases: np.ndarray  # (R, N)
    damping: np.ndarray  # (R,)
    extra: dict = field(default_factory=dict)


# ======================================================================
# Shared helper: per-channel amplitude / phase from a set of frequencies.
# ======================================================================
def _fit_amp_phase(X, t, frequencies, damping=None, mask=None):
    """Least-squares fit of x_n(t) = sum_r a_{nr} cos(w_r t - phi_{nr}) e^{-s_r t}.

    Returns
    -------
    amplitudes : (R, N)
    phases     : (R, N)   (radians, in (-pi, pi])
    """
    T_, N_ = X.shape
    R = len(frequencies)
    w = 2 * np.pi * np.asarray(frequencies)
    s = np.zeros(R) if damping is None else np.asarray(damping)

    # Build design matrix with cos/sin pair per mode.
    env = np.exp(-np.outer(t, s))  # (T, R)
    cos_part = env * np.cos(np.outer(t, w))  # (T, R)
    sin_part = env * np.sin(np.outer(t, w))  # (T, R)
    basis = np.concatenate([cos_part, sin_part], axis=1)  # (T, 2R)

    amps = np.zeros((R, N_))
    phis = np.zeros((R, N_))
    for n in range(N_):
        if mask is None:
            y = X[:, n]
            B = basis
        else:
            m = mask[:, n]
            y = X[m, n]
            B = basis[m]
        if len(y) < 2 * R:
            amps[:, n] = np.nan
            phis[:, n] = np.nan
            continue
        coeffs, *_ = np.linalg.lstsq(B, y, rcond=None)  # (2R,)
        A, Bc = coeffs[:R], coeffs[R:]
        amps[:, n] = np.sqrt(A**2 + Bc**2)
        phis[:, n] = np.arctan2(Bc, A)
    return amps, phis


def _build_hankel(X, L):
    """Row-block Hankel with rows ordered (delay, channel)."""
    T, N = X.shape
    M = T - L + 1
    # windows[m, n, i] = X[m + i, n], shape (M, N, L)
    windows = sliding_window_view(X, window_shape=L, axis=0)
    # want H[i*N + n, m] = X[i + m, n]  ->  axes (i, n, m)
    return np.ascontiguousarray(windows.transpose(2, 1, 0).reshape(L * N, M))


def _build_page(Y, n_win, L, N):
    """Stacked Page: for each channel, (L, n_win) blocks concatenated."""
    # Y.reshape(n_win, L, N)[win, i, n] = Y[win*L + i, n]
    # want Page[i, n*n_win + win] = that  ->  axes (i, n, win)
    return Y.reshape(n_win, L, N).transpose(1, 2, 0).reshape(L, N * n_win).copy()


def _unpack_page(Page_hat, n_win, L, N):
    """Inverse of _build_page."""
    return Page_hat.reshape(L, N, n_win).transpose(2, 0, 1).reshape(n_win * L, N).copy()


def _truncated_svd(A, k, n_oversamples=10, n_iter=2, rng=None):
    """Top-k SVD via randomized subspace iteration.
    Falls back to full SVD when k is not small relative to min(A.shape)."""
    m, n = A.shape
    min_dim = min(m, n)
    if k >= min_dim or k > min_dim // 2 or min_dim < 64:
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        return U[:, :k], s[:k], Vt[:k]
    if rng is None:
        rng = np.random.default_rng(0)
    l = min(k + n_oversamples, n)
    Q = rng.standard_normal((n, l))
    # Subspace iteration with re-orthogonalization (stable for flat spectra)
    for _ in range(n_iter):
        Q, _ = np.linalg.qr(A @ Q)
        Q, _ = np.linalg.qr(A.T @ Q)
    Q, _ = np.linalg.qr(A @ Q)  # final left basis, shape (m, l)
    B = Q.T @ A  # small, (l, n)
    Ub, s, Vt = np.linalg.svd(B, full_matrices=False)
    return (Q @ Ub)[:, :k], s[:k], Vt[:k]


# ======================================================================
# 1.  FFT / Lomb-Scargle periodogram
# ======================================================================
def find_periodicity_fft(
    X,
    *,
    dt=None,
    t=None,
    R=1,
    f_min=None,
    f_max=None,
    n_scan=4000,
    min_peak_separation_hz=None,
    verbose=False,
):
    """Find dominant frequencies via mean-periodogram peak picking.

    Uniform sampling  (dt given):   per-channel FFT, averaged in power.
    Irregular sampling (t given):   per-channel Lomb-Scargle, averaged.

    Parameters
    ----------
    X : (T, N) array
        Multivariate signal. NaNs allowed only in the Lomb-Scargle case.
    dt : float, optional
        Sampling interval (seconds). Use for uniformly spaced data.
    t : (T,) array, optional
        Timestamps (seconds). Use for irregularly sampled data.
    R : int
        Number of frequencies to return.
    f_min, f_max : float
        Frequency search range (Hz). Defaults cover the meaningful band.
    n_scan : int
        Resolution of the Lomb-Scargle frequency grid.
    min_peak_separation_hz : float
        Minimum spacing between reported peaks. Default is
        (f_max - f_min) / 50.

    Returns
    -------
    PeriodicityResult
    """
    if (dt is None) == (t is None):
        raise ValueError("Provide exactly one of `dt` or `t`.")
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    T_, N_ = X.shape

    if dt is not None:
        t = np.arange(T_) * dt
        uniform = True
    else:
        t = np.asarray(t, dtype=float)
        uniform = np.allclose(np.diff(t), np.diff(t).mean(), rtol=1e-4)
        dt = np.median(np.diff(t))

    nyquist = 0.5 / dt
    if f_min is None:
        f_min = 0.5 / (t[-1] - t[0])
    if f_max is None:
        f_max = 0.9 * nyquist
    if min_peak_separation_hz is None:
        min_peak_separation_hz = (f_max - f_min) / 50.0

    # Build the mean periodogram.
    if uniform:
        # Standard FFT: per-channel power, averaged, interpolated onto scan grid.
        X_dem = X - np.nanmean(X, axis=0, keepdims=True)
        F = np.fft.rfft(np.nan_to_num(X_dem), axis=0)
        freqs_fft = np.fft.rfftfreq(T_, d=dt)
        power = (np.abs(F) ** 2).mean(axis=1)
        mask_band = (freqs_fft >= f_min) & (freqs_fft <= f_max)
        f_scan = freqs_fft[mask_band]
        pgram = power[mask_band]
    else:
        from scipy.signal import lombscargle

        f_scan = np.linspace(f_min, f_max, n_scan)
        w_scan = 2 * np.pi * f_scan
        pgrams = np.zeros((N_, len(f_scan)))
        for n in tqdm(range(N_), disable=not verbose, leave=False):
            col = X[:, n]
            valid = ~np.isnan(col)
            if valid.sum() < 8:
                continue
            y = col[valid] - col[valid].mean()
            pgrams[n] = lombscargle(t[valid], y, w_scan, normalize=True)
        pgram = pgrams.mean(axis=0)

    # Peak picking: take local maxima ordered by power, enforce separation.
    from scipy.signal import find_peaks

    sep_idx = max(1, int(min_peak_separation_hz / np.mean(np.diff(f_scan))))
    peak_locs, _ = find_peaks(pgram, distance=sep_idx)
    peak_locs = peak_locs[np.argsort(-pgram[peak_locs])][:R]
    peaks = np.sort(f_scan[peak_locs]) if len(peak_locs) else np.array([])

    # If we found fewer peaks than R, pad with NaN.
    if len(peaks) < R:
        peaks = np.concatenate([peaks, np.full(R - len(peaks), np.nan)])

    # Fit per-channel amplitude and phase given the recovered frequencies.
    finite = np.isfinite(peaks)
    amps = np.full((R, N_), np.nan)
    phis = np.full((R, N_), np.nan)
    if finite.any():
        mask = ~np.isnan(X) if not uniform else None
        X_fit = np.nan_to_num(X) if mask is None else X
        a, p = _fit_amp_phase(X_fit, t, peaks[finite], mask=mask)
        amps[finite] = a
        phis[finite] = p

    return PeriodicityResult(
        frequencies=peaks,
        amplitudes=amps,
        phases=phis,
        damping=np.zeros(R),
        extra={"f_scan": f_scan, "pgram": pgram, "uniform": uniform},
    )


# ======================================================================
# 2.  Multichannel ESPRIT on a row-block Hankel matrix
# ======================================================================


def find_periodicity_hankel(
    X,
    *,
    dt,
    R=1,
    L=None,
    include_damping=True,
    keep_unit_circle_tol=0.25,
    verbose=False,
):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    T_, N_ = X.shape
    if np.isnan(X).any():
        raise ValueError(
            "Hankel/ESPRIT needs complete data. Use find_periodicity_page."
        )

    if L is None:
        L = max(2 * R + 2, T_ // 3)
    if L >= T_:
        raise ValueError(f"L={L} must be < T={T_}.")

    K_modes = 2 * R
    H = _build_hankel(X, L)

    # Only the top-K_modes left singular vectors are needed.
    U_sig, sv, _ = _truncated_svd(H, K_modes, n_iter=2)

    U_up, U_down = U_sig[:-N_], U_sig[N_:]
    Phi, *_ = np.linalg.lstsq(U_up, U_down, rcond=None)
    eigvals = np.linalg.eigvals(Phi)

    freqs_all = np.angle(eigvals) / (2 * np.pi * dt)
    damp_all = -np.log(np.abs(eigvals)) / dt
    keep = (freqs_all > 1e-6) & (np.abs(np.abs(eigvals) - 1) < keep_unit_circle_tol)

    f_kept, d_kept = freqs_all[keep], damp_all[keep]
    order = np.argsort(f_kept)
    f_kept, d_kept = f_kept[order], d_kept[order]
    if len(f_kept) >= R:
        f_out, d_out = f_kept[:R], d_kept[:R]
    else:
        pad = np.full(R - len(f_kept), np.nan)
        f_out = np.concatenate([f_kept, pad])
        d_out = np.concatenate([d_kept, pad])

    if not include_damping:
        d_out = np.zeros_like(d_out)

    t = np.arange(T_) * dt
    finite = np.isfinite(f_out)
    amps = np.full((R, N_), np.nan)
    phis = np.full((R, N_), np.nan)
    if finite.any():
        a, p = _fit_amp_phase(
            X,
            t,
            f_out[finite],
            damping=(d_out[finite] if include_damping else None),
        )
        amps[finite] = a
        phis[finite] = p

    return PeriodicityResult(
        frequencies=f_out,
        amplitudes=amps,
        phases=phis,
        damping=d_out,
        extra={"sv": sv, "eigvals": eigvals, "L": L},
    )


def find_periodicity_page(
    X,
    *,
    dt,
    R=1,
    L=None,
    rank=None,
    mask=None,
    n_iter=200,
    tol=1e-5,
    verbose=False,
    rng=None,
):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    T_, N_ = X.shape
    if mask is None:
        mask = ~np.isnan(X)
    if L is None:
        L = max(2 * R + 2, T_ // 10)
    if rank is None:
        rank = 4 * R  # = R * G * slack, G=2, slack=2

    n_win = T_ // L
    if n_win < 2:
        raise ValueError(f"Not enough windows: T/L = {n_win} < 2. Reduce L.")

    Y = X[: n_win * L]
    M = mask[: n_win * L]

    Page = _build_page(Y, n_win, L, N_)
    PageMask = _build_page(M.astype(bool), n_win, L, N_)

    # Observed entries: NaN-safe constant.
    Page0 = np.where(PageMask, Page, 0.0)

    # Initial fill: per-row mean across observed columns.
    row_counts = PageMask.sum(axis=1)
    row_means = np.where(
        row_counts > 0,
        Page0.sum(axis=1) / np.maximum(row_counts, 1),
        0.0,
    )
    Z = np.where(PageMask, Page0, row_means[:, None])

    # Iterative hard-thresholded SVD with observed-entry refit.
    prev_norm = np.linalg.norm(Z)
    iterator = range(n_iter)
    if verbose:
        from tqdm import tqdm

        iterator = tqdm(iterator, leave=False)

    for _ in iterator:
        U, s, Vt = _truncated_svd(Z, rank, n_iter=1, rng=rng)
        Z_new = (U * s) @ Vt
        Z_next = np.where(PageMask, Page0, Z_new)

        delta = np.linalg.norm(Z_next - Z) / (prev_norm + 1e-12)
        Z = Z_next
        prev_norm = np.linalg.norm(Z)
        if delta < tol:
            break

    X_denoised = _unpack_page(Z, n_win, L, N_)

    result = find_periodicity_hankel(X_denoised, dt=dt, R=R)
    result.extra.update(
        {
            "X_denoised": X_denoised,
            "page_shape": Page.shape,
            "observed_fraction": float(mask.mean()),
        }
    )
    return result
