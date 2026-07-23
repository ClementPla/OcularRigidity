"""Aggregators: collapse ``(T_kept, K)`` candidate traces into one signal.

Kept as its own axis rather than folded into the phase estimators: otherwise
every (phase method × aggregation) pair would need its own class. A phase
estimator holds an aggregator and stays agnostic about how the traces were
reduced.

All aggregators return the aggregated trace **on the full uniform grid**, NaN
where the sample was not kept, so phase code can reason about gaps directly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from ocularrigidity.motion.pulsation.rate import RateEstimate, lomb_scargle_power
from ocularrigidity.motion.pulsation.traces import Traces


def _standardized(values: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance per trace, so no trace dominates by scale."""
    mu = values.mean(axis=0, keepdims=True)
    sd = values.std(axis=0, keepdims=True)
    return (values - mu) / np.where(sd > 0, sd, 1.0)


class AbstractTraceAggregator(ABC):
    """Reduces the candidate traces to the single signal phase is read from."""

    @abstractmethod
    def aggregate(self, traces: Traces, rate: Optional[RateEstimate]) -> np.ndarray:
        """Return a ``(T_uniform,)`` trace, NaN outside ``traces.kept_mask``."""


class SelectBestComponent(AbstractTraceAggregator):
    """Keep only the trace the rate estimator judged most cardiac.

    This is what the original pipeline did. Requires a rate estimate that
    ranks traces (``best_index``).
    """

    def aggregate(self, traces: Traces, rate: Optional[RateEstimate]) -> np.ndarray:
        if rate is None or rate.best_index is None:
            raise ValueError(
                "SelectBestComponent needs a RateEstimate with a best_index. "
                "Use a ranking rate estimator (e.g. LombScargleRateEstimator), "
                "or switch to MeanTrace/SingleTrace aggregation."
            )
        return traces.full(rate.best_index)


class SingleTrace(AbstractTraceAggregator):
    """Keep one fixed trace, by index. Useful when K == 1 or for debugging."""

    def __init__(self, index: int = 0):
        self.index = index

    def aggregate(self, traces: Traces, rate: Optional[RateEstimate]) -> np.ndarray:
        return traces.full(self.index)


class MeanTrace(AbstractTraceAggregator):
    """Plain average across traces. Needs no rate estimate."""

    def __init__(self, standardize: bool = True):
        self.standardize = standardize

    def aggregate(self, traces: Traces, rate: Optional[RateEstimate]) -> np.ndarray:
        values = _standardized(traces.values) if self.standardize else traces.values
        return traces.embed(values.mean(axis=1))


class PowerWeightedMean(AbstractTraceAggregator):
    """Average across traces, weighted by the rate estimator's quality scores.

    A softer :class:`SelectBestComponent`: instead of betting everything on the
    top-scoring trace, it keeps the runners-up in proportion to their score.
    ``power`` sharpens (``> 1``) or flattens (``< 1``) the weighting.
    """

    def __init__(self, standardize: bool = True, power: float = 1.0):
        self.standardize = standardize
        self.power = power

    def aggregate(self, traces: Traces, rate: Optional[RateEstimate]) -> np.ndarray:
        if rate is None or rate.weights is None:
            raise ValueError(
                "PowerWeightedMean needs a RateEstimate carrying per-trace weights."
            )
        w = np.asarray(rate.weights, dtype=float) ** self.power
        total = w.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Rate estimate weights are all zero; cannot aggregate.")
        w = w / total
        values = _standardized(traces.values) if self.standardize else traces.values
        return traces.embed(values @ w)


@dataclass
class SpectralCombinationConfig:
    """Knobs for :class:`OptimizedSpectralCombination`."""

    # A candidate is accepted if its own Lomb-Scargle peak lands within this
    # many bpm of the rate estimate's frequency -- tighter than the (usually
    # wide) band the rate estimator scored candidates over in the first
    # place, mirroring the `heartBeat +/- 0.08 Hz` acceptance window in
    # detectPulseUsingSVD.m.
    accept_tol_bpm: float = 6.0
    # Discard candidates with no real in-band peak of their own.
    min_concentration: float = 0.15
    # Cap on how many candidates enter the optimization.
    max_candidates: int = 20
    # Ridge term on ||w||^2 in the objective; a no-op once the equality
    # constraint pins ||w||_2 = 1, kept only because detectPulseUsingSVD.m
    # exposes the same knob (called with lambda=0 there too).
    l2_regularization: float = 0.0
    # Local optimizer restarts: one from the phase-corrected warm start, the
    # rest from random points on the unit sphere -- the practical substitute
    # for MATLAB's GlobalSearch/MultiStart.
    n_restarts: int = 8
    random_state: int = 0
    maxiter: int = 200


@dataclass
class SpectralCombinationResult:
    """Diagnostics from the last :meth:`OptimizedSpectralCombination.aggregate` call."""

    selected_indices: np.ndarray  # indices into the source Traces' columns
    weights: np.ndarray  # (len(selected_indices),), ||weights||_2 == 1
    objective: float
    spatial_pattern: Optional[np.ndarray]  # traces.mixing[:, selected] @ weights


class OptimizedSpectralCombination(AbstractTraceAggregator):
    """Linear combination of candidate traces optimized to concentrate power
    in the cardiac band.

    Python port of the right-singular-vector rotation in SANSORI's MATLAB
    ``detectPulseUsingSVD.m`` (``periodicprojection`` + ``fmincon``/
    ``GlobalSearch``): rather than keeping a single "best" trace
    (:class:`SelectBestComponent`) or averaging with fixed weights
    (:class:`PowerWeightedMean`), this searches for weights ``w`` (one per
    selected candidate, ``||w||_2 == 1``) that maximize in-band Lomb-Scargle
    power relative to out-of-band power. It is agnostic to what produced the
    candidate traces -- SVD/ICA/PCA components (the intended use, pairing with
    ``DecompositionConfig(method="svd")``) or, with few enough columns, raw
    per-trace signals.

    Needs a :class:`RateEstimate` carrying full periodogram diagnostics (so,
    concretely, a :class:`~ocularrigidity.motion.pulsation.rate.lomb_scargle.LombScargleRateEstimator`)
    both to pick candidates and to know the target frequency; there is no
    finger-pulse ground truth in this pipeline the way there is in the MATLAB
    script, so the already-estimated ``rate.freq`` stands in for it.
    """

    def __init__(self, config: Optional[SpectralCombinationConfig] = None):
        self.config = config or SpectralCombinationConfig()
        self.last_result: Optional[SpectralCombinationResult] = None

    def aggregate(self, traces: Traces, rate: Optional[RateEstimate]) -> np.ndarray:
        cfg = self.config
        required_keys = {"freqs", "power", "peak_freq", "concentration"}
        if rate is None or not required_keys <= rate.diagnostics.keys():
            raise ValueError(
                "OptimizedSpectralCombination needs a RateEstimate with full "
                "periodogram diagnostics (freqs/power/peak_freq/concentration); "
                "use LombScargleRateEstimator as the rate estimator."
            )

        diag = rate.diagnostics
        peak_freq = np.asarray(diag["peak_freq"], dtype=float)
        concentration = np.asarray(diag["concentration"], dtype=float)
        quality = np.asarray(diag.get("quality", concentration), dtype=float)

        f0 = rate.freq
        f0_bpm = f0 * 60.0
        near_target = np.abs(peak_freq * 60.0 - f0_bpm) <= cfg.accept_tol_bpm
        candidates = np.where(near_target & (concentration >= cfg.min_concentration))[
            0
        ]
        if candidates.size == 0:
            raise ValueError(
                f"No candidate trace has a peak within {cfg.accept_tol_bpm:.1f} bpm "
                f"of {f0_bpm:.1f} bpm with concentration >= {cfg.min_concentration}; "
                "cannot build an optimized combination."
            )
        if candidates.size > cfg.max_candidates:
            top = np.argsort(quality[candidates])[::-1][: cfg.max_candidates]
            candidates = np.sort(candidates[top])

        V = traces.values[:, candidates]  # (T_kept, n_cand)
        t = traces.time
        freqs = np.asarray(diag["freqs"], dtype=float)
        tol_hz = cfg.accept_tol_bpm / 60.0
        in_band = np.abs(freqs - f0) <= tol_hz
        if not in_band.any():
            in_band = np.zeros_like(in_band)
            in_band[int(np.argmin(np.abs(freqs - f0)))] = True

        w0 = self._initial_weights(V, t, f0, quality[candidates])
        if V.shape[1] == 1:
            w_opt, objective = w0, self._objective(w0, V, t, freqs, in_band, 0.0)
        else:
            w_opt, objective = self._optimize(V, t, freqs, in_band, w0, cfg)

        spatial_pattern = (
            traces.mixing[:, candidates] @ w_opt if traces.mixing is not None else None
        )
        self.last_result = SpectralCombinationResult(
            selected_indices=candidates,
            weights=w_opt,
            objective=objective,
            spatial_pattern=spatial_pattern,
        )

        return traces.embed(V @ w_opt)

    # -- warm start -------------------------------------------------------
    @staticmethod
    def _initial_weights(
        V: np.ndarray, t: np.ndarray, f0: float, own_quality: np.ndarray
    ) -> np.ndarray:
        """Sign-correct each candidate against the first one via its phasor at
        ``f0`` (the narrowband analogue of MATLAB's bandpass-then-covariance-sign
        step), weight by its own peak quality, then project onto the unit
        sphere so the optimizer starts feasible."""
        c = np.cos(2 * np.pi * f0 * t)
        s = np.sin(2 * np.pi * f0 * t)
        centered = V - V.mean(axis=0, keepdims=True)
        phasors = centered.T @ c + 1j * (centered.T @ s)  # (n_cand,)
        ref = phasors[0] if phasors[0] != 0 else 1.0 + 0j
        signs = np.sign(np.real(phasors * np.conj(ref)))
        signs[signs == 0] = 1.0

        weight_mag = np.clip(own_quality - own_quality.min(), 1e-9, None)
        w0 = signs * weight_mag
        norm = np.linalg.norm(w0)
        return w0 / norm if norm > 0 else np.ones(len(w0)) / np.sqrt(len(w0))

    # -- objective + constrained multi-start optimization ------------------
    @staticmethod
    def _objective(
        w: np.ndarray,
        V: np.ndarray,
        t: np.ndarray,
        freqs: np.ndarray,
        in_band: np.ndarray,
        lam: float,
    ) -> float:
        y = V @ w
        y = y - y.mean()
        power = lomb_scargle_power(t, y, freqs)
        e_in = power[in_band].mean()
        e_out = power[~in_band].mean() if (~in_band).any() else 0.0
        return float(e_out - e_in + lam * np.sum(w**2))

    def _optimize(
        self,
        V: np.ndarray,
        t: np.ndarray,
        freqs: np.ndarray,
        in_band: np.ndarray,
        w0: np.ndarray,
        cfg: SpectralCombinationConfig,
    ) -> tuple[np.ndarray, float]:
        n = V.shape[1]
        bounds = [(-1.0, 1.0)] * n
        constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w**2) - 1.0)}]

        rng = np.random.default_rng(cfg.random_state)
        starts = [w0]
        for _ in range(max(cfg.n_restarts - 1, 0)):
            r = rng.normal(size=n)
            rn = np.linalg.norm(r)
            starts.append(r / rn if rn > 0 else w0)

        results = [
            minimize(
                self._objective,
                w_start,
                args=(V, t, freqs, in_band, cfg.l2_regularization),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": cfg.maxiter},
            )
            for w_start in starts
        ]
        pool = [r for r in results if r.success] or results
        best = min(pool, key=lambda r: r.fun)

        w_opt = best.x
        norm = np.linalg.norm(w_opt)
        if norm > 0:
            w_opt = w_opt / norm
        return w_opt, float(best.fun)
