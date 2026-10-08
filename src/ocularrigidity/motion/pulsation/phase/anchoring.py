"""Anchor phase 0 on the choroid-thickness minimum."""

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ocularrigidity.motion.pulsation.phase.base import (
    AbstractPhaseEstimator,
    PhaseTrack,
)
from ocularrigidity.motion.pulsation.rate import RateEstimate
from ocularrigidity.motion.pulsation.traces import (
    AbstractUniformTraceSource,
    Traces,
)


@dataclass
class ThicknessAnchorConfig:
    n_harmonics: int = 3
    # Width (σ) of the per-column baseline removed before the fit, in cardiac periods.
    detrend_cycles: float = 1.0
    grid_size: int = 720
    # How columns are combined into the fitted trace.
    column_reduce: str = "median"
    # Below this coherence a note is left
    min_coherence: float = 0.3
    # Anchor each of this many equal-duration segments separately.
    n_segments: int = 1


def thickness_minimum_phase(
    phase_uniform: np.ndarray,
    good_uniform: np.ndarray,
    thickness: np.ndarray,
    dt: float,
    freq: float,
    config: ThicknessAnchorConfig | None = None,
) -> dict:
    """Phase (radians) of minimal thickness under ``phase_uniform``, plus diagnostics."""
    cfg = config or ThicknessAnchorConfig()
    ref = np.asarray(thickness, dtype=float)
    if ref.ndim == 1:
        ref = ref[:, None]

    valid = ~np.isnan(ref)
    sigma = cfg.detrend_cycles / freq / dt
    num = gaussian_filter1d(np.where(valid, ref, 0.0), sigma, axis=0, mode="nearest")
    den = gaussian_filter1d(valid.astype(float), sigma, axis=0, mode="nearest")
    dev = ref - num / np.where(den > 0, den, np.nan)

    use = valid & good_uniform[:, None] & np.isfinite(phase_uniform)[:, None]
    use &= np.isfinite(dev)
    phase = np.nan_to_num(phase_uniform)

    c_w = np.where(use, dev * np.exp(-1j * phase)[:, None], 0.0).sum(0)
    coherence = float(np.abs(c_w.sum()) / max(np.abs(c_w).sum(), 1e-12))

    n_used = use.sum(1)
    ok = n_used > 0
    grid = np.linspace(0.0, 2 * np.pi, cfg.grid_size, endpoint=False)
    empty = dict(
        offset=float("nan"), coherence=coherence, depth=float("nan"),
        grid=grid, curve=np.full_like(grid, np.nan), n_samples=int(ok.sum()),
    )  # fmt: skip
    if ok.sum() < 4 * cfg.n_harmonics + 2:
        return empty

    if cfg.column_reduce == "median":
        with np.errstate(all="ignore"):
            s = np.nanmedian(np.where(use, dev, np.nan)[ok], axis=1)
    else:
        s = np.where(use, dev, 0.0).sum(1)[ok] / n_used[ok]
    phi = phase[ok]

    def design(p):
        k = np.arange(1, cfg.n_harmonics + 1)
        return np.hstack(
            [np.ones((len(p), 1)), np.cos(np.outer(p, k)), np.sin(np.outer(p, k))]
        )

    beta, *_ = np.linalg.lstsq(design(phi), s, rcond=None)
    curve = design(grid) @ beta
    return dict(
        offset=float(grid[np.argmin(curve)]),
        coherence=coherence,
        depth=float(curve.max() - curve.min()),
        grid=grid,
        curve=curve,
        n_samples=int(ok.sum()),
    )


class ThicknessAnchoredPhaseEstimator(AbstractPhaseEstimator):
    """Any phase estimator, with phase 0 moved onto minimal choroid thickness."""

    def __init__(
        self,
        inner: AbstractPhaseEstimator,
        reference: AbstractUniformTraceSource,
        config: ThicknessAnchorConfig | None = None,
    ):
        super().__init__(inner.aggregator, inner.per_trace)
        self.inner = inner
        self.reference = reference
        self.config = config or ThicknessAnchorConfig()
        self.requires_rate = inner.requires_rate
        self.anchor: dict | None = None

    def phase_from_trace(self, trace, traces, rate):
        return self.inner.phase_from_trace(trace, traces, rate)

    def estimate(
        self, traces: Traces, rate: Optional[RateEstimate] = None
    ) -> PhaseTrack:
        track = self.inner.estimate(traces, rate)
        self.notes = list(self.inner.notes)
        cfg = self.config

        thickness = self.reference.interpolated_signal
        if len(thickness) != len(track.phase_uniform):
            raise ValueError(
                f"Reference has {len(thickness)} uniform samples, phase has "
                f"{len(track.phase_uniform)}; they must share an aligner."
            )

        def fit(good):
            return thickness_minimum_phase(
                track.phase_uniform, good, thickness, traces.dt, track.freq, cfg
            )

        overall = fit(track.good_uniform)
        if not np.isfinite(overall["offset"]):
            self.notes.append(
                "Thickness anchor: too few good samples; phase left as is."
            )
            self.anchor = dict(overall=overall, segments=[])
            return track

        ts = traces.timestamps_seconds
        n = max(1, cfg.n_segments)
        dur = (ts[-1] - ts[0]) / n

        def segment_of(t):
            return np.clip(((t - ts[0]) // dur).astype(int), 0, n - 1)

        seg_u = segment_of(traces.uniform_time)
        seg_f = segment_of(ts)

        offsets, segments = np.empty(n), []
        for k in range(n):
            a = fit(track.good_uniform & (seg_u == k)) if n > 1 else overall
            if not np.isfinite(a["offset"]):
                self.notes.append(
                    f"Thickness anchor, segment {k + 1}/{n}: too few good "
                    "samples; using the whole-video offset."
                )
                a = overall
            offsets[k] = a["offset"]
            segments.append(a)
            if a["coherence"] < cfg.min_coherence:
                self.notes.append(
                    f"Thickness anchor, segment {k + 1}/{n}: low column coherence "
                    f"({a['coherence']:.2f}); phase 0 may not be the thickness minimum."
                )
        self.anchor = dict(overall=overall, segments=segments, offsets=offsets)

        desc = ", ".join(
            f"{np.degrees(o):.0f}° (depth {a['depth']:.2f} px, coh {a['coherence']:.2f})"
            for o, a in zip(offsets, segments)
        )
        self.notes.append(
            f"Phase anchored on thickness minimum, {n} segment(s): shifted {desc}."
        )
        return replace(
            track,
            phase_uniform=np.mod(track.phase_uniform - offsets[seg_u], 2 * np.pi),
            phase_per_frame=np.mod(
                track.phase_per_frame - offsets[seg_f] / (2 * np.pi), 1.0
            ),
        )
