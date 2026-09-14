"""Coherence-based A-scan selection as a trace-source *wrapper*.

Every other trace score judges a trace **in isolation** — its own periodogram
power / concentration / FAP, or its instantaneous envelope — so a trace that is
strongly periodic on its own (a noise resonance, a local artifact) but
incoherent with the ensemble goes uncaught. This wrapper keeps the A-scans whose
cardiac phase stays in a *constant* relation to the ensemble over time —
**coherent even when not in phase** — and drops the rest.

The primitive is the phase-locking value (PLV)::

    C_jk = | mean_t  w(t) · exp( i (phi_j(t) - phi_k(t)) ) |

which is invariant to a *constant* phase offset by construction: two A-scans
carrying the same pulsation at a fixed lag score C = 1, while a drifting or
random relationship decays to 0. That offset-invariance is exactly what PCA/ICA
lack — a lagged pulsation splits across a sin/cos pair of components there, but
stays a single coherent group here.

It consumes and produces the same :class:`Traces` contract, so it composes with
the other sources. Put it *before* the rate/phase stages, so Lomb-Scargle and
the phase estimator run on an already-coherent subset::

    coherent = CoherentTraceSource(MaskThicknessTraceSource(reg, aligner))
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from scipy.signal import hilbert

from ocularrigidity.motion.pulsation.traces.base import AbstractTraceSource, Traces


@dataclass
class CoherenceConfig:
    # How membership is scored.
    #   "consensus"   — PLV of each trace to an iteratively refined ensemble
    #                   phase; O(K), robust, the default.
    #   "eigenvector" — leading eigenvector of the K×K coherence matrix; also
    #                   yields an eigengap that says whether there is *one*
    #                   coherent population or several.
    mode: Literal["consensus", "eigenvector"] = "consensus"

    # Weight each instant by the trace envelope, so moments where a trace is
    # momentarily weak (noise, dropout) count less toward its coherence.
    weight_by_envelope: bool = True

    # Standardize each trace to unit variance before the analytic signal, so no
    # single high-amplitude trace dominates the consensus by scale alone.
    standardize: bool = True

    # consensus mode: decontamination iterations. The consensus is recomputed
    # with the previous PLV as a membership weight, so incoherent traces stop
    # polluting the reference after the first pass.
    n_iter: int = 3

    # How the score becomes a selection.
    selection: Literal["threshold", "quantile", "top_k"] = "threshold"
    plv_threshold: float = 0.5      # selection="threshold"
    keep_quantile: float = 0.5      # selection="quantile" (keep the top half)
    top_k: Optional[int] = None     # selection="top_k"

    # Never return fewer than this many traces — guards against over-pruning a
    # case where coherence is genuinely low everywhere.
    min_traces: int = 8

    verbose: bool = True


class CoherentTraceSource(AbstractTraceSource):
    """Wraps a source and returns only its coherently-pulsating traces.

    Diagnostics populated by :meth:`compute` (all indexed by the *base* source's
    trace order):

    ``scores``      per-trace coherence in [0, 1].
    ``selected``    indices kept, sorted.
    ``eigengap``    λ1/λ2 of the coherence matrix (eigenvector mode only); large
                    ⇒ a single dominant coherent population.
    ``coherent_fraction``  λ1 / trace (eigenvector mode only).
    """

    def __init__(self, source: AbstractTraceSource, config: Optional[CoherenceConfig] = None):
        super().__init__()
        self.source = source
        self.config = config or CoherenceConfig()
        self.scores: Optional[np.ndarray] = None
        self.selected: Optional[np.ndarray] = None
        self.eigengap: Optional[float] = None
        self.coherent_fraction: Optional[float] = None

    # -- analytic signal ------------------------------------------------
    def _analytic(self, values: np.ndarray):
        """Per-trace wrapped phase and envelope of the (already bandpassed) traces.

        ``values`` is ``(T_kept, K)`` and gap-free, so the Hilbert transform's
        narrowband assumption holds directly. The kept samples are treated as
        evenly spaced; any small phase distortion across bridged gaps is common
        to every trace and cancels in the phase *differences* the scores use.
        """
        x = values - values.mean(axis=0, keepdims=True)
        if self.config.standardize:
            sd = x.std(axis=0, keepdims=True)
            x = x / np.where(sd > 0, sd, 1.0)
        z = hilbert(x, axis=0)               # analytic signal per trace
        return np.angle(z), np.abs(z)        # wrapped phase is fine for PLV

    # -- scorers --------------------------------------------------------
    def _consensus_scores(self, phase: np.ndarray, env: np.ndarray) -> np.ndarray:
        cfg = self.config
        _, K = phase.shape
        P = np.exp(1j * phase)                               # (T, K) unit phasors
        amp = env if cfg.weight_by_envelope else np.ones_like(env)
        m = np.ones(K)                                       # membership weight
        plv = np.ones(K)
        for _ in range(max(1, cfg.n_iter)):
            # Envelope- and membership-weighted ensemble phase per instant.
            acc = (P * amp * m[None, :]).sum(axis=1)         # (T,)
            ref = np.exp(-1j * np.angle(acc))                # conj consensus phasor
            # PLV of each trace to the consensus; a constant lag cancels here.
            num = np.abs((P * ref[:, None] * amp).sum(axis=0))   # (K,)
            den = amp.sum(axis=0) + 1e-12
            plv = num / den
            m = plv
        return plv

    def _eigen_scores(self, phase: np.ndarray, env: np.ndarray) -> np.ndarray:
        cfg = self.config
        amp = env if cfg.weight_by_envelope else np.ones_like(env)
        A = amp * np.exp(1j * phase)                         # (T, K)
        M = A.conj().T @ A                                   # (K, K) Hermitian, PSD
        norm = np.sqrt(np.real(np.diag(M)))                  # ||amp_k||
        M = M / (norm[:, None] * norm[None, :] + 1e-12)      # unit diagonal (coherence)
        evals, evecs = np.linalg.eigh(M)                     # ascending
        v = evecs[:, -1]
        self.eigengap = float(evals[-1] / (evals[-2] + 1e-12)) if len(evals) > 1 else float("inf")
        self.coherent_fraction = float(evals[-1] / (evals.sum() + 1e-12))
        score = np.abs(v)                                    # |v_k| = membership
        return score / (score.max() + 1e-12)

    # -- selection ------------------------------------------------------
    def _select(self, scores: np.ndarray) -> np.ndarray:
        cfg = self.config
        K = len(scores)
        if cfg.selection == "top_k" and cfg.top_k:
            keep = np.argsort(scores)[::-1][: cfg.top_k]
        elif cfg.selection == "quantile":
            keep = np.where(scores >= np.quantile(scores, cfg.keep_quantile))[0]
        else:  # threshold
            keep = np.where(scores >= cfg.plv_threshold)[0]
        floor = min(cfg.min_traces, K)
        if len(keep) < floor:
            keep = np.argsort(scores)[::-1][:floor]
        return np.sort(keep)

    # -- contract -------------------------------------------------------
    def compute(self) -> Traces:
        cfg = self.config
        base = self.source.traces
        phase, env = self._analytic(base.values)

        if cfg.mode == "eigenvector":
            scores = self._eigen_scores(phase, env)
        else:
            scores = self._consensus_scores(phase, env)

        keep = self._select(scores)
        self.scores = scores
        self.selected = keep

        if cfg.verbose:
            note = (
                f"Coherent selection ({cfg.mode}): kept {len(keep)}/{base.n_traces} "
                f"traces, score ≥ {scores[keep].min():.2f}"
            )
            if self.eigengap is not None:
                note += (
                    f"; eigengap λ1/λ2 = {self.eigengap:.1f}, "
                    f"coherent fraction = {self.coherent_fraction:.2f}"
                )
            self.notes.append(note)

        mixing = base.mixing[:, keep] if base.mixing is not None else None
        return Traces(
            values=base.values[:, keep],
            uniform_time=base.uniform_time,
            kept_mask=base.kept_mask,
            gap_mask=base.gap_mask,
            timestamps_seconds=base.timestamps_seconds,
            mixing=mixing,
            source_map=base.source_map,
        )

    def reset(self) -> None:
        super().reset()
        self.source.reset()
        self.scores = None
        self.selected = None
        self.eigengap = None
        self.coherent_fraction = None

    @property
    def notes_all(self) -> list[str]:
        return list(self.source.notes) + list(self.notes)
