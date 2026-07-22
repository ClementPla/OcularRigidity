"""ICA/PCA as a trace-source *wrapper* rather than a separate leaf class.

Because it consumes and produces the same ``Traces`` contract, any source can be
decomposed without a parallel class hierarchy::

    raw = MaskThicknessTraceSource(registrator, aligner)      # K = W
    ica = DecomposedTraceSource(raw, DecompositionConfig())   # K = n_components
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from ocularrigidity.motion.projection._1d import project_into_separable_components
from ocularrigidity.motion.pulsation.traces.base import AbstractTraceSource, Traces


@dataclass
class DecompositionConfig:
    method: Literal["ICA", "PCA", "SVD", "ica", "pca", "svd"] = "ICA"
    n_components: int = 16
    random_state: int = 0
    standardize_sign: bool = True
    max_iter: int = 5000
    whiten: str = "unit-variance"
    tol: float = 0.001
    fun: str = "cube"


class DecomposedTraceSource(AbstractTraceSource):
    """Wraps another source and returns its ICA/PCA/SVD components as the traces.

    The mixing matrix and the sign convention live here because this is the only
    place they mean anything: component sign is arbitrary out of FastICA, so each
    component (and its mixing column) is flipped to correlate positively with the
    mean of the underlying physical signal.

    SVD is deliberately not run on ``base.values``: that is the trace source's
    spatially-smoothed, temporally-bandpassed output (see
    ``AbstractUniformTraceSource.filtered_signal``), whereas SVD is meant to see
    the user's data directly. It instead decomposes ``source.interpolated_signal``
    — resampled onto the uniform grid, gaps marked, but otherwise untouched.
    """

    def __init__(
        self,
        source: AbstractTraceSource,
        config: Optional[DecompositionConfig] = None,
    ):
        super().__init__()
        self.source = source
        self.config = config or DecompositionConfig()

    def compute(self) -> Traces:
        cfg = self.config
        base = self.source.traces
        method = cfg.method.lower()

        if method == "svd":
            if not hasattr(self.source, "interpolated_signal"):
                raise TypeError(
                    "SVD decomposition needs a source exposing "
                    "`interpolated_signal` (e.g. AbstractUniformTraceSource); "
                    f"got {type(self.source).__name__}."
                )
            source_map = self.source.interpolated_signal
            kept_mask = ~np.isnan(source_map).any(axis=1)
            signal_in = source_map[kept_mask]
        else:
            source_map = base.source_map
            kept_mask = base.kept_mask
            signal_in = base.values

        components, mixing = project_into_separable_components(
            signal_in,
            method=method,
            n_components=cfg.n_components,
            random_state=cfg.random_state,
            max_iter=cfg.max_iter,
            whiten=cfg.whiten,
            tol=cfg.tol,
            fun=cfg.fun,
        )

        if cfg.standardize_sign:
            ref = np.nanmean(signal_in, axis=1)
            for k in range(components.shape[1]):
                c = np.corrcoef(components[:, k], ref)[0, 1]
                if np.isfinite(c) and c < 0:
                    components[:, k] *= -1
                    mixing[:, k] *= -1

        return Traces(
            values=components,
            uniform_time=base.uniform_time,
            kept_mask=kept_mask,
            gap_mask=base.gap_mask,
            timestamps_seconds=base.timestamps_seconds,
            mixing=mixing,
            source_map=source_map,
        )

    def reset(self) -> None:
        super().reset()
        self.source.reset()

    @property
    def notes_all(self) -> list[str]:
        return list(self.source.notes) + list(self.notes)
