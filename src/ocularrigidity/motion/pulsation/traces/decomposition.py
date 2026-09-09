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
from ocularrigidity.motion.pulsation.traces.base import (
    AbstractTraceSource,
    AbstractUniformTraceSource,
    Traces,
)


@dataclass
class DecompositionConfig:
    method: Literal[
        "ICA", "PCA", "SVD", "SPARSE-PCA", "ica", "pca", "svd", "sparse-pca"
    ] = "ICA"
    n_components: int = 16
    random_state: int = 0
    standardize_sign: bool = True
    max_iter: int = 5000
    whiten: str = "unit-variance"
    tol: float = 0.001
    fun: str = "cube"


class DecomposedTraceSource(AbstractUniformTraceSource):
    """Wraps another source and returns its ICA/PCA components as the traces.

    The mixing matrix and the sign convention live here because this is the only
    place they mean anything: component sign is arbitrary out of FastICA, so each
    component (and its mixing column) is flipped to correlate positively with the
    mean of the underlying physical signal.
    """

    def __init__(
        self,
        source: AbstractTraceSource,
        config: DecompositionConfig | None = None,
    ):
        super().__init__(aligner=source.aligner)
        self.source = source
        self.config = config or DecompositionConfig()
        self._mixing = None

    def compute(self) -> Traces:
        components = self.raw_signal()
        base = self.source

        return Traces(
            values=components,
            uniform_time=base.uniform_time,
            kept_mask=base.kept_mask,
            gap_mask=base.gap_mask,
            timestamps_seconds=base.timestamps_seconds,
            mixing=self._mixing,
            source_map=base.source_map,
        )

    def raw_signal(self) -> np.ndarray:
        """The raw signal from the underlying source."""
        cfg = self.config
        base = self.source.traces
        components, mixing = project_into_separable_components(
            base.values,
            method=cfg.method.lower(),
            n_components=cfg.n_components,
            random_state=cfg.random_state,
            max_iter=cfg.max_iter,
            whiten=cfg.whiten,
            tol=cfg.tol,
            fun=cfg.fun,
        )

        if cfg.standardize_sign:
            ref = np.nanmean(base.values, axis=1)
            for k in range(components.shape[1]):
                c = np.corrcoef(components[:, k], ref)[0, 1]
                if np.isfinite(c) and c < 0:
                    components[:, k] *= -1
                    mixing[:, k] *= -1
        self._mixing = mixing
        return components

    def reset(self) -> None:
        super().reset()
        self.source.reset()

    @property
    def notes_all(self) -> list[str]:
        return list(self.source.notes) + list(self.notes)

    @property
    def gap_mask(self) -> np.ndarray:
        """Gap mask of the underlying source, propagated to the decomposed traces."""
        return self.source.traces.gap_mask
