"""The orchestrator: trace source → (optional) rate estimator → phase estimator.

``PulseExtractor`` owns no signal processing of its own. It wires three
collaborators together, caches their results and exposes the small surface the
rest of the codebase consumes (``cardiac_bpm``, ``phase_per_frame``,
``good_per_frame``, …):

    PulseExtractor(
        trace_source   = DecomposedTraceSource(MaskThicknessTraceSource(reg, aligner)),
        rate_estimator = LombScargleRateEstimator(),
        phase_estimator= IQDemodPhaseEstimator(aggregator=SelectBestComponent()),
    )

Swap any one of the three without touching the others. ``rate_estimator=None``
is a valid configuration for phase estimators that recover the rate themselves.
"""

import numpy as np

from ocularrigidity.motion.pulsation.phase import (
    AbstractPhaseEstimator,
    AbstractTraceAggregator,
    IQDemodPhaseEstimator,
    PhaseTrack,
    SelectBestComponent,
)
from ocularrigidity.motion.pulsation.rate import (
    AbstractRateEstimator,
    LombScargleConfig,
    LombScargleRateEstimator,
    RateEstimate,
)
from ocularrigidity.motion.pulsation.band import CardiacBand
from ocularrigidity.motion.pulsation.traces import (
    AbstractTraceSource,
    BandPassFilterTraceConfig,
    BandPassFilterTraceSource,
    DecomposedTraceSource,
    DecompositionConfig,
    MaskThicknessTraceSource,
    MaskTraceConfig,
    Traces,
)


class PulseExtractor:
    def __init__(
        self,
        trace_source: AbstractTraceSource,
        phase_estimator: AbstractPhaseEstimator,
        rate_estimator: AbstractRateEstimator | None = None,
        registered_video=None,
        aligner=None,
    ):
        self.trace_source = trace_source
        self.phase_estimator = phase_estimator
        self.rate_estimator = rate_estimator

        # Both are only needed by consumers (folding, viewers); default to
        # whatever the trace source was built on.
        self.registered_video = registered_video or self._find(
            trace_source, "registered_video"
        )
        self.aligner = aligner or self._find(trace_source, "aligner")

        self._rate: RateEstimate | None = None
        self._phase: PhaseTrack | None = None
        self._freq_override: float | None = None

    @staticmethod
    def _find(source, attr):
        """Walk a chain of trace-source decorators looking for ``attr``."""
        seen = set()
        while source is not None and id(source) not in seen:
            seen.add(id(source))
            if hasattr(source, attr):
                return getattr(source, attr)
            source = getattr(source, "source", None)
        return None

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------
    @property
    def traces(self) -> Traces:
        return self.trace_source.traces

    @property
    def rate(self) -> RateEstimate | None:
        """The rate estimate, or None when the rate stage was bypassed."""
        if self.rate_estimator is None:
            return None
        if self._rate is None:
            self._rate = self.rate_estimator.estimate(self.traces)
            if self._freq_override is not None:
                self._rate.freq = self._freq_override
        return self._rate

    @property
    def phase(self) -> PhaseTrack:
        if self._phase is None:
            self._phase = self.phase_estimator.estimate(self.traces, self.rate)
        return self._phase

    def reset(self, *, traces: bool = False) -> None:
        """Drop cached results. ``traces=True`` also re-runs the trace source."""
        self._rate = None
        self._phase = None
        if traces:
            self.trace_source.reset()

    # ------------------------------------------------------------------
    # Consumer-facing surface
    # ------------------------------------------------------------------
    @property
    def cardiac_freq(self) -> float:
        if self._freq_override is not None:
            return self._freq_override
        rate = self.rate
        return rate.freq if rate is not None else self.phase.freq

    @cardiac_freq.setter
    def cardiac_freq(self, value: float) -> None:
        """Manual override; invalidates the downstream phase."""
        self._freq_override = float(value)
        self._rate = None
        self._phase = None

    @property
    def cardiac_bpm(self) -> float:
        return self.cardiac_freq * 60.0

    @property
    def phase_uniform(self) -> np.ndarray:
        return self.phase.phase_uniform

    @property
    def good_uniform(self) -> np.ndarray:
        return self.phase.good_uniform

    @property
    def phase_per_frame(self) -> np.ndarray:
        return self.phase.phase_per_frame

    @property
    def good_per_frame(self) -> np.ndarray:
        return self.phase.good_per_frame

    @property
    def inst_bpm(self) -> np.ndarray:
        return self.phase.inst_bpm

    @property
    def confidence(self) -> str:
        rate = self.rate
        return rate.confidence if rate is not None else "unknown"

    @property
    def notes(self) -> list[str]:
        out = list(self._collect_notes(self.trace_source))
        if self._rate is not None:
            out += list(self._rate.notes)
        out += list(self.phase_estimator.notes)
        return out

    @classmethod
    def _collect_notes(cls, source) -> list[str]:
        out: list[str] = []
        seen = set()
        while source is not None and id(source) not in seen:
            seen.add(id(source))
            out = list(getattr(source, "notes", [])) + out
            source = getattr(source, "source", None)
        return out

    # -- timeline delegations -------------------------------------------
    @property
    def timestamps_seconds(self):
        return self.aligner.timestamps_seconds

    @property
    def uniform_time(self):
        return self.aligner.uniform_time

    @property
    def dt(self) -> float:
        return self.aligner.dt

    @property
    def fs(self) -> float:
        return self.aligner.fs

    @property
    def gap_mask(self):
        return self.traces.gap_mask

    @property
    def gap_fraction(self) -> float:
        return float(self.gap_mask.mean())

    @property
    def registered_frames(self):
        return self.registered_video.registered_frames

    @property
    def registered_masks(self):
        return self.registered_video.registered_masks

    # ------------------------------------------------------------------
    # Default recipe
    # ------------------------------------------------------------------
    @classmethod
    def from_masks(
        cls,
        registered_video,
        aligner,
        *,
        trace_config: MaskTraceConfig | None = None,
        filter_config: BandPassFilterTraceConfig | None = None,
        decomposition: DecompositionConfig | None = None,
        rate_config: LombScargleConfig | None = None,
        phase_estimator: AbstractPhaseEstimator | None = None,
        aggregator: AbstractTraceAggregator | None = None,
        override_bpm: float | None = None,
    ) -> "PulseExtractor":
        """The historical pipeline: thickness → bandpass → ICA → Lomb-Scargle → phase.

        ``decomposition=None`` skips the ICA/PCA step and works directly on the
        per-A-scan thickness traces (in which case pair it with a ``MeanTrace``
        aggregator, since there is no "best component" to select).
        ``filter_config=None`` skips the bandpass, feeding raw thickness on.
        """
        trace_config = trace_config or MaskTraceConfig()
        source: AbstractTraceSource = MaskThicknessTraceSource(
            registered_video, aligner, trace_config
        )
        if filter_config is not None:
            source = BandPassFilterTraceSource(source, filter_config)
        if decomposition is not None:
            source = DecomposedTraceSource(source, decomposition)

        # The band lives on the filter config now, not on the trace config.
        band = filter_config.band if filter_config is not None else CardiacBand()
        rate_config = rate_config or LombScargleConfig(band=band)
        rate_estimator = LombScargleRateEstimator(
            rate_config, override_bpm=override_bpm
        )

        if phase_estimator is None:
            phase_estimator = IQDemodPhaseEstimator(
                aggregator=aggregator or SelectBestComponent()
            )
        elif aggregator is not None:
            phase_estimator.aggregator = aggregator

        return cls(
            trace_source=source,
            phase_estimator=phase_estimator,
            rate_estimator=rate_estimator,
            registered_video=registered_video,
            aligner=aligner,
        )
