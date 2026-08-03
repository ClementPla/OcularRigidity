"""End-to-end orchestration: paths → registration → extraction → folding.

Wires the collaborators together:

    VideoRegistrator → VideoTimelineAligner → MaskPulseExtractor
                                            → NCycleReconstructor

and packages the outcome as a :class:`CardiacPipelineResults`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ocularrigidity.motion.pulsation.extractor import PulseExtractor
from ocularrigidity.motion.pulsation.legacy import (
    MaskPulseExtractor,
    PulseExtractionConfig,
)
from ocularrigidity.motion.pulsation.n_cycle_reconstructor import (
    NCycleConfig,
    NCycleReconstructor,
)
from ocularrigidity.motion.pulsation.phase import (
    IQDemodPhaseEstimator,
    SelectBestComponent,
)
from ocularrigidity.motion.pulsation.rate import LombScargleRateEstimator
from ocularrigidity.motion.pulsation.traces import (
    BandPassFilterTraceSource,
    CoherentTraceSource,
    DecomposedTraceSource,
    MaskThicknessTraceSource,
)
from ocularrigidity.motion.video_timeline_aligner import TimeUnits, VideoTimelineAligner
from ocularrigidity.registration.config import RegistrationConfig
from ocularrigidity.registration.registration_engine import VideoRegistrator

if TYPE_CHECKING:
    from ocularrigidity.motion.pipeline_results import CardiacPipelineResults


def build_extractor(registrator, aligner, stage_configs: dict) -> PulseExtractor:
    """The composed chain, from the per-stage configs.

        thickness -> bandpass -> coherent selection -> PCA
                  -> Lomb-Scargle rate -> IQ phase on the best component

    ``stage_configs`` is what ``PulsationConfig.chain_for_video`` returns, so
    the study config stays the single place the recipe is written down.
    """
    source = MaskThicknessTraceSource(registrator, aligner, stage_configs["trace"])
    source = BandPassFilterTraceSource(source, stage_configs["bandpass"])
    source = CoherentTraceSource(source, stage_configs["coherence"])
    source = DecomposedTraceSource(source, stage_configs["decomposition"])
    return PulseExtractor(
        trace_source=source,
        rate_estimator=LombScargleRateEstimator(stage_configs["rate"]),
        phase_estimator=IQDemodPhaseEstimator(
            stage_configs["phase"], aggregator=SelectBestComponent()
        ),
        registered_video=registrator,
        aligner=aligner,
    )


def run_composed_pipeline(
    video_relpath: str,
    *,
    root_masks: str,
    root_data: str,
    timestamps_path: str,
    stage_configs: dict,
    fold_config: Optional[NCycleConfig] = None,
    registration_config: Optional[RegistrationConfig] = None,
    cache_dir: Optional[Path] = None,
    units_in_timestamps: TimeUnits = TimeUnits.MICROSECONDS,
    compute_n_cycle_video: bool = True,
    verbose: bool = True,
) -> CardiacPipelineResults:
    """End-to-end run of the composed chain, packaged as results.

    The composed counterpart of :func:`run_cardiac_pipeline`, which drives the
    superseded flat ``PulseExtractionConfig`` recipe.
    """
    from ocularrigidity.motion.pipeline_results import CardiacPipelineResults

    registrator = VideoRegistrator(
        video=video_relpath,
        root_data=Path(root_data),
        root_masks=Path(root_masks),
        config=registration_config,
        verbose=verbose,
        cache_dir=cache_dir,
    )
    aligner = VideoTimelineAligner(
        registrator, timestamps_path, units_in_timestamps=units_in_timestamps
    )
    extractor = build_extractor(registrator, aligner, stage_configs)

    if verbose:
        ts = extractor.timestamps_seconds
        print(f"fs = {extractor.fs:.2f} Hz, duration = {ts[-1]:.1f}s, T = {len(ts)}")
        print(f"Gap fraction on uniform grid: {extractor.gap_fraction:.2%}")

    _ = extractor.phase_per_frame

    if verbose:
        T = len(extractor.timestamps_seconds)
        print(f"Good frames for folding: {int(extractor.good_per_frame.sum())} / {T}")
        print(
            f"Cardiac rate: {extractor.cardiac_bpm:.1f} bpm "
            f"(confidence={extractor.confidence})"
        )

    reconstructor = None
    if compute_n_cycle_video:
        reconstructor = NCycleReconstructor(extractor, fold_config)
        reconstructor.compute()

    return CardiacPipelineResults.from_composed(
        extractor, reconstructor, stage_configs=stage_configs
    )


def run_cardiac_pipeline(
    video_relpath: str,
    *,
    root_masks: str,
    root_data: str,
    timestamps_path: str,
    config: Optional[PulseExtractionConfig] = None,
    fold_config: Optional[NCycleConfig] = None,
    registration_config: Optional[RegistrationConfig] = None,
    cache_dir: Optional[Path] = None,
    units_in_timestamps: TimeUnits = TimeUnits.MICROSECONDS,
    # --- Orchestration ----------------------------------------------------
    compute_n_cycle_video: bool = False,
    verbose: bool = True,
) -> CardiacPipelineResults:
    """Run the mask-based cardiac pipeline and return the packaged results.

    ``config`` bundles the extraction knobs, ``fold_config`` the folding knobs
    and ``registration_config`` the registration knobs (all default to their
    dataclass defaults). ``cache_dir`` is forwarded to :class:`VideoRegistrator`:
    registration is deterministic across ICA/PCA and phase methods, so caching
    lets repeated runs of the same video reuse the registered frames/masks
    instead of recomputing them.
    """
    # Imported lazily to avoid a circular import: pipeline_results imports the
    # pulsation package (for its config types), which imports this module.
    from ocularrigidity.motion.pipeline_results import CardiacPipelineResults

    if config is None:
        config = PulseExtractionConfig(verbose=verbose)

    registrator = VideoRegistrator(
        video=video_relpath,
        root_data=Path(root_data),
        root_masks=Path(root_masks),
        config=registration_config,
        verbose=verbose,
        cache_dir=cache_dir,
    )
    aligner = VideoTimelineAligner(
        registrator, timestamps_path, units_in_timestamps=units_in_timestamps
    )
    extractor = MaskPulseExtractor(registrator, aligner, config)

    if verbose:
        ts = extractor.timestamps_seconds
        print(f"fs = {extractor.fs:.2f} Hz, duration = {ts[-1]:.1f}s, T = {len(ts)}")
        print(f"Gap fraction on uniform grid: {extractor.gap_fraction:.2%}")

    _ = extractor.phase_per_frame

    if verbose:
        T = len(extractor.timestamps_seconds)
        print(f"Good frames for folding: {int(extractor.good_per_frame.sum())} / {T}")
        print(
            f"Cardiac rate: {extractor.cardiac_bpm:.1f} bpm "
            f"(confidence={extractor.confidence})"
        )

    reconstructor = None
    if compute_n_cycle_video:
        reconstructor = NCycleReconstructor(extractor, fold_config)
        reconstructor.compute()

    return CardiacPipelineResults.from_objects(extractor, reconstructor)
