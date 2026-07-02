"""End-to-end orchestration: paths → registration → extraction → folding.

Wires the collaborators together:

    VideoRegistrator → VideoTimelineAligner → MaskPulseExtractor
                                            → NCycleReconstructor

and packages the outcome as a :class:`CardiacPipelineResults`.
"""

from pathlib import Path
from typing import Optional

from ocularrigidity.motion.pipeline_results import CardiacPipelineResults
from ocularrigidity.motion.pulsation.config import NCycleConfig, PulseExtractionConfig
from ocularrigidity.motion.pulsation.mask_pulse_extractor import MaskPulseExtractor
from ocularrigidity.motion.pulsation.n_cycle_reconstructor import NCycleReconstructor
from ocularrigidity.motion.video_timeline_aligner import TimeUnits, VideoTimelineAligner
from ocularrigidity.registration.registration_engine import VideoRegistrator


def run_cardiac_pipeline(
    video_relpath: str,
    *,
    root_masks: str,
    root_data: str,
    timestamps_path: str,
    config: Optional[PulseExtractionConfig] = None,
    fold_config: Optional[NCycleConfig] = None,
    # --- Registration -----------------------------------------------------
    skip_first_n_frames: int = 3,
    drop_last_n_frames: int = 0,
    flatten_rpe: bool = True,
    correct_transversal: bool = True,
    use_encoded_video: bool = True,
    cache_dir: Optional[Path] = None,
    lateral_method: str = "xcorr",
    subpixel: bool = True,
    units_in_timestamps: TimeUnits = TimeUnits.MICROSECONDS,
    # --- Orchestration ----------------------------------------------------
    compute_n_cycle_video: bool = False,
    verbose: bool = True,
) -> CardiacPipelineResults:
    """Run the mask-based cardiac pipeline and return the packaged results.

    ``config`` bundles the extraction knobs and ``fold_config`` the folding
    knobs (both default to their dataclass defaults). ``cache_dir`` is forwarded
    to :class:`VideoRegistrator`: registration is deterministic across
    ICA/PCA and phase methods, so caching lets repeated runs of the same video
    reuse the registered frames/masks instead of recomputing them.
    """
    if config is None:
        config = PulseExtractionConfig(verbose=verbose)

    registrator = VideoRegistrator(
        video=video_relpath,
        root_data=Path(root_data),
        root_masks=Path(root_masks),
        skip_first_n_frames=skip_first_n_frames,
        drop_last_n_frames=drop_last_n_frames,
        flatten_rpe=flatten_rpe,
        correct_transversal=correct_transversal,
        verbose=verbose,
        use_encoded_video=use_encoded_video,
        cache_dir=cache_dir,
        lateral_method=lateral_method,
        subpixel=subpixel,
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
