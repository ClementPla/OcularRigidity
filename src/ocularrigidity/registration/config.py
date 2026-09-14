"""Parameters of the registration engine.

Lives with the code it configures: ``VideoRegistrator`` and ``register_videos``
take this directly, so the registration package must not have to import the
study-level ``pipeline_config`` to know its own arguments. The cohort-wide
instance in force for the pipeline is the ``REGISTRATION`` singleton in
:mod:`ocularrigidity.pipeline_config`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ocularrigidity.consts import REGISTRATION_BATCH_SIZE, REGISTRATOR_CHECKPOINT


@dataclass()
class RegistrationConfig:
    # Which estimator produces (dx, dy). ``"classical"`` is the correlation +
    # BM-difference pipeline in ``rigid.py``; ``"learned"`` is the trained
    # ``RegistrationRegressor`` reading the frozen segmentation encoder, which
    # is what the fused single-pass stage runs. The default stays classical so
    # existing callers of ``register_videos`` are unaffected; the cohort's
    # choice is pinned in ``pipeline_config.REGISTRATION``.
    method: Literal["classical", "learned"] = "classical"

    skip_first_n_frames: int = 20
    drop_last_n_frames: int = 10
    use_encoded_video: bool = True
    correct_transversal: bool = True
    correct_axial: bool = True
    flatten_rpe: bool = False
    axial_refinement: bool = False
    fovea_correction_enabled: bool = False

    lateral_method: Literal["xcorr", "fullframe", "both"] = "fullframe"
    max_lateral_shift: int = 8
    smooth_transversal: bool = False
    smooth_transversal_sigma: float = 2.0
    crop_factor: float = (
        0.5  # fraction of the frame width to keep for lateral registration
    )
    scale_factor: float = 1.0  # downscale factor for lateral registration
    transversal_bandpass: tuple[float, float] = (0.02, 0.5)
    axial_bandpass: tuple[float, float] = (0.02, 0.5)
    max_axial_shift: int = 7

    # General.
    subpixel: bool = True
    batch_size: int = REGISTRATION_BATCH_SIZE

    # --- Learned registration (method="learned") -------------------------
    # Weights of the RegistrationRegressor. It reads the *frozen* encoder of
    # the segmentation checkpoint, so the two must be the pair that was
    # trained together: swapping CHECKPOINT_PATH invalidates these weights.
    registrator_checkpoint: Path = REGISTRATOR_CHECKPOINT

    # Frames the probe pass segments to choose the reference frame. The
    # regressor registers every frame *onto that reference*, so the reference
    # has to be known before the single encode-once pass starts — and the rule
    # it is chosen by (mask area closest to the temporal median) needs mask
    # areas that do not exist yet. Sampling uniformly across the volume
    # estimates that median from `probe_frames` frames instead of all of them,
    # at the cost of one extra encode each. The chosen frame's percentile in
    # the full area distribution is logged, so too small an N is visible.
    probe_frames: int = 64

    # How the reference is chosen among the probe frames.
    #
    # "area" is the historical rule: mask area closest to the temporal median.
    # On a subsample that rule is not enough — a frame can sit exactly at the
    # median *area* and still be at the extreme of the eye's axial drift, and
    # registering the whole volume onto it then asks for the largest warp every
    # other frame could need. On 1242919/2023-04-20/OD the area rule picked a
    # frame at area-percentile 53 that pushed 73 of 2973 frames entirely off
    # the canvas.
    #
    # "motion_medoid" keeps the area rule only to pick a pivot, then measures
    # dy from that pivot to every probe frame and takes the one sitting at the
    # median of the axial trajectory. The probe features are already in memory,
    # so this costs one regressor call per probe frame (<1 s).
    # An int fixes the reference frame index directly and skips the probe.
    reference_selection: Literal["area", "motion_medoid"] | int = "motion_medoid"

    # Frames per forward pass of the fused stage. Separate from `batch_size`
    # (which sizes the classical warp): this one also holds encoder
    # activations for a 1536x1024 input, so it is the memory-sensitive one.
    fused_batch_size: int = 8

    # Blank the A-scan columns whose BM is unreliable, in frames and masks
    # alike. Not a registration estimate — a mask-quality postprocess the
    # downstream trace source expects — so it survives method="learned".
    filter_bad_columns: bool = True
    keep_largest_cc: bool = True
