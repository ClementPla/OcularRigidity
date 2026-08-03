"""Parameters of the registration engine.

Lives with the code it configures: ``VideoRegistrator`` and ``register_videos``
take this directly, so the registration package must not have to import the
study-level ``pipeline_config`` to know its own arguments. The cohort-wide
instance in force for the pipeline is the ``REGISTRATION`` singleton in
:mod:`ocularrigidity.pipeline_config`.
"""

from dataclasses import dataclass
from typing import Literal

from ocularrigidity.consts import REGISTRATION_BATCH_SIZE


@dataclass()
class RegistrationConfig:
    skip_first_n_frames: int = 20
    drop_last_n_frames: int = 10
    use_encoded_video: bool = True
    # What to correct.
    correct_transversal: bool = False
    correct_axial: bool = True
    flatten_rpe: bool = False
    axial_refinement: bool = False
    fovea_correction_enabled: bool = True

    # Transversal (x) parameters.
    lateral_method: Literal["xcorr", "fullframe", "both"] = "fullframe"
    max_lateral_shift: int = 16
    smooth_transversal: bool = False
    smooth_transversal_sigma: float = 2.0
    crop_factor: float = (
        0.66  # fraction of the frame width to keep for lateral registration
    )
    scale_factor: float = 1.0  # downscale factor for lateral registration
    transversal_bandpass: tuple[float, float] = (0.02, 0.5)
    axial_bandpass: tuple[float, float] = (0.02, 0.5)
    # Axial (y) parameters. ``max_axial_shift`` is the RPE-refinement pass's
    # maximal tested vertical shift (px).
    max_axial_shift: int = 7

    # General.
    subpixel: bool = True
    # Frames warped per grid_sample call. Purely a memory/throughput trade-off:
    # it does not change the result. Set OCULARRIGIDITY_REGISTRATION_BATCH to
    # tune it for the card.
    batch_size: int = REGISTRATION_BATCH_SIZE
