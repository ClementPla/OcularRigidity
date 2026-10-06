"""Parameters of the registration engine."""

from dataclasses import dataclass
from typing import Literal

from ocularrigidity.consts import REGISTRATION_BATCH_SIZE, REGISTRATOR_REVISION


@dataclass()
class RegistrationConfig:
    # Which estimator produces (dx, dy).
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
    # Hub revision of the RegistrationRegressor weights.
    registrator_revision: str = REGISTRATOR_REVISION

    # Frames the probe pass segments to choose the reference frame.
    probe_frames: int = 64

    # How the reference is chosen among the probe frames.
    reference_selection: Literal["area", "motion_medoid"] | int = "motion_medoid"

    # Frames per forward pass of the fused stage.
    fused_batch_size: int = 8

    # Blank the A-scan columns whose BM is unreliable, in frames and masks alike.
    filter_bad_columns: bool = True
    keep_largest_cc: bool = True

    dy_bulk_only: bool = False

    dy_align_bm: bool = False
    # Smoothing of that correction along x, in columns (median, then Gaussian of this sigma).
    dy_align_bm_sigma: float = 5.0
