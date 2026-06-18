from pathlib import Path
from ocularrigidity.data.compression import read_gray
from ocularrigidity.data.io import load_mask
from ocularrigidity.motion.displacement import (
    compute_delta_A_from_displacements,
    compute_minimal_A,
    extract_displacement_at_boundaries,
)
from ocularrigidity.segmentation.closing_structures import trim_choroid
import matplotlib.pyplot as plt


def extract_displacement(
    video_path: Path, mask_path: Path, N_cycles: int = 3, method="optical_flow"
):
    video = read_gray(video_path)
    mask = load_mask(mask_path)

    trimmed_masks = trim_choroid(
        mask,
        75,
    )

    frame_per_cycle = video.shape[0] // N_cycles
    deltaA_per_cycle = []
    minA_per_cycle = []
    displacement_per_cycle = []
    reference_coordinates_per_cycle = []
    for i in range(N_cycles):
        start_frame = i * frame_per_cycle
        end_frame = (i + 1) * frame_per_cycle
        video_cycle = video[start_frame:end_frame]
        mask_cycle = trimmed_masks[start_frame:end_frame]
        displacement, reference_border_coordinates = extract_displacement_at_boundaries(
            video_cycle,
            mask_cycle,
            smooth_window=11,
            lk_window=35,
            method=method,
        )
        delta_a_differential = compute_delta_A_from_displacements(
            reference_border_coordinates, displacement
        )
        deltaA_per_cycle.append(delta_a_differential)
        minA = compute_minimal_A(reference_border_coordinates, displacement)
        minA_per_cycle.append(minA)
        displacement_per_cycle.append(displacement)
        reference_coordinates_per_cycle.append(reference_border_coordinates)
    return (
        deltaA_per_cycle,
        minA_per_cycle,
        displacement_per_cycle,
        reference_coordinates_per_cycle,
    )


if __name__ == "__main__":
    mask_path = Path(
        "/media/clement/HD/Santiago/OcularRigidity/outputs/CardiacPipeline/measures_ica_peak_locked/433755/2023-03-09/Rigidity/OS/segmented_cycles.npz"
    )
    video_path = Path(
        "/media/clement/HD/Santiago/OcularRigidity/outputs/CardiacPipeline/one_cycle_ica_peak_locked/433755/2023-03-09/Rigidity/OS/one_cycle.mkv"
    )
    deltaA_per_cycle, minA_per_cycle = extract_displacement(video_path, mask_path)
