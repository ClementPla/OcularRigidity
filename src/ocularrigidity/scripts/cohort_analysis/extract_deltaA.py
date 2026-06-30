from pathlib import Path
from ocularrigidity.consts import ROOT_CARDIAC_PIPELINE
from ocularrigidity.pipeline_config import DELTA_A
from ocularrigidity.data.compression import read_gray
from ocularrigidity.data.io import load_mask
from ocularrigidity.motion.displacement import (
    compute_delta_A_from_displacements,
    compute_minimal_A,
    extract_displacement_at_boundaries,
)
from ocularrigidity.segmentation.closing_structures import trim_choroid
import pickle
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

def extract_displacement(
    video_path: Path,
    mask_path: Path,
    N_cycles: int = DELTA_A.n_cycles,
    method=DELTA_A.method,
    smooth_window: int = DELTA_A.smooth_window,
    lk_window: int = DELTA_A.lk_window,
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
            smooth_window=smooth_window,
            lk_window=lk_window,
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
    input_dir = ROOT_CARDIAC_PIPELINE
    segmented_dirs = list(input_dir.rglob("**/*segmented_cycles.npz"))
    for segmented_dir in tqdm(segmented_dirs):
        result_filepath = segmented_dir.parent / "deltaA_per_cycle.pkl"
        if result_filepath.exists():
            continue
        try:
            relative_path = segmented_dir.relative_to(input_dir)
            video_path = input_dir / relative_path.parent / "one_cycle.mkv"
            # We need to replace "measures_" with "one_cycle_" in the relative path to get the corresponding video path
            video_path = Path(str(video_path).replace("measures_", "one_cycle_"))
            (
                deltaA_per_cycle,
                minA_per_cycle,
                displacement_per_cycle,
                reference_coordinates_per_cycle,
            ) = extract_displacement(video_path, segmented_dir)
            results = {
                "deltaA_per_cycle": deltaA_per_cycle,
                "minA_per_cycle": minA_per_cycle,
                "displacement_per_cycle": displacement_per_cycle,
                "reference_coordinates_per_cycle": reference_coordinates_per_cycle,}
            with open(result_filepath, "wb") as f:
                pickle.dump(results, f)
        except Exception as e:
            print(f"Error processing {segmented_dir}: {e}")