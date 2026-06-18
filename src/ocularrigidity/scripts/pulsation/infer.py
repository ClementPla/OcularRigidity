import numpy as np

from ocularrigidity.motion.one_cycle import estimate_cardiac_amplitude
from ocularrigidity.motion.pulsation import run_cardiac_pipeline
from ocularrigidity.consts import (
    CHECKPOINT_PATH,
    ROOT_MASKS,
    ROOT_DATA_MNT,
    ROOT_COMPRESSED_VIDEO,
)
from ocularrigidity.data.compression import cube_to_mkv_lossless, read_gray
from pathlib import Path
from ocularrigidity.data.measurements.dataframe import load_measurements
from tqdm.auto import tqdm
import pandas as pd
from ocularrigidity.motion.results import CardiacPipelineResults
from ocularrigidity.rigidity.features import compute_deltaY_masks
from ocularrigidity.segmentation.inference import infer
from ocularrigidity.segmentation.trainer.pl_module import ChoroidSegmentationModule

OVERWRITE = False


def compute_one_cycle(
    root_one_cycle, root_measures, method="pca", phase_method_for_fold="iq"
):
    df = load_measurements(include_HR=True)
    for index, row in tqdm(df.iterrows(), total=len(df)):
        video = Path(row["MeasureValue"])
        # Convert to unix path
        video = video.as_posix().replace("\\", "/")
        HR = row["HR"]
        if np.isnan(HR) or HR <= 0:
            HR = None

        if not (ROOT_COMPRESSED_VIDEO / video / "cube.mp4").exists():
            continue
        measure_path = root_measures / video / "measure.pkl"
        if measure_path.exists() and not OVERWRITE:
            continue
        one_cycle_path = root_one_cycle / video / "one_cycle.mkv"
        if one_cycle_path.exists() and not OVERWRITE:
            continue
        try:
            result: CardiacPipelineResults = run_cardiac_pipeline(
                video_relpath=video,
                root_masks=ROOT_MASKS,
                root_data=ROOT_COMPRESSED_VIDEO,
                timestamps_path=ROOT_DATA_MNT / video / "timestamp.txt",
                skip_first_n_frames=10,
                drop_last_n_frames=10,
                compute_n_cycle_video=True,
                flatten=False,
                horizontal_alignment=True,
                verbose=True,
                ICA_or_PCA=method,
                use_encoded_video=True,
                bpm_range=(30, 150),
                sigma_col=1,
                expected_bpm=HR,
                n_bins=30,
                col_slice=None,
                one_cycle_fold_method="median",
                n_cycle=3,
                phase_method_for_fold=phase_method_for_fold,
            )
        except Exception as e:
            print(f"Error processing {video}: {e}")
            continue
        one_cycle_path.parent.mkdir(parents=True, exist_ok=True)
        cube_to_mkv_lossless(
            result.cycles,
            str(one_cycle_path.parent / "one_cycle.mkv"),
            fps=30,
        )
        measure_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(measure_path, include_cycles=False)


def extract_deltaY_from_one_cycle(output_file: Path, input_one_cycle: Path, n_cycles=3):
    df = load_measurements(include_HR=True)
    model = ChoroidSegmentationModule.load_from_checkpoint(CHECKPOINT_PATH).cuda()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    for index, row in tqdm(df.iterrows(), total=len(df)):
        if not output_file.exists():
            df: pd.DataFrame = pd.DataFrame(
                columns=["video", "deltaY", "Amplitudes", "Fits", "cycle"]
            )
        else:
            df: pd.DataFrame = pd.read_pickle(output_file)
        video = Path(row["MeasureValue"]).as_posix().replace("\\", "/")
        if not OVERWRITE and video in df["video"].values:
            continue
        try:
            data = read_gray(input_one_cycle / video / "one_cycle.mkv")
            masks = infer(
                model,
                data,
                batch_size=32,
                scale_factor=(1.0, 1.0),
                return_logit=False,
                use_graphcut=True,
                use_amp=True,
                verbose=True,
                graphcut_kwargs=dict(
                    temporal_smooth=False,
                    temporal_iterations=4,
                    temporal_mu=1.0,
                    temporal_sigma=2.0,
                    lambda_smooth=1.0,
                ),
            )
            thickness = compute_deltaY_masks(masks)
            # thickness_smoothed = smooth_boundary_2d(
            #     thickness, sigma_time=3, sigma_col=5.0
            # )
            T = thickness.shape[0]
            for cycle in range(n_cycles):
                current_cycle = thickness[
                    cycle * T // n_cycles : (cycle + 1) * T // n_cycles
                ]
                fits, _ = estimate_cardiac_amplitude(
                    current_cycle,
                    n_harmonics=1,
                    residual_threshold_percentile=75,
                    amplitude_threshold_percentile=50,
                )

                amplitude = fits.max(axis=0) - fits.min(axis=0)
                mean_amplitude = np.mean(amplitude)
                row = pd.DataFrame(
                    {
                        "video": [video],
                        "cycle": [cycle],
                        "deltaY": [mean_amplitude],
                        "Amplitudes": [amplitude],
                        "Fits": [fits],
                    }
                )
                df = pd.concat([df, row], ignore_index=True)
            df.to_pickle(output_file)
        except Exception as e:
            print(f"Error processing {video}: {e}")
            continue


if __name__ == "__main__":
    for method in ["pca"]:
        for phase_method in [
            "peak_locked",
            "iq",
        ]:
            root_one_cycle = Path(
                f"/media/clement/HD/Santiago/OcularRigidity/outputs/CardiacPipeline_V2/one_cycle_{method}_{phase_method}/"
            )
            root_measures = Path(
                f"/media/clement/HD/Santiago/OcularRigidity/outputs/CardiacPipeline_V2/measures_{method}_{phase_method}/"
            )
            compute_one_cycle(
                root_one_cycle,
                root_measures,
                method=method,
                phase_method_for_fold=phase_method,
            )
            extract_deltaY_from_one_cycle(
                Path(
                    f"/media/clement/HD/Santiago/OcularRigidity/outputs/CardiacPipeline_V2/deltaY_{method}_{phase_method}.pkl"
                ),
                root_one_cycle,
            )
