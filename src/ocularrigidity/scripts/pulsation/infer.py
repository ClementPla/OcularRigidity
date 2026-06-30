import numpy as np

from ocularrigidity.motion.one_cycle import estimate_cardiac_amplitude
from ocularrigidity.motion.pulsation import run_cardiac_pipeline
from ocularrigidity.consts import (
    CHECKPOINT_PATH,
    ROOT_MASKS,
    ROOT_DATA_MNT,
    ROOT_COMPRESSED_VIDEO,
    ROOT_REGISTERED_CACHE,
    ROOT_CARDIAC_PIPELINE,
)
from ocularrigidity.pipeline_config import REGISTRATION, PULSATION, DELTA_Y
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
    root_one_cycle,
    root_measures,
    method="pca",
    phase_method_for_fold="iq",
    cache_dir=None,
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
                skip_first_n_frames=REGISTRATION.skip_first_n_frames,
                drop_last_n_frames=REGISTRATION.drop_last_n_frames,
                compute_n_cycle_video=True,
                flatten=REGISTRATION.flatten,
                horizontal_alignment=REGISTRATION.horizontal_alignment,
                lateral_method=REGISTRATION.lateral_method,
                subpixel=REGISTRATION.subpixel,
                use_encoded_video=REGISTRATION.use_encoded_video,
                verbose=True,
                ICA_or_PCA=method,
                sigma_col=PULSATION.sigma_col,
                expected_bpm=HR,
                expected_bpm_band_frac=PULSATION.expected_bpm_band_frac,
                n_bins=PULSATION.n_bins,
                col_slice=PULSATION.col_slice,
                one_cycle_fold_method=PULSATION.one_cycle_fold_method,
                n_cycle=PULSATION.n_cycle,
                phase_method_for_fold=phase_method_for_fold,
                cache_dir=cache_dir,
            )
        except Exception as e:
            print(f"Error processing {video}: {e}")
            continue
        one_cycle_path.parent.mkdir(parents=True, exist_ok=True)
        cube_to_mkv_lossless(
            result.cycles,
            str(one_cycle_path.parent / "one_cycle.mkv"),
            fps=PULSATION.output_fps,
        )
        measure_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(measure_path, include_cycles=False)


def extract_deltaY_from_one_cycle(
    output_file: Path, input_one_cycle: Path, n_cycles=DELTA_Y.n_cycles
):
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
                batch_size=DELTA_Y.batch_size,
                scale_factor=(1.0, 1.0),
                return_logit=False,
                use_graphcut=True,
                use_amp=True,
                verbose=True,
                graphcut_kwargs=DELTA_Y.graphcut_kwargs,
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
                    n_harmonics=DELTA_Y.n_harmonics,
                    residual_threshold_percentile=DELTA_Y.residual_threshold_percentile,
                    amplitude_threshold_percentile=DELTA_Y.amplitude_threshold_percentile,
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
    # Registration is identical across method/phase combos, so a shared cache
    # (sibling of the compressed/ and masks/ roots) is computed once and reused.
    for method in PULSATION.methods:
        for phase_method in PULSATION.phase_methods:
            root_one_cycle = (
                ROOT_CARDIAC_PIPELINE / f"one_cycle_{method}_{phase_method}"
            )
            root_measures = (
                ROOT_CARDIAC_PIPELINE / f"measures_{method}_{phase_method}"
            )
            compute_one_cycle(
                root_one_cycle,
                root_measures,
                method=method,
                phase_method_for_fold=phase_method,
                cache_dir=ROOT_REGISTERED_CACHE,
            )
            extract_deltaY_from_one_cycle(
                ROOT_CARDIAC_PIPELINE / f"deltaY_{method}_{phase_method}.pkl",
                root_one_cycle,
            )
