from ocularrigidity.consts import (
    ROOT_MASKS,
    ROOT_COMPRESSED_VIDEO,
    ROOT_REGISTERED_CACHE,
)
from ocularrigidity.pipeline_config import REGISTRATION
from pathlib import Path
from ocularrigidity.data.measurements.dataframe import load_measurements
from tqdm.auto import tqdm
from ocularrigidity.registration.registration_engine import VideoRegistrator
from ocularrigidity.scripts.exceptions_videos import PROCESS_ANYWAY

OVERWRITE = False


def register_all(cache_dir=None):
    df = load_measurements()
    for index, row in tqdm(df.iterrows(), total=len(df)):
        video = Path(row["MeasureValue"])
        # Convert to unix path
        video = video.as_posix().replace("\\", "/")

        try:
            registrator = VideoRegistrator(
                video=video,
                root_data=Path(ROOT_COMPRESSED_VIDEO),
                root_masks=Path(ROOT_MASKS),
                skip_first_n_frames=REGISTRATION.skip_first_n_frames,
                drop_last_n_frames=REGISTRATION.drop_last_n_frames,
                correct_transversal=REGISTRATION.correct_transversal,
                correct_axial=REGISTRATION.correct_axial,
                flatten_rpe=REGISTRATION.flatten_rpe,
                axial_refinement=REGISTRATION.axial_refinement,
                fovea_correction_enabled=REGISTRATION.fovea_correction_enabled,
                lateral_method=REGISTRATION.lateral_method,
                max_lateral_shift=REGISTRATION.max_lateral_shift,
                smooth_transversal=REGISTRATION.smooth_transversal,
                smooth_transversal_sigma=REGISTRATION.smooth_transversal_sigma,
                max_axial_shift=REGISTRATION.max_axial_shift,
                subpixel=REGISTRATION.subpixel,
                use_encoded_video=REGISTRATION.use_encoded_video,
                batch_size=REGISTRATION.batch_size,
                cache_dir=cache_dir,
                overwrite_cache=OVERWRITE or (Path(video) in PROCESS_ANYWAY),
                verbose=True,
            )
            paths = registrator._cache_paths()
            if Path(video) in PROCESS_ANYWAY:
                print(f"Processing {video} anyway (in PROCESS_ANYWAY)")
            if all(p.exists() for p in paths.values()) and (
                Path(video) not in PROCESS_ANYWAY
            ):
                continue
            registrator.compute_registration()
        except Exception as e:
            print(f"Error registering {video}: {e}")
            continue


if __name__ == "__main__":
    # Registration is identical across method/phase combos, so a shared cache
    # (sibling of the compressed/ and masks/ roots) is computed once and reused.
    register_all(
        cache_dir=ROOT_REGISTERED_CACHE,
    )
