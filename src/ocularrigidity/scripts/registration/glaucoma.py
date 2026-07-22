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
                config=REGISTRATION,
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
