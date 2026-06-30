from ocularrigidity.consts import (
    ROOT_MASKS,
    ROOT_COMPRESSED_VIDEO,
    ROOT_REGISTERED_CACHE,
)
from ocularrigidity.pipeline_config import REGISTRATION
from pathlib import Path
from ocularrigidity.data.measurements.dataframe import load_measurements
from tqdm.auto import tqdm
from ocularrigidity.motion.registered_video import RegisteredVideo

OVERWRITE = False


def register_all(cache_dir=None):
    df = load_measurements()
    for index, row in tqdm(df.iterrows(), total=len(df)):
        video = Path(row["MeasureValue"])
        # Convert to unix path
        video = video.as_posix().replace("\\", "/")
        try:
            registrator = RegisteredVideo(
                video=video,
                root_data=Path(ROOT_COMPRESSED_VIDEO),
                root_masks=Path(ROOT_MASKS),
                skip_first_n_frames=REGISTRATION.skip_first_n_frames,
                drop_last_n_frames=REGISTRATION.drop_last_n_frames,
                flatten=REGISTRATION.flatten,
                horizontal_alignment=REGISTRATION.horizontal_alignment,
                verbose=True,
                use_encoded_video=REGISTRATION.use_encoded_video,
                cache_dir=cache_dir,
                lateral_method=REGISTRATION.lateral_method,
                subpixel=REGISTRATION.subpixel,
                batch_size=REGISTRATION.batch_size,
            )
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
