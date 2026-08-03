from pathlib import Path
from functools import lru_cache
from tqdm.auto import tqdm
from ocularrigidity.consts import CHECKPOINT_PATH, ROOT_CARDIAC_PIPELINE
from ocularrigidity.pipeline_config import SEGMENTATION
from ocularrigidity.data.compression import cube_to_mp4, cube_to_mp4_fastest, read_gray
from ocularrigidity.data.io import save_mask
from ocularrigidity.scripts.exceptions_videos import PROCESS_ANYWAY
from ocularrigidity.segmentation.inference import infer
from ocularrigidity.segmentation.postprocess.temporal_smoothing import (
    smooth_masks_temporal,
)
from ocularrigidity.segmentation.trainer.pl_module import ChoroidSegmentationModule
import numpy as np
import pickle

OVERWRITE = True


@lru_cache(maxsize=1)
def get_model():
    """The same weights that segmented the raw videos (consts.CHECKPOINT_PATH).

    Previously this pulled the published Hugging Face checkpoint, which meant
    the folded cycles were measured with different weights than the ones the
    registration was built from — the two are not comparable.
    """
    model = ChoroidSegmentationModule.load_from_checkpoint(CHECKPOINT_PATH)
    return model.eval().cuda()


def segment_videos(video_path: Path, cardiac_freq: float, n_cycles: int) -> Path:
    """Segment videos into individual cycles."""
    data = read_gray(video_path)
    T, H, W = data.shape
    timestamps = (np.linspace(0, 1.0, T) / cardiac_freq) * n_cycles
    model = get_model()
    mask = infer(
        model,
        data,
        batch_size=SEGMENTATION.batch_size,
        device="cuda:0",
        use_graphcut=False,
    )
    one_cardiac_period = 1.0 / cardiac_freq
    # We filter to a window of 1/5 of a cardiac period around each timestamp
    mask = smooth_masks_temporal(
        mask, timestamps, sigma_time=one_cardiac_period / 5.0, sigma_col=0
    )
    return mask


def process_cohort(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    pbar = tqdm(
        list(input_dir.rglob("*.mkv")),
        desc="Processing videos",
        leave=False,
        position=1,
    )
    for video_path in pbar:
        relative_path = video_path.relative_to(input_dir)
        pbar.set_postfix_str(f"Processing {relative_path.parent}")
        output_path = output_dir / relative_path.parent / "segmented_cycles.npz"
        input_measures = output_dir / relative_path.parent / "measure.pkl"

        if (
            output_path.exists()
            and (Path(relative_path.parent) not in PROCESS_ANYWAY)
            and not OVERWRITE
        ):
            continue
        try:
            # A video with no measure.pkl is one the pulsation stage skipped;
            # that is a normal outcome for a rejected case, not a reason to stop.
            with open(input_measures, "rb") as f:
                measures = pickle.load(f)
            mask = segment_videos(video_path, measures.cardiac_freq, measures.n_cycle)
            save_mask(mask, output_path)
            cube_to_mp4_fastest(
                (mask * 255).astype(np.uint8),
                output_path.with_suffix(".mp4"),
                fps=30,
                cq=20,
            )
        except Exception as e:
            print(f"Error segmenting {relative_path.parent}: {type(e).__name__}: {e}")
            continue


if __name__ == "__main__":
    input_dir = ROOT_CARDIAC_PIPELINE
    one_cycle_dirs = list(input_dir.glob("one_cycle*"))
    pbar = tqdm(one_cycle_dirs, desc="Processing cycle directories")
    for one_cycle_dir in pbar:
        pbar.set_postfix_str(f"Processing {one_cycle_dir.name}")
        dirname = one_cycle_dir.name.replace("one_cycle", "measures")
        output_dir = input_dir / dirname
        process_cohort(one_cycle_dir, output_dir)
