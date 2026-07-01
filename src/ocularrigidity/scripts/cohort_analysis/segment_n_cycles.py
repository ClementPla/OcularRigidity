from pathlib import Path
from functools import lru_cache
from tqdm.auto import tqdm
from ocularrigidity.consts import ROOT_CARDIAC_PIPELINE
from ocularrigidity.pipeline_config import SEGMENTATION
from ocularrigidity.data.compression import read_gray
from ocularrigidity.data.io import save_mask
from ocularrigidity.scripts.exceptions_videos import PROCESS_ANYWAY
from ocularrigidity.segmentation.inference import infer
from ocularrigidity.segmentation.utils import get_choroid_segmentation_model


@lru_cache(maxsize=1)
def get_model():
    model = get_choroid_segmentation_model()
    return model.eval().cuda()


def segment_videos(video_path: Path) -> Path:
    """Segment videos into individual cycles."""
    data = read_gray(video_path)
    model = get_model()
    mask = infer(model, data, batch_size=SEGMENTATION.batch_size, device="cuda:0")
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
        if output_path.exists() and (Path(relative_path.parent) not in PROCESS_ANYWAY):
            continue
        mask = segment_videos(video_path)
        save_mask(mask, output_path)


if __name__ == "__main__":
    input_dir = ROOT_CARDIAC_PIPELINE
    one_cycle_dirs = list(input_dir.glob("one_cycle*"))
    pbar = tqdm(one_cycle_dirs, desc="Processing cycle directories")
    for one_cycle_dir in pbar:
        pbar.set_postfix_str(f"Processing {one_cycle_dir.name}")
        dirname = one_cycle_dir.name.replace("one_cycle", "measures")
        output_dir = input_dir / dirname
        process_cohort(one_cycle_dir, output_dir)
