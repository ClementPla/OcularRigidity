import os

# MUST be set at the very top before importing numpy, cv2, or torch
# to prevent C++ OpenMP background thread locks.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import logging
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ocularrigidity.consts import (
    CHECKPOINT_PATH,
    ROOT_COMPRESSED_VIDEO,
    ROOT_DATA_MNT,
    ROOT_MASKS,
)
from ocularrigidity.data.compression import mp4_to_cube
from ocularrigidity.data.io import load_cube, save_mask
from ocularrigidity.data.measurements.dataframe import load_measurements
from ocularrigidity.pipeline_config import SEGMENTATION
from ocularrigidity.segmentation.inference import infer
from ocularrigidity.segmentation.trainer.pl_module import ChoroidSegmentationModule

OUTPUT_FOLDER = ROOT_MASKS
LOG_FILE = OUTPUT_FOLDER / "processing.log"


def setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("batch_infer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def output_path_for(measure_value: str, output_folder: Path) -> Path:
    rel = Path(measure_value.lstrip("/"))
    if rel.suffix == ".bin":
        rel = rel.parent
    return output_folder / rel / "mask.npz"


def identity_collate(batch):
    return batch[0]


class VolumeDataset(Dataset):
    """Prefetches and decodes MP4/bin cubes in isolated CPU worker processes."""

    def __init__(self, tasks):
        self.tasks = tasks

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        measure_value = self.tasks[idx]
        encoded = ROOT_COMPRESSED_VIDEO / measure_value / "cube.mp4"
        if encoded.exists():
            data = mp4_to_cube(encoded)
        else:
            data = load_cube(ROOT_DATA_MNT / measure_value)
        return measure_value, data


def main():
    logger = setup_logging(LOG_FILE)
    logger.info("Starting batch inference with PyTorch DataLoader")

    torch.backends.cudnn.benchmark = True

    model = (
        ChoroidSegmentationModule.load_from_checkpoint(CHECKPOINT_PATH).cuda().eval()
    )
    model = torch.compile(model)
    logger.info(f"Loaded checkpoint: {CHECKPOINT_PATH}")

    df = load_measurements(include_HR=False)

    tasks = []
    skipped = 0
    for _, row in df.iterrows():
        measure_value = str(Path(row["MeasureValue"])).replace("\\", "/")
        out_path = output_path_for(measure_value, OUTPUT_FOLDER)
        if out_path.exists():
            skipped += 1
        else:
            tasks.append(measure_value)

    logger.info(f"Total: {len(df)} | Skipped: {skipped} | To Process: {len(tasks)}")

    if not tasks:
        logger.info("Nothing to process.")
        return

    dataset = VolumeDataset(tasks)

    # DataLoader isolates MP4 decoding into CPU workers and passes arrays
    # to the GPU via Linux Shared Memory (/dev/shm)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=8,  # Adjust based on CPU cores (4 is usually optimal)
        shuffle=False,
        prefetch_factor=2,  # Keep up to 2 items in shared memory ahead of GPU
        collate_fn=identity_collate,  # Return raw tuple without automatic batch stacking
    )

    n_ok, n_fail = 0, 0

    for measure_value, data in tqdm(loader, total=len(tasks), desc="Processing"):
        try:
            with torch.inference_mode():
                mask = infer(
                    model,
                    data,
                    batch_size=SEGMENTATION.batch_size,
                    scale_factor=(1.0, 1.0),
                    return_logit=False,
                    use_graphcut=False,
                    graphcut_kwargs={"max_step": 1, "prob_threshold": 0.1},
                    verbose=True,
                )

            out_path = output_path_for(measure_value, OUTPUT_FOLDER)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_mask(mask, out_path)
            n_ok += 1

        except Exception as e:
            logger.error(
                f"FAILED: {measure_value}\n"
                f"  Error: {type(e).__name__}: {e}\n"
                f"  Traceback:\n{traceback.format_exc()}"
            )
            n_fail += 1

    logger.info("=" * 60)
    logger.info(f"Done. {n_ok} succeeded, {n_fail} failed, {skipped} skipped.")


if __name__ == "__main__":
    main()
