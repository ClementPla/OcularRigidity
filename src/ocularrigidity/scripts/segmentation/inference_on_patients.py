import logging
from pathlib import Path
import traceback

import numpy as np
from tqdm import tqdm

from ocularrigidity.segmentation.trainer.pl_module import ChoroidSegmentationModule
from ocularrigidity.segmentation.inference import infer
from ocularrigidity.data.io import load_cube, save_mask
from ocularrigidity.data.measurements.dataframe import load_measurements
from ocularrigidity.consts import CHECKPOINT_PATH, ROOT_DATA_SMB, ROOT_DATA_MNT


OUTPUT_FOLDER = Path("/media/clement/HD/Santiago/OcularRigidity/outputs/masks")
LOG_FILE = OUTPUT_FOLDER / "processing.log"


def setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("batch_infer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers if re-run in a notebook

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def output_path_for(measure_value: str, output_folder: Path) -> Path:
    """Mirror the input path structure under output_folder, saving as .npz."""
    rel = Path(measure_value.lstrip("/"))
    # If MeasureValue points at a folder (containing cube.bin), use that folder name.
    # If it points at cube.bin directly, use its parent.
    if rel.suffix == ".bin":
        rel = rel.parent
    return output_folder / rel / "mask.npz"


def process_row(row, model, output_folder: Path, logger: logging.Logger) -> bool:
    measure_value = str(Path(row["MeasureValue"])).replace(
        "\\", "/"
    )  # normalize Windows paths
    out_path = output_path_for(measure_value, output_folder)

    if out_path.exists():
        logger.info(f"SKIP (already exists): {out_path}")
        return True

    try:
        data = load_cube(ROOT_DATA_MNT + measure_value)
        mask = infer(
            model,
            data,
            batch_size=64,
            scale_factor=(1.0, 1.0),
            return_logit=False,
            use_graphcut=True,
            graphcut_kwargs={"max_step": 1, "prob_threshold": 0.1},
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Save as packed bits to shrink file size ~8x (bool → bit per pixel)
        save_mask(mask, out_path)

        logger.info(f"OK: {measure_value} → {out_path} (shape={mask.shape})")
        return True

    except Exception as e:
        logger.error(
            f"FAILED: {measure_value}\n"
            f"  Error: {type(e).__name__}: {e}\n"
            f"  Traceback:\n{traceback.format_exc()}"
        )
        return False


def main():
    logger = setup_logging(LOG_FILE)
    logger.info("=" * 60)
    logger.info("Starting batch inference")

    model = ChoroidSegmentationModule.load_from_checkpoint(CHECKPOINT_PATH).cuda()
    logger.info(f"Loaded checkpoint: {CHECKPOINT_PATH}")

    df = load_measurements(include_HR=False)
    logger.info(f"Loaded {len(df)} measurements")

    n_ok, n_fail = 0, 0
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        success = process_row(row, model, OUTPUT_FOLDER, logger)
        if success:
            n_ok += 1
        else:
            n_fail += 1

    logger.info("=" * 60)
    logger.info(f"Done. {n_ok} succeeded, {n_fail} failed.")
    logger.info(f"Full log: {LOG_FILE}")


if __name__ == "__main__":
    main()
