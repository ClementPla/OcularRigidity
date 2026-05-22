import logging
from pathlib import Path
import traceback

import numpy as np
from tqdm import tqdm

from ocularrigidity.data.io import load_cube
from ocularrigidity.data.compression import cube_to_mp4, cube_to_mp4_fastest
from ocularrigidity.data.measurements.dataframe import load_measurements
from ocularrigidity.consts import ROOT_DATA_SMB, ROOT_DATA_MNT


OUTPUT_FOLDER = Path("/media/clement/HD/Santiago/OcularRigidity/outputs/compressed")
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
    return output_folder / rel / "cube.mp4"


def process_row(
    row, output_folder: Path, logger: logging.Logger, fps: int = 30
) -> bool:
    measure_value = Path(row["MeasureValue"]).as_posix().replace("\\", "/")
    out_path = output_path_for(measure_value, output_folder)

    if out_path.exists():
        logger.info(f"SKIP (already exists): {out_path}")
        return True

    try:
        data = load_cube(ROOT_DATA_MNT / Path(measure_value))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        cube_to_mp4_fastest(data, out_path, fps=fps, cq=18)

        logger.info(f"OK: {measure_value} → {out_path} (shape={data.shape})")
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
    logger.info("Starting compression")

    fps = 30  # arbitrary, since this is not really a video.
    df = load_measurements()
    logger.info(f"Loaded {len(df)} measurements")
    logger.info(f"Using cq={15}, fps={fps} for compression")

    n_ok, n_fail = 0, 0
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        success = process_row(row, OUTPUT_FOLDER, logger, fps=fps)
        if success:
            n_ok += 1
        else:
            n_fail += 1

    logger.info("=" * 60)
    logger.info(f"Done. {n_ok} succeeded, {n_fail} failed.")
    logger.info(f"Full log: {LOG_FILE}")


if __name__ == "__main__":
    main()
