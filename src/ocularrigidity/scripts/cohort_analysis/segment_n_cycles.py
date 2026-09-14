import os

from ocularrigidity.segmentation.utils import get_choroid_segmentation_model

# MUST be set at the very top before importing numpy, cv2, or torch
# to prevent C++ OpenMP background thread locks.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import pickle
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ocularrigidity.consts import CHECKPOINT_PATH, ROOT_CARDIAC_PIPELINE
from ocularrigidity.data.compression import cube_to_mp4_fastest, read_gray
from ocularrigidity.data.io import save_mask
from ocularrigidity.pipeline_config import SEGMENTATION
from ocularrigidity.scripts._logging import setup_logging
from ocularrigidity.scripts.dataset import PrefetchDataset, identity_collate
from ocularrigidity.scripts.exceptions_videos import PROCESS_ANYWAY
from ocularrigidity.segmentation.inference import infer
from ocularrigidity.segmentation.postprocess.blob import (
    keep_largest_connected_component,
)
from ocularrigidity.segmentation.postprocess.temporal_smoothing import (
    smooth_masks_temporal,
)

OUTPUT_FOLDER = ROOT_CARDIAC_PIPELINE

NUM_DECODE_WORKERS = 2
PREFETCH_FACTOR = 2


@lru_cache(maxsize=1)
def get_model(device: str = "cuda"):
    """The frozen choroid segmentation model, loaded straight onto ``device``.

    Never load and then move. Moving an already-loaded instance across GPUs --
    ``get_model().to("cuda:1")`` -- silently zeroes every parameter: no error,
    no warning, and the encoder then returns all-zero features for any input.
    Downstream that looks like a *model* failure (a registration that predicts
    the same constant dx/dy for every frame) rather than a loading one, which
    is exactly how it hid. Loading directly onto the device is fine, so the
    device is a parameter here and part of the lru_cache key.
    """
    model = get_choroid_segmentation_model().eval().to(device)
    if float(next(model.parameters()).abs().max()) == 0.0:
        raise RuntimeError(
            f"segmentation weights are all zero after loading onto {device!r} -- "
            "the checkpoint did not load; refusing to run with a dead encoder"
        )
    return model


@dataclass(frozen=True)
class Task:
    """One folded clip to segment, with the sidecar it needs and where it goes."""

    name: str  # cohort-relative id, for logs
    video_path: Path  # one_cycle.mkv
    measure_path: Path  # measure.pkl, for the cardiac frequency
    output_path: Path  # segmented_cycles.npz


def load_clip(task: Task):
    """Decode one folded clip plus its measures. Runs in a DataLoader worker.

    A clip with no measure.pkl is one the pulsation stage skipped; that raises
    here and is reported as a per-clip failure, not a reason to stop.
    """
    with open(task.measure_path, "rb") as f:
        measures = pickle.load(f)
    data = read_gray(task.video_path)
    tensor = torch.from_numpy(np.ascontiguousarray(data))
    return tensor, float(measures.cardiac_freq), int(measures.n_cycle)


def postprocess(raw_mask: np.ndarray, cardiac_freq: float, n_cycles: int) -> np.ndarray:
    """Largest-CC + temporal smoothing, the mask as it is written to disk."""
    mask = keep_largest_connected_component(raw_mask)

    T = mask.shape[0]
    timestamps = (np.linspace(0, 1.0, T) / cardiac_freq) * n_cycles
    one_cardiac_period = 1.0 / cardiac_freq
    # We filter to a window of 1/5 of a cardiac period around each timestamp
    return smooth_masks_temporal(
        mask, timestamps, sigma_time=one_cardiac_period / 5.0, sigma_col=0
    )


def write_cycle_mask(mask: np.ndarray, out_path: Path) -> None:
    """The npz + the preview mp4 next to it."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write then rename, so a run killed mid-write never leaves a truncated
    # npz that the next pass would skip as "already done". The name keeps the
    # .npz suffix because np.savez appends one otherwise.
    tmp_path = out_path.with_name(out_path.name + ".tmp.npz")
    save_mask(mask, tmp_path)
    tmp_path.replace(out_path)

    cube_to_mp4_fastest(
        (mask * 255).astype(np.uint8),
        out_path.with_suffix(".mp4"),
        fps=30,
        cq=20,
    )


def finalize(
    raw_mask: np.ndarray,
    cardiac_freq: float,
    n_cycles: int,
    out_path: Path,
) -> None:
    """Post-process + write. Runs on a background thread while the GPU moves on."""
    write_cycle_mask(postprocess(raw_mask, cardiac_freq, n_cycles), out_path)


def build_tasks(input_root: Path, overwrite: bool) -> tuple[list[Task], int]:
    """Every folded clip under ``input_root``, minus the ones already done.

    The whole cohort is collected up front, across all ``one_cycle*`` sweep
    directories, so the run is one DataLoader over one model load rather than
    one per directory.
    """
    tasks, skipped = [], 0
    for one_cycle_dir in sorted(input_root.glob("one_cycle*")):
        output_dir = input_root / one_cycle_dir.name.replace("one_cycle", "measures")
        for video_path in sorted(one_cycle_dir.rglob("*.mkv")):
            video = video_path.relative_to(one_cycle_dir).parent
            output_path = output_dir / video / "segmented_cycles.npz"
            process_anyway = Path(video) in PROCESS_ANYWAY

            if output_path.exists() and not overwrite and not process_anyway:
                skipped += 1
                continue

            tasks.append(
                Task(
                    name=f"{one_cycle_dir.name}/{video}",
                    video_path=video_path,
                    measure_path=output_dir / video / "measure.pkl",
                    output_path=output_path,
                )
            )
    return tasks, skipped


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-segment clips that already have a segmented_cycles.npz.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=SEGMENTATION.batch_size,
        help="Frames per GPU forward pass.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(
        "segment_n_cycles", OUTPUT_FOLDER / "logs" / "segment_n_cycles.log"
    )
    logger.info(
        f"Starting per-cycle segmentation | batch_size={args.batch_size} "
        f"| CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}"
    )

    torch.backends.cudnn.benchmark = True

    tasks, skipped = build_tasks(OUTPUT_FOLDER, args.overwrite)
    logger.info(
        f"Total: {len(tasks) + skipped} | Skipped: {skipped} | Remaining: {len(tasks)}"
    )
    if not tasks:
        logger.info("Nothing to process.")
        return

    model = get_model()
    logger.info(f"Loaded checkpoint: {CHECKPOINT_PATH}")

    loader = DataLoader(
        PrefetchDataset(tasks, load_clip),
        batch_size=1,
        num_workers=NUM_DECODE_WORKERS,
        shuffle=False,
        prefetch_factor=PREFETCH_FACTOR,
        collate_fn=identity_collate,  # return the raw tuple, no batch stacking
    )

    n_ok, n_fail = 0, 0
    pending = None  # in-flight finalize(), one clip behind the GPU

    # One writer thread: the CC pass is already internally parallel, and the
    # nvenc preview encode is a subprocess, so a deeper queue would only hold
    # more masks in RAM.
    with ThreadPoolExecutor(max_workers=1) as writer:

        def drain(job):
            nonlocal n_ok, n_fail
            if job is None:
                return
            future, name = job
            try:
                future.result()
                n_ok += 1
            except Exception as e:
                logger.error(
                    f"FAILED (write): {name}\n  Error: {type(e).__name__}: {e}"
                )
                n_fail += 1

        t_mark = time.perf_counter()
        for i, (task, payload, load_error) in enumerate(loader, start=1):
            # Time blocked on the loader. Near zero means the decode workers are
            # keeping up and the GPU is the wall.
            t_wait = time.perf_counter() - t_mark

            if load_error is not None:
                logger.error(f"FAILED (load): {task.name}\n{load_error}")
                n_fail += 1
                t_mark = time.perf_counter()
                continue

            data, cardiac_freq, n_cycles = payload
            n_frames = len(data)
            t_gpu = time.perf_counter()
            try:
                raw_mask = infer(
                    model,
                    data,
                    batch_size=args.batch_size,
                    device="cuda:0",
                    use_graphcut=False,
                    post_process=False,  # done on the writer thread below
                )
                t_gpu = time.perf_counter() - t_gpu
            except Exception as e:
                logger.error(
                    f"FAILED (inference): {task.name}\n"
                    f"  Error: {type(e).__name__}: {e}\n"
                    f"  Traceback:\n{traceback.format_exc()}"
                )
                n_fail += 1
                t_mark = time.perf_counter()
                continue
            finally:
                del data, payload

            # Collect the previous clip before queueing this one, so at most one
            # mask is in flight. A non-zero wait here means the post-process +
            # write has become the wall rather than the GPU.
            t_writer = time.perf_counter()
            drain(pending)
            t_writer = time.perf_counter() - t_writer

            pending = (
                writer.submit(
                    finalize, raw_mask, cardiac_freq, n_cycles, task.output_path
                ),
                task.name,
            )
            del raw_mask

            logger.info(
                f"[{i}/{len(tasks)}] {task.name} | {n_frames} frames | "
                f"load-wait {t_wait:5.1f}s | gpu {t_gpu:5.1f}s | "
                f"writer-wait {t_writer:5.1f}s"
            )
            t_mark = time.perf_counter()

        drain(pending)

    logger.info("=" * 60)
    logger.info(f"Done. {n_ok} succeeded, {n_fail} failed, {skipped} skipped.")


if __name__ == "__main__":
    main()
