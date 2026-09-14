"""Segment every raw cohort video.

Decode (CPU), inference (GPU) and largest-CC + save (CPU) are pipelined so the
GPU is never idle: decode runs in DataLoader workers, which hand back a torch
tensor rather than a numpy array so the multi-GB cube travels through shared
memory instead of being pickled down a socket; largest-CC + save run on a
background thread.

The GPU pass is already saturated per batch, so the only way past it is more
cards: ``--num-shards``/``--shard`` split the volumes over several GPUs, one
process each (see scripts/pipeline.sh).

Resumable: volumes whose ``mask.npz`` exists are skipped, and a volume that
fails is logged and stepped over rather than aborting the run.
"""

import os

from ocularrigidity.scripts.dataset import VolumeDataset, identity_collate

# MUST be set at the very top before importing numpy, cv2, or torch
# to prevent C++ OpenMP background thread locks.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ocularrigidity.consts import (
    CHECKPOINT_PATH,
    ROOT_MASKS,
)
from ocularrigidity.data.io import save_mask
from ocularrigidity.data.measurements.dataframe import load_measurements
from ocularrigidity.pipeline_config import SEGMENTATION
from ocularrigidity.scripts._logging import setup_logging
from ocularrigidity.segmentation.inference import infer
from ocularrigidity.segmentation.postprocess.blob import (
    keep_largest_connected_component,
)
from ocularrigidity.segmentation.trainer.pl_module import ChoroidSegmentationModule

OUTPUT_FOLDER = ROOT_MASKS

# One cube in flight is ~4.7 GB, so the queue depth is bounded deliberately:
# two decode workers already cover the GPU pass, and each extra prefetched item
# is another 4.7 GB of shared memory (times the number of shards).
NUM_DECODE_WORKERS = 2
PREFETCH_FACTOR = 1


def output_path_for(measure_value: str, output_folder: Path) -> Path:
    rel = Path(measure_value.lstrip("/"))
    if rel.suffix == ".bin":
        rel = rel.parent
    return output_folder / rel / "mask.npz"


def finalize(raw_mask: np.ndarray, out_path: Path) -> None:
    """Largest-CC + write. Runs on a background thread while the GPU moves on."""
    mask = keep_largest_connected_component(raw_mask)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write then rename, so a run killed mid-write never leaves a truncated
    # mask.npz that the next pass would skip as "already done". The name keeps
    # the .npz suffix because np.savez appends one otherwise.
    tmp_path = out_path.with_name(out_path.name + ".tmp.npz")
    save_mask(mask, tmp_path)
    tmp_path.replace(out_path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the cohort into N disjoint shards (one process per GPU).",
    )
    p.add_argument(
        "--shard",
        type=int,
        default=0,
        help="Which shard this process handles, 0-based.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=SEGMENTATION.batch_size,
        help="Frames per GPU forward pass.",
    )
    p.add_argument(
        "--no-compile",
        action="store_true",
        help="Skip torch.compile. Falls back to eager, which costs throughput "
        "but skips the one-off compile at startup.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.shard < args.num_shards:
        raise SystemExit(f"--shard must be in [0, {args.num_shards})")

    sharded = args.num_shards > 1
    rank_tag = f" [shard {args.shard}]" if sharded else ""
    log_name = f"processing.shard{args.shard}.log" if sharded else "processing.log"
    logger = setup_logging("batch_infer", OUTPUT_FOLDER / log_name, rank_tag)
    logger.info(
        f"Starting batch inference | batch_size={args.batch_size} "
        f"| shard {args.shard + 1}/{args.num_shards} "
        f"| CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}"
    )

    torch.backends.cudnn.benchmark = True

    model = (
        ChoroidSegmentationModule.load_from_checkpoint(CHECKPOINT_PATH).cuda().eval()
    )
    logger.info(f"Loaded checkpoint: {CHECKPOINT_PATH}")

    if not args.no_compile:
        # dynamic=False plus pad_last_batch below means exactly one input shape
        # for the whole run, so this compiles once rather than specialising per
        # volume. The cost lands on the first volume's `gpu` figure — expect it
        # to read a minute or two high, and judge throughput from the second.
        # A compile failure degrades to eager instead of killing a run that is
        # otherwise going to be left alone for an hour.
        torch._dynamo.config.suppress_errors = True
        model = torch.compile(model, dynamic=False)
        logger.info("torch.compile enabled (dynamic=False, fixed batch shape)")

    df = load_measurements(include_HR=False)

    tasks = []
    skipped = 0
    for _, row in df.iterrows():
        measure_value = str(Path(row["MeasureValue"])).replace("\\", "/")
        if output_path_for(measure_value, OUTPUT_FOLDER).exists():
            skipped += 1
        else:
            tasks.append(measure_value)

    # Shard *after* the skip check: splitting the cohort instead would hand one
    # shard everything left to do whenever the remaining volumes happen to
    # cluster on one side of the split.
    remaining = len(tasks)
    tasks = tasks[args.shard :: args.num_shards]

    logger.info(
        f"Total: {len(df)} | Skipped: {skipped} | Remaining: {remaining} "
        f"| This shard: {len(tasks)}"
    )

    if not tasks:
        logger.info("Nothing to process.")
        return

    dataset = VolumeDataset(tasks)

    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=NUM_DECODE_WORKERS,
        shuffle=False,
        prefetch_factor=PREFETCH_FACTOR,
        # No pin_memory: page-locking a 4.7 GB cube costs ~4 s and saves ~1 s of
        # copy, and the locked pages are RAM the kernel cannot reclaim while two
        # shards and a desktop compete for it.
        collate_fn=identity_collate,  # return raw tuple, no batch stacking
    )

    n_ok, n_fail = 0, 0
    pending = None  # in-flight finalize(), one volume behind the GPU

    # One writer thread: the CC pass is already internally parallel, and a
    # deeper queue would just hold more 4.7 GB masks in RAM.
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
        for i, (measure_value, data, load_error) in enumerate(loader, start=1):
            # Time blocked on the loader. Near zero means the decode workers are
            # keeping up and the GPU is the wall; large means they are not, and
            # the fix is more workers rather than more cards.
            t_wait = time.perf_counter() - t_mark

            if load_error is not None:
                logger.error(f"FAILED (load): {measure_value}\n{load_error}")
                n_fail += 1
                t_mark = time.perf_counter()
                continue

            n_frames = len(data)
            t_gpu = time.perf_counter()
            try:
                with torch.inference_mode():
                    raw_mask = infer(
                        model,
                        data,
                        batch_size=args.batch_size,
                        scale_factor=1.0,
                        return_logit=False,
                        use_graphcut=False,
                        graphcut_kwargs={"max_step": 1, "prob_threshold": 0.1},
                        post_process=False,  # done on the writer thread below
                        pin_memory=False,  # see the DataLoader note above
                        accumulate_on_cpu=True,
                        pad_last_batch=not args.no_compile,
                        verbose=True,
                    )
                t_gpu = time.perf_counter() - t_gpu
            except Exception as e:
                logger.error(
                    f"FAILED: {measure_value}\n"
                    f"  Error: {type(e).__name__}: {e}\n"
                    f"  Traceback:\n{traceback.format_exc()}"
                )
                n_fail += 1
                t_mark = time.perf_counter()
                continue
            finally:
                del data

            # Collect the previous volume before queueing this one, so at most
            # one mask is in flight. A non-zero wait here means largest-CC + save
            # has become the wall rather than the GPU.
            t_writer = time.perf_counter()
            drain(pending)
            t_writer = time.perf_counter() - t_writer

            pending = (
                writer.submit(
                    finalize, raw_mask, output_path_for(measure_value, OUTPUT_FOLDER)
                ),
                measure_value,
            )
            del raw_mask

            # One line per volume: without it the stage prints nothing for
            # minutes at a time and looks hung.
            logger.info(
                f"[{i}/{len(tasks)}] {measure_value} | {n_frames} frames | "
                f"load-wait {t_wait:5.1f}s | gpu {t_gpu:5.1f}s | "
                f"writer-wait {t_writer:5.1f}s"
            )
            t_mark = time.perf_counter()

        drain(pending)

    logger.info("=" * 60)
    logger.info(f"Done. {n_ok} succeeded, {n_fail} failed, {skipped} skipped.")


if __name__ == "__main__":
    main()
