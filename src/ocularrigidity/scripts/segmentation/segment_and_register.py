"""Segment and register every cohort video in one GPU pass per volume.

One encoder pass feeds both the segmentation decoder and the trained
:class:`RegistrationRegressor`, roughly halving the GPU time against running
the two stages separately (see :mod:`ocularrigidity.registration.fused`).

It writes the same artifacts, in the same places, as the standalone
segmentation and registration stages, so nothing downstream changes:

    masks/<video>/mask.npz                    raw (unregistered) segmentation
    registered_frames/<video>/cube.mp4        registered frames
    registered_masks/<video>/mask.npz         registered masks
    registered_masks/<video>/transform.npz    dx, dy + the config they came from

The write goes through :meth:`VideoRegistrator.prime_from_result`, so the cache
is produced by exactly the code that reads it back, and the old registration
path is never entered.

Pipelined like the stages it replaces: decode runs in DataLoader workers ahead
of the GPU, and the encode + mask compression run on a writer thread so the GPU
starts the next volume immediately. ``--num-shards``/``--shard`` split the
cohort over several GPUs, one process each.

Resumable: a volume whose registration cache is valid *and* whose raw mask
exists is skipped, and a volume that fails is logged and stepped over rather
than aborting the cohort.
"""

import os

# MUST be set at the very top before importing numpy, cv2, or torch
# to prevent C++ OpenMP background thread locks.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import gc
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ocularrigidity.consts import (
    CHECKPOINT_PATH,
    ROOT_MASKS,
    ROOT_REGISTERED_CACHE,
)
from ocularrigidity.data.measurements.dataframe import load_measurements
from ocularrigidity.pipeline_config import REGISTRATION
from ocularrigidity.registration.deep_learning.inference import (
    load_registration_regressor,
)
from ocularrigidity.registration.fused import segment_and_register
from ocularrigidity.registration.registration_engine import (
    VideoRegistrator,
    cache_is_valid,
)
from ocularrigidity.scripts._logging import setup_logging
from ocularrigidity.scripts.dataset import (
    PrefetchDataset,
    identity_collate,
    load_source_frames,
    source_path,
)
from ocularrigidity.scripts.exceptions_videos import PROCESS_ANYWAY
from ocularrigidity.scripts.segmentation.inference_on_patients import (
    finalize as finalize_raw_mask,
)

# A raw cube is ~4.7 GB, so the queue is one item deep: the volume being
# processed already holds several arrays that size.
NUM_DECODE_WORKERS = 2
PREFETCH_FACTOR = 1


def write_outputs(result, video: str, config) -> None:
    """Raw mask + registration cache. Runs on the writer thread.

    The registration goes out through a registrator primed with it, so the
    files are written by the same code that reads them back.
    """
    finalize_raw_mask(result.raw_masks, ROOT_MASKS / video / "mask.npz")

    registrator = VideoRegistrator(
        video=Path(video),
        config=config,
        cache_dir=ROOT_REGISTERED_CACHE,
        verbose=False,
    )
    registrator.prime_from_result(
        result.registered_frames, result.registered_masks, result.transform
    )
    registrator.flush_cache()


def build_tasks(config, overwrite: bool, logger: logging.Logger) -> list[str]:
    df = load_measurements(include_HR=False)

    tasks, skipped, missing = [], 0, 0
    for _, row in df.iterrows():
        video = Path(str(row["MeasureValue"]).replace("\\", "/")).as_posix()
        if not source_path(video).exists():
            missing += 1
            continue
        done = (ROOT_MASKS / video / "mask.npz").exists() and cache_is_valid(
            ROOT_REGISTERED_CACHE, Path(video), config
        )
        if done and not overwrite and Path(video) not in PROCESS_ANYWAY:
            skipped += 1
            continue
        tasks.append(video)

    logger.info(
        f"Total: {len(df)} | Up to date: {skipped} | No source video: {missing} "
        f"| To process: {len(tasks)}"
    )
    return tasks


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-shards", type=int, default=1,
                   help="Split the cohort into N disjoint shards (one process per GPU).")
    p.add_argument("--shard", type=int, default=0,
                   help="Which shard this process handles, 0-based.")
    p.add_argument("--overwrite", action="store_true",
                   help="Redo volumes whose outputs already exist.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N volumes (smoke run).")
    p.add_argument("--batch-size", type=int, default=REGISTRATION.fused_batch_size,
                   help="Frames per fused forward pass.")
    return p.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.shard < args.num_shards:
        raise SystemExit(f"--shard must be in [0, {args.num_shards})")

    sharded = args.num_shards > 1
    rank_tag = f" [shard {args.shard}]" if sharded else ""
    log_name = f"segreg.shard{args.shard}.log" if sharded else "segreg.log"
    logger = setup_logging(
        "segment_and_register", ROOT_REGISTERED_CACHE / "logs" / log_name, rank_tag
    )

    config = REGISTRATION
    if config.method != "learned":
        raise SystemExit(
            "The fused stage runs the learned registrator; "
            f'REGISTRATION.method is "{config.method}".'
        )
    if args.batch_size != config.fused_batch_size:
        from dataclasses import replace

        config = replace(config, fused_batch_size=args.batch_size)

    logger.info(
        f"Fused segmentation+registration | shard {args.shard + 1}/{args.num_shards} "
        f"| batch {config.fused_batch_size} "
        f"| CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}"
    )

    tasks = build_tasks(config, args.overwrite, logger)
    # Shard *after* the skip check, so a shard is not handed everything left to
    # do whenever the remaining volumes cluster on one side of the split.
    tasks = tasks[args.shard :: args.num_shards]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    logger.info(f"This shard: {len(tasks)}")
    if not tasks:
        logger.info("Nothing to process.")
        return

    torch.backends.cudnn.benchmark = True

    from ocularrigidity.scripts.cohort_analysis.segment_n_cycles import get_model

    seg_model = get_model()
    logger.info(f"Segmentation checkpoint: {CHECKPOINT_PATH}")
    reg_model, reg_meta = load_registration_regressor(config.registrator_checkpoint)
    logger.info(
        f"Registration checkpoint: {config.registrator_checkpoint} "
        f"(epoch {reg_meta.get('epoch')}, val {reg_meta.get('val_loss')})"
    )

    loader = DataLoader(
        PrefetchDataset(tasks, partial(load_source_frames, config=config)),
        batch_size=1,
        num_workers=NUM_DECODE_WORKERS,
        shuffle=False,
        prefetch_factor=PREFETCH_FACTOR,
        collate_fn=identity_collate,
    )

    n_ok, n_fail = 0, 0
    pending = None  # in-flight write, one volume behind the GPU

    # One writer thread: it holds a whole volume's arrays until the encode is
    # done, and a deeper queue would just hold several.
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
                logger.error(f"FAILED (write): {name}\n  Error: {type(e).__name__}: {e}")
                n_fail += 1

        t_mark = time.perf_counter()
        for i, (video, frames, load_error) in enumerate(loader, start=1):
            # Time blocked on the loader. Near zero means the decode workers are
            # keeping up and the GPU is the wall.
            t_wait = time.perf_counter() - t_mark

            if load_error is not None:
                logger.error(f"FAILED (load): {video}\n{load_error}")
                n_fail += 1
                t_mark = time.perf_counter()
                continue

            n_frames = len(frames)
            try:
                result = segment_and_register(
                    frames, seg_model, reg_model, config, verbose=False
                )
            except Exception as e:
                logger.error(
                    f"FAILED (fused): {video}\n"
                    f"  Error: {type(e).__name__}: {e}\n"
                    f"  Traceback:\n{traceback.format_exc()}"
                )
                n_fail += 1
                # Drop the 4.7 GB cube now rather than at the next rebind, but
                # by rebinding rather than `del`: the name is read again below
                # on the success path.
                frames = None
                gc.collect()
                torch.cuda.empty_cache()
                t_mark = time.perf_counter()
                continue

            # Collect the previous write before queueing this one, so at most
            # one volume is in flight. A non-zero wait here means the encodes
            # have become the wall rather than the GPU.
            t_writer = time.perf_counter()
            drain(pending)
            t_writer = time.perf_counter() - t_writer

            pending = (writer.submit(write_outputs, result, video, config), video)

            stages = " ".join(f"{k} {v:.1f}s" for k, v in result.timings.items())
            logger.info(
                f"[{i}/{len(tasks)}] {video} | {n_frames} frames | "
                f"ref {result.ref_idx} (p{result.ref_percentile:.0f}) | "
                f"load-wait {t_wait:5.1f}s | {stages} | "
                f"writer-wait {t_writer:5.1f}s"
            )

            del frames, result
            gc.collect()
            torch.cuda.empty_cache()
            t_mark = time.perf_counter()

        drain(pending)

    logger.info("=" * 60)
    logger.info(f"Done. {n_ok} succeeded, {n_fail} failed.")


if __name__ == "__main__":
    main()
