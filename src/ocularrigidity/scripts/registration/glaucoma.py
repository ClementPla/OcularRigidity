"""Batch registration pipeline with preloading and RAM safety."""

import os


import argparse
import gc
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ocularrigidity.consts import (
    ROOT_REGISTERED_CACHE,
)
from ocularrigidity.data.measurements.dataframe import load_measurements
from ocularrigidity.pipeline_config import REGISTRATION
from ocularrigidity.registration.registration_engine import VideoRegistrator
from ocularrigidity.scripts._logging import setup_logging
from ocularrigidity.scripts.dataset import VolumeDataset, identity_collate
from ocularrigidity.scripts.exceptions_videos import PROCESS_ANYWAY

OUTPUT_FOLDER = ROOT_REGISTERED_CACHE

NUM_DECODE_WORKERS = 2
PREFETCH_FACTOR = 1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split cohort into N disjoint shards (one process per GPU).",
    )
    p.add_argument(
        "--shard",
        type=int,
        default=0,
        help="Which shard this process handles (0-based).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Force overwrite existing registered caches.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.shard < args.num_shards:
        raise SystemExit(f"--shard must be in [0, {args.num_shards})")

    sharded = args.num_shards > 1
    rank_tag = f" [shard {args.shard}]" if sharded else ""
    log_name = f"registration.shard{args.shard}.log" if sharded else "registration.log"
    logger = setup_logging(
        "batch_registration", OUTPUT_FOLDER / "logs" / log_name, rank_tag
    )

    logger.info(
        f"Starting batch registration | shard {args.shard + 1}/{args.num_shards} "
        f"| CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}"
    )

    df = load_measurements()

    tasks = []
    skipped = 0
    for _, row in df.iterrows():
        measure_value = str(Path(row["MeasureValue"])).replace("\\", "/")
        is_exception = Path(measure_value) in PROCESS_ANYWAY

        # Instantiate dummy registrator spec to check true output cache paths
        dummy_reg = VideoRegistrator(
            video=measure_value,
            config=REGISTRATION,
            frames=None,
            masks=None,
            cache_dir=OUTPUT_FOLDER,
            overwrite_cache=args.overwrite or is_exception,
            verbose=False,
        )
        cache_paths = dummy_reg._cache_paths()
        cache_exists = all(p.exists() for p in cache_paths.values())

        if cache_exists and not args.overwrite and not is_exception:
            skipped += 1
        else:
            tasks.append(measure_value)

    # Shard task list for multi-GPU runs
    remaining = len(tasks)
    tasks = tasks[args.shard :: args.num_shards]

    logger.info(
        f"Total: {len(df)} | Skipped: {skipped} | Remaining: {remaining} "
        f"| This shard: {len(tasks)}"
    )

    if not tasks:
        logger.info("Nothing to process.")
        return

    dataset = VolumeDataset(tasks, return_masks=True)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=NUM_DECODE_WORKERS,
        shuffle=False,
        prefetch_factor=PREFETCH_FACTOR,
        collate_fn=identity_collate,
    )

    n_ok, n_fail = 0, 0
    pending = None  # in-flight cache write, one volume behind the registration

    # One writer thread: a deeper queue would just hold more ~9.4 GB
    # registrations in RAM waiting to be encoded.
    with ThreadPoolExecutor(max_workers=1) as writer:

        def drain(job):
            """Collect the previous volume's cache write."""
            nonlocal n_ok, n_fail
            if job is None:
                return
            future, name = job
            try:
                future.result()
                n_ok += 1
            except Exception as e:
                logger.error(
                    f"FAILED (cache write): {name}\n  Error: {type(e).__name__}: {e}"
                )
                n_fail += 1

        t_mark = time.perf_counter()

        for i, (measure_value, payload, load_error) in enumerate(
            tqdm(loader, desc="Processing", total=len(tasks)), start=1
        ):
            # Time blocked on the loader. Near zero means the decode worker is
            # keeping up and registration is the wall.
            t_wait = time.perf_counter() - t_mark

            frames = masks = registrator = None
            try:
                # Safe error check BEFORE tuple unpacking
                if load_error is not None:
                    logger.error(f"FAILED (load): {measure_value}\n{load_error}")
                    n_fail += 1
                    continue

                frames, masks = payload
                t_reg = time.perf_counter()

                try:
                    registrator = VideoRegistrator(
                        video=measure_value,
                        config=REGISTRATION,
                        frames=frames,
                        masks=masks,
                        cache_dir=OUTPUT_FOLDER,
                        overwrite_cache=args.overwrite
                        or (Path(measure_value) in PROCESS_ANYWAY),
                        verbose=True,
                    )

                    # The write is handed to the writer thread below.
                    registrator.compute_registration(save_cache=False)
                    t_reg = time.perf_counter() - t_reg

                except Exception as e:
                    logger.error(
                        f"FAILED (registration): {measure_value}\n"
                        f"  Error: {type(e).__name__}: {e}\n"
                        f"  Traceback:\n{traceback.format_exc()}"
                    )
                    n_fail += 1
                    continue

                # Collect the previous write before queueing this one, so at
                # most one registration is in flight. A non-zero wait here means
                # the encode has become the wall rather than the registration.
                t_writer = time.perf_counter()
                drain(pending)
                t_writer = time.perf_counter() - t_writer

                # `registrator` stays referenced by the job until the write is
                # collected — it owns the arrays being encoded.
                pending = (writer.submit(registrator.flush_cache), measure_value)

                # One line per volume: without it the stage prints nothing for
                # minutes at a time and looks hung.
                logger.info(
                    f"[{i}/{len(tasks)}] {measure_value} | "
                    f"load-wait {t_wait:5.1f}s | reg {t_reg:5.1f}s | "
                    f"writer-wait {t_writer:5.1f}s"
                )
            finally:
                # One release point for every path out of the iteration. These
                # are the ~4.7 GB volume arrays: drop this iteration's
                # references before the loader hands over the next volume, or
                # two are resident at once. Rebinding rather than `del` leaves
                # the names defined on the paths that never bound them.
                frames = masks = payload = registrator = None
                gc.collect()
                torch.cuda.empty_cache()
                t_mark = time.perf_counter()

        drain(pending)

    logger.info("=" * 60)
    logger.info(f"Done. {n_ok} succeeded, {n_fail} failed, {skipped} skipped.")


if __name__ == "__main__":
    main()
