"""Segment and register videos in one GPU pass per volume (pipeline stages 1 and 2).

Two video lists:

- ``--videos cohort`` (default): the cohort (``load_measurements``), written
  where the rest of the pipeline reads (``OCULARRIGIDITY_MASKS`` /
  ``OCULARRIGIDITY_REGISTERED_CACHE``).
- ``--videos plex``: every PLEX video of the measurement database, macular and
  ONH (``--kinds``), written under its own root (``--out``):

      <out>/masks/<video>/mask.npz                      raw masks (stage 1)
      <out>/registered_frames/<video>/cube.mp4          registration cache (stage 2)
      <out>/registered_masks/<video>/{mask,transform}.npz
                        transform.npz: dx (T,), dy (T, W), ref_idx, and iou (T,),
                        each registered mask's IoU with the reference's
      <out>/index/videos.shard<k>.csv                   one row per video written
      <out>/logs/segreg.shard<k>.log

The registration cache has the layout ``run_cohort.py --reuse-segreg`` reads.
Videos already done are skipped, so a stopped run resumes where it was.

    # Every PLEX video, one process per GPU
    CUDA_VISIBLE_DEVICES=0 python -m ocularrigidity.scripts.segmentation.segment_and_register \\
        --videos plex --num-shards 2 --shard 0 &
    CUDA_VISIBLE_DEVICES=1 python -m ocularrigidity.scripts.segmentation.segment_and_register \\
        --videos plex --num-shards 2 --shard 1 &
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import csv
import gc
import logging
import sqlite3
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import partial
from itertools import zip_longest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ocularrigidity.consts import (
    MEASUREMENTS_PATH,
    OUTPUT_ROOT,
    REGISTRATOR_REVISION,
    ROOT_MASKS,
    ROOT_REGISTERED_CACHE,
    SEGMENTATION_REVISION,
)
from ocularrigidity.data.measurements.dataframe import load_measurements
from ocularrigidity.pipeline_config import REGISTRATION
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
from ocularrigidity.segmentation.utils import (
    get_model,
    get_registration_model,
    write_raw_mask,
)

NUM_DECODE_WORKERS = 2
PREFETCH_FACTOR = 1


PLEX_KINDS = {
    "plex_macular": "Plex Macular Video",
    "plex_onh": "Plex ONH Video",
}
DEFAULT_PLEX_OUT = OUTPUT_ROOT / "SegRegPlexAll"

INDEX_COLUMNS = [
    "video", "kind", "patient", "n_frames", "ref_idx", "ref_percentile",
    "median_iou", "segmentation_revision", "registration_revision",
]


@dataclass(frozen=True)
class Outputs:
    """Where the raw masks and the registration cache go."""

    masks: Path
    cache: Path
    logs: Path
    index: Path | None = None  # per-video table, plex mode only

    def mask(self, video: str) -> Path:
        return self.masks / video / "mask.npz"


@dataclass(frozen=True)
class Video:
    video: str
    kind: str = "cohort"
    patient: str = ""


def write_outputs(result, video: str, config, outputs: Outputs) -> None:
    """Raw mask + registration cache."""
    write_raw_mask(result.raw_masks, outputs.mask(video))

    registrator = VideoRegistrator(
        video=Path(video),
        config=config,
        cache_dir=outputs.cache,
        verbose=False,
    )
    # Saved in transform.npz next to dx/dy, to filter outlier frames later
    transform = {**result.transform, "ref_idx": result.ref_idx, "iou": result.iou}
    registrator.prime_from_result(
        result.registered_frames, result.registered_masks, transform
    )
    registrator.flush_cache()


def cohort_videos() -> list[Video]:
    df = load_measurements(include_HR=False)
    return [
        Video(Path(str(v).replace("\\", "/")).as_posix())
        for v in df["MeasureValue"]
    ]


def plex_videos(kinds) -> list[Video]:
    """Every PLEX video of the database of these kinds, each video once."""
    names = {PLEX_KINDS[k].lower(): k for k in kinds}
    with sqlite3.connect(f"file:{MEASUREMENTS_PATH}?mode=ro", uri=True) as con:
        df = pd.read_sql_query(
            "SELECT PatientId, MeasureName, MeasureValue FROM measurements "
            f"WHERE lower(MeasureName) IN ({','.join('?' * len(names))})",
            con,
            params=list(names),
        )
    df = df.dropna(subset=["MeasureValue"])
    df = df[~df["MeasureValue"].str.startswith("\\\\Usereve")]
    df["video"] = (
        df["MeasureValue"].str.replace("\\", "/", regex=False).str.strip().str.strip("/")
    )
    df["kind"] = df["MeasureName"].str.lower().map(names)
    # ONH first: the order the kinds alternate in
    df = df.drop_duplicates("video").sort_values(["kind", "video"], ascending=[False, True])
    return [
        Video(r.video, r.kind, str(r.PatientId))
        for r in df.itertuples(index=False)
    ]


def build_tasks(
    videos: list[Video], shard: int, num_shards: int, config, outputs: Outputs,
    overwrite: bool, logger: logging.Logger,
) -> list[Video]:
    # Shard first, every existence check below being a stat on the share. Each
    # kind is sharded on its own, then the kinds alternate (ONH, macular, ONH,
    # ...), so the progress of both shows from the start.
    by_kind: dict[str, list[Video]] = {}
    for v in videos:
        by_kind.setdefault(v.kind, []).append(v)
    queues = [vs[shard::num_shards] for vs in by_kind.values()]
    mine = [v for group in zip_longest(*queues) for v in group if v is not None]
    tasks, skipped, missing = [], 0, 0
    for v in mine:
        done = outputs.mask(v.video).exists() and cache_is_valid(
            outputs.cache, Path(v.video), config
        )
        if done and not overwrite:
            skipped += 1
            continue
        if not source_path(v.video).exists():
            missing += 1
            continue
        tasks.append(v)

    logger.info(
        f"Total: {len(videos)} | This shard: {len(mine)} | Up to date: {skipped} "
        f"| No source video: {missing} | To process: {len(tasks)}"
    )
    return tasks


def append_index(path: Path, row: dict) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n", 1)[1],
    )
    p.add_argument("--videos", choices=("cohort", "plex"), default="cohort",
                   help="cohort: the measurement cohort; plex: every PLEX video in the database.")
    p.add_argument("--kinds", nargs="+", choices=list(PLEX_KINDS), default=list(PLEX_KINDS),
                   help="With --videos plex: which PLEX videos.")
    p.add_argument("--out", type=Path, default=None,
                   help=f"With --videos plex: output root (default {DEFAULT_PLEX_OUT}).")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Split the videos into N disjoint shards (one process per GPU).")
    p.add_argument("--shard", type=int, default=0,
                   help="Which shard this process handles, 0-based.")
    p.add_argument("--overwrite", action="store_true",
                   help="Redo volumes whose outputs already exist.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N volumes (smoke run).")
    p.add_argument("--dry-run", action="store_true",
                   help="List what would run and exit, without loading models.")
    p.add_argument("--batch-size", type=int, default=REGISTRATION.fused_batch_size,
                   help="Frames per fused forward pass.")
    return p.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.shard < args.num_shards:
        raise SystemExit(f"--shard must be in [0, {args.num_shards})")
    if args.videos == "cohort" and args.out is not None:
        raise SystemExit(
            "--out is for --videos plex; the cohort run writes where "
            "OCULARRIGIDITY_MASKS / OCULARRIGIDITY_REGISTERED_CACHE point."
        )

    if args.videos == "plex":
        root = args.out or DEFAULT_PLEX_OUT
        outputs = Outputs(
            masks=root / "masks", cache=root, logs=root / "logs", index=root / "index"
        )
    else:
        outputs = Outputs(
            masks=ROOT_MASKS, cache=ROOT_REGISTERED_CACHE,
            logs=ROOT_REGISTERED_CACHE / "logs",
        )

    sharded = args.num_shards > 1
    rank_tag = f" [shard {args.shard}]" if sharded else ""
    log_name = f"segreg.shard{args.shard}.log" if sharded else "segreg.log"
    logger = setup_logging("segment_and_register", outputs.logs / log_name, rank_tag)

    config = REGISTRATION
    if args.videos == "plex":
        # Columns are kept as they are; outlier frames are filtered later on the
        # saved per-frame IoU instead
        config = replace(config, filter_bad_columns=False)
    if config.method != "learned":
        raise SystemExit(
            "The fused stage runs the learned registrator; "
            f'REGISTRATION.method is "{config.method}".'
        )
    if args.batch_size != config.fused_batch_size:
        config = replace(config, fused_batch_size=args.batch_size)

    logger.info(
        f"Fused segmentation+registration | videos={args.videos}"
        + (f" ({' '.join(args.kinds)})" if args.videos == "plex" else "")
        + f" | shard {args.shard + 1}/{args.num_shards} | batch {config.fused_batch_size} "
        f"| CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}"
    )
    logger.info(f"masks -> {outputs.masks} | registration cache -> {outputs.cache}")

    videos = plex_videos(args.kinds) if args.videos == "plex" else cohort_videos()
    tasks = build_tasks(
        videos, args.shard, args.num_shards, config, outputs, args.overwrite, logger
    )
    if args.limit is not None:
        tasks = tasks[: args.limit]
        logger.info(f"--limit {args.limit}: processing {len(tasks)}")
    if args.videos == "plex":
        counts = pd.Series([t.kind for t in tasks]).value_counts().to_dict()
        logger.info(f"To process by kind: {counts}")
    if args.dry_run:
        logger.info("--dry-run: stopping before any work.")
        return
    if not tasks:
        logger.info("Nothing to process.")
        return

    torch.backends.cudnn.benchmark = True

    seg_model = get_model()
    reg_model = get_registration_model(config.registrator_revision).to("cuda")
    logger.info(
        f"Models from the Hub: segmentation {SEGMENTATION_REVISION}, "
        f"registration {config.registrator_revision}"
    )

    by_video = {t.video: t for t in tasks}
    loader = DataLoader(
        PrefetchDataset(
            [t.video for t in tasks], partial(load_source_frames, config=config)
        ),
        batch_size=1,
        num_workers=NUM_DECODE_WORKERS,
        shuffle=False,
        prefetch_factor=PREFETCH_FACTOR,
        collate_fn=identity_collate,
    )
    index_path = (
        outputs.index / f"videos.shard{args.shard}.csv" if outputs.index else None
    )

    n_ok, n_fail = 0, 0
    pending = None  # in-flight write, one volume behind the GPU

    with ThreadPoolExecutor(max_workers=1) as writer:
        def drain(job):
            nonlocal n_ok, n_fail
            if job is None:
                return
            future, name, row = job
            try:
                future.result()
                n_ok += 1
            except Exception as e:
                logger.error(f"FAILED (write): {name}\n  Error: {type(e).__name__}: {e}")
                n_fail += 1
                return
            # Only once the files are on disk: the index lists what can be read back
            if index_path is not None:
                append_index(index_path, row)

        t_mark = time.perf_counter()
        for i, (video, frames, load_error) in enumerate(loader, start=1):
            # Time blocked on the loader.
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
                frames = None
                gc.collect()
                torch.cuda.empty_cache()
                t_mark = time.perf_counter()
                continue

            task = by_video[video]
            row = {
                "video": video,
                "kind": task.kind,
                "patient": task.patient,
                "n_frames": n_frames,
                "ref_idx": result.ref_idx,
                "ref_percentile": result.ref_percentile,
                "median_iou": (
                    float(np.median(result.iou)) if result.iou is not None else None
                ),
                "segmentation_revision": SEGMENTATION_REVISION,
                "registration_revision": config.registrator_revision or REGISTRATOR_REVISION,
            }

            # Collect the previous write before queueing this one, so at
            # most one volume is in flight.
            t_writer = time.perf_counter()
            drain(pending)
            t_writer = time.perf_counter() - t_writer

            pending = (
                writer.submit(write_outputs, result, video, config, outputs), video, row
            )

            stages = " ".join(f"{k} {v:.1f}s" for k, v in result.timings.items())
            ref_p = (
                f" (p{result.ref_percentile:.0f})"
                if result.ref_percentile is not None else ""
            )
            logger.info(
                f"[{i}/{len(tasks)}] {task.kind} {video} | {n_frames} frames | "
                f"ref {result.ref_idx}{ref_p} | "
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
