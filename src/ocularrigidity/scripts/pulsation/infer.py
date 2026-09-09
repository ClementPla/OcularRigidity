"""Fold every cohort video into ``N_CYCLES`` cardiac cycles, then measure deltaY.

Two stages, pipelined the same way as the registration pass
(scripts/registration/glaucoma.py), because the per-video costs have the same
shape — a big decode (CPU), the actual work, then an encode + write (CPU):

* stage 1 folds each video. Its input is the *registered* cache written by the
  registration stage, and decoding that (~25 s of mp4 + zstd per volume) runs in
  a DataLoader worker ahead of the fold; the decoded arrays are handed to the
  registrator with ``prime_from_cache``, so the pipeline sees exactly what it
  would have loaded itself. The lossless one_cycle.mkv encode and the measure
  pickle then go to a writer thread, so the next fold starts immediately.
* stage 2 segments each folded clip and fits the cardiac amplitude. The clip
  decode runs in workers, the GPU pass on the main thread, and the deltaY table
  is checkpointed on the writer thread instead of being re-read and re-written
  once per video.

Single GPU: no ``--num-shards`` here. The fold is dominated by CPU work
(Lomb-Scargle, IQ demodulation, the FFV1 encode), so a second card would sit
idle; more cores is what this stage wants.

Resumable: videos whose outputs exist are skipped unless ``--overwrite``, and a
video that fails is logged and stepped over rather than aborting the cohort.
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
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ocularrigidity.consts import (
    ROOT_CARDIAC_PIPELINE,
    ROOT_DATA_MNT,
    ROOT_MASKS,
    ROOT_REGISTERED_CACHE,
)
from ocularrigidity.data.compression import cube_to_mkv_lossless, read_gray
from ocularrigidity.data.measurements.dataframe import load_measurements
from ocularrigidity.motion.one_cycle import estimate_cardiac_amplitude
from ocularrigidity.motion.pipeline_results import CardiacPipelineResults
from ocularrigidity.motion.pulsation.pipeline import run_composed_pipeline
from ocularrigidity.pipeline_config import DELTA_Y, PULSATION, REGISTRATION
from ocularrigidity.registration.config import RegistrationConfig
from ocularrigidity.registration.registration_engine import (
    VideoRegistrator,
    cache_is_valid,
    read_cache_payload,
)
from ocularrigidity.scripts._logging import setup_logging
from ocularrigidity.scripts.cohort_analysis.segment_n_cycles import get_model
from ocularrigidity.scripts.dataset import PrefetchDataset, identity_collate
from ocularrigidity.scripts.exceptions_videos import PROCESS_ANYWAY
from ocularrigidity.segmentation.inference import infer
from ocularrigidity.thickness.features import compute_deltaY_masks

# A registered volume is ~9.4 GB of frames + masks, so the queue depth is
# bounded deliberately: each extra prefetched item is another 9.4 GB of RAM
# held next to the one being folded.
#
# Two workers, not one: measured on this cohort the cache decode is ~39 s per
# volume against ~24 s for the fold itself, so a single worker leaves the stage
# decode-bound and the fold idle a third of the time. (The one-worker setting
# dated from when the fold was minutes of CPU, which the composed chain no
# longer is.) Drop back to 1 if RAM is contended.
NUM_DECODE_WORKERS = 2
PREFETCH_FACTOR = 1

# The folded clips are ~90 frames (~140 MB), so stage 2 can afford a deeper
# queue than stage 1.
DELTAY_DECODE_WORKERS = 2
DELTAY_PREFETCH_FACTOR = 2

# How often the deltaY table is flushed to disk. Rewriting it once per video is
# what the previous version did and it is quadratic in cohort size; losing at
# most this many videos to a crash is the trade.
CHECKPOINT_EVERY = 10


# --- Stage 1: fold ---------------------------------------------------------


@dataclass(frozen=True)
class FoldTask:
    """One video to fold, plus where its two outputs go."""

    video: str  # cohort-relative id, also the registration cache key
    expected_bpm: Optional[float]
    one_cycle_path: Path
    measure_path: Path


def load_registration(
    task: FoldTask, cache_dir: Path, config: RegistrationConfig
) -> Optional[dict]:
    """Decode this video's registration cache. Runs in a DataLoader worker.

    Returns None on a cache miss (or a cache written with different
    registration parameters); the fold then registers the video itself, in the
    main process, exactly as it did before.
    """
    if cache_dir is None:
        return None
    payload = read_cache_payload(cache_dir, Path(task.video), config)
    if payload is None:
        return None
    # Tensors, so the ~9.4 GB travels through shared memory rather than being
    # pickled down the worker socket.
    payload["frames"] = torch.from_numpy(np.ascontiguousarray(payload["frames"]))
    payload["masks"] = torch.from_numpy(np.ascontiguousarray(payload["masks"]))
    return payload


def write_fold_outputs(
    result: CardiacPipelineResults,
    one_cycle_path: Path,
    measure_path: Path,
    fps: int,
) -> None:
    """Lossless cycle video + measures pickle. Runs on a background thread."""
    one_cycle_path.parent.mkdir(parents=True, exist_ok=True)
    cube_to_mkv_lossless(result.cycles, str(one_cycle_path), fps=fps)
    measure_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(measure_path, include_cycles=False)


def build_fold_tasks(
    root_one_cycle: Path, root_measures: Path, overwrite: bool, cache_dir: Path
) -> tuple[list[FoldTask], int, int]:
    """The videos still to fold, plus the skipped and no-registration counts.

    The fold's input is the *registration cache*, not the source video, so that
    is what it is gated on. Under ``method="learned"`` a cache miss is fatal
    rather than merely slow — the registrator cannot fall back to registering
    the volume itself — so a video without one has to be excluded here rather
    than discovered one exception at a time.
    """
    df = load_measurements(include_HR=True)

    tasks, skipped, missing = [], 0, 0
    for _, row in df.iterrows():
        video = Path(row["MeasureValue"]).as_posix().replace("\\", "/")
        process_anyway = Path(video) in PROCESS_ANYWAY

        if not cache_is_valid(cache_dir, Path(video), REGISTRATION):
            missing += 1
            continue

        one_cycle_path = root_one_cycle / video / "one_cycle.mkv"
        measure_path = root_measures / video / "measure.pkl"
        done = one_cycle_path.exists() or measure_path.exists()
        if done and not overwrite and not process_anyway:
            skipped += 1
            continue

        HR = row["HR"]
        if HR is None or np.isnan(HR) or HR <= 0:
            HR = None

        tasks.append(
            FoldTask(
                video=video,
                expected_bpm=HR,
                one_cycle_path=one_cycle_path,
                measure_path=measure_path,
            )
        )
    return tasks, skipped, missing


def compute_one_cycle(
    root_one_cycle: Path,
    root_measures: Path,
    logger: logging.Logger,
    cache_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> None:
    """Fold every cohort video into ``N_CYCLES`` cardiac cycles.

    The recipe is the composed chain pinned in ``PULSATION.chain`` — the one
    validated in notebooks/pipeline/test.ipynb. It does not sweep method/phase,
    so the outputs are plain ``one_cycle`` / ``measures``: which decomposition
    and which demodulation produced them are fields of the stage configs saved
    in every ``measure.pkl``, not something a directory name can record.
    """
    tasks, skipped, missing = build_fold_tasks(
        root_one_cycle, root_measures, overwrite, cache_dir
    )
    logger.info(
        f"Fold -> {root_one_cycle.name} | Skipped: {skipped} | "
        f"No registration: {missing} | Remaining: {len(tasks)}"
    )
    if not tasks:
        logger.info("Nothing to fold.")
        return

    loader = DataLoader(
        PrefetchDataset(
            tasks, partial(load_registration, cache_dir=cache_dir, config=REGISTRATION)
        ),
        batch_size=1,
        num_workers=NUM_DECODE_WORKERS,
        shuffle=False,
        prefetch_factor=PREFETCH_FACTOR,
        collate_fn=identity_collate,
    )

    n_ok, n_fail = 0, 0
    pending = None  # in-flight write, one video behind the fold

    # One writer thread: a deeper queue would just hold more folded cycles in
    # RAM waiting on the FFV1 encode.
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
        for i, (task, payload, load_error) in enumerate(loader, start=1):
            # Time blocked on the loader. Near zero means the decode worker is
            # keeping up and the fold is the wall.
            t_wait = time.perf_counter() - t_mark

            registrator = result = None
            try:
                if load_error is not None:
                    logger.error(f"FAILED (load): {task.video}\n{load_error}")
                    n_fail += 1
                    continue

                t_fold = time.perf_counter()
                try:
                    registrator = VideoRegistrator(
                        video=Path(task.video),
                        root_data=ROOT_DATA_MNT,
                        root_masks=ROOT_MASKS,
                        config=REGISTRATION,
                        cache_dir=cache_dir,
                        verbose=True,
                    )
                    if payload is not None:
                        registrator.prime_from_cache(payload)

                    stage_configs, fold_config = PULSATION.chain_for_video(
                        expected_bpm=task.expected_bpm
                    )
                    result: CardiacPipelineResults = run_composed_pipeline(
                        video_relpath=task.video,
                        root_masks=ROOT_MASKS,
                        root_data=ROOT_DATA_MNT,
                        timestamps_path=ROOT_DATA_MNT / task.video / "timestamp.txt",
                        stage_configs=stage_configs,
                        fold_config=fold_config,
                        registration_config=REGISTRATION,
                        compute_n_cycle_video=True,
                        cache_dir=cache_dir,
                        registrator=registrator,
                    )
                    t_fold = time.perf_counter() - t_fold
                except Exception as e:
                    logger.error(
                        f"FAILED (fold): {task.video}\n"
                        f"  Error: {type(e).__name__}: {e}\n"
                        f"  Traceback:\n{traceback.format_exc()}"
                    )
                    n_fail += 1
                    continue

                # Collect the previous write before queueing this one, so at
                # most one folded result is in flight. A non-zero wait here
                # means the encode has become the wall rather than the fold.
                t_writer = time.perf_counter()
                drain(pending)
                t_writer = time.perf_counter() - t_writer

                pending = (
                    writer.submit(
                        write_fold_outputs,
                        result,
                        task.one_cycle_path,
                        task.measure_path,
                        PULSATION.output_fps,
                    ),
                    task.video,
                )

                # One line per video: without it the stage prints nothing for
                # minutes at a time and looks hung.
                logger.info(
                    f"[{i}/{len(tasks)}] {task.video} | "
                    f"{'cache' if payload is not None else 'no-cache'} | "
                    f"load-wait {t_wait:5.1f}s | fold {t_fold:5.1f}s | "
                    f"writer-wait {t_writer:5.1f}s"
                )
            finally:
                # One release point for every path out of the iteration. The
                # decoded volume is several GB: drop this iteration's references
                # before the loader hands over the next one. `result` stays
                # referenced by the writer job until drain() collects it; the
                # registered volume behind it does not have to. Rebinding rather
                # than `del` leaves the names defined on the paths that never
                # bound them.
                payload = registrator = result = None
                gc.collect()
                torch.cuda.empty_cache()
                t_mark = time.perf_counter()

        drain(pending)

    logger.info(f"Fold done. {n_ok} succeeded, {n_fail} failed, {skipped} skipped.")


# --- Stage 2: deltaY on the folded cycles ----------------------------------


def load_one_cycle(video: str, root: Path) -> torch.Tensor:
    """Decode one folded clip. Runs in a DataLoader worker."""
    data = read_gray(root / video / "one_cycle.mkv")
    return torch.from_numpy(np.ascontiguousarray(data))


def write_table(table: pd.DataFrame, output_file: Path) -> None:
    """Checkpoint the deltaY table. Runs on a background thread."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_file.with_name(output_file.name + ".tmp")
    table.to_pickle(tmp_path)
    tmp_path.replace(output_file)


def measure_deltaY(masks: np.ndarray, video: str, n_cycles: int) -> list[dict]:
    """Per-cycle cardiac amplitude of one segmented clip."""
    thickness = compute_deltaY_masks(masks)
    T = thickness.shape[0]

    rows = []
    for cycle in range(n_cycles):
        current_cycle = thickness[cycle * T // n_cycles : (cycle + 1) * T // n_cycles]
        fits, _ = estimate_cardiac_amplitude(
            current_cycle,
            n_harmonics=DELTA_Y.n_harmonics,
            residual_threshold_percentile=DELTA_Y.residual_threshold_percentile,
            amplitude_threshold_percentile=DELTA_Y.amplitude_threshold_percentile,
        )
        amplitude = fits.max(axis=0) - fits.min(axis=0)
        rows.append(
            {
                "video": video,
                "cycle": cycle,
                "deltaY": float(np.mean(amplitude)),
                "Amplitudes": amplitude,
                "Fits": fits,
            }
        )
    return rows


def extract_deltaY_from_one_cycle(
    output_file: Path,
    input_one_cycle: Path,
    logger: logging.Logger,
    overwrite: bool = False,
    n_cycles: int = DELTA_Y.n_cycles,
) -> None:
    """Segment each folded clip and fit its per-cycle cardiac amplitude."""
    df = load_measurements(include_HR=True)

    columns = ["video", "deltaY", "Amplitudes", "Fits", "cycle"]
    if output_file.exists():
        table = pd.read_pickle(output_file)
    else:
        table = pd.DataFrame(columns=columns)

    tasks, skipped, missing = [], 0, 0
    done = set(table["video"].values)
    for _, row in df.iterrows():
        video = Path(row["MeasureValue"]).as_posix().replace("\\", "/")
        if not (input_one_cycle / video / "one_cycle.mkv").exists():
            missing += 1
            continue
        if video in done and not overwrite and Path(video) not in PROCESS_ANYWAY:
            skipped += 1
            continue
        tasks.append(video)

    logger.info(
        f"deltaY -> {output_file.name} | Skipped: {skipped} | "
        f"No folded clip: {missing} | Remaining: {len(tasks)}"
    )
    if not tasks:
        logger.info("Nothing to measure.")
        return

    model = get_model()
    loader = DataLoader(
        PrefetchDataset(tasks, partial(load_one_cycle, root=input_one_cycle)),
        batch_size=1,
        num_workers=DELTAY_DECODE_WORKERS,
        shuffle=False,
        prefetch_factor=DELTAY_PREFETCH_FACTOR,
        collate_fn=identity_collate,
    )

    n_ok, n_fail = 0, 0
    rows: list[dict] = []
    pending = None  # in-flight checkpoint write

    with ThreadPoolExecutor(max_workers=1) as writer:

        def drain(job):
            if job is None:
                return
            try:
                job.result()
            except Exception as e:
                logger.error(f"FAILED (checkpoint write): {type(e).__name__}: {e}")

        def snapshot() -> pd.DataFrame:
            """The table as it stands, built on the main thread.

            Only the videos actually re-measured lose their previous rows — a
            video that was re-run and then failed keeps the ones it had, which
            is why the drop happens here rather than up front.
            """
            recomputed = {r["video"] for r in rows}
            kept = table[~table["video"].isin(recomputed)]
            return pd.concat(
                [kept, pd.DataFrame(rows, columns=columns)], ignore_index=True
            )

        t_mark = time.perf_counter()
        for i, (video, data, load_error) in enumerate(loader, start=1):
            t_wait = time.perf_counter() - t_mark

            if load_error is not None:
                logger.error(f"FAILED (load): {video}\n{load_error}")
                n_fail += 1
                t_mark = time.perf_counter()
                continue

            t_gpu = time.perf_counter()
            try:
                masks = infer(
                    model,
                    data,
                    batch_size=DELTA_Y.batch_size,
                    scale_factor=(1.0, 1.0),
                    return_logit=False,
                    use_graphcut=True,
                    use_amp=True,
                    verbose=True,
                    graphcut_kwargs=DELTA_Y.graphcut_kwargs,
                )
                t_gpu = time.perf_counter() - t_gpu
                rows.extend(measure_deltaY(masks, video, n_cycles))
                n_ok += 1
            except Exception as e:
                logger.error(
                    f"FAILED (deltaY): {video}\n"
                    f"  Error: {type(e).__name__}: {e}\n"
                    f"  Traceback:\n{traceback.format_exc()}"
                )
                n_fail += 1
                t_mark = time.perf_counter()
                continue
            finally:
                del data

            logger.info(
                f"[{i}/{len(tasks)}] {video} | load-wait {t_wait:5.1f}s | "
                f"gpu {t_gpu:5.1f}s"
            )

            if i % CHECKPOINT_EVERY == 0 and rows:
                drain(pending)
                pending = writer.submit(write_table, snapshot(), output_file)

            t_mark = time.perf_counter()

        drain(pending)
        if rows:
            write_table(snapshot(), output_file)

    logger.info(f"deltaY done. {n_ok} succeeded, {n_fail} failed, {skipped} skipped.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-fold and re-measure videos whose outputs already exist.",
    )
    p.add_argument(
        "--stage",
        choices=("all", "fold", "deltaY"),
        default="all",
        help="Run only one of the two stages.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(
        "pulsation", ROOT_CARDIAC_PIPELINE / "logs" / "pulsation.log"
    )
    logger.info(
        f"Starting pulsation | stage={args.stage} | overwrite={args.overwrite} "
        f"| CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}"
    )

    root_one_cycle = ROOT_CARDIAC_PIPELINE / "one_cycle"
    root_measures = ROOT_CARDIAC_PIPELINE / "measures"

    if args.stage in ("all", "fold"):
        compute_one_cycle(
            root_one_cycle,
            root_measures,
            logger,
            cache_dir=ROOT_REGISTERED_CACHE,
            overwrite=args.overwrite,
        )
    if args.stage in ("all", "deltaY"):
        extract_deltaY_from_one_cycle(
            ROOT_CARDIAC_PIPELINE / "deltaY.pkl",
            root_one_cycle,
            logger,
            overwrite=args.overwrite,
        )

    logger.info("=" * 60)
    logger.info("Pulsation stage complete.")


if __name__ == "__main__":
    main()
