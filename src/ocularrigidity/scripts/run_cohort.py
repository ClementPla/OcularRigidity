import os

# MUST be set at the very top before importing numpy, cv2, or torch
# to prevent C++ OpenMP background thread locks.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import gc
import logging
import pickle
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ocularrigidity.consts import (
    ROOT_CARDIAC_PIPELINE,
    ROOT_DATA_MNT,
    ROOT_MASKS,
    ROOT_REGISTERED_CACHE,
    SEGMENTATION_BATCH_SIZE,
    SEGMENTATION_REVISION,
)
from ocularrigidity.data.compression import read_gray
from ocularrigidity.data.io import load_mask
from ocularrigidity.data.measurements.dataframe import load_measurements
from ocularrigidity.motion.pipeline_results import (
    CardiacPipelineResults,
    write_fold_outputs,
)
from ocularrigidity.motion.pulsation.pipeline import run_composed_pipeline
from ocularrigidity.pipeline_config import (
    DELTA_Y,
    N_CYCLES,
    PULSATION,
    REGISTRATION,
)
from ocularrigidity.registration.fused import segment_and_register
from ocularrigidity.registration.registration_engine import (
    VideoRegistrator,
    cache_is_valid,
    read_cache_payload,
)
from ocularrigidity.scripts._logging import setup_logging
from ocularrigidity.motion.displacement import extract_displacement
from ocularrigidity.registration.qc import compute_qc_metrics, evaluate_flag
from ocularrigidity.segmentation.postprocess.temporal_smoothing import (
    postprocess_cycles,
)
from ocularrigidity.segmentation.utils import (
    get_model,
    get_registration_model,
    write_cycle_mask,
    write_raw_mask,
)
from ocularrigidity.scripts.dataset import (
    PrefetchDataset,
    identity_collate,
    load_source_frames,
    source_path,
)
from ocularrigidity.segmentation.inference import infer
from ocularrigidity.segmentation.postprocess.blob import (
    keep_largest_connected_component,
)
from ocularrigidity.thickness.features import measure_deltaY

# In pipeline order. Everything downstream of a stage that runs, runs too.
STAGES = ("segment", "register", "fold", "cycles", "deltaY", "deltaA", "qc")


NUM_DECODE_WORKERS = 4
PREFETCH_FACTOR = 1

# How often the two cohort-level tables (deltaY, QC) are flushed.
CHECKPOINT_EVERY = 10


# --- Where everything goes --------------------------------------------------


@dataclass(frozen=True)
class CohortPaths:
    """Every output location for this run.
    """

    @property
    def root_one_cycle(self) -> Path:
        return ROOT_CARDIAC_PIPELINE / "one_cycle"

    @property
    def root_measures(self) -> Path:
        return ROOT_CARDIAC_PIPELINE / "measures"

    @property
    def deltaY_table(self) -> Path:
        return ROOT_CARDIAC_PIPELINE / "deltaY.pkl"

    @property
    def qc_table(self) -> Path:
        return ROOT_CARDIAC_PIPELINE / "misregistration_flags.csv"

    def source(self, video: str) -> Path:
        return source_path(video)

    def raw_mask(self, video: str) -> Path:
        return ROOT_MASKS / video / "mask.npz"

    def one_cycle(self, video: str) -> Path:
        return self.root_one_cycle / video / "one_cycle.mkv"

    def measure(self, video: str) -> Path:
        return self.root_measures / video / "measure.pkl"

    def cycle_mask(self, video: str) -> Path:
        return self.root_measures / video / "segmented_cycles.npz"

    def deltaA(self, video: str) -> Path:
        return self.root_measures / video / "deltaA_per_cycle.pkl"


@dataclass(frozen=True)
class VideoTask:
    """One video and the stages it still needs, in pipeline order."""

    video: str
    expected_bpm: Optional[float]
    stages: tuple[str, ...]
    paths: CohortPaths

    @property
    def start(self) -> str:
        return self.stages[0]


def plan_video(
    video: str,
    paths: CohortPaths,
    force_from: Optional[str],
    done_deltaY: set,
    done_qc: set,
    reuse_segreg: bool = False,
) -> tuple[str, ...]:
    """The stages this video still needs. Empty means there is nothing to do.
    Only file existence (plus the registration cache's config check) is
    consulted here.
    """
    first = STAGES.index("fold") if reuse_segreg else 0

    present = {
        "segment": paths.raw_mask(video).exists(),
        "register": cache_is_valid(ROOT_REGISTERED_CACHE, Path(video), REGISTRATION),
        "fold": paths.one_cycle(video).exists() and paths.measure(video).exists(),
        "cycles": paths.cycle_mask(video).exists(),
        "deltaY": video in done_deltaY,
        "deltaA": paths.deltaA(video).exists(),
        "qc": video in done_qc,
    }
    forced = STAGES.index(force_from) if force_from else len(STAGES)
    forced = max(forced, first)

    for i, stage in enumerate(STAGES):
        if i < first:
            continue
        if i >= forced or not present[stage]:
            return STAGES[i:]
    return ()


def build_tasks(
    paths: CohortPaths,
    force_from: Optional[str],
    limit: Optional[int],
    logger: logging.Logger,
    reuse_segreg: bool = False,
) -> list[VideoTask]:
    df = load_measurements(include_HR=True)

    done_deltaY = set()
    if paths.deltaY_table.exists():
        done_deltaY = set(pd.read_pickle(paths.deltaY_table)["video"].values)
    done_qc = set()
    if paths.qc_table.exists():
        done_qc = set(pd.read_csv(paths.qc_table)["video"].astype(str).values)

    tasks, skipped, missing = [], 0, 0
    starts = Counter()
    for _, row in df.iterrows():
        video = Path(row["MeasureValue"]).as_posix().replace("\\", "/")
        if reuse_segreg and not cache_is_valid(
            ROOT_REGISTERED_CACHE, Path(video), REGISTRATION
        ):
            missing += 1
            continue
        stages = plan_video(
            video, paths, force_from, done_deltaY, done_qc, reuse_segreg
        )
        if not stages:
            skipped += 1
            continue
        # The raw cube is only needed by the first two stages; a video resuming
        # further down the chain does not care that its source is offline.
        if stages[0] in ("segment", "register") and not paths.source(video).exists():
            missing += 1
            continue

        HR = row["HR"]
        if pd.isna(HR) or HR <= 0:
            HR = None

        starts[stages[0]] += 1
        tasks.append(
            VideoTask(video=video, expected_bpm=HR, stages=stages, paths=paths)
        )

    logger.info(
        f"Total: {len(df)} | Up to date: {skipped} | Missing input: {missing} "
        f"| To process: {len(tasks)}"
    )
    logger.info(
        "Starting stage: "
        + (", ".join(f"{s}={starts[s]}" for s in STAGES if starts[s]) or "none")
    )

    if limit is not None:
        tasks = tasks[:limit]
        logger.info(f"--limit {limit}: processing the first {len(tasks)}")
    return tasks


def load_inputs(task: VideoTask) -> dict:
    """Load exactly what this video's first stage needs. Runs in a worker.

    """
    todo, paths, video = set(task.stages), task.paths, task.video
    inputs = {}

    if todo & {"segment", "register"}: # Only load the raw input if segment or register is needed
        inputs["frames"] = load_source_frames(video, REGISTRATION)

    if "fold" in todo and "register" not in todo:
        payload = read_cache_payload(ROOT_REGISTERED_CACHE, Path(video), REGISTRATION)
        if payload is None:
            raise FileNotFoundError(f"No usable registration cache for {video}")
        payload["frames"] = torch.from_numpy(np.ascontiguousarray(payload["frames"]))
        payload["masks"] = torch.from_numpy(np.ascontiguousarray(payload["masks"]))
        inputs["registration"] = payload

    if todo & {"cycles", "deltaY", "deltaA"} and "fold" not in todo:
        cycles = read_gray(paths.one_cycle(video))
        inputs["cycles"] = torch.from_numpy(np.ascontiguousarray(cycles))
        with open(paths.measure(video), "rb") as f:
            measures = pickle.load(f)
        inputs["cardiac_freq"] = float(measures.cardiac_freq)
        inputs["n_cycle"] = int(measures.n_cycle)

    if todo & {"deltaA", "qc"} and "cycles" not in todo:
        inputs["cycle_masks"] = load_mask(paths.cycle_mask(video))

    return inputs


def run_jobs(jobs: list[tuple[str, Callable[[], None]]]) -> None:
    """Run one video's writes in order, labelling whichever one fails."""
    for label, job in jobs:
        try:
            job()
        except Exception as e:
            raise RuntimeError(f"{label}: {type(e).__name__}: {e}") from e


def compute_and_write_deltaA(cycles: np.ndarray, masks: np.ndarray, out: Path) -> None:
    """Boundary displacement / area change. CPU only, so it rides with the writes."""
    deltaA, minA, displacement, reference = extract_displacement(
        video=cycles, mask=masks, N_cycles=N_CYCLES
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(
            {
                "deltaA_per_cycle": deltaA,
                "minA_per_cycle": minA,
                "displacement_per_cycle": displacement,
                "reference_coordinates_per_cycle": reference,
            },
            f,
        )


@dataclass
class VideoOutcome:
    """What one video produced: writes to queue, table rows, and stage timings.
    """

    jobs: list = field(default_factory=list)
    deltaY_rows: list = field(default_factory=list)
    qc_row: Optional[dict] = None
    timings: dict = field(default_factory=dict)


def process_video(
    task: VideoTask, inputs: dict, model, reg_model, batch_size: int
) -> VideoOutcome:
    """Carry one video through its stages, handing every write back to the caller.
    """
    out = VideoOutcome()
    todo, paths, video = set(task.stages), task.paths, task.video

    def timed(stage, fn):
        t = time.perf_counter()
        value = fn()
        out.timings[stage] = time.perf_counter() - t
        return value

    # 1+2. Segment and register in one pass ----------------------------------
    registrator = None
    if todo & {"segment", "register"}:
        result = timed(
            "segreg",
            lambda: segment_and_register(
                inputs["frames"], model, reg_model, REGISTRATION, verbose=False
            ),
        )
        out.jobs.append(
            (
                "mask.npz",
                partial(write_raw_mask, result.raw_masks, paths.raw_mask(video)),
            )
        )
        registrator = VideoRegistrator(
            video=video,
            root_data=ROOT_DATA_MNT,
            root_masks=ROOT_MASKS,
            config=REGISTRATION,
            cache_dir=ROOT_REGISTERED_CACHE,
            verbose=False,
        )

        registrator.prime_from_result(
            result.registered_frames, result.registered_masks, result.transform
        )
        out.jobs.append(("registration cache", registrator.flush_cache))
        out.timings["ref_percentile"] = result.ref_percentile
        del result
    elif "fold" in todo:
        registrator = VideoRegistrator(
            video=video,
            root_data=ROOT_DATA_MNT,
            root_masks=ROOT_MASKS,
            config=REGISTRATION,
            cache_dir=ROOT_REGISTERED_CACHE,
            verbose=False,
        )
        registrator.prime_from_cache(inputs["registration"])

    # The raw volume is dead once registration is done; the mask lives on in the
    # queued write, the registered arrays in the registrator.
    inputs.pop("frames", None)
    inputs.pop("registration", None)

    # 3. Fold into N_CYCLES cardiac cycles -----------------------------------
    if "fold" in todo:
        stage_configs, fold_config = PULSATION.chain_for_video(
            expected_bpm=task.expected_bpm, verbose=False
        )
        result: CardiacPipelineResults = timed(
            "fold",
            lambda: run_composed_pipeline(
                video_relpath=video,
                root_masks=ROOT_MASKS,
                root_data=ROOT_DATA_MNT,
                timestamps_path=ROOT_DATA_MNT / video / "timestamp.txt",
                stage_configs=stage_configs,
                fold_config=fold_config,
                registration_config=REGISTRATION,
                compute_n_cycle_video=True,
                cache_dir=ROOT_REGISTERED_CACHE,
                registrator=registrator,
                verbose=False,
            ),
        )
        cycles = np.ascontiguousarray(result.cycles, dtype=np.uint8)
        cardiac_freq, n_cycle = result.cardiac_freq, result.n_cycle
        out.jobs.append(
            (
                "one_cycle.mkv + measure.pkl",
                partial(
                    write_fold_outputs,
                    result,
                    paths.one_cycle(video),
                    paths.measure(video),
                    PULSATION.output_fps,
                ),
            )
        )
    else:
        cycles = inputs.get("cycles")
        cardiac_freq = inputs.get("cardiac_freq")
        n_cycle = inputs.get("n_cycle")

    # 4. Segment the folded cycles -------------------------------------------
    raw_cycle_masks = None
    if "cycles" in todo:
        raw_cycle_masks = timed(
            "cycles",
            lambda: infer(
                model,
                cycles,
                batch_size=batch_size,
                device="cuda:0",
                post_process=False,
            ),
        )
        # Post-processed here rather than on the writer thread because deltaA
        # and QC both measure the smoothed mask.
        cycle_masks = postprocess_cycles(raw_cycle_masks, cardiac_freq, n_cycle)
        out.jobs.append(
            (
                "segmented_cycles.npz",
                partial(write_cycle_mask, cycle_masks, paths.cycle_mask(video)),
            )
        )
    else:
        cycle_masks = inputs.get("cycle_masks")

    # 5. deltaY: the cardiac amplitude fit, on the unsmoothed cycle masks ------
    if "deltaY" in todo:
        if raw_cycle_masks is None:
            raw_cycle_masks = infer(
                model,
                cycles,
                batch_size=batch_size,
                device="cuda:0",
                post_process=False,
            )
        deltaY_masks = timed(
            "deltaY", lambda: keep_largest_connected_component(raw_cycle_masks)
        )
        out.deltaY_rows = measure_deltaY(deltaY_masks, video, N_CYCLES, DELTA_Y)
        del deltaY_masks
    raw_cycle_masks = None

    # 6. deltaA: boundary displacement / area change (CPU, on the writer) -----
    if "deltaA" in todo:
        cycles_np = cycles.numpy() if isinstance(cycles, torch.Tensor) else cycles
        out.jobs.append(
            (
                "deltaA_per_cycle.pkl",
                partial(
                    compute_and_write_deltaA,
                    cycles_np,
                    cycle_masks,
                    paths.deltaA(video),
                ),
            )
        )

    # 7. QC: is this registration trustworthy? -------------------------------
    if "qc" in todo:
        metrics = timed("qc", lambda: compute_qc_metrics(cycle_masks))
        flag, reasons = evaluate_flag(metrics)
        out.qc_row = {
            "source": paths.root_measures.name,
            "video": video,
            **metrics,
            "flag": flag,
            "reasons": reasons,
        }

    return out


def merge_rows(table: pd.DataFrame, rows: list[dict], key: str = "video"):
    """New rows replace the ones they supersede; everything else is left alone."""
    if not rows:
        return table
    new = pd.DataFrame(rows)
    if table.empty:
        return new
    kept = table[~table[key].isin(new[key])]
    return pd.concat([kept, new], ignore_index=True)


def flush_tables(paths: CohortPaths, deltaY_rows: list, qc_rows: list) -> None:
    """Write both cohort tables, each merged into whatever is already there."""
    if deltaY_rows:
        table = (
            pd.read_pickle(paths.deltaY_table)
            if paths.deltaY_table.exists()
            else pd.DataFrame(
                columns=["video", "deltaY", "Amplitudes", "Fits", "cycle"]
            )
        )
        # One row per cycle, so a video's whole set of rows is replaced at once.
        merged = merge_rows(table, deltaY_rows)
        tmp = paths.deltaY_table.with_name(paths.deltaY_table.name + ".tmp")
        merged.to_pickle(tmp)
        tmp.replace(paths.deltaY_table)

    if qc_rows:
        table = (
            pd.read_csv(paths.qc_table)
            if paths.qc_table.exists()
            else pd.DataFrame(columns=["video"])
        )
        merged = merge_rows(table, qc_rows)
        tmp = paths.qc_table.with_name(paths.qc_table.name + ".tmp")
        merged.to_csv(tmp, index=False)
        tmp.replace(paths.qc_table)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--force-from",
        choices=STAGES,
        default=None,
        help="Recompute from this stage on, whatever is already on disk.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Shorthand for --force-from segment: redo everything.",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="Process at most N videos (smoke run)."
    )
    p.add_argument(
        "--reuse-segreg",
        action="store_true",
        help=(
            "Never segment or register: start every video at the fold, from the "
            "masks and registration cache OCULARRIGIDITY_MASKS / "
            "OCULARRIGIDITY_REGISTERED_CACHE point at (typically another run's). "
            "Videos without a valid cache there are skipped."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run, per stage, and exit without touching a GPU.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=SEGMENTATION_BATCH_SIZE,
        help="Frames per GPU forward pass.",
    )
    return p.parse_args()


def main():
    # Set the environment variable OCULARRIGIDITY_OUTPUT_ROOT
    args = parse_args()
    paths = CohortPaths()
    logger = setup_logging(
        "run_cohort", ROOT_CARDIAC_PIPELINE / "logs" / "run_cohort.log"
    )

    force_from = "segment" if args.overwrite else args.force_from
    logger.info(
        f"Cohort run | outputs -> {ROOT_CARDIAC_PIPELINE} "
        f"| registration={REGISTRATION.method} "
        f"| force-from={force_from or 'none'} | batch_size={args.batch_size}"
    )
    logger.info(
        f"masks <- {ROOT_MASKS} | registration cache <- {ROOT_REGISTERED_CACHE} "
        f"| reuse-segreg={args.reuse_segreg} "
        f"| pulse model={PULSATION.sinc_revision or 'thickness traces'} "
        f"| phase anchoring={'on' if PULSATION.anchor else 'off'}"
    )

    if args.reuse_segreg and force_from in ("segment", "register"):
        raise SystemExit(
            "--reuse-segreg cannot be combined with forcing segment/register"
        )
    tasks = build_tasks(paths, force_from, args.limit, logger, args.reuse_segreg)
    if args.dry_run:
        logger.info("--dry-run: stopping before any work.")
        return
    if not tasks:
        logger.info("Nothing to process.")
        return

    torch.backends.cudnn.benchmark = True
    model = get_model()
    reg_model = get_registration_model(REGISTRATION.registrator_revision).to("cuda")
    logger.info(
        f"Models from the Hub: segmentation {SEGMENTATION_REVISION}, "
        f"registration {REGISTRATION.registrator_revision}"
    )

    loader = DataLoader(
        PrefetchDataset(tasks, load_inputs),
        batch_size=1,
        num_workers=NUM_DECODE_WORKERS,
        shuffle=False,
        prefetch_factor=PREFETCH_FACTOR,
        collate_fn=identity_collate,
    )

    n_ok, n_fail = 0, 0
    deltaY_rows: list[dict] = []
    qc_rows: list[dict] = []
    pending = None  # in-flight writes, one video behind the GPU

    # One writer thread: it holds a whole video's arrays until it is done, and a
    # deeper queue would just hold several.
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
                logger.error(f"FAILED (write): {name}\n  Error: {e}")
                n_fail += 1

        t_mark = time.perf_counter()
        for i, (task, inputs, load_error) in enumerate(loader, start=1):
            # Time blocked on the loader. Near zero means the decode worker is
            # keeping up and the compute is the wall.
            t_wait = time.perf_counter() - t_mark

            outcome = None
            try:
                if load_error is not None:
                    logger.error(f"FAILED (load): {task.video}\n{load_error}")
                    n_fail += 1
                    continue

                try:
                    outcome = process_video(
                        task, inputs, model, reg_model, args.batch_size
                    )
                except Exception as e:
                    logger.error(
                        f"FAILED ({task.start}): {task.video}\n"
                        f"  Error: {type(e).__name__}: {e}\n"
                        f"  Traceback:\n{traceback.format_exc()}"
                    )
                    n_fail += 1
                    continue

                # Collect the previous video's writes before queueing this one,
                # so at most one video is in flight. A non-zero wait here means
                # the encodes have become the wall rather than the GPU.
                t_writer = time.perf_counter()
                drain(pending)
                t_writer = time.perf_counter() - t_writer

                pending = (writer.submit(run_jobs, outcome.jobs), task.video)

                deltaY_rows += outcome.deltaY_rows
                if outcome.qc_row is not None:
                    qc_rows.append(outcome.qc_row)

                stages = " ".join(
                    f"{k} {v:.1f}" + ("" if k == "ref_percentile" else "s")
                    for k, v in outcome.timings.items()
                )
                logger.info(
                    f"[{i}/{len(tasks)}] {task.video} | from {task.start} | "
                    f"load-wait {t_wait:5.1f}s | {stages} | "
                    f"writer-wait {t_writer:5.1f}s"
                )

                if i % CHECKPOINT_EVERY == 0:
                    flush_tables(paths, deltaY_rows, qc_rows)
                    deltaY_rows, qc_rows = [], []
            finally:
                # One release point for every path out of the iteration. The
                # decoded volume is several GB: drop this iteration's references
                # before the loader hands over the next one, or two are resident
                # at once. Rebinding rather than `del` leaves the names defined
                # on the paths that never bound them.
                inputs = outcome = None
                gc.collect()
                torch.cuda.empty_cache()
                t_mark = time.perf_counter()

        drain(pending)

    flush_tables(paths, deltaY_rows, qc_rows)

    logger.info("=" * 60)
    logger.info(f"Done. {n_ok} videos written, {n_fail} failed.")
    logger.info(f"Outputs under {ROOT_CARDIAC_PIPELINE}")


if __name__ == "__main__":
    main()
