"""Pulsation amplitude of the historical Spectralis rigidity acquisitions."""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import fcntl
import gc
import json
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import torch
from scipy.io import loadmat

from ocularrigidity.consts import (
    AXIAL_PIXEL_SIZE_MM,
    CHOROID_TRIM_PX,
    OUTPUT_ROOT,
    ROOT_DATA_MNT,
    SINC_REVISION,
    TRANVERSAL_PIXEL_SIZE_MM,
)
from ocularrigidity.data.compression import read_gray
from ocularrigidity.motion.pipeline_results import (
    CardiacPipelineResults,
    write_fold_outputs,
)
from ocularrigidity.motion.pulsation.n_cycle_reconstructor import NCycleReconstructor
from ocularrigidity.motion.pulsation.phase.anchoring import ThicknessAnchorConfig
from ocularrigidity.motion.pulsation.pipeline import build_extractor
from ocularrigidity.motion.pulsation.traces.mask import MaskTraceConfig
from ocularrigidity.motion.video_timeline_aligner import TimeUnits, VideoTimelineAligner
from ocularrigidity.pipeline_config import DELTA_A, DELTA_Y, PULSATION, REGISTRATION
from ocularrigidity.registration.fused import segment_and_register
from ocularrigidity.registration.registration_engine import (
    VideoRegistrator,
    cache_is_valid,
)
from ocularrigidity.scripts._logging import setup_logging
from ocularrigidity.motion.tracking.displacement import extract_displacement
from ocularrigidity.segmentation.postprocess.temporal_smoothing import (
    postprocess_cycles,
)
from ocularrigidity.segmentation.utils import get_model, get_registration_model
from ocularrigidity.thickness.features import measure_deltaY
from ocularrigidity.segmentation.closing_structures import trim_choroid
from ocularrigidity.segmentation.inference import infer
from ocularrigidity.thickness.delta import mask_ct_series_mm, measure_delta_ct_from_disp

DEFAULT_TABLE = (
    Path(__file__).resolve().parents[4] / "notebooks/diane_replication/df_rigidity.pkl"
)
DEFAULT_OUTPUT = OUTPUT_ROOT / "DianeReplication"
REQUIRED_FILES = ("ProcessedImages.tif", "Data Files/ImageList.mat")

PLEX_WIDTH = 1024
PLEX_FS = 87.6

N_CYCLE = 1
N_BINS = 20

# Savitzky-Golay window (frames) smoothing each tracked boundary point along the cycle.
SMOOTH_WINDOW = 7

# Columns dropped on each side of the folded-cycle mask before the
# displacement-based ΔCT: neither tracked nor measured. None: on 45 same-day
# pairs, 100 / 50 / 20 / 0 gave the same repeatability (ICC of log ΔCT 0.61 /
# 0.59 / 0.59 / 0.60) and no tracking failure, and the untrimmed thickness is
# the closest to a full-width one.
TRIM = CHOROID_TRIM_PX

# When an acquisition comes with an expected heart rate, SiNC's input is
# band-passed to that rate ± this fraction (none of the historical acquisitions
# is run that way: the launcher gives no expected rate).
SINC_INPUT_BAND_FRAC = 0.1

# Which boundary points the displacement-based ΔCT tracks: one per column on
# each interface (`reference_boundary_points`). Recorded with each measurement
# so that --remeasure can tell the ones made on the earlier mask outline.
BOUNDARY_POINTS = "per_column"

DY_ALIGN_BM = True

LOCATE_THREADS = 16


# --- Loading ---------------------------------------------------------------


def load_spectralis(eye_dir: Path):
    """One Spectralis acquisition: B-scans ``(T, H, W)`` uint8 in time order, timestamps (s, from 0) and the axial / lateral pixel sizes (mm)."""
    frames = tifffile.imread(eye_dir / "ProcessedImages.tif")
    if frames.ndim != 3 or frames.dtype != np.uint8:
        raise ValueError(f"Expected (T, H, W) uint8, got {frames.shape} {frames.dtype}")
    mat = loadmat(
        eye_dir / "Data Files" / "ImageList.mat",
        squeeze_me=True,
        struct_as_record=False,
    )
    meta = pd.DataFrame(
        [
            {f: getattr(s, f) for f in s._fieldnames}
            for s in np.atleast_1d(mat["ImageList"])
        ]
    )
    if len(meta) != len(frames):
        raise ValueError(f"{len(frames)} frames but {len(meta)} ImageList entries")

    t = (meta.hour * 3600 + meta.minute * 60 + meta.second).to_numpy(dtype=float)
    order = np.argsort(t, kind="stable")
    frames, t, meta = frames[order], t[order], meta.iloc[order].reset_index(drop=True)

    m = meta.iloc[0]
    # ScaleX is stored rounded to 0.1 µm
    lateral_mm = float(np.hypot(m.endX - m.startX, m.endY - m.startY)) / m.width
    return frames, t - t[0], float(m.scaleY), lateral_mm, meta


def to_plex_grid(frames, axial_mm, lateral_mm, width=PLEX_WIDTH):
    """Resample B-scans to the PLEX pixel size the networks were trained on."""
    T, H, W = frames.shape
    new_h = int(round(H * axial_mm / AXIAL_PIXEL_SIZE_MM / 32)) * 32
    new_w = int(round(W * lateral_mm / TRANVERSAL_PIXEL_SIZE_MM))
    out = np.stack(
        [cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_CUBIC) for f in frames]
    )
    width = min(width, new_w // 32 * 32)
    x0 = (new_w - width) // 2
    out = np.ascontiguousarray(out[:, :, x0 : x0 + width])
    return out, H * axial_mm / new_h, W * lateral_mm / new_w


def registration_config(batch_size: int = REGISTRATION.fused_batch_size):
    """The cohort's learned registration, minus the frame trimming (a Spectralis series is short and has no ramp-up to discard) and with every A-scan aligned on the mask's upper interface."""
    return replace(
        REGISTRATION,
        skip_first_n_frames=0,
        drop_last_n_frames=0,
        fused_batch_size=batch_size,
        dy_align_bm=DY_ALIGN_BM,
    )


# --- One acquisition -------------------------------------------------------


def measure_displacement(
    cycles_u8: np.ndarray,
    cardiac_freq: float,
    seg_model,
    batch_size: int,
    px: dict,
    trim: int = TRIM,
) -> tuple[list, np.ndarray]:
    """Displacement-based ΔCT of each folded cycle."""
    cycle_masks = postprocess_cycles(
        infer(
            seg_model,
            cycles_u8,
            batch_size=batch_size,
            post_process=False,
        ),
        cardiac_freq,
        N_CYCLE,
    )
    _, _, displacements, references = extract_displacement(
        video=cycles_u8,
        mask=cycle_masks,
        N_cycles=N_CYCLE,
        smooth_window=SMOOTH_WINDOW,
        trim=trim,
    )
    trimmed_masks = trim_choroid(cycle_masks, trim)
    frames_per_cycle = len(cycles_u8) // N_CYCLE
    results = [
        measure_delta_ct_from_disp(
            displacements[i],
            references[i],
            trimmed_masks[i * frames_per_cycle : i * frames_per_cycle + 1],
            reference_frame_idx=0,
            **px,
        )
        for i in range(N_CYCLE)
    ]
    return results, mask_ct_series_mm(trimmed_masks, **px)


@contextmanager
def exclusive(lock_path: Path):
    """Hold ``lock_path`` against the other worker processes."""
    with open(lock_path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def process_video(
    video: Path,
    output: Path,
    seg_model,
    reg_model,
    reg_config,
    sinc_revision: str,
    batch_size: int,
    gpu_lock: Path,
    cache_dir: Optional[Path] = None,
    acquisition: Optional[tuple] = None,
    expected_bpm: Optional[float] = None,
) -> tuple[pd.DataFrame, dict]:
    """Seg+reg, SiNC fold and pulsation amplitude of one acquisition.

    ``acquisition`` is ``(frames, timestamps, axial_mm, lateral_mm)`` as
    `load_spectralis` returns them (B-scans in time order, seconds from 0,
    native pixel sizes); given, it is used instead of reading
    ``ROOT_DATA_MNT / video``, and ``video`` only names the outputs.

    ``expected_bpm``, if given, band-passes SiNC's input around it
    (``SINC_INPUT_BAND_FRAC``); the rate is still the one read off the SiNC
    waveform.
    """
    measure_dir = output / "measures" / video
    timings: dict = {}
    t0 = time.perf_counter()

    def lap(name):
        nonlocal t0
        now = time.perf_counter()
        timings[name] = now - t0
        t0 = now

    if acquisition is None:
        acquisition = load_spectralis(ROOT_DATA_MNT / video)[:4]
    raw_frames, timestamps, axial_mm, lateral_mm = acquisition
    frames, axial_mm_px, lateral_mm_px = to_plex_grid(raw_frames, axial_mm, lateral_mm)
    px = dict(axial_mm_per_px=axial_mm_px, transversal_mm_per_px=lateral_mm_px)
    lap("load")

    with exclusive(gpu_lock):
        lap("gpu-wait")
        fused = segment_and_register(
            frames, seg_model, reg_model, reg_config, verbose=False
        )
        torch.cuda.empty_cache()
    registrator = VideoRegistrator(
        video=video, config=reg_config, cache_dir=cache_dir, verbose=False
    )
    registrator.prime_from_result(
        fused.registered_frames,
        fused.registered_masks,
        fused.transform,
        raw_masks=fused.raw_masks,
    )
    lap("segreg")
    if cache_dir is not None:
        registrator.flush_cache()
        lap("cache")

    aligner = VideoTimelineAligner(
        registrator,
        timestamps,
        units_in_timestamps=TimeUnits.SECONDS,
        target_fs=PLEX_FS,
    )
    W = frames.shape[-1]
    pulsation = replace(
        PULSATION,
        trace=MaskTraceConfig(col_slice=slice(100, W - 100)),
        sinc_revision=sinc_revision,
        sinc_input_band_frac=SINC_INPUT_BAND_FRAC if expected_bpm else None,
        anchor=ThicknessAnchorConfig(n_segments=N_CYCLE),
        fold=replace(PULSATION.fold, n_cycle=N_CYCLE, n_bins=N_BINS),
    )
    stage_configs, fold_config = pulsation.chain_for_video(
        expected_bpm=expected_bpm, verbose=False
    )
    extractor = build_extractor(registrator, aligner, stage_configs)
    _ = extractor.phase_per_frame
    lap("pulse")

    reconstructor = NCycleReconstructor(extractor, fold_config)
    cycles, counts = reconstructor.compute()
    result = CardiacPipelineResults.from_composed(
        extractor, reconstructor, stage_configs=stage_configs
    )
    write_fold_outputs(
        result,
        output / "one_cycle" / video / "one_cycle.mkv",
        measure_dir / "measure.pkl",
        PULSATION.output_fps,
    )
    lap("fold")

    cycles_u8 = np.ascontiguousarray(cycles, dtype=np.uint8)
    frames_per_cycle = len(cycles_u8) // N_CYCLE

    deltaY_masks = infer(seg_model, cycles_u8, batch_size=batch_size)
    deltaY = pd.DataFrame(measure_deltaY(deltaY_masks, str(video), N_CYCLE, DELTA_Y))
    lap("deltaY")

    results, thickness_mm = measure_displacement(
        cycles_u8, extractor.cardiac_freq, seg_model, batch_size, px
    )

    dt = np.diff(timestamps)
    rows = []
    for i, res in enumerate(results):
        cycle = slice(i * frames_per_cycle, (i + 1) * frames_per_cycle)
        rows.append(
            {
                "video": str(video),
                "cycle": i,
                "bpm": extractor.cardiac_bpm,
                "expected_bpm": expected_bpm if expected_bpm else np.nan,
                "confidence": extractor.confidence,
                "deltaCT_um": res.deltaCT_um,  # displacement-based
                "deltaCT_mask_um": deltaY.deltaY[i] * axial_mm_px * 1e3,  # mask-based
                "deltaCT_mask_thickness_um": (
                    np.nanmax(thickness_mm[cycle]) - np.nanmin(thickness_mm[cycle])
                )
                * 1e3,
                "minCT_um": res.min_ct_mm * 1e3,
                "rpe_residual_um": res.rpe_residual_um,
                "smooth_window": SMOOTH_WINDOW,
                "boundary_points": BOUNDARY_POINTS,
                "trim": TRIM,
                "dy_align_bm": reg_config.dy_align_bm,
                "n_frames": len(timestamps),
                "n_good_frames": int(extractor.good_per_frame.sum()),
                "min_frames_per_bin": int(counts.min()),
                "duration_s": timestamps[-1],
                "median_fs_hz": 1 / np.median(dt),
                "spectralis_axial_um_px": axial_mm * 1e3,
                "spectralis_lateral_um_px": lateral_mm * 1e3,
                "axial_um_px": axial_mm_px * 1e3,
                "lateral_um_px": lateral_mm_px * 1e3,
                "notes": " | ".join(extractor.notes),
            }
        )
    table = pd.DataFrame(rows)
    lap("deltaCT")

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    ax = axes[0, 0]
    ax.imshow(frames[fused.ref_idx], cmap="gray")
    ax.contour(fused.raw_masks[fused.ref_idx], levels=[0.5], colors="r", linewidths=1)
    ax.set_title(f"Reference frame {fused.ref_idx} (PLEX grid) + choroid mask")
    ax.axis("off")
    ax = axes[0, 1]
    ax.plot(
        timestamps,
        fused.transform["dy"].mean(axis=1) * axial_mm_px * 1e3,
        label="axial (dy)",
    )
    ax.plot(timestamps, fused.transform["dx"] * lateral_mm_px * 1e3, label="lateral (dx)")
    ax.set(xlabel="time (s)", ylabel="shift (µm)", title="Registration transform")
    ax.legend()
    ax = axes[1, 0]
    ax.plot(extractor.uniform_time, extractor.traces.source_map[:, 0])
    ax.set(
        xlabel="time (s)",
        ylabel="SiNC waveform",
        title=f"{extractor.cardiac_bpm:.1f} bpm (confidence={extractor.confidence})",
    )
    ax = axes[1, 1]
    ax.plot(np.arange(len(thickness_mm)) / frames_per_cycle, thickness_mm * 1e3)
    ax.set(
        xlabel="cardiac phase (cycles)",
        ylabel="mean choroidal thickness (µm)",
        title=f"ΔCT {table.deltaCT_um[0]:.1f} µm (displacement), "
        f"{table.deltaCT_mask_um[0]:.1f} µm (mask)",
    )
    fig.suptitle(str(video))
    fig.tight_layout()
    fig.savefig(measure_dir / "qc.png", dpi=100)
    plt.close(fig)
    lap("qc")

    table.to_csv(measure_dir / "pulsation.csv", index=False)
    return table, timings


def remeasure_video(video: Path, output: Path, seg_model, batch_size: int) -> None:
    """Redo the displacement-based ΔCT of one acquisition from its one_cycle.mkv."""
    csv_path = output / "measures" / video / "pulsation.csv"
    table = pd.read_csv(csv_path)
    px = dict(
        axial_mm_per_px=(
            table.axial_um_px[0] / 1e3 if "axial_um_px" in table else AXIAL_PIXEL_SIZE_MM
        ),
        transversal_mm_per_px=(
            table.lateral_um_px[0] / 1e3
            if "lateral_um_px" in table
            else TRANVERSAL_PIXEL_SIZE_MM
        ),
    )
    cycles_u8 = np.ascontiguousarray(
        read_gray(output / "one_cycle" / video / "one_cycle.mkv"), dtype=np.uint8
    )
    results, _ = measure_displacement(
        cycles_u8, table.bpm[0] / 60.0, seg_model, batch_size, px
    )

    table["deltaCT_um_previous"] = table["deltaCT_um"]
    table["smooth_window_previous"] = table.get("smooth_window", DELTA_A.smooth_window)
    for i, res in enumerate(results):
        table.loc[i, "deltaCT_um"] = res.deltaCT_um
        table.loc[i, "minCT_um"] = res.min_ct_mm * 1e3
        table.loc[i, "rpe_residual_um"] = res.rpe_residual_um
    table["smooth_window"] = SMOOTH_WINDOW
    table["boundary_points"] = BOUNDARY_POINTS
    table["trim"] = TRIM

    tmp_path = csv_path.with_name(csv_path.name + ".tmp")
    table.to_csv(tmp_path, index=False)
    tmp_path.replace(csv_path)


def build_remeasure_tasks(output: Path) -> tuple[list[str], int]:
    """Finished acquisitions measured with another window, other boundary
    points or another trim, and the up-to-date count."""
    tasks, current = [], 0
    root = output / "measures"
    for csv_path in sorted(root.rglob("pulsation.csv")):
        table = pd.read_csv(csv_path)
        current_settings = {
            "smooth_window": SMOOTH_WINDOW,
            "boundary_points": BOUNDARY_POINTS,
            "trim": TRIM,
        }
        if all(
            name in table and (table[name] == value).all()
            for name, value in current_settings.items()
        ):
            current += 1
            continue
        tasks.append(csv_path.parent.relative_to(root).as_posix())
    return tasks, current


# --- Planning --------------------------------------------------------------


def find_eye_dir(row) -> Optional[Path]:
    """The acquisition folder of one table row, or None if it is not there."""
    patient_dir = ROOT_DATA_MNT / str(row.patient)
    if not patient_dir.is_dir():
        return None
    for visit_dir in patient_dir.iterdir():
        eye_dir = visit_dir / row.modality / row.eye
        if not eye_dir.is_dir():
            continue
        try:
            date = pd.to_datetime(visit_dir.name, dayfirst=True)
        except (ValueError, TypeError):
            continue
        if date == row.date:
            return eye_dir
    return None


def _locate(row) -> tuple[Optional[str], Optional[dict]]:
    """``(video, None)`` for a usable acquisition, ``(None, skip record)`` otherwise."""
    label = f"{row.patient}/{row.date:%Y-%m-%d}/{row.modality}/{row.eye}"
    try:
        eye_dir = find_eye_dir(row)
        if eye_dir is None:
            return None, {"video": label, "reason": "folder not found"}
        video = eye_dir.relative_to(ROOT_DATA_MNT).as_posix()
        missing = [f for f in REQUIRED_FILES if not (eye_dir / f).is_file()]
    except OSError as e:
        return None, {"video": label, "reason": f"{type(e).__name__}: {e}"}
    if missing:
        return None, {"video": video, "reason": f"missing {missing}"}
    return video, None


def build_tasks(
    table: Path, output: Path, overwrite: bool, cache_dir: Optional[Path] = None
) -> tuple[list[str], pd.DataFrame, int]:
    """The acquisitions still to process, the skipped ones, and the done count."""
    reg_config = registration_config()
    df = pd.read_pickle(table)
    df = df[df.spectralis == True]  # noqa: E712

    with ThreadPoolExecutor(LOCATE_THREADS) as pool:
        located = list(pool.map(_locate, df.itertuples()))

    skipped = pd.DataFrame(
        [s for _, s in located if s is not None], columns=["video", "reason"]
    )
    tasks, done = [], 0
    for video, _ in located:
        if video is None:
            continue
        measured = (output / "measures" / video / "pulsation.csv").exists()
        cached = cache_dir is None or cache_is_valid(cache_dir, Path(video), reg_config)
        if measured and cached and not overwrite:
            done += 1
            continue
        tasks.append(video)
    return tasks, skipped, done


# --- Worker ----------------------------------------------------------------


def run_shard(args) -> None:
    output = args.output
    logger = setup_logging(
        "spectralis",
        output / "logs" / f"spectralis.shard{args.shard}.log",
        f" [shard {args.shard}]",
    )
    tasks = json.loads(args.tasks.read_text())[args.shard :: args.num_shards]
    logger.info(
        f"{'Re-measuring ' if args.remeasure else ''}{len(tasks)} acquisitions "
        f"| batch {args.batch_size} "
        f"| CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}"
    )
    failures_path = output / "logs" / f"failures.shard{args.shard}.csv"
    failures_path.unlink(missing_ok=True)
    if not tasks:
        return

    torch.backends.cudnn.benchmark = True
    seg_model = get_model()
    if args.remeasure:
        def work(video: str) -> dict:
            remeasure_video(Path(video), output, seg_model, args.batch_size)
            return {}

    else:
        reg_config = registration_config(args.batch_size)
        reg_model = get_registration_model(reg_config.registrator_revision).to("cuda")
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "all").replace(",", "_")
        gpu_lock = output / "logs" / f"gpu{gpu}.lock"

        def work(video: str) -> dict:
            _, timings = process_video(
                Path(video),
                output,
                seg_model,
                reg_model,
                reg_config,
                args.sinc_revision,
                args.batch_size,
                gpu_lock,
                args.cache_dir,
            )
            return timings

    n_ok, failures = 0, []
    for i, video in enumerate(tasks, start=1):
        t0 = time.perf_counter()
        try:
            timings = work(video)
            n_ok += 1
            stages = " ".join(f"{k} {v:.1f}s" for k, v in timings.items())
            logger.info(
                f"[{i}/{len(tasks)}] {video} | {time.perf_counter() - t0:5.1f}s | {stages}"
            )
        except Exception as e:
            logger.error(f"[{i}/{len(tasks)}] FAILED {video}: {type(e).__name__}: {e}")
            failures.append(
                {
                    "video": video,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                }
            )
            pd.DataFrame(failures).to_csv(failures_path, index=False)
        finally:
            plt.close("all")
            gc.collect()
            torch.cuda.empty_cache()

    logger.info(f"Done. {n_ok} succeeded, {len(failures)} failed.")


# --- Launcher --------------------------------------------------------------


def summarise(output: Path) -> None:
    """One table of every finished acquisition, and this run's failures."""
    logs = output / "logs"
    tables = [pd.read_csv(p) for p in sorted((output / "measures").rglob("pulsation.csv"))]
    if tables:
        pd.concat(tables, ignore_index=True).to_csv(
            output / "pulsation_all.csv", index=False
        )
    failures = [pd.read_csv(p) for p in sorted(logs.glob("failures.shard*.csv"))]
    pd.concat(
        failures or [pd.DataFrame(columns=["video", "error", "traceback"])],
        ignore_index=True,
    ).to_csv(output / "failures.csv", index=False)
    print(
        f"{len(tables)} acquisitions in {output / 'pulsation_all.csv'} | "
        f"{sum(len(f) for f in failures)} failed this run"
    )


def launch(args) -> None:
    output = args.output
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    if args.remeasure:
        tasks, current = build_remeasure_tasks(output)
        if args.limit is not None:
            tasks = tasks[: args.limit]
        print(
            f"To re-measure: {len(tasks)} | "
            f"Already at smooth_window={SMOOTH_WINDOW}, "
            f"boundary_points={BOUNDARY_POINTS}, trim={TRIM}: {current}"
        )
    else:
        tasks, skipped, done = build_tasks(
            args.table, output, args.overwrite, args.cache_dir
        )
        skipped.to_csv(output / "skipped.csv", index=False)
        if args.limit is not None:
            tasks = tasks[: args.limit]
        print(
            f"To process: {len(tasks)} | Already done: {done} | "
            f"Missing folder/data: {len(skipped)}"
        )
    if not tasks:
        summarise(output)
        return

    tasks_file = logs / "tasks.json"
    tasks_file.write_text(json.dumps(tasks))
    for stale in logs.glob("failures.shard*.csv"):
        stale.unlink()

    gpus = args.gpus.split(",")
    n = min(args.workers, len(tasks))
    procs = []
    for shard in range(n):
        cmd = [
            sys.executable,
            "-m",
            "ocularrigidity.scripts.pulsation.spectralis",
            "--tasks", str(tasks_file),
            "--num-shards", str(n),
            "--shard", str(shard),
            "--output", str(output),
            "--sinc-revision", args.sinc_revision,
            "--batch-size", str(args.batch_size),
        ]  # fmt: skip
        if args.remeasure:
            cmd.append("--remeasure")
        if args.cache_dir is not None:
            cmd += ["--cache-dir", str(args.cache_dir)]
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=gpus[shard % len(gpus)],
            NUMBA_NUM_THREADS=str(max(1, (os.cpu_count() or 1) // n)),
        )
        procs.append(subprocess.Popen(cmd, env=env))
    print(f"Started {n} workers on GPUs {args.gpus} | logs in {logs}")

    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait()
        print("Interrupted; finished acquisitions are kept and will be skipped.")
    finally:
        summarise(output)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table", type=Path, default=DEFAULT_TABLE,
                   help="df_rigidity.pkl; rows with spectralis == True are run.")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--sinc-revision", default=SINC_REVISION)
    p.add_argument("--workers", type=int, default=6,
                   help="Worker processes, spread round-robin over --gpus.")
    p.add_argument("--gpus", default="0,1",
                   help="Comma-separated GPU indices.")
    p.add_argument("--batch-size", type=int, default=REGISTRATION.fused_batch_size,
                   help="Frames per GPU forward pass (memory only, not speed).")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="Also save the registered frames, masks and transform "
                        "under this directory (e.g. the --output one).")
    p.add_argument("--overwrite", action="store_true",
                   help="Redo acquisitions that already have a pulsation.csv.")
    p.add_argument("--remeasure", action="store_true",
                   help="Only redo the displacement-based ΔCT of finished "
                        "acquisitions, from their one_cycle.mkv.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N acquisitions (smoke run).")
    # Set by the launcher on the workers it starts.
    p.add_argument("--tasks", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--shard", type=int, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def main():
    args = parse_args()
    if args.shard is None:
        launch(args)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
