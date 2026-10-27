"""Dump frames for unsupervised training of the registration regressor.

No classical registration is needed: each video gives one fixed frame and
``n_frames - 1`` moving frames, drawn at random (excluding the start-up and end
frames that the registration config trims), with no transforms. The layout is
the one ``train_registration.py`` reads::

    <output>/images/fixed/<name>_<idx>.png     one per video (middle drawn frame)
    <output>/images/moving/<name>_<idx>.png
    <output>/manifest.csv                      one row per video (patient, source, ...)
    <output>/skipped.csv                       videos that could not be read, and why

``<name>`` is ``<kind>_<video path with / and spaces as _>`` and ``<idx>`` is the
frame index in time order. The fixed frame is written last, so a run killed
mid-video never leaves a fixed frame with missing moving frames.

Videos come from the measurements database, shuffled with ``--seed``. Every
patient's first video comes before any second one, so the 400 videos cover as
many patients as possible. Reruns resume: videos already in ``manifest.csv`` or
``skipped.csv`` are not redone, and each kind is topped up to ``--n-videos``.

PLEX frames are read from the local MP4 cache when it exists, otherwise only the
drawn frames are read from ``cube.bin`` on the share. Spectralis B-scans are
resampled to the PLEX pixel size (as in ``scripts/pulsation/spectralis.py``) and
zero-padded to the PLEX frame size, so batches can mix devices.

    python -m ocularrigidity.registration.deep_learning.dataset.prepare_unsupervised_data
    python -m ocularrigidity.registration.deep_learning.dataset.prepare_unsupervised_data \\
        --kinds spectralis_macular --n-videos 50 --workers 4
"""

import argparse
import os
import sqlite3
import zlib
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ocularrigidity.consts import MEASUREMENTS_PATH, ROOT_COMPRESSED_VIDEO, ROOT_DATA_MNT
from ocularrigidity.data.compression import mp4_to_cube
from ocularrigidity.pipeline_config import REGISTRATION

OUTPUT_DIR = Path(
    "/home/clement/Documents/data/OcularRigidityAnnotated/dataset/Registration/"
)

# kind -> (MeasureName in the database, device)
KINDS = {
    "plex_macular": ("Plex Macular Video", "plex"),
    "plex_onh": ("Plex ONH Video", "plex"),
    "spectralis_macular": ("Spectralis Macular Volume Path", "spectralis"),
}

# Raw cube.bin layout (see data.io.load_cube): frames stored (H, W), returned transposed
CUBE_H, CUBE_W = 1024, 1536
# Frame size the trainer sees: (depth rows, A-scans)
FRAME_SIZE = (CUBE_W, CUBE_H)

MANIFEST_COLUMNS = [
    "name", "kind", "device", "patient", "video", "source",
    "n_frames_video", "n_saved", "fixed_idx", "height", "width",
]


# --- Candidates --------------------------------------------------------------


def candidates(kind: str, seed: int) -> list[str]:
    """Video paths of one kind, shuffled, patients' first videos first."""
    measure_name, _ = KINDS[kind]
    with sqlite3.connect(f"file:{MEASUREMENTS_PATH}?mode=ro", uri=True) as con:
        df = pd.read_sql_query(
            "SELECT PatientId, MeasureValue FROM measurements "
            "WHERE lower(MeasureName) = lower(?)",
            con,
            params=(measure_name,),
        )
    df = df.dropna(subset=["MeasureValue"])
    df = df[~df["MeasureValue"].str.startswith("\\\\Usereve")]
    df["video"] = df["MeasureValue"].str.replace("\\", "/", regex=False).str.strip("/")
    df = df.drop_duplicates("video").sample(frac=1.0, random_state=seed)
    df["rank"] = df.groupby("PatientId").cumcount()
    return df.sort_values("rank", kind="stable")["video"].tolist()


def video_name(kind: str, video: str) -> str:
    return f"{kind}_" + video.replace("/", "_").replace(" ", "_")


# --- Frame reading -----------------------------------------------------------


def draw_indices(n: int, n_frames: int, skip: int, drop: int, rng) -> np.ndarray:
    """Sorted time indices of the frames to keep."""
    valid = np.arange(skip, n - drop)
    if len(valid) < 2:
        raise ValueError(f"only {len(valid)} usable frames out of {n}")
    k = min(n_frames, len(valid))
    return np.sort(rng.choice(valid, size=k, replace=False))


def read_plex(video: str, n_frames: int, rng, source: str):
    """``(indices, frames (k, 1536, 1024) uint8, n_total, source)``."""
    skip, drop = REGISTRATION.skip_first_n_frames, REGISTRATION.drop_last_n_frames
    mp4 = ROOT_COMPRESSED_VIDEO / video / "cube.mp4"
    if source == "mp4" or (source == "auto" and mp4.exists()):
        cube = mp4_to_cube(mp4, verbose=False)
        idx = draw_indices(len(cube), n_frames, skip, drop, rng)
        return idx, np.ascontiguousarray(cube[idx]), len(cube), "mp4"

    folder = ROOT_DATA_MNT / video
    cube_path = folder / "cube.bin"
    frame_bytes = CUBE_H * CUBE_W
    size = os.path.getsize(cube_path)
    if size % frame_bytes:
        raise ValueError(f"cube.bin size {size} is not a multiple of {CUBE_H}x{CUBE_W}")
    n = size // frame_bytes

    # Same time ordering as load_cube
    order = np.arange(n)
    ts_path = folder / "timestamp.txt"
    if ts_path.exists():
        with open(ts_path) as f:
            ts = np.array([float(line) for line in f if line.strip()])
        if len(ts) == n:
            order = np.argsort(ts, kind="stable")

    idx = draw_indices(n, n_frames, skip, drop, rng)
    frames = np.empty((len(idx), CUBE_W, CUBE_H), dtype=np.uint8)
    with open(cube_path, "rb") as f:
        for k, raw in enumerate(order[idx]):
            f.seek(int(raw) * frame_bytes)
            buf = f.read(frame_bytes)
            if len(buf) != frame_bytes:
                raise ValueError(f"short read at frame {raw}")
            frames[k] = np.frombuffer(buf, np.uint8).reshape(CUBE_H, CUBE_W).T
    return idx, frames, n, "raw"


def pad_to(frame: np.ndarray, size) -> np.ndarray:
    """Zero-pad (or crop) to ``size``: top-aligned in depth, centred laterally."""
    H, W = size
    out = np.zeros((H, W), dtype=frame.dtype)
    h, w = min(frame.shape[0], H), min(frame.shape[1], W)
    x_src = (frame.shape[1] - w) // 2
    x_dst = (W - w) // 2
    out[:h, x_dst : x_dst + w] = frame[:h, x_src : x_src + w]
    return out


def read_spectralis(video: str, n_frames: int, rng):
    # Heavy module (torch, models); only imported when Spectralis is asked for
    from ocularrigidity.scripts.pulsation.spectralis import load_spectralis, to_plex_grid

    frames, _, axial_mm, lateral_mm, _ = load_spectralis(ROOT_DATA_MNT / video)
    # A Spectralis series has no start-up to discard
    idx = draw_indices(len(frames), n_frames, 0, 0, rng)
    resampled, _, _ = to_plex_grid(frames[idx], axial_mm, lateral_mm)
    out = np.stack([pad_to(f, FRAME_SIZE) for f in resampled])
    return idx, out, len(frames), "spectralis"


# --- One video ---------------------------------------------------------------


def process(kind: str, video: str, output: Path, n_frames: int, seed: int, source: str) -> dict:
    device = KINDS[kind][1]
    rng = np.random.default_rng([seed, zlib.crc32(video.encode())])
    if device == "plex":
        idx, frames, n_total, src = read_plex(video, n_frames, rng, source)
    else:
        idx, frames, n_total, src = read_spectralis(video, n_frames, rng)
    if frames.shape[1:] != FRAME_SIZE:
        raise ValueError(f"frame size {frames.shape[1:]} != {FRAME_SIZE}")

    name = video_name(kind, video)
    fixed = len(idx) // 2
    moving_dir = output / "images" / "moving"
    for k, (i, frame) in enumerate(zip(idx, frames)):
        if k != fixed:
            cv2.imwrite(str(moving_dir / f"{name}_{i}.png"), frame)
    # Written last: the trainer only uses videos that have a fixed frame
    cv2.imwrite(str(output / "images" / "fixed" / f"{name}_{idx[fixed]}.png"), frames[fixed])

    return {
        "name": name,
        "kind": kind,
        "device": device,
        "patient": video.split("/")[0],
        "video": video,
        "source": src,
        "n_frames_video": n_total,
        "n_saved": len(idx),
        "fixed_idx": int(idx[fixed]),
        "height": FRAME_SIZE[0],
        "width": FRAME_SIZE[1],
    }


# --- Bookkeeping -------------------------------------------------------------


def append_row(path: Path, row: dict, columns: list[str]):
    pd.DataFrame([row], columns=columns).to_csv(
        path, mode="a", header=not path.exists(), index=False
    )


def read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=columns)


def run_kind(kind: str, args, manifest: pd.DataFrame, skipped: pd.DataFrame):
    manifest_path = args.output / "manifest.csv"
    skipped_path = args.output / "skipped.csv"
    n_done = int((manifest["kind"] == kind).sum())
    seen = set(manifest.loc[manifest["kind"] == kind, "video"])
    if not args.retry_skipped:
        seen |= set(skipped.loc[skipped["kind"] == kind, "video"])
    todo = iter(v for v in candidates(kind, args.seed) if v not in seen)
    target = args.n_videos - n_done
    if target <= 0:
        print(f"{kind}: {n_done} videos already done")
        return

    ok, n_skipped = 0, 0
    pbar = tqdm(total=args.n_videos, initial=n_done, desc=kind)
    with ThreadPoolExecutor(args.workers) as pool:
        pending = {}

        def fill():
            while len(pending) < args.workers and ok + len(pending) < target:
                video = next(todo, None)
                if video is None:
                    return
                fut = pool.submit(
                    process, kind, video, args.output, args.n_frames, args.seed, args.source
                )
                pending[fut] = video

        fill()
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                video = pending.pop(fut)
                try:
                    append_row(manifest_path, fut.result(), MANIFEST_COLUMNS)
                    ok += 1
                    pbar.update(1)
                except Exception as e:  # noqa: BLE001 - logged, video skipped
                    n_skipped += 1
                    reason = f"{type(e).__name__}: {e}".replace("\n", " ")
                    append_row(
                        skipped_path,
                        {"kind": kind, "video": video, "reason": reason},
                        ["kind", "video", "reason"],
                    )
                    pbar.set_postfix(skipped=n_skipped)
            fill()
    pbar.close()
    if n_done + ok < args.n_videos:
        print(f"{kind}: ran out of candidates at {n_done + ok}/{args.n_videos} videos")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--kinds", nargs="+", choices=list(KINDS), default=list(KINDS))
    ap.add_argument("--n-videos", type=int, default=400, help="videos per kind")
    ap.add_argument("--n-frames", type=int, default=40, help="frames per video, fixed included")
    ap.add_argument(
        "--source",
        choices=("auto", "mp4", "raw"),
        default="auto",
        help="PLEX only: auto = the MP4 cache when it exists, else cube.bin",
    )
    ap.add_argument("--workers", type=int, default=8, help="videos read in parallel")
    ap.add_argument("--retry-skipped", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for sub in ("images/fixed", "images/moving"):
        (args.output / sub).mkdir(parents=True, exist_ok=True)

    for kind in args.kinds:
        manifest = read_csv(args.output / "manifest.csv", MANIFEST_COLUMNS)
        skipped = read_csv(args.output / "skipped.csv", ["kind", "video", "reason"])
        run_kind(kind, args, manifest, skipped)

    manifest = read_csv(args.output / "manifest.csv", MANIFEST_COLUMNS)
    print(
        manifest.groupby("kind").agg(
            videos=("name", "size"),
            patients=("patient", "nunique"),
            frames=("n_saved", "sum"),
        )
    )
    print(f"{manifest.patient.nunique()} distinct patients overall")


if __name__ == "__main__":
    main()
