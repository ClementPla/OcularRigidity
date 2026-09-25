"""Training data for SiNC: a compact cache of every video, and random clips.

``build_cache`` reads each ``measure.pkl`` once (~100 MB each, mostly arrays
SiNC does not need) and keeps the pooled BM/CSI deviations on the uniform grid
as float16, ~4 MB a video. The index CSV next to it carries what evaluation
needs: timing, gap fraction, measured HR and the stored pipeline estimate.

Splits are by patient, so no eye of a validation patient is seen in training.
"""

import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from ocularrigidity.motion.pulsation.sinc.preprocess import (
    N_COLS,
    boundaries_to_uniform,
)

INDEX_NAME = "index.csv"


def video_key(video: str) -> str:
    return video.replace("/", "__")


def patient_of(video: str) -> str:
    return video.split("/")[0]


def split_of(video: str, val_percent: int = 15) -> str:
    """Deterministic patient-level split (md5, so stable across runs/machines)."""
    h = int(hashlib.md5(patient_of(video).encode()).hexdigest(), 16)
    return "val" if h % 100 < val_percent else "train"


def _cache_one(args) -> dict | None:
    import pickle

    path, measures_root, cache_dir, n_cols = args
    video = path.parent.relative_to(measures_root).as_posix()
    out = cache_dir / f"{video_key(video)}.npz"
    try:
        with open(path, "rb") as f:
            m = pickle.load(f)
        if not out.exists():
            x = boundaries_to_uniform(
                m.registered_boundaries,
                m.timestamps_seconds,
                m.uniform_time,
                m.gap_mask,
                n_cols,
            )
            np.savez(out, x=x.astype(np.float16))
        ut = m.uniform_time
        return dict(
            video=video,
            n_uniform=len(ut),
            dt=float(ut[1] - ut[0]),
            gap_frac=float(np.mean(m.gap_mask)),
            pipeline_bpm=float(m.cardiac_freq) * 60.0,
            pipeline_confidence=m.confidence,
        )
    except Exception as e:  # a corrupt pickle should not stop the cohort
        print(f"skip {video}: {e}")
        return None


def build_cache(
    measures_root: Path,
    cache_dir: Path,
    n_cols: int = N_COLS,
    workers: int = 4,
    val_percent: int = 15,
    limit: int | None = None,
) -> pd.DataFrame:
    """Cache every ``measure.pkl`` under ``measures_root``; returns the index."""
    from ocularrigidity.data.measurements.dataframe import load_measurements

    measures_root, cache_dir = Path(measures_root), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(measures_root.rglob("measure.pkl"))[:limit]
    jobs = [(p, measures_root, cache_dir, n_cols) for p in files]
    with ProcessPoolExecutor(workers) as pool:
        rows = [
            r
            for r in tqdm(pool.map(_cache_one, jobs), total=len(jobs), desc="cache")
            if r is not None
        ]
    index = pd.DataFrame(rows)

    hr = load_measurements(include_HR=True)
    hr = pd.DataFrame(
        {
            "video": hr["MeasureValue"].map(lambda s: str(s).replace("\\", "/")),
            "HR": hr["HR"].where(hr["HR"] > 0),
        }
    ).drop_duplicates("video")
    index = index.merge(hr, on="video", how="left")
    index["split"] = index["video"].map(lambda v: split_of(v, val_percent))
    index.to_csv(cache_dir / INDEX_NAME, index=False)
    return index


def load_index(cache_dir: Path) -> pd.DataFrame:
    return pd.read_csv(Path(cache_dir) / INDEX_NAME)


def load_video(cache_dir: Path, video: str) -> np.ndarray:
    """``(T_uniform, 2, n_cols)`` float16, NaN on gaps and holes."""
    return np.load(Path(cache_dir) / f"{video_key(video)}.npz")["x"]


class SiNCClipDataset(Dataset):
    """Random raw windows, long enough to be resampled to ``clip_len`` at any
    speed up to ``max_speed`` (resampling itself happens on the GPU).

    Every video is held in memory (~4 MB each). One epoch is
    ``samples_per_epoch`` draws; windows with more than ``max_gap_frac`` of
    gap samples are redrawn.
    """

    def __init__(
        self,
        cache_dir: Path,
        index: pd.DataFrame,
        clip_len: int = 870,
        max_speed: float = 1.4,
        max_gap_frac: float = 0.3,
        samples_per_epoch: int = 4096,
        seed: int | None = None,
    ):
        self.clip_len = clip_len
        self.raw_len = int(np.ceil(clip_len * max_speed)) + 2
        self.max_gap_frac = max_gap_frac
        self.samples_per_epoch = samples_per_epoch
        self.rng = np.random.default_rng(seed)

        self.videos, self.dts, self.data = [], [], []
        for row in index.itertuples():
            if row.n_uniform < self.raw_len:
                continue
            self.videos.append(row.video)
            self.dts.append(row.dt)
            self.data.append(load_video(cache_dir, row.video))

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, _):
        for _attempt in range(50):
            v = int(self.rng.integers(len(self.data)))
            x = self.data[v]
            s = int(self.rng.integers(len(x) - self.raw_len + 1))
            w = x[s : s + self.raw_len]
            gap = np.isnan(w).all(axis=(1, 2))
            if gap.mean() <= self.max_gap_frac:
                break
        return torch.from_numpy(w.astype(np.float32)), torch.tensor(
            self.dts[v], dtype=torch.float32
        )


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker its own random stream."""
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed % 2**32)
