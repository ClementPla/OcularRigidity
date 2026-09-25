"""Rate estimates for comparison: SiNC versus the current chain.

The stored ``measure.pkl`` rate is not a fair baseline wherever an HR was
measured: the chain's search band is clamped to ±30 % of that HR, so it cannot
miss by more. ``open_band_bpm`` reruns the same chain (same stage configs, same
thickness) with no expected BPM, which is the question SiNC has to answer.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ocularrigidity.motion.pulsation.sinc.data import load_index, load_video
from ocularrigidity.motion.pulsation.sinc.module import SiNCModule


class _MeasureStub:
    """Just enough of a VideoRegistrator for the aligner and mask source."""

    skip_first_n_frames = 0
    drop_last_n_frames = 0

    def __init__(self, boundaries: np.ndarray):
        self.thickness = (boundaries[:, 1] - boundaries[:, 0]).astype(np.float32)


def open_band_bpm(measure_path: Path) -> float:
    """The composed chain's rate on one video, with no expected BPM."""
    from ocularrigidity.motion.pulsation.rate import LombScargleRateEstimator
    from ocularrigidity.motion.pulsation.traces import (
        BandPassFilterTraceSource,
        MaskThicknessTraceSource,
    )
    from ocularrigidity.motion.video_timeline_aligner import (
        TimeUnits,
        VideoTimelineAligner,
    )
    from ocularrigidity.pipeline_config import PULSATION

    with open(measure_path, "rb") as f:
        m = pickle.load(f)
    stub = _MeasureStub(np.asarray(m.registered_boundaries))
    aligner = VideoTimelineAligner(
        stub, m.timestamps_seconds, units_in_timestamps=TimeUnits.SECONDS
    )
    stages = PULSATION.chain.for_video(expected_bpm=None, verbose=False)
    source = MaskThicknessTraceSource(stub, aligner, stages["trace"])
    source = BandPassFilterTraceSource(source, stages["bandpass"])
    return LombScargleRateEstimator(stages["rate"]).estimate(source.traces).bpm


def open_band_table(
    cache_dir: Path, measures_root: Path, split: str = "val", out: Path | None = None
) -> pd.DataFrame:
    """Open-band chain rates for one split; cached to ``out`` if given."""
    if out is not None and Path(out).exists():
        return pd.read_csv(out)
    index = load_index(cache_dir)
    index = index[index["split"] == split]
    rows = []
    for video in tqdm(index["video"], desc="open-band chain"):
        try:
            bpm = open_band_bpm(Path(measures_root) / video / "measure.pkl")
        except Exception as e:
            print(f"skip {video}: {e}")
            bpm = np.nan
        rows.append(dict(video=video, open_band_bpm=bpm))
    df = pd.DataFrame(rows)
    if out is not None:
        df.to_csv(out, index=False)
    return df


def sinc_table(module: SiNCModule, cache_dir: Path, split: str = "val") -> pd.DataFrame:
    """SiNC rate for every video of ``split``."""
    index = load_index(cache_dir)
    index = index[index["split"] == split]
    module.eval()
    rows = [
        dict(
            video=r.video,
            sinc_bpm=module.estimate_bpm(load_video(cache_dir, r.video), r.dt),
        )
        for r in tqdm(index.itertuples(), total=len(index), desc="SiNC")
    ]
    return pd.DataFrame(rows)
