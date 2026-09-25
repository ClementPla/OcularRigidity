"""Per-video inspection of a trained SiNC model against the thickness it reads.

For one video, :func:`load_view` gathers on the uniform grid:

* the source thickness — the stored ``interpolated_signal`` (the chain's
  ``col_slice``), each column's slow baseline removed, and its column median
  (robust to the few columns whose segmentation breaks), lightly smoothed for
  display;
* the SiNC waveform, its rate, and its IQ phase anchored on minimal thickness;
* the trace the stored chain picked (``filtered_signal[:, best_component_idx]``,
  HR-anchored bandpass), its rate, and its stored phase, anchored the same way.

:func:`plot_view` draws them; :func:`browser` wraps both in patient/video
dropdowns and a time window for a notebook.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.timeseries import LombScargle
from scipy.ndimage import gaussian_filter1d

from ocularrigidity.motion.pulsation.phase import (
    IQDemodPhaseEstimator,
    thickness_minimum_phase,
)
from ocularrigidity.motion.pulsation.rate import RateEstimate
from ocularrigidity.motion.pulsation.sinc.data import load_index, load_video
from ocularrigidity.motion.pulsation.sinc.losses import peak_frequency
from ocularrigidity.motion.pulsation.sinc.module import SiNCModule
from ocularrigidity.motion.pulsation.sinc.preprocess import boundaries_to_uniform
from ocularrigidity.motion.pulsation.traces import Traces

# Categorical slots 1–3 of the reference palette (validated: CVD ΔE ≥ 9.2).
# Aqua is under 3:1 on white, so every series also has its own dash + legend.
COLORS = dict(thickness="#2a78d6", sinc="#eb6834", original="#1baf7a")
STYLES = dict(thickness="-", sinc="-", original="--")
LABELS = dict(
    thickness="Thickness (column median, smoothed)",
    sinc="SiNC output",
    original="Chain's chosen trace",
)
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


@dataclass
class VideoView:
    video: str
    split: str
    hr: float  # measured, may be NaN
    time: np.ndarray  # (T_u,) s
    gap: np.ndarray  # (T_u,) bool
    thickness_dev: np.ndarray  # (T_u, W) px, baseline removed, smoothed, NaN on gaps
    thickness_raw: np.ndarray  # (T_u,) px, column median, unsmoothed
    thickness: np.ndarray  # (T_u,) px, column median, smoothed
    sinc: np.ndarray  # (T_u,) z-scored, sign-aligned to thickness, NaN on gaps
    sinc_flipped: bool
    sinc_bpm: float
    sinc_phase: np.ndarray  # (T_u,) anchored, radians, NaN where not good
    sinc_anchor: dict
    original: np.ndarray  # (T_u,) z-scored
    original_column: int  # index into the thickness map's columns
    pipeline_bpm: float
    pipeline_confidence: str
    pipeline_phase: np.ndarray  # (T_u,) stored phase, anchored
    pipeline_anchor: dict

    @property
    def dt(self) -> float:
        return float(self.time[1] - self.time[0])


def _zscore(x: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x)
    return (x - np.nanmean(x[ok])) / (np.nanstd(x[ok]) + 1e-12)


def _detrend(th: np.ndarray, sigma: float) -> np.ndarray:
    """Remove each column's slow baseline (NaN-aware Gaussian of ``sigma`` samples)."""
    valid = np.isfinite(th)
    num = gaussian_filter1d(np.where(valid, th, 0.0), sigma, axis=0, mode="nearest")
    den = gaussian_filter1d(valid.astype(float), sigma, axis=0, mode="nearest")
    return th - num / np.where(den > 0, den, np.nan)


def _smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    """NaN-aware Gaussian smoothing along time; NaN stays NaN."""
    valid = np.isfinite(x)
    num = gaussian_filter1d(np.where(valid, x, 0.0), sigma, axis=0, mode="nearest")
    den = gaussian_filter1d(valid.astype(float), sigma, axis=0, mode="nearest")
    return np.where(valid, num / np.where(den > 0, den, 1.0), np.nan)


def _anchored(phase_u, good_u, thickness, dt, freq):
    a = thickness_minimum_phase(phase_u, good_u, thickness, dt, freq)
    shifted = np.mod(phase_u - a["offset"], 2 * np.pi)
    return np.where(good_u, shifted, np.nan), a


def _video_info(video: str, cache_dir: Path | None) -> dict | None:
    """The training index row for ``video``, or None if it is not in the cache."""
    if cache_dir is None or not (Path(cache_dir) / "index.csv").exists():
        return None
    index = load_index(cache_dir).set_index("video")
    return index.loc[video].to_dict() if video in index.index else None


def _expected_bpm(m) -> float:
    """The HR the chain was anchored on, if any (NaN otherwise)."""
    rate = m.config.get("rate") if isinstance(m.config, dict) else None
    bpm = getattr(getattr(rate, "band", None), "expected_bpm", None)
    return float(bpm) if bpm is not None else float("nan")


def load_view(
    video: str,
    module: SiNCModule,
    cache_dir: Path | None,
    measures_root: Path,
    phase_config=None,
) -> VideoView:
    """``video`` is the path of its folder relative to ``measures_root``.

    Videos in the training cache reuse its input and report their split and
    measured HR. Any other ``measure.pkl`` (another study, say) is run from the
    pickle itself, through the same preprocessing; its split is "external" and
    its HR is the one the chain was anchored on, if any.
    """
    info = _video_info(video, cache_dir)
    with open(Path(measures_root) / video / "measure.pkl", "rb") as f:
        m = pickle.load(f)

    t = np.asarray(m.uniform_time)
    t = t - t[0]
    dt = float(t[1] - t[0])
    gap = np.asarray(m.gap_mask, bool)
    th = np.asarray(m.interpolated_signal, float)

    # SiNC: waveform on the same uniform grid, rate from its in-band peak.
    if info is not None:
        x = load_video(cache_dir, video)
    else:
        x = boundaries_to_uniform(
            m.registered_boundaries, m.timestamps_seconds, m.uniform_time, m.gap_mask
        )
    if len(x) != len(t):
        raise ValueError(f"cache has {len(x)} samples, measure.pkl {len(t)}")
    module.eval()
    y, mask, tt = module.predict_video(x, dt)
    f_sinc = float(peak_frequency(y[None], tt[None], mask[None])[0][0])
    kept = mask.cpu().numpy() & ~gap
    sinc = np.where(kept, y.float().cpu().numpy(), np.nan)

    period = (1.0 / f_sinc) / dt  # samples
    th_dev = _detrend(th, sigma=period)
    with np.errstate(all="ignore"):
        th_raw = np.nanmedian(th_dev, axis=1)
    th_raw[gap] = np.nan
    # Display smoothing: σ = 1/12 cycle keeps the pulse, drops frame jitter.
    th_mean = _smooth(th_raw, period / 12)

    sinc_z = _zscore(sinc)
    ok = np.isfinite(sinc_z) & np.isfinite(th_mean)
    flipped = bool(np.corrcoef(sinc_z[ok], th_mean[ok])[0, 1] < 0)
    if flipped:
        sinc_z = -sinc_z

    # SiNC phase: IQ demodulation at its own rate, then thickness anchoring.
    traces = Traces(
        values=sinc[kept][:, None],
        uniform_time=np.asarray(m.uniform_time),
        kept_mask=kept,
        gap_mask=gap,
        timestamps_seconds=np.asarray(m.timestamps_seconds),
        source_map=sinc[:, None],
    )
    if phase_config is None:
        from ocularrigidity.pipeline_config import PULSATION

        phase_config = PULSATION.chain.phase
    track = IQDemodPhaseEstimator(phase_config).estimate(
        traces, RateEstimate(freq=f_sinc)
    )
    sinc_phase, sinc_anchor = _anchored(
        track.phase_uniform, track.good_uniform, th, dt, f_sinc
    )

    best = int(m.best_component_idx)
    original = np.asarray(m.filtered_signal, float)[:, best]
    pipe_phase, pipe_anchor = _anchored(
        np.asarray(m.phase_uniform),
        np.asarray(m.good_uniform, bool),
        th,
        dt,
        float(m.cardiac_freq),
    )

    return VideoView(
        video=video,
        split=str(info["split"]) if info is not None else "external",
        hr=float(info["HR"]) if info is not None else _expected_bpm(m),
        time=t,
        gap=gap,
        thickness_dev=_smooth(th_dev, period / 12),
        thickness_raw=th_raw,
        thickness=th_mean,
        sinc=sinc_z,
        sinc_flipped=flipped,
        sinc_bpm=f_sinc * 60.0,
        sinc_phase=sinc_phase,
        sinc_anchor=sinc_anchor,
        original=_zscore(original),
        original_column=best if th.shape[1] == m.filtered_signal.shape[1] else -1,
        pipeline_bpm=float(m.cardiac_freq) * 60.0,
        pipeline_confidence=str(m.confidence),
        pipeline_phase=pipe_phase,
        pipeline_anchor=pipe_anchor,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def _shade_gaps(ax, t, gap, lo, hi):
    sel = (t >= lo) & (t <= hi)
    g = gap & sel
    edges = np.flatnonzero(np.diff(np.r_[0, g.astype(int), 0]))
    for s, e in zip(edges[::2], edges[1::2]):
        ax.axvspan(t[s], t[min(e, len(t) - 1)], color=GRID, alpha=0.8, lw=0)


def _spectrum(t, x, freqs):
    ok = np.isfinite(x)
    p = LombScargle(t[ok], x[ok]).power(freqs)
    return p / p.max()


def _fold(phase, x, n_bins=30):
    ok = np.isfinite(phase) & np.isfinite(x)
    b = (phase[ok] / (2 * np.pi) * n_bins).astype(int) % n_bins
    v = x[ok]
    med = np.array(
        [np.median(v[b == k]) if (b == k).any() else np.nan for k in range(n_bins)]
    )
    q1 = np.array(
        [
            np.percentile(v[b == k], 25) if (b == k).any() else np.nan
            for k in range(n_bins)
        ]
    )
    q3 = np.array(
        [
            np.percentile(v[b == k], 75) if (b == k).any() else np.nan
            for k in range(n_bins)
        ]
    )
    centers = (np.arange(n_bins) + 0.5) / n_bins
    return centers, med, q1, q3


def plot_view(view: VideoView, start: float = 0.0, window: float = 10.0):
    """Overlay, thickness map, spectra and folded thickness for one video."""
    t = view.time
    lo, hi = start, min(start + window, t[-1])
    sel = (t >= lo) & (t <= hi)

    fig = plt.figure(figsize=(13, 10.5), facecolor="white")
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.0, 1.1], hspace=0.45, wspace=0.22)
    ax_o = fig.add_subplot(gs[0, :])
    ax_h = fig.add_subplot(gs[1, :], sharex=ax_o)
    ax_s = fig.add_subplot(gs[2, 0])
    ax_f = fig.add_subplot(gs[2, 1])

    has_hr = np.isfinite(view.hr)
    hr_name = "HR given to chain" if view.split == "external" else "HR measured"
    hr = f"{hr_name} {view.hr:.0f} bpm" if has_hr else "no HR"
    anchored = "HR-anchored" if has_hr else "open band"
    fig.suptitle(
        f"{view.video}  ({view.split})\n{hr} · SiNC {view.sinc_bpm:.1f} bpm · chain "
        f"{view.pipeline_bpm:.1f} bpm ({view.pipeline_confidence}, {anchored})",
        x=0.06, ha="left", fontsize=11.5, color=INK, linespacing=1.5,
    )  # fmt: skip

    # 1. Overlay (z-scores on one axis). The raw column median sits faintly
    # behind its smoothed version, on the smoothed version's scale.
    _shade_gaps(ax_o, t, view.gap, lo, hi)
    mu, sd = np.nanmean(view.thickness), np.nanstd(view.thickness) + 1e-12
    ax_o.plot(
        t[sel], ((view.thickness_raw - mu) / sd)[sel], "-",
        color=COLORS["thickness"], lw=0.6, alpha=0.3,
    )  # fmt: skip
    bold = {
        "thickness": (view.thickness - mu) / sd,
        "sinc": view.sinc,
        "original": view.original,
    }
    for key, y in bold.items():
        lw = 2.4 if key == "thickness" else 1.6
        ax_o.plot(
            t[sel], y[sel], STYLES[key], color=COLORS[key], lw=lw, label=LABELS[key]
        )
    span = np.nanpercentile(
        np.abs(np.concatenate([y[sel] for y in bold.values()])), 99.5
    )
    ax_o.set_ylim(-1.25 * span, 1.35 * span)
    flip = " (sign flipped to match thickness)" if view.sinc_flipped else ""
    ax_o.set_title(
        f"Signals, z-scored · grey = gaps{flip}", loc="left", fontsize=10, color=MUTED
    )
    ax_o.set_ylabel("z-score")
    ax_o.legend(loc="upper right", ncols=3, fontsize=9, frameon=False)
    _style(ax_o)

    # 2. Thickness map, baseline removed (diverging, 0 = column baseline).
    dev = view.thickness_dev[sel]
    # Colour range from typical columns: a few broken ones swing ±70 px.
    with np.errstate(all="ignore"):
        col_sd = np.nanstd(dev, axis=0)
    lim = 2.5 * np.nanmedian(col_sd) if np.isfinite(col_sd).any() else 1.0
    ax_h.imshow(
        dev.T, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower",
        extent=[t[sel][0], t[sel][-1], 0, dev.shape[1]], interpolation="nearest",
    )  # fmt: skip
    if view.original_column >= 0:
        W = dev.shape[1]
        below = view.original_column > 0.85 * W
        ax_h.axhline(view.original_column + 0.5, color=INK, lw=1, ls="--")
        ax_h.text(
            t[sel][0], view.original_column + (-4 if below else 4),
            " chain's chosen column", color=INK, fontsize=8,
            va="top" if below else "bottom",
        )  # fmt: skip
    ax_h.set_title(
        f"Thickness − column baseline, smoothed (px, red = thicker, colour clipped at ±{lim:.1f} px)",
        loc="left", fontsize=10, color=MUTED,
    )  # fmt: skip
    ax_h.set_ylabel("column (in col_slice)")
    ax_h.set_xlabel("time (s)")
    _style(ax_h)
    ax_h.grid(False)
    ax_o.set_xlim(lo, hi)

    # 3. Spectra over the whole video.
    freqs = np.linspace(0.5, 3.0, 1000)
    for key, y in (
        ("thickness", view.thickness_raw),
        ("sinc", view.sinc),
        ("original", view.original),
    ):
        ax_s.plot(
            freqs * 60,
            _spectrum(t, y, freqs),
            STYLES[key],
            color=COLORS[key],
            lw=1.6,
            label=LABELS[key],
        )
    if np.isfinite(view.hr):
        ax_s.axvline(view.hr, color=INK, lw=1, ls=":")
        ax_s.text(
            view.hr, 1.02, " HR", color=INK, fontsize=8, ha="left", va="bottom"
        )
    ax_s.set_title(
        "Periodogram, whole video (peak = 1)", loc="left", fontsize=10, color=MUTED
    )
    ax_s.set_xlabel("rate (bpm)")
    ax_s.set_ylabel("normalised power")
    ax_s.set_ylim(0, 1.12)
    ax_s.legend(loc="upper right", fontsize=8, frameon=False)
    _style(ax_s)

    # 4. Thickness folded on each anchored phase (median, IQR band).
    for key, phase, a in (
        ("sinc", view.sinc_phase, view.sinc_anchor),
        ("original", view.pipeline_phase, view.pipeline_anchor),
    ):
        c, med, q1, q3 = _fold(phase, view.thickness)
        name = "SiNC phase" if key == "sinc" else "chain phase"
        ax_f.fill_between(c, q1, q3, color=COLORS[key], alpha=0.15, lw=0)
        ax_f.plot(
            c, med, STYLES[key], color=COLORS[key], lw=2,
            label=f"{name} · depth {a['depth']:.2f} px · coherence {a['coherence']:.2f}",
        )  # fmt: skip
    ax_f.set_title(
        "Thickness folded on phase anchored at its minimum (median, IQR)",
        loc="left", fontsize=10, color=MUTED,
    )  # fmt: skip
    ax_f.set_xlabel("cardiac phase (cycles)")
    ax_f.set_ylabel("thickness − baseline (px)")
    ax_f.set_xlim(0, 1)
    ax_f.legend(loc="upper right", fontsize=8, frameon=False)
    _style(ax_f)
    return fig


# ---------------------------------------------------------------------------
# Notebook browser
# ---------------------------------------------------------------------------
def browser(module: SiNCModule, measures_root: Path, cache_dir: Path | None = None):
    """Patient → video dropdowns and a time window; views are cached per video.

    Lists every ``measure.pkl`` under ``measures_root`` (``patient/…/measure.pkl``).
    With ``cache_dir`` (the training cache), labels also show the split and
    measured HR of videos the cache knows.
    """
    import ipywidgets as w
    from IPython.display import display

    measures_root = Path(measures_root)
    videos = sorted(
        p.parent.relative_to(measures_root).as_posix()
        for p in measures_root.rglob("measure.pkl")
    )
    if not videos:
        raise FileNotFoundError(f"no measure.pkl under {measures_root}")
    index = None
    if cache_dir is not None and (Path(cache_dir) / "index.csv").exists():
        index = load_index(cache_dir).set_index("video")
    patients = sorted({v.split("/")[0] for v in videos})
    cache: dict[str, VideoView] = {}

    def video_options(patient):
        out = []
        for v in videos:
            if v.split("/")[0] != patient:
                continue
            label = "/".join(v.split("/")[1:])
            if index is not None and v in index.index:
                r = index.loc[v]
                hr = f"HR {r.HR:.0f}" if np.isfinite(r.HR) else "no HR"
                label += f"  [{r.split}, {hr}]"
            out.append((label, v))
        return out

    patient = w.Dropdown(options=patients, description="Patient")
    video = w.Dropdown(
        options=video_options(patients[0]),
        description="Video",
        layout=w.Layout(width="720px"),
    )
    start = w.FloatSlider(
        value=0.0,
        min=0.0,
        max=60.0,
        step=0.5,
        description="Start (s)",
        continuous_update=False,
    )
    window = w.FloatSlider(
        value=10.0,
        min=2.0,
        max=90.0,
        step=1.0,
        description="Window (s)",
        continuous_update=False,
    )
    out = w.Output()

    def on_patient(change):
        video.options = video_options(change["new"])

    def redraw(*_):
        v = video.value
        if v is None:
            return
        with out:
            out.clear_output(wait=True)
            if v not in cache:
                print(f"loading {v} …")
                cache[v] = load_view(v, module, cache_dir, measures_root)
                out.clear_output(wait=True)
            view = cache[v]
            start.max = max(float(view.time[-1]) - 2.0, 0.0)
            fig = plot_view(view, start.value, window.value)
            display(fig)
            plt.close(fig)

    patient.observe(on_patient, names="value")
    for c in (video, start, window):
        c.observe(redraw, names="value")
    display(w.VBox([w.HBox([patient, video]), w.HBox([start, window]), out]))
    redraw()
