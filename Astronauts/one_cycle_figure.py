# -*- coding: utf-8 -*-
"""
one_cycle_figure.py

Figure des one-cycles (``compute_one_cycle_plex.py``), par condition et par pouls :

    tous les one-cycles bout a bout sur l'axe des bins (presentation des notebooks) :
    nombre de frames par bin en barres grises (axe de droite), epaisseur de la choroide
    (um) des masques REPLIES par cycle + fondamental en tirets, et du one-cycle
    RESEGMENTE en pointille noir

C'est aussi le SEUL code de lecture des ``.npz`` de one-cycle : la page « One-cycle »
du GUI (``data_visualizer_GUI/pages/one_cycle.py``) passe par ``load_one_cycle`` et
``bin_table``, si bien que la page et la figure montrent les memes chiffres.

Figure interactive Plotly (regle CLAUDE.md), ecrite en HTML autonome.

Lancer (kernel pyOR, depuis la racine) :
    python Astronauts/one_cycle_figure.py --condition 210713001before_rigidity_OD --pulse ct
    python Astronauts/one_cycle_figure.py --condition 210713001before_rigidity_OD --pulse svd --cycle 2
    python Astronauts/one_cycle_figure.py --all          # toutes les conditions calculees, deux pouls
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

VARIANT_ROOT = Path("E:/NASA_Rigidity/SegmentationVariations/cascade_v9_plex_full")
# Sorties de compute_one_cycle_plex.py (6 battements par cycle) et de
# compute_one_cycle_plex_groups.py (1 ou 3 one-cycles par video).
VARIANTS = {"6beats": "one_cycle_6beats_10bins", "1per_video": "one_cycle_1per_video_10bins",
            "3per_video": "one_cycle_3per_video_10bins"}
ROOT = VARIANT_ROOT / VARIANTS["6beats"]
CYCLES_DIR = ROOT / "cycles"


def cycles_dir(variant: str = "6beats") -> Path:
    return VARIANT_ROOT / VARIANTS[variant] / "cycles"
FIG_DIR = ROOT / "figures"
PULSE_LABELS = {"ct": "pouls d'epaisseur (masques)", "svd": "pouls d'intensite (SVD)",
                "sinc": "pouls SiNC (appris)"}
COLORS = {"fold": "#d62728", "seg": "#1f77b4", "count": "#7f7f7f"}


def npz_path(astro: str, condition: str, pulse: str, root: Path = CYCLES_DIR) -> Path:
    return Path(root) / f"{astro}__{condition}__{pulse}.npz"


def find_npz(condition: str, pulse: str, root: Path = CYCLES_DIR) -> Path | None:
    hits = sorted(Path(root).glob(f"*__{condition}__{pulse}.npz"))
    return hits[0] if hits else None


def load_one_cycle(path, images: bool = True) -> dict:
    """Charge un ``.npz`` de one-cycle. ``images=False`` saute les cubes (rapide)."""
    d = np.load(path)
    out = {k: d[k] for k in d.files
           if k not in ("cycles", "occupancy", "seg_masks_packed", "seg_probs",
                         "reg_frames", "reg_masks_packed", "reg_probs")}
    for k in ("n_bins", "n_cycle", "n_groups"):
        out[k] = int(out[k])
    for k in ("pulse", "condition", "astro", "data_subdir", "pulse_info", "sinc_confidence"):
        if k in out:
            out[k] = str(out[k])
    if images:
        out["cycles"] = d["cycles"]
        out["occupancy"] = d["occupancy"].astype(np.float32) / 255.0
        shape = tuple(int(v) for v in d["seg_shape"])
        out["seg_masks"] = np.unpackbits(d["seg_masks_packed"])[: int(np.prod(shape))].reshape(shape).astype(bool)
        out["seg_probs"] = d["seg_probs"]
        if "reg_frames" in d.files:  # resegmentation cascade_v9 : one-cycle recale
            out["reg_frames"] = d["reg_frames"]
            out["reg_masks"] = np.unpackbits(d["reg_masks_packed"])[: int(np.prod(shape))].reshape(shape).astype(bool)
    return out


def bin_table(data: dict, cycle: int | None = None) -> pd.DataFrame:
    """Une ligne par (cycle, bin). ``cycle`` (1..n_groups) restreint a un one-cycle."""
    nb, ng = data["n_bins"], data["n_groups"]
    k = np.arange(nb * ng)
    df = pd.DataFrame({
        "slot": k, "cycle": k // nb + 1, "bin": k % nb, "n_frames": data["counts"],
        "thickness_fold_um": data["thickness_fold"], "thickness_fold_fit_um": data["thickness_fold_fit"],
        "thickness_seg_um": data["thickness_seg"], "thickness_seg_fit_um": data["thickness_seg_fit"],
        "dice": data["dice"]})
    if cycle is not None:
        df = df[df["cycle"] == cycle]
    return df.reset_index(drop=True)


def title_of(data: dict, cycle: int | None) -> str:
    which = f"cycle {cycle}/{data['n_groups']}" if cycle else f"{data['n_groups']} cycles superposes"
    return (f"{data['condition']} -- {PULSE_LABELS.get(data['pulse'], data['pulse'])} -- {which} "
            f"({data['n_cycle']} battements x {data['n_bins']} bins, FC {float(data['hr']):.0f} BPM)")


def cycle_colors(n: int) -> list[str]:
    """Une couleur par cycle (meme palette dans la figure et dans le GUI)."""
    import colorsys

    return ["#%02x%02x%02x" % tuple(int(255 * c) for c in colorsys.hsv_to_rgb(i / max(n, 1), 0.75, 0.8))
            for i in range(n)]


def build_figure(data: dict, cycle: int | None = None):
    """Figure Plotly, presentation de ``notebook/demons_sitk_variants.ipynb`` : tous les
    one-cycles BOUT A BOUT sur l'axe des bins ; en fond, le nombre de frames par bin
    (barres grises, axe de droite) ; par-dessus, l'epaisseur des masques REPLIES (une
    couleur par cycle, son fondamental en tirets) et celle du one-cycle RESEGMENTE
    (pointille noir). ``cycle`` restreint a un seul one-cycle."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = bin_table(data, cycle)
    nb, ng = data["n_bins"], data["n_groups"]
    colors = cycle_colors(ng)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df["slot"], y=df["n_frames"], name="frames par bin",
                         marker=dict(color="lightgray", line=dict(width=0)), opacity=0.6,
                         customdata=df["bin"],
                         hovertemplate="bin %{customdata} : %{y} frames<extra></extra>"),
                  secondary_y=True)
    for c in sorted(df["cycle"].unique()):
        sub = df[df["cycle"] == c]
        lab, col = f"cycle {c}", colors[c - 1]
        fig.add_trace(go.Scatter(x=sub["slot"], y=sub["thickness_fold_um"], mode="lines+markers",
                                 name=lab, legendgroup=lab, line=dict(color=col, width=2),
                                 marker=dict(size=6), customdata=sub["bin"],
                                 hovertemplate=lab + ", bin %{customdata}<br>%{y:.1f} um (replie)<extra></extra>"),
                      secondary_y=False)
        fig.add_trace(go.Scatter(x=sub["slot"], y=sub["thickness_fold_fit_um"], mode="lines",
                                 name=lab, legendgroup=lab, showlegend=False,
                                 line=dict(color=col, width=1.5, dash="dash"),
                                 hovertemplate=lab + "<br>%{y:.1f} um (fondamental)<extra></extra>"),
                      secondary_y=False)
        fig.add_trace(go.Scatter(x=sub["slot"], y=sub["thickness_seg_um"], mode="lines+markers",
                                 name="one-cycle resegmente", legendgroup="seg",
                                 showlegend=bool(c == df["cycle"].min()),
                                 line=dict(color="black", width=1.5, dash="dot"),
                                 marker=dict(size=5, symbol="square"), customdata=sub["bin"],
                                 hovertemplate=lab + ", bin %{customdata}<br>%{y:.1f} um (resegmente)<extra></extra>"),
                      secondary_y=False)
        if c > df["cycle"].min():
            fig.add_vline(x=(c - 1) * nb - 0.5, line=dict(color="gray", dash="dot", width=1))
    fig.update_yaxes(title_text="epaisseur (um) -- plein : masques replies, tirets : fondamental, "
                                "pointille noir : one-cycle resegmente", secondary_y=False)
    fig.update_yaxes(title_text="frames par bin", secondary_y=True, showgrid=False, rangemode="tozero")
    fig.update_xaxes(title_text=f"bin ({ng} one-cycles de {nb} bins, bout a bout ; "
                                "bin 0 de chaque cycle = maximum d'epaisseur)")
    fig.update_layout(title=title_of(data, cycle), template="plotly_white", height=480,
                      barmode="overlay", legend=dict(groupclick="togglegroup"))
    return fig



def write_figure(path_npz: Path, cycle: int | None = None, out_dir: Path = FIG_DIR) -> Path:
    data = load_one_cycle(path_npz, images=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"__cycle{cycle}" if cycle else ""
    out = out_dir / f"{Path(path_npz).stem}{suffix}.html"
    build_figure(data, cycle).write_html(out, include_plotlyjs="cdn")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Figure des one-cycles (frames et epaisseur par bin).")
    ap.add_argument("--condition", help="ex. 210713001before_rigidity_OD")
    ap.add_argument("--pulse", choices=list(PULSE_LABELS), default="ct")
    ap.add_argument("--cycle", type=int, default=None, help="1..n ; absent = tous superposes")
    ap.add_argument("--all", action="store_true", help="toutes les conditions calculees")
    args = ap.parse_args(argv)
    if args.all:
        paths = sorted(CYCLES_DIR.glob("*.npz"))
    elif args.condition:
        p = find_npz(args.condition, args.pulse)
        if p is None:
            print(f"non calcule : {args.condition} ({args.pulse})")
            return 1
        paths = [p]
    else:
        ap.error("--condition ou --all")
    for p in paths:
        print(write_figure(p, args.cycle))
    return 0


if __name__ == "__main__":
    sys.exit(main())
