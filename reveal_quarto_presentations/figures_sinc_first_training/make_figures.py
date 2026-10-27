# -*- coding: utf-8 -*-
"""
Figures et tables -- page « First training » (section Sinc on intensity).

Ce script NE LIT QUE des sorties deja ecrites ; il ne recadre rien, n'entraine rien.

Entrees
-------
    Astronauts/pulsation/sinc_intensity/datasets/metadata/OCT.csv   videos retenues + roles
    E:/NASA_Rigidity/Reproducibility/choroid_crops/summary.csv      toutes les videos recadrees
    E:/SANSORI/Reproducibility/choroid_crops/<SLUG>/crop.npz         crops + corrections
    E:/NASA_Rigidity/Reproducibility/sinc_intensity/first_training/fold0_seed0/
        arg_obj.txt, losses_train.csv, losses_val.csv               entrainement (train.py)
        predictions/predictions.csv, predictions/<slug>.npz          evaluation (evaluate_oct.py)
    ../train.log                                                    journal du lancement

Sorties (fragments Quarto, inclus par ``sinc-first-training.qmd``)
-----------------------------------------------------------------
    tab_videos.qmd          videos retenues (role, cadence) et videos ecartees
    fig_crop_moyennes.qmd   fenetre moyenne avant / apres correction, crop, patchs (menu par video)
    fig_corrections.qmd     gain par A-scan a_x et gain par frame alpha_t
    fig_crop_videos.qmd     les crops en video (.mp4)
    tab_entrainement.qmd    reglages, echantillons, duree, meilleure epoque
    fig_losses.qmd          pertes d'entrainement et de validation
    fig_hr.qmd              FC SiNC contre FC de reference, 23 videos
    fig_test_ondes.qmd      GRAZIANA : onde SiNC, pouls d'epaisseur, spectres
    tab_test.qmd            GRAZIANA : chiffres
    plotlyjs.html           plotly.js, insere une seule fois par page
    resume.txt              les chiffres cites dans la prose

Lancer DEPUIS LA RACINE DU DEPOT (ce dossier est une JONCTION vers E:) :
    python reveal_quarto_presentations/figures_sinc_first_training/make_figures.py
"""

import base64
import io
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots
from PIL import Image

# --------------------------------------------------------------------------- #
# Entrees / sorties
# --------------------------------------------------------------------------- #
REPO = Path(__file__).absolute().parents[2]
META = REPO / "Astronauts/pulsation/sinc_intensity/datasets/metadata/OCT.csv"
SUMMARY = Path("E:/NASA_Rigidity/Reproducibility/choroid_crops/summary.csv")
EXP_ROOT = Path("E:/NASA_Rigidity/Reproducibility/sinc_intensity/first_training")
EXP = EXP_ROOT / "fold0_seed0"
PRED = EXP / "predictions"

# `.absolute()` et surtout PAS `.resolve()` : ce dossier est une JONCTION vers E:.
SORTIE = Path(__file__).absolute().parent
DOSSIER = "figures_sinc_first_training"

COULEUR_TEXTE = "#c9c9c9"
GRILLE = "rgba(150,150,150,0.18)"
C_SPLIT = {"train": "#2a78d6", "val": "#e08a1e", "test": "#27a567"}
LIB_SPLIT = {"train": "entraînement", "val": "validation", "test": "test"}
C_LOSS = {"total": "#c9c9c9", "bandwidth": "#2a78d6", "sparsity": "#e08a1e", "variance": "#9b59b6"}
STRETCH_VIDEO = 3  # etirement vertical des videos de crops
PATCH = 64

CONFIG = {"displaylogo": False, "responsive": True,
          "modeBarButtonsToRemove": ["select2d", "lasso2d"]}

FICHIERS: list[str] = []
RESUME: dict = {}


# --------------------------------------------------------------------------- #
# Habillage commun (le meme que figures_repro_cascade)
# --------------------------------------------------------------------------- #
def mise_en_page(fig, titre, hauteur=440, legende=True):
    fig.update_layout(
        title={"text": titre, "font": {"size": 14}, "x": 0.01, "xanchor": "left"},
        template="none",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COULEUR_TEXTE, "size": 11},
        height=hauteur,
        margin={"l": 60, "r": 20, "t": 80, "b": 80 if legende else 50},
        showlegend=legende,
        legend={"font": {"size": 10}, "orientation": "h", "yanchor": "top",
                "y": -0.12, "xanchor": "left", "x": 0},
        hoverlabel={"font": {"size": 11}},
    )
    fig.update_xaxes(gridcolor=GRILLE, zerolinecolor=GRILLE,
                     linecolor=GRILLE, ticks="outside", tickcolor=GRILLE)
    fig.update_yaxes(gridcolor=GRILLE, zerolinecolor=GRILLE,
                     linecolor=GRILLE, ticks="outside", tickcolor=GRILLE)
    for a in fig.layout.annotations or ():
        if a.text and a.font is not None:
            a.font.size = 11
            a.font.color = COULEUR_TEXTE
    return fig


def enregistrer(fig, nom):
    html = fig.to_html(full_html=False, include_plotlyjs=False,
                       config=CONFIG, div_id=f"plot-{nom}")
    (SORTIE / f"{nom}.qmd").write_text("```{=html}\n" + html + "\n```\n",
                                       encoding="utf-8")
    FICHIERS.append(f"{nom}.qmd")
    print(f"  {nom}.qmd")


def ecrire_table(texte, nom):
    entete = ("<!-- Genere par figures_sinc_first_training/make_figures.py"
              " -- ne pas editer a la main. -->" + chr(10) + chr(10))
    (SORTIE / f"{nom}.qmd").write_text(entete + texte + chr(10), encoding="utf-8")
    FICHIERS.append(f"{nom}.qmd")
    print(f"  {nom}.qmd")


def absente(nom, pourquoi):
    """Un ``{{< include >}}`` sur un fichier absent fait ECHOUER le rendu du site
    entier : une figure impossible laisse donc un encadre lisible a sa place."""
    ecrire_table(f"::: {{.callout-warning}}{chr(10)}"
                 f"## Figure indisponible{chr(10)}{chr(10)}{pourquoi}{chr(10)}:::",
                 nom)


def fr(x, chiffres=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{chiffres}f}".replace(".", ",")


def nom_participant(p):
    return p.replace("_", " ")


def png_uri(img, lo=None, hi=None):
    """Image 2D -> PNG niveaux de gris en data URI (bornes 1-99 % par defaut)."""
    img = np.asarray(img, dtype=np.float32)
    if lo is None:
        lo, hi = np.percentile(img, [1, 99])
    u8 = np.clip((img - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(u8, mode="L").save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def resoudre_ffmpeg() -> str:
    """PATH d'abord, `imageio_ffmpeg` en secours (comme scripts/registration/astronauts.py)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


# --------------------------------------------------------------------------- #
# Donnees
# --------------------------------------------------------------------------- #
def charger():
    meta = pd.read_csv(META)
    crops = {r.slug: np.load(r.path) for r in meta.itertuples()}
    return meta, crops


# --------------------------------------------------------------------------- #
# Methodes : videos retenues
# --------------------------------------------------------------------------- #
def tab_videos(meta):
    s = pd.read_csv(SUMMARY)
    s_ok = s[s["status"] == "ok"].copy()
    s_ok["fps_r"] = s_ok["fps"].round(2)
    lignes = ["| participant | réplicats (OD) | cadence (Hz) | frames | durée (s) | rôle |",
              "|:--|:--|--:|--:|--:|:--|"]
    for p, g in meta.groupby("participant", sort=True):
        reps = ", ".join(f"OD{int(r)}" for r in g["replicate"])
        cad = " / ".join(sorted({fr(f, 2) for f in g["fps"]}))
        dur = (g["T"] / g["fps"]).sum()
        lignes.append(f"| {nom_participant(p)} | {reps} | {cad} | {int(g['T'].sum())} | "
                      f"{fr(dur, 0)} | {LIB_SPLIT[g['split'].iloc[0]]} |")
    ecartees = s_ok[~s_ok["slug"].isin(meta["slug"])]
    par_cadence = ecartees.groupby("fps_r").agg(n=("slug", "size"),
                                                 oeil=("eye", lambda e: "/".join(sorted(set(e)))))
    texte = "\n".join(lignes)
    texte += ("\n\n: Vidéos retenues (9,5–9,6 Hz). Les rôles sont attribués par participant. "
              "{#tbl-videos}\n\nÉcartées pour leur cadence : "
              + ", ".join(f"{int(r.n)} à {fr(f, 2)} Hz ({r.oeil})" for f, r in par_cadence.iterrows())
              + f" ; et `MARTINEZ_ALEJANDRA_OS3`, sans vidéo recalée `plex_full`.")
    ecrire_table(texte, "tab_videos")
    n = meta.groupby("split").agg(v=("slug", "size"), frames=("T", "sum"),
                                  s=("T", lambda t: float((t / meta.loc[t.index, "fps"]).sum())))
    RESUME["videos"] = {k: {"n": int(r.v), "frames": int(r.frames), "secondes": round(r.s, 1)}
                        for k, r in n.iterrows()}
    RESUME["n_ecartees"] = int(len(ecartees))


# --------------------------------------------------------------------------- #
# Crops
# --------------------------------------------------------------------------- #
def formes_crop(above, depth, c0, c1, W, xref, yref):
    x0 = (c1 - c0 - ((c1 - c0) // PATCH) * PATCH) // 2 + c0
    n = (c1 - c0) // PATCH
    formes = [dict(type="rect", xref=xref, yref=yref, x0=c0 - 0.5, x1=c1 - 0.5,
                   y0=above - 0.5, y1=above + depth - 0.5, line=dict(color="#f1c40f", width=1.2))]
    formes.append(dict(type="line", xref=xref, yref=yref, x0=-0.5, x1=W - 0.5,
                       y0=above - 0.5, y1=above - 0.5, line=dict(color="#e74c3c", width=0.8, dash="dot")))
    for k in range(n + 1):
        x = x0 + k * PATCH - 0.5
        formes.append(dict(type="line", xref=xref, yref=yref, x0=x, x1=x, y0=above - 0.5,
                           y1=above + depth - 0.5, line=dict(color="#27a567", width=0.8, dash="dash")))
    return formes


def formes_patchs(depth, ncols, xref, yref):
    n = ncols // PATCH
    x0 = (ncols - n * PATCH) // 2
    return [dict(type="line", xref=xref, yref=yref, x0=x0 + k * PATCH - 0.5, x1=x0 + k * PATCH - 0.5,
                 y0=-0.5, y1=depth - 0.5, line=dict(color="#27a567", width=1, dash="dash"))
            for k in range(n + 1)]


def fig_crop_moyennes(meta, crops):
    slugs = list(meta["slug"])
    fig = make_subplots(rows=3, cols=1, vertical_spacing=0.09, row_heights=[0.36, 0.36, 0.28],
                        subplot_titles=("fenêtre moyenne brute (aplatie sur la BM)",
                                        "fenêtre moyenne corrigée A-scan par A-scan",
                                        "crop d'une frame corrigée (frame du milieu), découpé en patchs 64 × 64"))
    boutons = []
    for i, slug in enumerate(slugs):
        z = crops[slug]
        above, depth, c0, c1 = (int(z[k]) for k in ("fit_above", "depth", "c0", "c1"))
        mb, ma = z["mean_before"], z["mean_after"]
        valid = np.flatnonzero(z["frame_valid"])
        frame = z["video"][valid[len(valid) // 2]].astype(np.float32)
        W = mb.shape[1]
        vis = i == 0
        fig.add_trace(go.Image(source=png_uri(mb), visible=vis, name=slug,
                               hovertemplate="colonne %{x}, ligne %{y}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Image(source=png_uri(ma), visible=vis, name=slug,
                               hovertemplate="colonne %{x}, ligne %{y}<extra></extra>"), row=2, col=1)
        fig.add_trace(go.Image(source=png_uri(frame), visible=vis, name=slug,
                               hovertemplate="colonne %{x} du crop, profondeur %{y} px<extra></extra>"),
                      row=3, col=1)
        formes = (formes_crop(above, depth, c0, c1, W, "x", "y")
                  + formes_crop(above, depth, c0, c1, W, "x2", "y2")
                  + formes_patchs(depth, c1 - c0, "x3", "y3"))
        if i == 0:
            fig.update_layout(shapes=formes)
        visible = [False] * (3 * len(slugs))
        visible[3 * i:3 * i + 3] = [True] * 3
        split = meta.loc[meta["slug"] == slug, "split"].iloc[0]
        boutons.append(dict(label=f"{slug} ({LIB_SPLIT[split]})", method="update",
                            args=[{"visible": visible}, {"shapes": formes}]))
    fig.update_layout(updatemenus=[dict(buttons=boutons, direction="down", x=0.01, y=1.13,
                                        xanchor="left", yanchor="top", showactive=True,
                                        bgcolor="#2b2b2b", font={"color": COULEUR_TEXTE})])
    mise_en_page(fig, "", hauteur=760, legende=False)
    fig.update_layout(margin={"t": 110})
    for r in (1, 2, 3):
        fig.update_xaxes(showgrid=False, row=r, col=1)
        fig.update_yaxes(showgrid=False, autorange="reversed", row=r, col=1)
    fig.update_xaxes(title_text="colonne (A-scan, 5,9 µm/px)", row=2, col=1)
    fig.update_xaxes(title_text="colonne du crop", row=3, col=1)
    fig.update_yaxes(title_text="ligne (1,95 µm/px)", row=1, col=1)
    enregistrer(fig, "fig_crop_moyennes")


def fig_corrections(meta, crops):
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=("gain par A-scan a<sub>x</sub> (image moyenne)",
                                        "gain par frame α<sub>t</sub> (sur la moyenne corrigée)"))
    palette = ["#2a78d6", "#e08a1e", "#27a567", "#c2453f", "#9b59b6", "#16a085", "#d35400", "#7f8c8d"]
    couleur = {p: palette[i % len(palette)] for i, p in enumerate(sorted(meta["participant"].unique()))}
    a_med, al_iqr = [], []
    for r in meta.itertuples():
        z = crops[r.slug]
        a = np.where(z["col_valid"], z["a"], np.nan)
        al = np.where(z["frame_valid"], z["alpha"], np.nan)
        x_mm = np.arange(a.size) * float(z["lateral_um"]) / 1e3
        t = z["t_s"]
        a_med.append(np.nanmedian(a))
        al_iqr.append(np.nanpercentile(al, 75) - np.nanpercentile(al, 25))
        kw = dict(legendgroup=r.slug, line=dict(color=couleur[r.participant], width=1), opacity=0.8)
        fig.add_trace(go.Scattergl(x=x_mm, y=a, name=r.slug, **kw,
                                   hovertemplate=f"{r.slug}<br>%{{x:.2f}} mm : a = %{{y:.2f}}<extra></extra>"),
                      row=1, col=1)
        fig.add_trace(go.Scattergl(x=t, y=al, name=r.slug, showlegend=False, **kw,
                                   hovertemplate=f"{r.slug}<br>%{{x:.1f}} s : α = %{{y:.3f}}<extra></extra>"),
                      row=1, col=2)
    z0 = crops[meta["slug"].iloc[0]]
    lat = float(z0["lateral_um"]) / 1e3
    for c in (int(z0["c0"]), int(z0["c1"])):
        fig.add_vline(x=c * lat, line=dict(color="#f1c40f", width=1, dash="dash"), row=1, col=1)
    mise_en_page(fig, "", hauteur=480)
    fig.update_xaxes(title_text="position latérale (mm) — crop entre les pointillés", row=1, col=1)
    fig.update_xaxes(title_text="temps (s)", row=1, col=2)
    fig.update_yaxes(title_text="a<sub>x</sub>", row=1, col=1)
    fig.update_yaxes(title_text="α<sub>t</sub>", row=1, col=2)
    fig.update_layout(legend={"y": -0.2})
    enregistrer(fig, "fig_corrections")
    RESUME["a_mediane"] = [round(float(np.min(a_med)), 2), round(float(np.max(a_med)), 2)]
    RESUME["alpha_iqr_max"] = round(float(np.max(al_iqr)), 3)


def ecrire_mp4(cube, chemin, fps):
    lo, hi = np.percentile(cube, [1, 99])
    u8 = np.clip((cube - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8)
    u8 = np.repeat(u8, STRETCH_VIDEO, axis=1)
    T, h, w = u8.shape
    u8 = u8[:, : h - h % 2, : w - w % 2]
    T, h, w = u8.shape
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray",
           "-s", f"{w}x{h}", "-r", f"{fps:.4f}", "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28", "-movflags", "+faststart", str(chemin)]
    proc = subprocess.run(cmd, input=u8.tobytes(), capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="replace"))


def fig_crop_videos(meta, crops):
    (SORTIE / "videos").mkdir(exist_ok=True)
    blocs = []
    for split in ("test", "val", "train"):
        g = meta[meta["split"] == split]
        cellules = []
        for r in g.itertuples():
            z = crops[r.slug]
            nom = f"videos/{r.slug}.mp4"
            ecrire_mp4(z["video"].astype(np.float32), SORTIE / nom, float(z["fps"]))
            cellules.append(
                f'<figure style="margin:0"><video src="{DOSSIER}/{nom}" style="width:100%;height:auto" '
                f'controls loop muted playsinline preload="metadata"></video>'
                f'<figcaption style="font-size:0.8em">{r.slug} — {fr(float(z["fps"]), 2)} Hz, '
                f'{len(z["video"])} frames</figcaption></figure>')
        blocs.append(f"### {LIB_SPLIT[split].capitalize()}\n\n```{{=html}}\n"
                     '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px">\n'
                     + "\n".join(cellules) + "\n</div>\n```\n")
    ecrire_table("\n".join(blocs), "fig_crop_videos")


# --------------------------------------------------------------------------- #
# Entrainement
# --------------------------------------------------------------------------- #
def lire_args():
    args = {}
    for ligne in (EXP / "arg_obj.txt").read_text(encoding="utf-8").splitlines():
        if ligne.strip():
            k, *v = ligne.split()
            args[k] = " ".join(v)
    return args


def lire_journal():
    log = EXP_ROOT / "train.log"
    if not log.exists():
        return {}
    txt = log.read_text(encoding="utf-8", errors="replace")
    out = {}
    for split in ("train", "val"):
        m = re.search(rf"^{split}\s*\nVideos:\s+(\d+).*?\nSamples:\s+\((\d+), 3\)", txt, flags=re.M | re.S)
        if m:
            out[split] = {"videos": int(m.group(1)), "samples": int(m.group(2))}
    took = re.findall(r"Took ([\d.]+) seconds\.", txt)
    if took:
        out["secondes"] = float(took[-1])
        out["epoques_journal"] = len(took)
    out["fini"] = "Finished Training." in txt
    return out


def tab_entrainement():
    if not (EXP / "losses_val.csv").exists():
        absente("tab_entrainement", "L'entraînement n'a pas encore écrit `losses_val.csv`.")
        return None
    a, j = lire_args(), lire_journal()
    val = pd.read_csv(EXP / "losses_val.csv")
    best = int(val.loc[val["total"].idxmin(), "epoch"])
    n_ep = int(val["epoch"].max()) + 1
    duree = j.get("secondes", np.nan)
    lignes = [
        "| réglage | valeur |", "|:--|:--|",
        f"| modèle | PhysNet (SiNC), 1 canal, dropout {a.get('dropout')} |",
        f"| entrée | clips de {a.get('fpc')} frames (≈ {fr(int(a.get('fpc')) / float(a.get('fps')), 1)} s) "
        f"× 64 × 64 px, pas de {a.get('step')} frames entre clips |",
        f"| cadence des pertes | {a.get('fps')} Hz (moyenne de 9,52 et 9,62 Hz) |",
        f"| bande cardiaque | {fr(float(a.get('low_hz')) * 60, 0)}–{fr(float(a.get('high_hz')) * 60, 0)} BPM |",
        f"| pertes (entraînement) | bandwidth (IPR) + sparsity (SNR) + variance (EMD) |",
        f"| pertes (choix du modèle) | bandwidth + sparsity, sur la validation |",
        f"| augmentations | vitesse ×{a.get('speed_slow')}–{a.get('speed_fast')}, miroir latéral, "
        f"inversion du temps, bruit gaussien (σ = 0,05) |",
        f"| optimiseur | {a.get('optimizer')}, lr = {a.get('lr')}, lot de {a.get('batch_size')} |",
        f"| échantillons (vidéo × clip × patch) | entraînement {j.get('train', {}).get('samples', '—')}, "
        f"validation {j.get('val', {}).get('samples', '—')} |",
        f"| époques | {n_ep} / {a.get('epochs')}{'' if j.get('fini') else ' (en cours)'} |",
        f"| meilleure époque (validation) | {best} (perte {fr(float(val['total'].min()), 3)}) |",
        f"| durée | {fr(duree / 3600, 2)} h, soit {fr(duree / max(n_ep, 1), 0)} s par époque (RTX 3060) |",
    ]
    ecrire_table("\n".join(lignes) + "\n\n: Réglages et déroulement de l'entraînement. {#tbl-entrainement}",
                 "tab_entrainement")
    RESUME["entrainement"] = {"epoques": n_ep, "meilleure": best, "secondes": duree,
                              "fini": j.get("fini", False), "echantillons": {k: j[k] for k in ("train", "val") if k in j}}
    return best


def fig_losses(best):
    if not (EXP / "losses_train.csv").exists():
        absente("fig_losses", "L'entraînement n'a pas encore écrit `losses_train.csv`.")
        return
    tr = pd.read_csv(EXP / "losses_train.csv")
    val = pd.read_csv(EXP / "losses_val.csv")
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.09,
                        subplot_titles=("entraînement (moyenne par époque ; brut par itération masqué)",
                                        "validation (par époque)"))
    pertes = [c for c in ("total", "bandwidth", "sparsity", "variance") if c in tr.columns]
    par_ep = tr.groupby("epoch")[pertes].mean()
    n_it = tr.groupby("epoch").size().max()
    for c in pertes:
        x_it = tr["epoch"] + tr["step"] / n_it
        fig.add_trace(go.Scattergl(x=x_it, y=tr[c], name=f"{c} (itérations)", legendgroup=c,
                                   visible="legendonly", line=dict(color=C_LOSS[c], width=0.6), opacity=0.4),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=par_ep.index + 0.5, y=par_ep[c], name=c, legendgroup=c,
                                 line=dict(color=C_LOSS[c], width=2)), row=1, col=1)
    for c in [c for c in ("total", "bandwidth", "sparsity") if c in val.columns]:
        fig.add_trace(go.Scatter(x=val["epoch"], y=val[c], name=c, legendgroup=c, showlegend=False,
                                 line=dict(color=C_LOSS[c], width=2)), row=1, col=2)
    if best is not None:
        fig.add_vline(x=best, line=dict(color="#27a567", width=1, dash="dash"), row=1, col=2)
        fig.add_annotation(x=best, y=1, yref="y2 domain", text=f"meilleure : {best}", showarrow=False,
                           xanchor="left", font={"color": "#27a567"}, row=1, col=2)
    mise_en_page(fig, "", hauteur=460)
    fig.update_xaxes(title_text="époque", row=1, col=1)
    fig.update_xaxes(title_text="époque", row=1, col=2)
    fig.update_yaxes(title_text="perte", row=1, col=1)
    enregistrer(fig, "fig_losses")
    RESUME["pertes"] = {
        "train_premiere": {c: round(float(par_ep[c].iloc[0]), 4) for c in pertes},
        "train_derniere": {c: round(float(par_ep[c].iloc[-1]), 4) for c in pertes},
        "val_premiere": {c: round(float(val[c].iloc[0]), 4) for c in val.columns if c != "epoch"},
        "val_min": {c: round(float(val[c].min()), 4) for c in val.columns if c != "epoch"},
    }


# --------------------------------------------------------------------------- #
# Resultats
# --------------------------------------------------------------------------- #
def fig_hr():
    if not (PRED / "predictions.csv").exists():
        absente("fig_hr", "`evaluate_oct.py` n'a pas encore été lancé sur le modèle final.")
        return None
    p = pd.read_csv(PRED / "predictions.csv")
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=("SiNC contre la FC de référence (hr_prior)",
                                        "SiNC contre la FC du pouls d'épaisseur"))
    lim = [40, 120]
    for col, ref, lib in ((1, "hr_prior_BPM", "FC de référence"), (2, "hr_thickness", "FC épaisseur")):
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", showlegend=False, hoverinfo="skip",
                                 line=dict(color=GRILLE.replace("0.18", "0.6"), dash="dot")), row=1, col=col)
        for split in ("train", "val", "test"):
            g = p[p["split"] == split]
            fig.add_trace(go.Scatter(
                x=g[ref], y=g["hr_sinc"], mode="markers", name=LIB_SPLIT[split], legendgroup=split,
                showlegend=col == 1, marker=dict(color=C_SPLIT[split], size=9, line=dict(width=0)),
                error_y=dict(type="data", symmetric=False, array=g["hr_patch_max"] - g["hr_sinc"],
                             arrayminus=g["hr_sinc"] - g["hr_patch_min"], thickness=1, width=0,
                             color=C_SPLIT[split]),
                customdata=np.c_[g["slug"], g["hr_patch_min"], g["hr_patch_max"]],
                hovertemplate=("%{customdata[0]}<br>" + lib + " %{x:.1f}<br>SiNC %{y:.1f} BPM"
                               "<br>patchs %{customdata[1]:.1f}–%{customdata[2]:.1f}<extra></extra>")),
                row=1, col=col)
    mise_en_page(fig, "", hauteur=470)
    for col, lib in ((1, "FC de référence, hr_prior (BPM)"), (2, "FC du pouls d'épaisseur (BPM)")):
        fig.update_xaxes(title_text=lib, range=lim, row=1, col=col)
        fig.update_yaxes(title_text="FC SiNC (BPM)", range=lim, row=1, col=col)
    enregistrer(fig, "fig_hr")
    tol = 0.10
    p["ok_prior"] = (p["hr_sinc"] - p["hr_prior_BPM"]).abs() <= tol * p["hr_prior_BPM"]
    p["ok_th"] = (p["hr_sinc"] - p["hr_thickness"]).abs() <= tol * p["hr_thickness"]
    RESUME["hr"] = {s: {"n": int(len(g)), "a_10pc_prior": int(g["ok_prior"].sum()),
                        "a_10pc_epaisseur": int(g["ok_th"].sum()),
                        "mae_prior": round(float((g["hr_sinc"] - g["hr_prior_BPM"]).abs().mean()), 1)}
                    for s, g in p.groupby("split")}
    return p


def fig_test_ondes(p):
    if p is None:
        absente("fig_test_ondes", "`evaluate_oct.py` n'a pas encore été lancé sur le modèle final.")
        return
    g = p[p["split"] == "test"].sort_values("slug")
    n = len(g)
    fig = make_subplots(rows=n, cols=2, column_widths=[0.66, 0.34], horizontal_spacing=0.07,
                        vertical_spacing=0.09,
                        subplot_titles=sum(([f"{s} : onde SiNC et pouls d'épaisseur (filtrés)",
                                             "spectre de l'onde SiNC"] for s in g["slug"]), []))
    for i, r in enumerate(g.itertuples(), start=1):
        z = np.load(PRED / f"{r.slug}.npz")
        t = z["t_s"]
        std = lambda y: (y - y.mean()) / (y.std() + 1e-12)  # noqa: E731
        fig.add_trace(go.Scatter(x=t, y=std(z["wave_bp"]), name="SiNC", legendgroup="sinc", showlegend=i == 1,
                                 line=dict(color="#27a567", width=1.4)), row=i, col=1)
        fig.add_trace(go.Scatter(x=t, y=std(z["thickness_bp"]), name="épaisseur (masque)", legendgroup="th",
                                 showlegend=i == 1, line=dict(color="#e08a1e", width=1.2)), row=i, col=1)
        psd = z["psd"] / z["psd"].max()
        fig.add_trace(go.Scatter(x=z["psd_bpm"], y=psd, name="spectre SiNC", legendgroup="psd", showlegend=False,
                                 line=dict(color="#27a567", width=1.4)), row=i, col=2)
        for v, c, lib in ((r.hr_prior_BPM, "#c9c9c9", "hr_prior"), (r.hr_svd_BPM, "#9b59b6", "SVD"),
                          (r.hr_thickness, "#e08a1e", "épaisseur")):
            if np.isfinite(v):
                fig.add_vline(x=v, line=dict(color=c, width=1, dash="dash"), row=i, col=2)
        fig.update_xaxes(title_text="temps (s)" if i == n else None, row=i, col=1)
        fig.update_xaxes(title_text="BPM" if i == n else None, row=i, col=2)
    mise_en_page(fig, "", hauteur=300 * n + 80)
    enregistrer(fig, "fig_test_ondes")


def tab_test(p):
    if p is None:
        absente("tab_test", "`evaluate_oct.py` n'a pas encore été lancé sur le modèle final.")
        return
    g = p[p["split"] == "test"].sort_values("slug")
    lignes = ["| acquisition | FC SiNC | patchs (min–max) | FC épaisseur | hr_prior | FC SVD | r(onde, épaisseur) |",
              "|:--|--:|--:|--:|--:|--:|--:|"]
    for r in g.itertuples():
        lignes.append(f"| {r.slug} | {fr(r.hr_sinc)} | {fr(r.hr_patch_min)}–{fr(r.hr_patch_max)} | "
                      f"{fr(r.hr_thickness)} | {fr(r.hr_prior_BPM)} | {fr(r.hr_svd_BPM)} | "
                      f"{fr(r.r_wave_thickness, 2)} |")
    ecrire_table("\n".join(lignes) + "\n\n: Jeu de test (MODICA GRAZIANA, OD). FC en BPM. {#tbl-test}",
                 "tab_test")
    RESUME["test"] = g[["slug", "hr_sinc", "hr_patch_min", "hr_patch_max", "hr_thickness", "hr_prior_BPM",
                        "hr_svd_BPM", "r_wave_thickness"]].round(2).to_dict(orient="records")


# --------------------------------------------------------------------------- #
FFMPEG = resoudre_ffmpeg()


def main():
    print("->", SORTIE)
    (SORTIE / "plotlyjs.html").write_text(f'<script type="text/javascript">{get_plotlyjs()}</script>',
                                          encoding="utf-8")
    meta, crops = charger()
    tab_videos(meta)
    fig_crop_moyennes(meta, crops)
    fig_corrections(meta, crops)
    fig_crop_videos(meta, crops)
    best = tab_entrainement()
    fig_losses(best)
    p = fig_hr()
    fig_test_ondes(p)
    tab_test(p)
    (SORTIE / "resume.txt").write_text(json.dumps(RESUME, indent=2, ensure_ascii=False, default=float),
                                       encoding="utf-8")
    print("  resume.txt")


if __name__ == "__main__":
    main()
