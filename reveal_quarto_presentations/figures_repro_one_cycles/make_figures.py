# -*- coding: utf-8 -*-
"""
Figures et tables -- page « One-cycles des réplicats ».

Ce script NE LIT QUE des tables deja ecrites ; il ne replie ni ne recale rien.

Entrees
-------
    E:/NASA_Rigidity/Reproducibility/SegmentationVariations/
        cascade_v9_1536x1024_hr_mediane/one_cycle_6beats_6bins/
            conditions.csv   bins.csv   params.json
    (ecrits par ``Reproducibility/compute_one_cycle.py``)

Sorties (fragments inclus par ``repro-one-cycles.qmd``)
-------------------------------------------------------
    tab_resume.qmd, fig_epaisseur.qmd, plotlyjs.html, resume.txt

Lancer DEPUIS LA RACINE DU DEPOT (ce dossier est une JONCTION vers E:) :
    python reveal_quarto_presentations/figures_repro_one_cycles/make_figures.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------- #
# Entrees / sorties
# --------------------------------------------------------------------------- #
SOURCE = Path("E:/NASA_Rigidity/Reproducibility/SegmentationVariations/"
              "cascade_v9_1536x1024_hr_mediane/one_cycle_6beats_6bins")
CSV_CONDITIONS = SOURCE / "conditions.csv"
CSV_BINS = SOURCE / "bins.csv"
PARAMS = SOURCE / "params.json"

# `.absolute()` et surtout PAS `.resolve()` : ce dossier est une JONCTION vers E:.
SORTIE = Path(__file__).absolute().parent

COULEUR_TEXTE = "#c9c9c9"
GRILLE = "rgba(150,150,150,0.18)"
# Les cycles se succedent le long de l'axe x et ne se superposent jamais : la
# couleur ne sert qu'a suivre l'ordre, d'ou une rampe ORDINALE a une teinte (bleu,
# pas plus clair que 250 ni plus sombre que 600, lisible sur les deux themes).
RAMPE_CYCLES = [[0.0, "#86b6ef"], [1.0, "#184f95"]]
C_SEG = "#eb6834"                    # resegmentation : hors de la famille des cycles
C_FRAMES = "rgba(150,150,150,0.35)"  # occupation des bins, en fond

CONFIG = {"displaylogo": False, "responsive": True,
          "modeBarButtonsToRemove": ["select2d", "lasso2d"]}
NOMS = ("tab_resume", "fig_epaisseur")
RESUME: dict = {}
nl = chr(10)


# --------------------------------------------------------------------------- #
# Habillage commun (le meme que figures_repro_hr_mediane)
# --------------------------------------------------------------------------- #
def mise_en_page(fig, titre, hauteur=440, legende=True):
    fig.update_layout(
        title={"text": titre, "font": {"size": 14}, "x": 0.01, "xanchor": "left"},
        template="none",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COULEUR_TEXTE, "size": 11},
        height=hauteur,
        margin={"l": 60, "r": 60, "t": 80, "b": 80 if legende else 50},
        showlegend=legende,
        legend={"font": {"size": 10}, "orientation": "h", "yanchor": "top",
                "y": -0.16, "xanchor": "left", "x": 0},
        hoverlabel={"font": {"size": 11}},
    )
    fig.update_xaxes(gridcolor=GRILLE, zerolinecolor=GRILLE,
                     linecolor=GRILLE, ticks="outside", tickcolor=GRILLE)
    fig.update_yaxes(gridcolor=GRILLE, zerolinecolor=GRILLE,
                     linecolor=GRILLE, ticks="outside", tickcolor=GRILLE)
    return fig


def enregistrer(fig, nom):
    html = fig.to_html(full_html=False, include_plotlyjs=False,
                       config=CONFIG, div_id=f"plot-{nom}")
    (SORTIE / f"{nom}.qmd").write_text("```{=html}\n" + html + "\n```\n", encoding="utf-8")
    print(f"  {nom}.qmd")


def ecrire_table(texte, nom):
    entete = ("<!-- Genere par figures_repro_one_cycles/make_figures.py"
              " -- ne pas editer a la main. -->" + nl + nl)
    (SORTIE / f"{nom}.qmd").write_text(entete + texte + nl, encoding="utf-8")
    print(f"  {nom}.qmd")


def absente(nom, pourquoi):
    """Un ``{{< include >}}`` sur un fichier absent fait ECHOUER le rendu du site
    entier : une figure impossible laisse donc un encadre lisible a sa place."""
    ecrire_table(f"::: {{.callout-warning}}{nl}## Figure indisponible{nl}{nl}{pourquoi}{nl}:::",
                 nom)


def fr(x, chiffres=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{{:.{chiffres}f}}".format(x).replace(".", ",")


def med_iqr(s, chiffres=1):
    s = pd.Series(s, dtype=float).dropna()
    if s.empty:
        return "—"
    return (f"{fr(s.median(), chiffres)} [{fr(s.quantile(0.25), chiffres)} – "
            f"{fr(s.quantile(0.75), chiffres)}]")


# --------------------------------------------------------------------------- #
# Donnees
# --------------------------------------------------------------------------- #
manquants = [p for p in (CSV_CONDITIONS, CSV_BINS) if not p.exists()]
if manquants:
    pourquoi = ("Le calcul n'a pas encore produit ces tables :" + nl + nl
                + nl.join(f"- `{p}`" for p in manquants) + nl + nl
                + "Lancer `Reproducibility/compute_one_cycle.py`.")
    for nom in NOMS:
        absente(nom, pourquoi)
    (SORTIE / "plotlyjs.html").write_text("", encoding="utf-8")
    raise SystemExit(0)

COND = pd.read_csv(CSV_CONDITIONS)
COND_OK = COND[COND["status"] == "ok"].sort_values("slug").reset_index(drop=True)
BINS = pd.read_csv(CSV_BINS)
BINS = BINS[BINS["slug"].isin(COND_OK["slug"])]
N_BINS = json.loads(PARAMS.read_text(encoding="utf-8"))["n_bins"] if PARAMS.exists() \
    else int(BINS["bin"].max()) + 1
N_CYCLE = json.loads(PARAMS.read_text(encoding="utf-8"))["n_cycle"] if PARAMS.exists() else None
RESUME["n_ok"] = f"{len(COND_OK)} / {len(COND)}"
RESUME["echecs"] = " | ".join(f"{r.slug}: {r.reason}"
                              for r in COND[COND["status"] != "ok"].itertuples()) or "aucun"


# --------------------------------------------------------------------------- #
# 1. Resume par acquisition
# --------------------------------------------------------------------------- #
def tab_resume():
    lignes = ["| acquisition | FC (BPM) | one-cycles | frames par bin (min – méd. – max) "
              "| bins < 5 frames | ΔY harm. replié (µm) | ΔY harm. resegmenté (µm) "
              "| ΔY harm. sans recalage intra-bin (µm) | Dice méd. | dx intra-bin (px) |",
              "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in COND_OK.itertuples():
        um = r.um_per_px_y
        lignes.append(
            f"| {r.slug} | {fr(r.hr_BPM)} | {int(r.n_groups)} "
            f"| {int(r.frames_per_bin_min)} – {fr(r.frames_per_bin_median)} – "
            f"{int(r.frames_per_bin_max)} | {int(r.n_bins_lt5)} "
            f"| {fr(r.dY_fold_fit_px_med * um)} | {fr(r.dY_seg_fit_px_med * um)} "
            f"| {fr(r.dY_std_fit_px_med * um)} | {fr(r.dice_median, 3)} "
            f"| {fr(r.dx_ptp_px_median, 2)} |")
    um = COND_OK["um_per_px_y"]
    RESUME["one_cycles_par_acq"] = med_iqr(COND_OK["n_groups"], 0)
    RESUME["one_cycles_total"] = int(COND_OK["n_groups"].sum())
    RESUME["frames_par_bin_med"] = med_iqr(COND_OK["frames_per_bin_median"])
    RESUME["bins_lt5_total"] = (f"{int(COND_OK['n_bins_lt5'].sum())} / "
                                f"{int(COND_OK['n_bins_filled'].sum())}")
    RESUME["dY_replie_um"] = med_iqr(COND_OK["dY_fold_fit_px_med"] * um)
    RESUME["dY_resegmente_um"] = med_iqr(COND_OK["dY_seg_fit_px_med"] * um)
    RESUME["dY_standard_um"] = med_iqr(COND_OK["dY_std_fit_px_med"] * um)
    RESUME["dice_med"] = med_iqr(COND_OK["dice_median"], 3)
    RESUME["dx_intra_bin_ptp_med_px"] = med_iqr(COND_OK["dx_ptp_px_median"], 2)
    RESUME["epaisseur_replie_moins_resegmente_px"] = med_iqr(
        COND_OK["thickness_fold_px_mean"] - COND_OK["thickness_seg_px_mean"], 2)
    ecrire_table(nl.join(lignes) + nl + nl
                 + ": Une ligne par acquisition. ΔY : crête-à-crête du fondamental ajusté "
                   "sur chaque one-cycle, médiane sur les one-cycles de l'acquisition, converti "
                   "avec la taille de pixel axiale du XML. Dice : masque resegmenté contre "
                   "occupation repliée > 0,5, médiane sur les bins. dx intra-bin : "
                   "crête-à-crête du dx prédit par cascade_v9 dans un bin, médiane sur les "
                   "bins. {.striped}", "tab_resume")


# --------------------------------------------------------------------------- #
# 2. Epaisseur et occupation des bins, une acquisition a la fois
# --------------------------------------------------------------------------- #
def fig_epaisseur():
    if COND_OK.empty:
        absente("fig_epaisseur", "Aucune acquisition `ok` dans `conditions.csv`.")
        return
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    spans, titres, formes, titres_x = [], [], [], []
    for i, r in enumerate(COND_OK.itertuples()):
        b = BINS[BINS["slug"] == r.slug].sort_values("slot")
        n_g = int(r.n_groups)
        um = float(r.um_per_px_y)
        visible = i == 0
        debut = len(fig.data)

        fig.add_trace(go.Bar(
            x=b["slot"], y=b["n_frames"], name="frames par bin", legendgroup="frames",
            marker={"color": C_FRAMES, "line": {"width": 0}}, visible=visible,
            customdata=np.stack([b["cycle"], b["bin"]], axis=-1),
            hovertemplate="cycle %{customdata[0]} — bin %{customdata[1]} : "
                          "%{y} frames<extra></extra>"), secondary_y=True)

        couleurs = (sample_colorscale(RAMPE_CYCLES, np.linspace(0, 1, n_g))
                    if n_g > 1 else [RAMPE_CYCLES[0][1]])
        for g in range(1, n_g + 1):
            bg = b[b["cycle"] == g]
            lib = f"cycle {g}"
            cd = np.stack([bg["bin"], bg["n_frames"], bg["thickness_fold_px"] * um,
                           bg["thickness_seg_px"] * um, bg["thickness_fold_fit_px"] * um,
                           bg["dice"]], axis=-1)
            fig.add_trace(go.Scatter(
                x=bg["slot"], y=bg["thickness_fold_px"], mode="lines+markers", name=lib,
                legendgroup=lib, visible=visible, customdata=cd,
                line={"color": couleurs[g - 1], "width": 2}, marker={"size": 8},
                hovertemplate=(lib + " — bin de phase %{customdata[0]} "
                               "(%{customdata[1]} frames)<br>replié : %{y:.2f} px "
                               "(%{customdata[2]:.1f} µm)<extra></extra>")),
                secondary_y=False)
            fig.add_trace(go.Scatter(
                x=bg["slot"], y=bg["thickness_fold_fit_px"], mode="lines", name=lib,
                legendgroup=lib, showlegend=False, visible=visible, customdata=cd,
                line={"color": couleurs[g - 1], "width": 1.5, "dash": "dash"},
                hovertemplate=(lib + " — harmonique<br>%{y:.2f} px "
                               "(%{customdata[4]:.1f} µm)<extra></extra>")),
                secondary_y=False)
            fig.add_trace(go.Scatter(
                x=bg["slot"], y=bg["thickness_seg_px"], mode="lines+markers",
                name="segmentation du one-cycle", legendgroup="seg", showlegend=(g == 1),
                visible=visible, customdata=cd,
                line={"color": C_SEG, "width": 1.5, "dash": "dot"},
                marker={"size": 8, "symbol": "square"},
                hovertemplate=(lib + " — bin de phase %{customdata[0]}<br>resegmenté : "
                               "%{y:.2f} px (%{customdata[3]:.1f} µm)<br>Dice "
                               "%{customdata[5]:.3f}<extra></extra>")),
                secondary_y=False)
        spans.append((debut, len(fig.data)))
        formes.append([{"type": "line", "xref": "x", "yref": "paper",
                        "x0": g * N_BINS - 0.5, "x1": g * N_BINS - 0.5, "y0": 0, "y1": 1,
                        "line": {"color": "rgba(150,150,150,0.55)", "width": 1, "dash": "dot"}}
                       for g in range(1, n_g)])
        titres.append(f"{r.slug} — FC {fr(r.hr_BPM)} BPM · {n_g} one-cycles de "
                      f"{N_CYCLE} battements × {N_BINS} bins<br>plein : masques repliés · "
                      f"tirets : harmonique · pointillé orange : segmentation du one-cycle")
        titres_x.append(f"bin ({n_g} one-cycles de {N_BINS} bins, bout à bout)")

    boutons = []
    for i, r in enumerate(COND_OK.itertuples()):
        vis = [False] * len(fig.data)
        vis[spans[i][0]:spans[i][1]] = [True] * (spans[i][1] - spans[i][0])
        boutons.append({"label": r.slug, "method": "update",
                        "args": [{"visible": vis},
                                 {"title.text": titres[i], "shapes": formes[i],
                                  "xaxis.title.text": titres_x[i]}]})
    fig.update_yaxes(title_text="épaisseur choroïdienne (px)", secondary_y=False)
    fig.update_yaxes(title_text="frames par bin", secondary_y=True, showgrid=False,
                     rangemode="tozero")
    fig.update_xaxes(title_text=titres_x[0])
    fig = mise_en_page(fig, titres[0], hauteur=560)
    # Titre (deux lignes) colle en haut du conteneur, menu colle au-dessus du trace :
    # les deux ne se chevauchent plus quelle que soit la longueur du slug.
    fig.update_layout(
        barmode="overlay", shapes=formes[0], margin={"t": 140, "b": 130},
        title={"yref": "container", "y": 0.985, "yanchor": "top"},
        legend={"y": -0.26},
        updatemenus=[{"buttons": boutons, "direction": "down", "showactive": True,
                      "x": 1.0, "xanchor": "right", "y": 1.0, "yanchor": "bottom",
                      "pad": {"b": 8}, "bgcolor": "rgba(120,120,120,0.25)",
                      "bordercolor": GRILLE, "font": {"size": 11, "color": "#222"}}])
    enregistrer(fig, "fig_epaisseur")


tab_resume()
fig_epaisseur()

(SORTIE / "plotlyjs.html").write_text(
    '<script type="text/javascript">' + nl + get_plotlyjs() + nl + "</script>" + nl,
    encoding="utf-8")
print("  plotlyjs.html")
(SORTIE / "resume.txt").write_text(nl.join(f"{k} = {v}" for k, v in RESUME.items()) + nl,
                                   encoding="utf-8")
print("  resume.txt")
