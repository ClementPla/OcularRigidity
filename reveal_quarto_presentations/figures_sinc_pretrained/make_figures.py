# -*- coding: utf-8 -*-
"""
Figures et tables -- page « Pretrained PhysNet » (section Sinc on intensity).

Ce script NE LIT QUE des sorties deja ecrites ; il n'entraine ni n'evalue rien.

Entrees
-------
    E:/NASA_Rigidity/Reproducibility/sinc_intensity/
        pretrained_<poids>_<entree>/fold0_seed0/   losses_*.csv, predictions/   (run_pretrained_sweep.py)
        pretrained_<poids>_<entree>/train.log       journal (duree)
        first_training/fold0_seed0/                 reference : PhysNet de SiNC, 200 epoques
    Les metriques par entrainement sont celles de ``run_pretrained_sweep.metrics`` (importee).

Sorties (fragments Quarto, inclus par ``sinc-pretrained-physnet.qmd``)
----------------------------------------------------------------------
    tab_sweep.qmd        un entrainement par ligne : pertes, FC, accord avec hr_prior, GRAZIANA
    fig_grille.qmd       videos a +/-10 % de hr_prior et Spearman, grille poids x entree
    fig_val.qmd          pertes de validation par epoque, tous les entrainements
    fig_hr.qmd           FC SiNC contre hr_prior, menu par entrainement
    fig_test_ondes.qmd   GRAZIANA, meilleur entrainement : onde, pouls d'epaisseur, spectre
    plotlyjs.html, resume.txt

Lancer DEPUIS LA RACINE DU DEPOT (ce dossier est une JONCTION vers E:) :
    python reveal_quarto_presentations/figures_sinc_pretrained/make_figures.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

REPO = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(REPO / "Astronauts/pulsation/sinc_intensity"))
import run_pretrained_sweep as sweep  # noqa: E402

ROOT = sweep.ROOT
# `.absolute()` et surtout PAS `.resolve()` : ce dossier est une JONCTION vers E:.
SORTIE = Path(__file__).absolute().parent

COULEUR_TEXTE = "#c9c9c9"
GRILLE = "rgba(150,150,150,0.18)"
C_POIDS = {"scamps": "#27a567", "pure": "#2a78d6", "ubfc": "#e08a1e", "maubfc": "#c2453f",
           "bp4d": "#9b59b6", "scratch": "#8a94a6", "aucun": "#c9c9c9"}
LIB_POIDS = {"scamps": "SCAMPS", "pure": "PURE", "ubfc": "UBFC-rPPG", "maubfc": "MA-UBFC",
             "bp4d": "BP4D (pseudo-labels)", "scratch": "sans préentraînement", "aucun": "PhysNet SiNC"}
DASH = {"diffnorm": "solid", "zscore": "dot"}
C_SPLIT = {"train": "#2a78d6", "val": "#e08a1e", "test": "#27a567"}
LIB_SPLIT = {"train": "entraînement", "val": "validation", "test": "test"}
CONFIG = {"displaylogo": False, "responsive": True,
          "modeBarButtonsToRemove": ["select2d", "lasso2d"]}
FICHIERS: list[str] = []
RESUME: dict = {}


# --------------------------------------------------------------------------- #
# Habillage commun (le meme que figures_sinc_first_training)
# --------------------------------------------------------------------------- #
def mise_en_page(fig, titre, hauteur=440, legende=True):
    fig.update_layout(
        title={"text": titre, "font": {"size": 14}, "x": 0.01, "xanchor": "left"},
        template="none", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COULEUR_TEXTE, "size": 11}, height=hauteur,
        margin={"l": 60, "r": 20, "t": 80, "b": 80 if legende else 50},
        showlegend=legende,
        legend={"font": {"size": 10}, "orientation": "h", "yanchor": "top",
                "y": -0.12, "xanchor": "left", "x": 0},
        hoverlabel={"font": {"size": 11}},
    )
    fig.update_xaxes(gridcolor=GRILLE, zerolinecolor=GRILLE, linecolor=GRILLE, ticks="outside", tickcolor=GRILLE)
    fig.update_yaxes(gridcolor=GRILLE, zerolinecolor=GRILLE, linecolor=GRILLE, ticks="outside", tickcolor=GRILLE)
    for a in fig.layout.annotations or ():
        if a.text and a.font is not None:
            a.font.size = 11
            a.font.color = COULEUR_TEXTE
    return fig


def enregistrer(fig, nom):
    html = fig.to_html(full_html=False, include_plotlyjs=False, config=CONFIG, div_id=f"plot-{nom}")
    (SORTIE / f"{nom}.qmd").write_text("```{=html}\n" + html + "\n```\n", encoding="utf-8")
    FICHIERS.append(f"{nom}.qmd")
    print(f"  {nom}.qmd")


def ecrire_table(texte, nom):
    entete = ("<!-- Genere par figures_sinc_pretrained/make_figures.py"
              " -- ne pas editer a la main. -->" + chr(10) + chr(10))
    (SORTIE / f"{nom}.qmd").write_text(entete + texte + chr(10), encoding="utf-8")
    FICHIERS.append(f"{nom}.qmd")
    print(f"  {nom}.qmd")


def absente(nom, pourquoi):
    """Un ``{{< include >}}`` sur un fichier absent fait ECHOUER le rendu du site."""
    ecrire_table(f"::: {{.callout-warning}}{chr(10)}## Figure indisponible{chr(10)}{chr(10)}"
                 f"{pourquoi}{chr(10)}:::", nom)


def fr(x, chiffres=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{chiffres}f}".replace(".", ",")


# --------------------------------------------------------------------------- #
# Donnees
# --------------------------------------------------------------------------- #
def entrainements():
    """[(nom, dossier fold0_seed0, poids, entree)] des entrainements evalues, reference d'abord."""
    out = []
    ref = ROOT / "first_training" / "fold0_seed0"
    if (ref / "predictions" / "predictions.csv").exists():
        out.append(("first_training", ref, "aucun", "zscore"))
    for w in sweep.PRETRAINED:
        for inp in sweep.INPUTS:
            exp = sweep.run_dir(w, inp) / "fold0_seed0"
            if (exp / "predictions" / "predictions.csv").exists():
                out.append((sweep.run_dir(w, inp).name, exp, w, inp))
    return out


def duree_min(nom):
    log = ROOT / nom / "train.log"
    if not log.exists():
        return np.nan
    took = re.findall(r"Took ([\d.]+) seconds total", log.read_text(encoding="utf-8", errors="replace"))
    return float(took[-1]) / 60 if took else np.nan


def hasard(prior):
    """Nombre attendu de videos a +/-10 % de hr_prior pour une FC uniforme 40-120 BPM."""
    return float(np.sum([min(120, 1.1 * x) - max(40, 0.9 * x) for x in prior]) / 80)


def libelle(w, inp):
    return f"{LIB_POIDS[w]} · {inp}"


# --------------------------------------------------------------------------- #
def tab_sweep(df, n_attendus):
    lignes = ["| poids | entrée | époques (meilleure) | val. min | FC prédites (BPM) | Spearman | "
              "±10 % (entr. / val. / test / tout) | GRAZIANA OD1 / OD2 / OD3 | \\|r\\| onde–épaisseur | durée |",
              "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in df.itertuples():
        lignes.append(
            f"| {LIB_POIDS[r.pretrained]} | {r.input} | {r.epochs} ({r.best_epoch}) | {fr(r.val_loss_min, 3)} | "
            f"{fr(r.hr_min, 0)}–{fr(r.hr_max, 0)} | {fr(r.spearman_prior, 2)} | "
            f"{r.n_10pc_train} / {r.n_10pc_val} / {r.n_10pc_test} / **{r.n_10pc_all}** | "
            f"{fr(r.hr_OD1)} / {fr(r.hr_OD2)} / {fr(r.hr_OD3)} | {fr(r.abs_r_wave_thickness_median, 2)} | "
            f"{fr(r.duree_min, 0)} min |")
    att = (f"\n\n: Un entraînement par ligne ; la première est le premier entraînement (PhysNet de SiNC, "
           f"200 époques). ±10 % : vidéos dont la FC SiNC tombe à ±10 % de `hr_prior` "
           f"(17 / 3 / 3 / 23 vidéos ; une FC tirée au hasard entre 40 et 120 BPM en donnerait "
           f"{fr(n_attendus, 1)} sur 23). GRAZIANA : `hr_prior` = 60,7 BPM sur les trois réplicats. "
           "{#tbl-sweep}")
    ecrire_table("\n".join(lignes) + att, "tab_sweep")


def fig_grille(df):
    d = df[df["pretrained"] != "aucun"]
    if d.empty:
        absente("fig_grille", "Aucun entraînement du balayage n'est encore évalué.")
        return
    poids = [w for w in sweep.PRETRAINED if w in set(d["pretrained"])]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.18,
                        subplot_titles=("vidéos à ±10 % de hr_prior (sur 23)", "Spearman FC SiNC / hr_prior"))
    for col, champ, zmin, zmax, fmt in ((1, "n_10pc_all", 0, 23, "{:.0f}"), (2, "spearman_prior", -1, 1, "{:.2f}")):
        z = np.full((len(poids), len(sweep.INPUTS)), np.nan)
        for r in d.itertuples():
            z[poids.index(r.pretrained), sweep.INPUTS.index(r.input)] = getattr(r, champ)
        fig.add_trace(go.Heatmap(
            z=z, x=list(sweep.INPUTS), y=[LIB_POIDS[w] for w in poids], zmin=zmin, zmax=zmax,
            colorscale="Viridis" if col == 1 else "RdBu", reversescale=False, showscale=False,
            text=[[("" if np.isnan(v) else fmt.format(v)) for v in row] for row in z],
            texttemplate="%{text}", hovertemplate="%{y} · %{x} : %{text}<extra></extra>"), row=1, col=col)
    mise_en_page(fig, "", hauteur=380, legende=False)
    fig.update_yaxes(autorange="reversed")
    enregistrer(fig, "fig_grille")


def fig_val(runs):
    fig = go.Figure()
    for nom, exp, w, inp in runs:
        v = pd.read_csv(exp / "losses_val.csv")
        fig.add_trace(go.Scatter(x=v["epoch"], y=v["total"], name=libelle(w, inp), mode="lines",
                                 line=dict(color=C_POIDS[w], dash=DASH[inp], width=1.6),
                                 hovertemplate=libelle(w, inp) + "<br>époque %{x} : %{y:.3f}<extra></extra>"))
    mise_en_page(fig, "", hauteur=480)
    fig.update_xaxes(title_text="époque")
    fig.update_yaxes(title_text="perte de validation (bandwidth + sparsity)")
    fig.update_layout(legend={"y": -0.18})
    enregistrer(fig, "fig_val")


def fig_hr(runs, meilleur):
    fig = go.Figure()
    lim = [40, 125]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", showlegend=False, hoverinfo="skip",
                             line=dict(color="rgba(150,150,150,0.6)", dash="dot")))
    for x in (0.9, 1.1):
        fig.add_trace(go.Scatter(x=lim, y=[x * lim[0], x * lim[1]], mode="lines", showlegend=False,
                                 hoverinfo="skip", line=dict(color="rgba(150,150,150,0.3)", width=1)))
    boutons, n_fixe = [], 3
    i_best = 0
    for i, (nom, exp, w, inp) in enumerate(runs):
        p = pd.read_csv(exp / "predictions" / "predictions.csv")
        for split in ("train", "val", "test"):
            g = p[p["split"] == split]
            fig.add_trace(go.Scatter(
                x=g["hr_prior_BPM"], y=g["hr_sinc"], mode="markers", name=LIB_SPLIT[split],
                visible=False, marker=dict(color=C_SPLIT[split], size=9),
                error_y=dict(type="data", symmetric=False, array=g["hr_patch_max"] - g["hr_sinc"],
                             arrayminus=g["hr_sinc"] - g["hr_patch_min"], thickness=1, width=0, color=C_SPLIT[split]),
                customdata=np.c_[g["slug"], g["hr_patch_min"], g["hr_patch_max"]],
                hovertemplate=("%{customdata[0]}<br>hr_prior %{x:.1f}<br>SiNC %{y:.1f} BPM"
                               "<br>patchs %{customdata[1]:.1f}–%{customdata[2]:.1f}<extra></extra>")))
        if nom == meilleur:
            i_best = i
        vis = [True] * n_fixe + [False] * (3 * len(runs))
        vis[n_fixe + 3 * i:n_fixe + 3 * i + 3] = [True] * 3
        boutons.append(dict(label=libelle(w, inp), method="update", args=[{"visible": vis}]))
    for k in range(3):
        fig.data[n_fixe + 3 * i_best + k].visible = True
    fig.update_layout(updatemenus=[dict(buttons=boutons, direction="down", x=0.01, y=1.14, xanchor="left",
                                        yanchor="top", active=i_best, bgcolor="#2b2b2b",
                                        font={"color": COULEUR_TEXTE})])
    mise_en_page(fig, "", hauteur=520)
    fig.update_xaxes(title_text="FC de référence, hr_prior (BPM)", range=lim)
    fig.update_yaxes(title_text="FC SiNC (BPM)", range=lim)
    enregistrer(fig, "fig_hr")


def fig_test_ondes(runs, meilleur):
    run = [r for r in runs if r[0] == meilleur]
    if not run:
        absente("fig_test_ondes", "Aucun entraînement évalué.")
        return
    nom, exp, w, inp = run[0]
    pred = exp / "predictions"
    p = pd.read_csv(pred / "predictions.csv")
    g = p[p["split"] == "test"].sort_values("slug")
    n = len(g)
    fig = make_subplots(rows=n, cols=2, column_widths=[0.66, 0.34], horizontal_spacing=0.07, vertical_spacing=0.09,
                        subplot_titles=sum(([f"{s} : onde SiNC et pouls d'épaisseur (filtrés)", "spectre de l'onde SiNC"]
                                            for s in g["slug"]), []))
    std = lambda y: (y - y.mean()) / (y.std() + 1e-12)  # noqa: E731
    for i, r in enumerate(g.itertuples(), start=1):
        z = np.load(pred / f"{r.slug}.npz")
        fig.add_trace(go.Scatter(x=z["t_s"], y=std(z["wave_bp"]), name="SiNC", legendgroup="s", showlegend=i == 1,
                                 line=dict(color="#27a567", width=1.4)), row=i, col=1)
        fig.add_trace(go.Scatter(x=z["t_s"], y=std(z["thickness_bp"]), name="épaisseur (masque)", legendgroup="t",
                                 showlegend=i == 1, line=dict(color="#e08a1e", width=1.2)), row=i, col=1)
        fig.add_trace(go.Scatter(x=z["psd_bpm"], y=z["psd"] / z["psd"].max(), showlegend=False,
                                 line=dict(color="#27a567", width=1.4)), row=i, col=2)
        for v, c in ((r.hr_prior_BPM, "#c9c9c9"), (r.hr_svd_BPM, "#9b59b6"), (r.hr_thickness, "#e08a1e")):
            if np.isfinite(v):
                fig.add_vline(x=v, line=dict(color=c, width=1, dash="dash"), row=i, col=2)
        fig.update_xaxes(title_text="temps (s)" if i == n else None, row=i, col=1)
        fig.update_xaxes(title_text="BPM" if i == n else None, row=i, col=2)
    mise_en_page(fig, f"{libelle(w, inp)}", hauteur=300 * n + 100)
    enregistrer(fig, "fig_test_ondes")


# --------------------------------------------------------------------------- #
def main():
    print("->", SORTIE)
    (SORTIE / "plotlyjs.html").write_text(f'<script type="text/javascript">{get_plotlyjs()}</script>',
                                          encoding="utf-8")
    runs = entrainements()
    if not runs:
        for nom in ("tab_sweep", "fig_grille", "fig_val", "fig_hr", "fig_test_ondes"):
            absente(nom, "Aucun entraînement n'est encore évalué.")
        return
    rows = []
    for nom, exp, w, inp in runs:
        m = sweep.metrics(nom, exp, w, inp)
        m["duree_min"] = duree_min(nom)
        rows.append(m)
    df = pd.DataFrame(rows)
    for c in ("hr_OD1", "hr_OD2", "hr_OD3"):
        if c not in df:
            df[c] = np.nan
    prior = pd.read_csv(runs[0][1] / "predictions" / "predictions.csv")["hr_prior_BPM"]
    n_att = hasard(prior)
    balayage = df[df["pretrained"] != "aucun"]
    # Meilleur entrainement du balayage : le plus de videos a +/-10 %, puis le Spearman.
    meilleur_nom = (balayage.sort_values(["n_10pc_all", "spearman_prior"], ascending=False)["run"].iloc[0]
                    if not balayage.empty else runs[0][0])
    tab_sweep(df, n_att)
    fig_grille(df)
    fig_val(runs)
    fig_hr(runs, meilleur_nom)
    fig_test_ondes(runs, meilleur_nom)
    RESUME.update({"n_entrainements": int(len(balayage)), "hasard_10pc": round(n_att, 1), "meilleur": meilleur_nom,
                   "table": df.round(3).to_dict(orient="records")})
    (SORTIE / "resume.txt").write_text(json.dumps(RESUME, indent=2, ensure_ascii=False, default=float),
                                       encoding="utf-8")
    print("  resume.txt")


if __name__ == "__main__":
    main()
