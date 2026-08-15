"""Figures de la phase B (grille combinee) pour la page "Mixtures" du site.

Reprend notebook/simulate_svd_radial_gaussians_viewer.ipynb : la cellule 9
(courbes phase B, toutes conditions superposees) pour la vue d'ensemble, puis
les cellules 13 / 14 / 16 / 18 pour le detail d'UN replicat par condition.

Encodage impose par la commande : tout ce qui est courbe part en **SVG**
(vectoriel), tout ce qui est image (cartes spatiales) part en **PNG**.

Produit dans ce dossier :
  phaseB_overlay_jitter_y.svg   -- 3 metriques, jitter_y en abscisse
  phaseB_overlay_jitter_x.svg   -- idem, jitters echanges
  phaseB_overlay_noise.svg      -- bruit en abscisse (casiers de quantiles)
  cond_<slug>_svd.svg           -- spectre de S + U*S temporels + leur spectre
  cond_<slug>_spatial.png       -- vecteurs singuliers droits (cartes spatiales)
  cond_<slug>_pulse.svg         -- pulse reconstruit + periodogramme + poids
  cond_<slug>_phase.svg         -- phase imposee vs mesuree + FC instantanee
  _conditions.json              -- metadonnees par condition, pour les .qmd
  _sidebar_snippet.yml          -- entrees de nav a coller dans _quarto.yml

et les sous-pages `../mixtures/cond_<slug>.qmd` (une par condition).

Le replicat montre par condition est celui d'erreur de frequence MEDIANE parmi
les 15 : un cas representatif, ni le meilleur ni le pire.

Relancer (kernel pyOR, depuis la racine du depot) :
  C:/Users/transformer/anaconda3/envs/pyOR/python.exe \
      reveal_quarto_presentations/figures_mixtures/make_figures_mixtures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "Astronauts"))
import simulate_svd_radial_gaussians as sim  # noqa: E402

from ocularrigidity.motion.pulsation.rate import lomb_scargle_power  # noqa: E402

SORTIE = Path(__file__).parent
SORTIE.mkdir(exist_ok=True)
PAGES = SORTIE.parent / "mixtures"
PAGES.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# Style : identique aux autres dossiers de figures (fond transparent, texte
# gris clair -- le site s'ouvre en theme sombre).
# --------------------------------------------------------------------------- #
COULEUR_TEXTE = "#c9c9c9"
C_GRIS = "#9a9a9a"
C_SIGNAL = "#2a78d6"
C_BRUIT = "#eb6834"
C_ROUGE = "#d1495b"
C_VERT = "#48a878"
C_VIOLET = "#a06cd5"

plt.rcParams.update({
    "savefig.transparent": True,
    "savefig.dpi": 190,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "text.color": COULEUR_TEXTE,
    "axes.labelcolor": COULEUR_TEXTE,
    "axes.edgecolor": COULEUR_TEXTE,
    "xtick.color": COULEUR_TEXTE,
    "ytick.color": COULEUR_TEXTE,
    "axes.facecolor": "none",
    "figure.facecolor": "none",
    # Ces figures sont larges (3 panneaux) et s'affichent sur ~1100 px via
    # `.column-page` : a 100 px/pouce le rendu est ~1:1, donc la taille de
    # police ici est bien celle vue a l'ecran. Ne pas descendre en dessous.
    "font.size": 10,
    "axes.titlesize": 11,
    "legend.frameon": False,
    "legend.labelcolor": COULEUR_TEXTE,
    "grid.color": "#6a6a6a",
    # Texte reste du texte dans le SVG (selectionnable, et fichier plus leger
    # qu'avec des chemins) ; les identifiants ne changent pas d'un run a
    # l'autre, donc regenerer ne cree pas un diff artificiel.
    "svg.fonttype": "none",
    "svg.hashsalt": "mixtures",
})

FREQ_ERR_THRESHOLD = 0.10   # meme seuil que le notebook
N_SHOW = 10                 # composantes SVD affichees (+ les retenues au-dela)


def sauver_png_palette(fig, chemin: Path, dpi: int = 100) -> None:
    """PNG en palette 255 couleurs + index transparent.

    Les cartes spatiales sont du bruit colore : en RGBA elles ne se compressent
    pas (2 Mo par planche, 45 planches). La colormap n'a de toute facon que 256
    teintes, donc la palette ne perd rien de visible et divise le poids par ~2.5.
    L'index 255 sert de transparence, pour garder le fond transparent du site.
    """
    from io import BytesIO

    from PIL import Image

    tampon = BytesIO()
    fig.savefig(tampon, format="png", dpi=dpi)
    tampon.seek(0)
    im = Image.open(tampon).convert("RGBA")
    alpha = im.getchannel("A")
    palette = im.convert("RGB").quantize(colors=255, dither=Image.Dither.NONE)
    palette.paste(255, alpha.point(lambda a: 255 if a < 128 else 0))
    palette.info["transparency"] = 255
    palette.save(chemin, optimize=True)


# --------------------------------------------------------------------------- #
# Chargement
# --------------------------------------------------------------------------- #
def charger_summary() -> pd.DataFrame:
    df = pd.read_csv(sim.SUMMARY_CSV)
    df["abs_corr"] = df["corr_combined_vs_truth"].abs()
    df["detected"] = df["confidence"].isin(["high", "medium"]) & (
        df["freq_rel_err"] <= FREQ_ERR_THRESHOLD
    )
    return df


def resoudre_npz(row) -> Path:
    """`npz_path` de summary.csv pointe encore sur l'ancien emplacement (C:).

    Les .npz vivent desormais sous ``sim.OUT_ROOT`` (E:), cf. le commentaire
    "Sortie sur E: et non a cote du code" du script de simulation. On retombe
    donc sur le tag, qui lui est stable.
    """
    p = Path(row["npz_path"])
    if p.exists():
        return p
    sous_dossier = {
        "B_combined": "sweep_combined",
        "A_amplitude": "sweep_amplitude",
        "A_noise": "sweep_noise",
        "A_jitter_x": "sweep_jitter_x",
        "A_jitter_y": "sweep_jitter_y",
    }[row["phase"]]
    return sim.OUT_ROOT / sous_dossier / f"{row['tag']}.npz"


# --------------------------------------------------------------------------- #
# 1. Vue d'ensemble -- toutes les conditions superposees (cellule 9 du notebook)
# --------------------------------------------------------------------------- #
PHASE_B_METRICS = (
    ("detected",     "mean",   "taux de detection",            (-0.05, 1.05), None),
    ("abs_corr",     "mean",   "|corr(combine, verite)|",      (-0.05, 1.05), None),
    ("freq_rel_err", "median", "erreur relative de frequence", None,          "log"),
)

PARAM_LABELS = {
    "amplitude_um": "amplitude (um)",
    "jitter_x_px": "sigma jitter x (px)",
    "jitter_y_px": "sigma jitter y (px)",
    "noise_level": "niveau de bruit (1 = median mesure)",
}

STYLES = ("-", "--", ":", "-.")


def figure_overlay(df, x_param, hue, style, nom, min_runs=3, x_bins=None):
    sub = df[df["phase"] == "B_combined"]
    n_failed = int((sub["confidence"] == "failed").sum())

    if x_bins is not None:
        sub = sub.copy()
        casiers = pd.qcut(sub[x_param], x_bins, duplicates="drop")
        sub[x_param] = casiers.map(lambda iv: iv.mid).astype(float)

    hues = sorted(sub[hue].unique())
    styles = sorted(sub[style].unique())
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(hues)))

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4))
    for ax, (metric, how, ylabel, ylim, yscale) in zip(axes, PHASE_B_METRICS):
        for ci, h in enumerate(hues):
            for si, s in enumerate(styles):
                cellule = sub[(sub[hue] == h) & (sub[style] == s)]
                if cellule.empty:
                    continue
                g = cellule.groupby(x_param)[metric]
                y = g.mean() if how == "mean" else g.median()
                y = y[g.size() >= min_runs].dropna()
                if y.empty:
                    continue
                ax.plot(y.index, y.values, marker="o", ms=4, lw=1.5,
                        color=colors[ci], ls=STYLES[si % len(STYLES)])
        ax.set_xlabel(PARAM_LABELS.get(x_param, x_param))
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if yscale is not None:
            ax.set_yscale(yscale)
        ax.grid(alpha=0.25)

    legende_hue = axes[0].legend(
        handles=[Line2D([], [], color=colors[i], lw=2.2, label=f"{h:g}")
                 for i, h in enumerate(hues)],
        title=PARAM_LABELS.get(hue, hue), fontsize=8, title_fontsize=8, loc="best",
    )
    axes[0].add_artist(legende_hue)
    axes[1].legend(
        handles=[Line2D([], [], color="0.55", lw=1.6, ls=STYLES[i % len(STYLES)],
                        label=f"{s:g}") for i, s in enumerate(styles)],
        title=PARAM_LABELS.get(style, style), fontsize=8, title_fontsize=8, loc="best",
    )
    fig.tight_layout()
    fig.savefig(SORTIE / nom)
    plt.close(fig)
    print(f"  {nom}  ({len(sub)} runs, {n_failed} en echec)")


# --------------------------------------------------------------------------- #
# 2. Detail d'un replicat -- une condition
# --------------------------------------------------------------------------- #
def composantes_affichees(K, selected):
    """Les `N_SHOW` premieres composantes, plus les retenues au-dela."""
    return sorted(set(range(min(N_SHOW, K))) | set(selected))


def figure_svd(d, slug):
    """Spectre des valeurs singulieres + vecteurs temporels + leur spectre."""
    U, S = d["U"], d["S"]
    uniform_time, kept_mask = d["uniform_time"], d["kept_mask"]
    t_kept = uniform_time[kept_mask]
    selected = set(d["selected_indices"].tolist())
    K = U.shape[1]
    montrees = composantes_affichees(K, selected)

    freqs_bpm = d["rate_freqs"] * 60.0
    power = d["rate_power"]          # (n_freqs, K) : deja calcule par le pipeline
    f_retenue = float(d["rate_freq"]) * 60.0

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6),
                             gridspec_kw={"width_ratios": [1, 1.25, 1.25]})

    # -- spectre des valeurs singulieres ------------------------------------
    axes[0].semilogy(np.arange(K), S, "o-", color=C_GRIS, ms=3, lw=0.9)
    for k in sorted(selected):
        axes[0].plot(k, S[k], "o", color=C_ROUGE, ms=8)
    axes[0].set_xlabel("composante")
    axes[0].set_ylabel("valeur singuliere")
    axes[0].set_title("Spectre des valeurs singulieres")
    axes[0].grid(alpha=0.2)
    axes[0].legend(handles=[Line2D([], [], color=C_ROUGE, marker="o", ls="",
                                   label="retenue")], fontsize=8, loc="best")

    # -- vecteurs singuliers temporels U*S ----------------------------------
    US = U * S[None, :]
    ecart = 3 * float(np.nanstd(US[:, montrees]))
    for i, k in enumerate(montrees):
        retenue = k in selected
        axes[1].plot(t_kept, US[:, k] + i * ecart,
                     color=C_ROUGE if retenue else C_GRIS,
                     lw=1.3 if retenue else 0.7)
    axes[1].set_yticks([i * ecart for i in range(len(montrees))])
    axes[1].set_yticklabels([f"$u_{{{k}}}$" for k in montrees], fontsize=7)
    axes[1].set_xlabel("temps (s)")
    axes[1].set_title("Vecteurs singuliers temporels $u_k \\sigma_k$")

    # -- les memes, dans le domaine des frequences --------------------------
    for i, k in enumerate(montrees):
        p = power[:, k]
        p = p / np.nanmax(p) if np.nanmax(p) > 0 else p
        retenue = k in selected
        axes[2].plot(freqs_bpm, p + i * 1.15,
                     color=C_ROUGE if retenue else C_GRIS,
                     lw=1.3 if retenue else 0.7)
    axes[2].axvline(f_retenue, color=C_SIGNAL, lw=1.2, ls="--",
                    label=f"frequence retenue = {f_retenue:.1f} bpm")
    axes[2].set_yticks([i * 1.15 for i in range(len(montrees))])
    axes[2].set_yticklabels([f"$u_{{{k}}}$" for k in montrees], fontsize=7)
    axes[2].set_xlabel("BPM")
    axes[2].set_title("Lomb-Scargle de chaque composante\n(normalise par son maximum)")
    axes[2].legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(SORTIE / f"cond_{slug}_svd.svg")
    plt.close(fig)


def figure_spatiale(d, slug):
    """Vecteurs singuliers droits = cartes spatiales sur l'anneau (PNG)."""
    V = d["V"]
    uniform_time = d["uniform_time"]
    selected = set(d["selected_indices"].tolist())
    K = V.shape[1]
    montrees = composantes_affichees(K, selected)

    calib = sim.RealCalibration(
        scale_x_um=float(d["scale_x_um"]), scale_y_um=float(d["scale_y_um"]),
        fs_hz=float(d["fs_hz"]), n_frames=len(uniform_time),
        duration_s=(len(uniform_time) - 1) / float(d["fs_hz"]),
    )
    forme = sim.crop_shape(calib)
    masque = sim.build_fixed_mask(forme)

    # L'anneau, circulaire en microns, est une ellipse tres allongee dans la
    # grille de pixels (3.9 um/px axial contre 11.4 lateral) : un panneau est
    # ~3x plus haut que large, d'ou une figure haute. On rend donc a une
    # resolution moderee -- a 190 dpi la planche pese 1.5 Mo pour etre affichee
    # sur 700 px de large.
    n = len(montrees)
    ncol = min(5, n)
    nrow = int(np.ceil(n / ncol))
    largeur_panneau = 1.4
    fig, axes = plt.subplots(
        nrow, ncol,
        figsize=(largeur_panneau * ncol,
                 largeur_panneau * nrow * forme[0] / forme[1]),
        dpi=100,
    )
    axes = np.atleast_1d(axes).ravel()
    vmax = float(np.nanpercentile(np.abs(V[:, montrees]), 99.5))
    for ax, k in zip(axes, montrees):
        carte = np.full(masque.shape, np.nan)
        carte[masque] = V[:, k]
        ax.imshow(carte, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"$v_{{{k}}}$", color=C_ROUGE if k in selected else COULEUR_TEXTE)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    sauver_png_palette(fig, SORTIE / f"cond_{slug}_spatial.png")
    plt.close(fig)


def figure_pulse(d, slug):
    """Pulse reconstruit optimise, son periodogramme, et les poids."""
    uniform_time, kept_mask = d["uniform_time"], d["kept_mask"]
    t_kept = uniform_time[kept_mask]
    combined, r_true = d["combined_uniform"], d["r_true_uniform"]
    selected = d["selected_indices"]
    weights = d["weights"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8),
                             gridspec_kw={"width_ratios": [1.5, 1.2, 0.8]})

    # -- pulse reconstruit vs verite terrain --------------------------------
    signe = np.sign(np.corrcoef(combined[kept_mask], r_true[kept_mask])[0, 1]) or 1.0
    combined_scaled = (
        signe * combined / np.nanstd(combined[kept_mask]) * np.nanstd(r_true[kept_mask])
    )
    axes[0].plot(uniform_time, r_true, color=C_GRIS, lw=1.6,
                 label="deplacement radial impose (verite)")
    axes[0].plot(uniform_time, combined_scaled, color=C_SIGNAL, lw=1.0, ls="--",
                 label="pulse reconstruit (mis a l'echelle, signe aligne)")
    axes[0].set_xlabel("temps (s)")
    axes[0].set_ylabel("deplacement radial (um, echelle relative)")
    axes[0].legend(fontsize=8.5, loc="upper right")
    axes[0].set_title("Pulse reconstruit vs verite terrain")

    # -- periodogramme du pulse reconstruit ---------------------------------
    combined_kept = combined[kept_mask] - np.nanmean(combined[kept_mask])
    freqs = d["rate_freqs"]
    power = lomb_scargle_power(t_kept, combined_kept, freqs)
    f_true_mean_bpm = float(np.mean(d["f_true_uniform"])) * 60.0
    axes[1].semilogy(freqs * 60.0, power, color=C_SIGNAL, lw=1.2)
    axes[1].axvline(float(d["rate_freq"]) * 60.0, color=C_ROUGE, lw=1.4,
                    label=f"retenue = {float(d['rate_freq']) * 60:.1f} bpm")
    axes[1].axvline(f_true_mean_bpm, color=C_GRIS, ls="--", lw=1.2,
                    label=f"vraie (moyenne) = {f_true_mean_bpm:.1f} bpm")
    axes[1].set_xlabel("BPM")
    axes[1].set_ylabel("puissance Lomb-Scargle")
    axes[1].legend(fontsize=8.5, loc="best")
    axes[1].set_title("Periodogramme du pulse reconstruit")

    # -- poids de la combinaison optimisee ----------------------------------
    x = np.arange(len(selected))
    axes[2].bar(x, weights, color=C_ROUGE, width=0.6)
    axes[2].axhline(0, color=C_GRIS, lw=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f"#{int(k)}" for k in selected], fontsize=8)
    axes[2].set_xlabel("composante retenue")
    axes[2].set_ylabel("poids")
    axes[2].set_title("Poids de la combinaison")
    axes[2].grid(alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(SORTIE / f"cond_{slug}_pulse.svg")
    plt.close(fig)


def figure_phase(d, slug):
    """Phase imposee vs mesuree, et pulsation instantanee."""
    uniform_time = d["uniform_time"]
    phase_uniform, inst_bpm, good_u = d["phase_uniform"], d["inst_bpm"], d["good_uniform"]
    true_phase = sim.chirp_phase(uniform_time, float(uniform_time[-1])) % (2 * np.pi)
    f_true_bpm = d["f_true_uniform"] * 60.0

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True)
    axes[0].plot(uniform_time, phase_uniform, color=C_VERT, lw=0.9,
                 label="phase mesuree (Hilbert)")
    axes[0].plot(uniform_time, true_phase, color=C_GRIS, lw=1.2, ls=":",
                 label="phase imposee (chirp)")
    axes[0].fill_between(uniform_time, 0, 2 * np.pi, where=~good_u,
                         color="0.5", alpha=0.25, step="mid")
    axes[0].set_ylabel("phase (rad, mod $2\\pi$)")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_title("Phase imposee vs mesuree  (zones grisees : phase non fiable)")

    axes[1].plot(uniform_time, inst_bpm, color=C_VIOLET, lw=0.9,
                 label="pulsation instantanee mesuree")
    axes[1].plot(uniform_time, f_true_bpm, color=C_GRIS, lw=1.2, ls=":",
                 label="pulsation instantanee imposee (chirp)")
    axes[1].fill_between(uniform_time, np.nanmin(f_true_bpm), np.nanmax(f_true_bpm),
                         where=~good_u, color="0.5", alpha=0.25, step="mid")
    axes[1].set_xlabel("temps (s)")
    axes[1].set_ylabel("BPM instantane")
    axes[1].legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(SORTIE / f"cond_{slug}_phase.svg")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 3. Selection du replicat et generation des pages
# --------------------------------------------------------------------------- #
def slug_condition(amp, jx, jy) -> str:
    return f"A{amp:g}_Jx{jx:g}_Jy{jy:g}".replace(".", "p")


def replicat_median(groupe: pd.DataFrame) -> pd.Series:
    """Le replicat d'erreur de frequence mediane : un cas representatif.

    Prendre le meilleur donnerait une page flatteuse, le pire une page
    alarmiste ; la mediane est celle qu'on peut montrer sans commentaire.
    """
    ok = groupe[groupe["confidence"] != "failed"]
    ok = ok if not ok.empty else groupe
    ordonne = ok.sort_values("freq_rel_err")
    return ordonne.iloc[len(ordonne) // 2]


GABARIT_PAGE = """---
title: "{titre}"
subtitle: "Un replicat representatif — {sous_titre}"
# Pas de TOC : la page est courte et repetitive, et sans elle les figures
# `.column-page-right` disposent de la marge droite sans passer dessous.
toc: false
---

[← Retour à Mixtures](../sim-mixtures.qmd)

## La condition

| | |
|:--|:--|
| amplitude imposée | **{amplitude:g} µm** crête à crête |
| jitter $x$ (commun à l'image) | **{jitter_x:g} px** |
| jitter $y$ (par A-scan) | **{jitter_y:g} px** |
| réplicats de la condition | {n_reps} |
| taux de détection sur la condition | **{taux_detection:.0%}** |
| \\|corr\\| moyenne sur la condition | {corr_moyen:.2f} |
| erreur de fréquence médiane sur la condition | {err_mediane:.1%} |

Le réplicat montré ci-dessous est celui d'**erreur de fréquence médiane** parmi
les {n_reps} — ni le meilleur, ni le pire.

| | |
|:--|:--|
| tag | `{tag}` |
| niveau de bruit tiré | {noise_level:.2f} ({bruit_commentaire}) |
| confiance `LombScargle` | **{confidence}** |
| erreur relative de fréquence | **{freq_rel_err:.1%}** |
| corrélation avec la vérité terrain | **{corr:.3f}** |
| composantes retenues | {n_selected} — {liste_selected} |

## Valeurs singulières et vecteurs temporels

![](../figures_mixtures/cond_{slug}_svd.svg){{.column-page-right}}

Les {n_show} premières composantes, plus toute composante retenue au-delà. En
rouge : celles que `OptimizedSpectralCombination` a gardées. Le panneau de
droite est le périodogramme de Lomb-Scargle calculé par le pipeline pour chaque
composante — c'est sur lui que la sélection se joue.

## Vecteurs singuliers spatiaux

![](../figures_mixtures/cond_{slug}_spatial.png){{.column-page-right}}

Les vecteurs singuliers droits $v_k$, remis sur la géométrie de l'anneau. Une
composante qui porte la pulsation montre une alternance rouge/bleu régulière le
long du bord ; une composante de bruit ne montre aucune structure.

## Pulse reconstruit, périodogramme et poids

![](../figures_mixtures/cond_{slug}_pulse.svg){{.column-page-right}}

À gauche le pulse reconstruit superposé au déplacement radial imposé (mis à
l'échelle, signe aligné) ; au centre son périodogramme, avec la fréquence
retenue et la fréquence moyenne vraie du chirp ; à droite le poids accordé à
chaque composante retenue dans la combinaison.

## Phase imposée vs mesurée, et pulsation instantanée

![](../figures_mixtures/cond_{slug}_phase.svg){{.column-page-right}}

En haut la phase de Hilbert du pulse reconstruit contre la phase vraie du
chirp ; en bas la pulsation instantanée qui en dérive, contre la fréquence
instantanée imposée (1.0 → 1.3 Hz). Les zones grisées sont les segments que le
pipeline marque comme phase non fiable (`good_uniform` faux).

::: {{.source-note}}
`Astronauts/simulate_svd_radial_gaussians.py` — run `{tag}` —
figures : `figures_mixtures/make_figures_mixtures.py`
:::
"""


def commentaire_bruit(niveau: float) -> str:
    if niveau < 0.85:
        return "sous le bruit median mesure"
    if niveau > 1.15:
        return "au-dessus du bruit median mesure"
    return "proche du bruit median mesure"


def ecrire_table(conditions) -> None:
    """Table des 45 conditions, incluse par `sim-mixtures.qmd`.

    Nom prefixe par `_` : Quarto ne rend pas ces fichiers comme des pages, ils
    ne servent que d'inclusion.
    """
    lignes = [
        "<!-- Genere par figures_mixtures/make_figures_mixtures.py "
        "-- ne pas editer a la main. -->",
        "",
        "| amplitude | jitter $x$ | jitter $y$ | détection | \\|corr\\| | "
        "err. fréq. médiane | page |",
        "|--:|--:|--:|--:|--:|--:|:--|",
    ]
    for c in conditions:
        lignes.append(
            f"| {c['amplitude']:g} µm | {c['jitter_x']:g} px | {c['jitter_y']:g} px "
            f"| {c['taux_detection']:.0%} | {c['corr_moyen']:.2f} "
            f"| {c['err_mediane']:.1%} "
            f"| [voir](mixtures/cond_{c['slug']}.qmd) |"
        )
    (PAGES / "_table.qmd").write_text("\n".join(lignes) + "\n", encoding="utf-8")


def main() -> None:
    df = charger_summary()
    b = df[df["phase"] == "B_combined"]
    print(f"{len(b)} runs de phase B, {int((b['confidence'] == 'failed').sum())} en echec")

    print("\n--- Vue d'ensemble ---")
    figure_overlay(df, "jitter_y_px", "amplitude_um", "jitter_x_px",
                   "phaseB_overlay_jitter_y.svg")
    figure_overlay(df, "jitter_x_px", "amplitude_um", "jitter_y_px",
                   "phaseB_overlay_jitter_x.svg")
    figure_overlay(df, "noise_level", "amplitude_um", "jitter_y_px",
                   "phaseB_overlay_noise.svg", x_bins=4)

    print("\n--- Une page par condition ---")
    conditions = []
    for (amp, jx, jy), groupe in b.groupby(["amplitude_um", "jitter_x_px", "jitter_y_px"]):
        slug = slug_condition(amp, jx, jy)
        row = replicat_median(groupe)
        chemin = resoudre_npz(row)
        d = np.load(chemin, allow_pickle=False)

        figure_svd(d, slug)
        figure_spatiale(d, slug)
        figure_pulse(d, slug)
        figure_phase(d, slug)

        selected = d["selected_indices"].tolist()
        meta = dict(
            slug=slug, amplitude=float(amp), jitter_x=float(jx), jitter_y=float(jy),
            titre=f"A = {amp:g} µm, jitter x = {jx:g} px, jitter y = {jy:g} px",
            sous_titre=(f"amplitude {amp:g} µm, jitter x {jx:g} px, jitter y {jy:g} px"),
            n_reps=int(len(groupe)),
            taux_detection=float(groupe["detected"].mean()),
            corr_moyen=float(groupe["abs_corr"].mean()),
            err_mediane=float(groupe["freq_rel_err"].median()),
            tag=str(row["tag"]),
            noise_level=float(row["noise_level"]),
            bruit_commentaire=commentaire_bruit(float(row["noise_level"])),
            confidence=str(row["confidence"]),
            freq_rel_err=float(row["freq_rel_err"]),
            corr=float(row["corr_combined_vs_truth"]),
            n_selected=int(len(selected)),
            liste_selected=", ".join(f"#{k}" for k in selected),
            n_show=N_SHOW,
        )
        conditions.append(meta)
        (PAGES / f"cond_{slug}.qmd").write_text(
            GABARIT_PAGE.format(**meta), encoding="utf-8"
        )
        print(f"  {slug}  detection {meta['taux_detection']:.0%}  "
              f"|corr| {meta['corr_moyen']:.2f}")

    (SORTIE / "_conditions.json").write_text(
        json.dumps(conditions, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    ecrire_table(conditions)

    # Entrees de nav, groupees par amplitude : 45 pages a plat seraient
    # illisibles dans la sidebar.
    lignes = []
    for amp in sorted({c["amplitude"] for c in conditions}):
        lignes.append(f'          - section: "A = {amp:g} µm"')
        lignes.append("            contents:")
        for c in [c for c in conditions if c["amplitude"] == amp]:
            lignes.append(
                f'              - text: "jx {c["jitter_x"]:g} / jy {c["jitter_y"]:g}"'
            )
            lignes.append(f'                href: mixtures/cond_{c["slug"]}.qmd')
    (SORTIE / "_sidebar_snippet.yml").write_text("\n".join(lignes) + "\n", encoding="utf-8")

    print(f"\n{len(conditions)} conditions -> {PAGES}")
    print(f"snippet de nav : {SORTIE / '_sidebar_snippet.yml'}")


if __name__ == "__main__":
    main()
