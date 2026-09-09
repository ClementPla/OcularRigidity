# -*- coding: utf-8 -*-
"""
Figures -- section « Strain as a predictor of SANS (first experiment) ».

Ce script NE CALCULE RIEN d'autre que des statistiques de lecture (p-valeurs
deduites de r et n, niveaux du hasard par tirage). Tout le reste est lu dans les
tables ecrites par les scripts de lot.

Entrees
-------
    E:/NASA_Rigidity/SegmentationVariations/model1_scale_1.0_flatten_choroid_xcorr/
        demons_strain/conditions.csv        1 ligne / condition
        demons_strain/one_cycles.csv        1 ligne / (condition, one-cycle)
        demons_strain/bins.csv              + cas, sigma, bin
        demons_strain/regions.csv           + region (5 bandes laterales)
        demons_strain/ct_pulse_traces.npz   courbes CT-FIR et pouls-FIR
        sans_predictors/pooled_slopes.csv   pente, r, p par (condition, region)
        sans_predictors/repeatability.csv   ICC + composantes de variance
        sans_predictors/sans_correlations.csv  r/rho, p, q par issue et par vue

Sorties : un fragment Quarto par figure, nomme ``p<page><n>_<sujet>.qmd``, avec
le HTML Plotly dans un bloc brut ```{=html} et SANS plotly.js -- la bibliotheque
est inseree une fois par page via ``plotlyjs.html``. Les tables sont des tables
markdown pipe. ``resume.txt`` reprend tous les chiffres cites dans la prose.

Quatre pages, quatre prefixes :
    p1_  Pulse comparison: mask vs intensities
    p2_  Stress-strain
    p3_  Repeatability
    p4_  Prediction of SANS

Le fond est transparent et le texte gris : les memes figures passent sur le
theme clair et sur le theme sombre du site.

Lancer DEPUIS LA RACINE DU DEPOT (ce dossier est une JONCTION vers E:, un chemin
relatif calcule ici sortirait du depot) :
    python reveal_quarto_presentations/figures_sans_predictors/make_figures.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots
from scipy import stats

# --------------------------------------------------------------------------- #
# Entrees / sorties
# --------------------------------------------------------------------------- #
VARIANT = Path("E:/NASA_Rigidity/SegmentationVariations/model1_scale_1.0_flatten_choroid_xcorr")
DEMONS = VARIANT / "demons_strain"
SANS = VARIANT / "sans_predictors"

# `.absolute()` et surtout PAS `.resolve()` : ce dossier est une JONCTION vers
# E:, la resolution ferait sortir SORTIE du depot.
SORTIE = Path(__file__).absolute().parent

LABEL_CAS = {"local": "référence locale (bin le plus fin de ce one-cycle)",
             "global": "référence globale (moyenne des bins les plus fins)"}
COURT_CAS = {"local": "locale", "global": "globale"}
ORDRE_REGIONS = ["colonne +2", "colonne +1", "colonne 0 (centre)",
                 "colonne -1", "colonne -2"]

COULEUR_SIGMA = {5.0: "#2a78d6", 10.0: "#27a567", 12.0: "#d0a72b", 15.0: "#c2453f"}
COULEUR_CAS = {"local": "#2a78d6", "global": "#eb6834"}
COULEUR_REGION = {"colonne +2": "#2a78d6", "colonne +1": "#27a567",
                  "colonne 0 (centre)": "#9a9a9a", "colonne -1": "#d0a72b",
                  "colonne -2": "#c2453f"}

# Les TROIS agregats du strain RETINIEN, tous au meme rang : deux mesures de
# centre et une mesure de queue. Le p95 est celui de la MAGNITUDE, rendu avec
# son signe (voir `compute_demons_strain.strain_p95_signe`) : si la plus grande
# magnitude est negative, le p95 est negatif.
#
# Le strain CHOROIDIEN n'apparait qu'a UN endroit de la section, la figure de
# validation `p2_validation`, ou il est la seule grandeur a posseder une valeur
# attendue (la reference etant le bin de choroide la plus fine) : controle de
# methode, jamais marqueur. Il ne figure plus ni dans `regions.csv`, ni dans les
# pentes, ni dans les predicteurs. L'EPAISSEUR choroidienne, elle, reste
# l'abscisse de toute l'analyse.
AGREGATS = [("strain_retine", "moyenne", "moy", "#2a78d6"),
            ("strain_retine_med", "médiane", "med", "#27a567"),
            ("strain_retine_p95", "p95 signé", "p95", "#eb6834")]
ORDRE_AGREGATS = [lab for _, lab, _, _ in AGREGATS]
COULEUR_AGREGAT = {lab: coul for _, lab, _, coul in AGREGATS}
COULEUR_FAMILLE = {"strain": "#2a78d6", "pente": "#27a567", "pente_pool": "#8e5bd0",
                   "ct_pouls": "#eb6834", "pouls": "#d0a72b", "rigidite": "#c2453f",
                   "qc": "#9a9a9a"}


def agregat_de(nom):
    """Agregat du strain porte par un nom de predicteur, quelle que soit la
    famille : ``strain_retine_p95`` comme ``poolpente_CT_um_strain_retine_p95``.

    Les suffixes sont testes du PLUS LONG au plus court, sinon
    ``..._strain_retine_med`` serait pris pour la moyenne. Rend ``None`` sur un
    nom inconnu -- c'est ce qui ecarte silencieusement les anciennes colonnes
    choroidiennes si une table d'avant la refonte traine encore.
    """
    for col, lab, _, _ in sorted(AGREGATS, key=lambda a: -len(a[0])):
        if nom == col or nom.endswith("_" + col):
            return lab
    return None


COULEUR_TEXTE = "#c9c9c9"
C_REF = "#9a9a9a"
C_SEUIL = "#c2453f"
GRILLE = "rgba(150,150,150,0.18)"

CONFIG = {"displaylogo": False, "responsive": True,
          "modeBarButtonsToRemove": ["select2d", "lasso2d"]}

FICHIERS: list[str] = []
RESUME: dict = {}


def mise_en_page(fig, titre, hauteur=440, legende=True):
    """Habillage commun : fond transparent, texte gris, grille discrete."""
    fig.update_layout(
        title={"text": titre, "font": {"size": 14}, "x": 0.01, "xanchor": "left"},
        template="none",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COULEUR_TEXTE, "size": 11},
        height=hauteur,
        margin={"l": 70, "r": 20, "t": 70, "b": 95 if legende else 50},
        showlegend=legende,
        legend={"font": {"size": 10}, "orientation": "h", "yanchor": "top",
                "y": -0.10, "xanchor": "left", "x": 0},
        hoverlabel={"font": {"size": 11}},
    )
    fig.update_xaxes(gridcolor=GRILLE, zerolinecolor=GRILLE,
                     linecolor=GRILLE, ticks="outside", tickcolor=GRILLE)
    fig.update_yaxes(gridcolor=GRILLE, zerolinecolor=GRILLE,
                     linecolor=GRILLE, ticks="outside", tickcolor=GRILLE)
    for a in fig.layout.annotations or ():
        if a.text and a.font is not None:
            a.font.size = 11
    return fig


def enregistrer(fig, nom):
    """Fragment Quarto : le HTML Plotly dans un bloc brut ```{=html}.

    Sans ce bloc, `{{< include >}}` livre le fragment au lecteur markdown de
    Pandoc, qui prend le `<div>` de Plotly pour un div natif et se plaint de ne
    pas trouver sa fermeture.
    """
    html = fig.to_html(full_html=False, include_plotlyjs=False,
                       config=CONFIG, div_id=f"plot-{nom}")
    chemin = SORTIE / f"{nom}.qmd"
    chemin.write_text("```{=html}\n" + html + "\n```\n", encoding="utf-8")
    FICHIERS.append(chemin.name)
    print(f"  {chemin.name}")


def ecrire_table(texte, nom):
    entete = ("<!-- Genere par figures_sans_predictors/make_figures.py"
              " -- ne pas editer a la main. -->" + chr(10) + chr(10))
    (SORTIE / f"{nom}.qmd").write_text(entete + texte + chr(10), encoding="utf-8")
    FICHIERS.append(f"{nom}.qmd")
    print(f"  {nom}.qmd")


def cellule(txt) -> str:
    """Echappe les barres verticales d'un libelle destine a une CELLULE de table
    pipe.

    Plusieurs libelles du panel en contiennent une paire -- ``|u|max``,
    ``|r| CT / pouls`` -- et pandoc les lit comme des separateurs de cellules :
    la ligne se decale silencieusement d'autant de colonnes, sans erreur de
    rendu. L'echappement se fait ICI, au moment d'ecrire le markdown, et pas
    dans la constante : les memes libelles servent d'etiquette d'axe et de texte
    de survol a plotly, ou un antislash s'afficherait tel quel.
    """
    # chr(92) plutot qu'un antislash litteral : la chaine doit contenir UN
    # antislash, et l'ecrire echappe dans un source deja plein de LaTeX est
    # une source d'erreur classique.
    return str(txt).replace("|", chr(92) + "|")


def fmt_sci(x, chiffres=2, signe=False):
    """Notation scientifique en LaTeX, virgule decimale : 1,51 x 10^-4.

    ``f"{x:.2e}"`` donnerait ``1.51e-04``, qui s'affiche tel quel dans une table
    markdown -- illisible a cote d'une prose qui ecrit 2,5 x 10^-3.
    """
    if x is None or not np.isfinite(x):
        return "n/a"
    if x == 0:
        return "$0$"
    exposant = int(np.floor(np.log10(abs(x))))
    mantisse = x / 10.0 ** exposant
    fmt = f"{{:+.{chiffres}f}}" if signe else f"{{:.{chiffres}f}}"
    m = fmt.format(mantisse).replace(".", "{,}")
    return f"${m} \\times 10^{{{exposant}}}$"


def fmt_num(x, chiffres=2, signe=False):
    """Nombre decimal a virgule, dans une formule pour rester aligne."""
    if x is None or not np.isfinite(x):
        return "n/a"
    fmt = f"{{:+.{chiffres}f}}" if signe else f"{{:.{chiffres}f}}"
    return "$" + fmt.format(x).replace(".", "{,}") + "$"


def med_iqr(s, fmt="{:.3g}"):
    s = pd.Series(s).dropna()
    if s.empty:
        return "n/a"
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return f"{fmt.format(s.median())} [{fmt.format(q1)} – {fmt.format(q3)}]"


def p_de_r(r, n):
    """p bilateral du coefficient de correlation, deduit de r et n.

    ``one_cycles.csv`` porte r et n mais pas p : la statistique de Student
    t = r sqrt((n-2)/(1-r^2)) suit une loi de Student a n-2 degres de liberte
    sous l'hypothese nulle, ce qui evite de recharger les traces pour un chiffre
    que r et n determinent entierement.
    """
    r = np.asarray(r, dtype=float)
    n = np.asarray(n, dtype=float)
    ok = np.isfinite(r) & np.isfinite(n) & (n > 2) & (np.abs(r) < 1)
    p = np.full(r.shape, np.nan)
    rr = r[ok]
    nn = n[ok]
    t = rr * np.sqrt((nn - 2) / np.maximum(1 - rr ** 2, 1e-15))
    p[ok] = 2 * stats.t.sf(np.abs(t), nn - 2)
    return p


def bh_fdr(p):
    """q-valeurs de Benjamini-Hochberg, les p non finis ecartes du decompte.

    Copie fidele de ``compute_sans_predictors._bh_fdr`` (qui n'utilise pas
    statsmodels, absent de l'environnement pyOR). Elle n'est employee ICI qu'en
    REPLI, quand ``sans_correlations.csv`` ne porte pas encore la colonne
    ``pearson_q_strain`` : la figure doit pouvoir etre refaite sans relancer un
    lot de quarante minutes. Les deux chemins portent sur le meme jeu de lignes
    et donnent donc les memes nombres.
    """
    p = np.asarray(p, dtype=float)
    out = np.full(p.size, np.nan)
    fini = np.isfinite(p)
    if not fini.any():
        return out
    pf = p[fini]
    n = pf.size
    order = np.argsort(pf)
    ranked = pf[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n, dtype=float)
    q[order] = np.clip(ranked, 0.0, 1.0)
    out[fini] = q
    return out


# --------------------------------------------------------------------------- #
# Chargement
# --------------------------------------------------------------------------- #
def _lire(path, quoi):
    if not path.exists():
        raise SystemExit(f"{path} absent -- lancer d'abord {quoi}")
    return pd.read_csv(path)


_LOT = "Astronauts/compute_demons_strain.py"
_SANS = "Astronauts/compute_sans_predictors.py"

conditions = _lire(DEMONS / "conditions.csv", _LOT)
one_cycles = _lire(DEMONS / "one_cycles.csv", _LOT)
bins = _lire(DEMONS / "bins.csv", _LOT)
regions = _lire(DEMONS / "regions.csv", _LOT)
pooled = _lire(SANS / "pooled_slopes.csv", _SANS)
repeat = _lire(SANS / "repeatability.csv", _SANS)
corr = _lire(SANS / "sans_correlations.csv", _SANS)
stab = _lire(SANS / "stability.csv", _SANS)
repeat_oc = _lire(SANS / "repeatability_one_cycles.csv", _SANS)

traces_path = DEMONS / "ct_pulse_traces.npz"
if not traces_path.exists():
    raise SystemExit(f"{traces_path} absent -- lancer Astronauts/compute_ct_pulse_traces.py")
traces = np.load(traces_path, allow_pickle=True)

gate = _lire(SANS / "one_cycle_filter.csv", _SANS)
# Les six issues seules, une ligne par sujet (4 ko). C'est la seule table du
# lot ou les issues se lisent SANS les predicteurs : `subject_values.csv` les
# porte aussi mais pese 245 Mo, illisible ici.
outcomes = _lire(SANS / "outcomes.csv", _SANS)

# --------------------------------------------------------------------------- #
# Le portail de validite des one-cycles
# --------------------------------------------------------------------------- #
# `one_cycle_filter.csv` est la SEULE definition du portail ; elle est produite
# par `compute_sans_predictors.py`, qui filtre ses predicteurs avec. On la relit
# ici pour filtrer les lectures DIRECTES des tables du lot, faute de quoi le
# site montrerait des nuages non filtres a cote de statistiques filtrees.
#
# Exception assumee : la page « Pulse comparison » travaille sur les donnees
# BRUTES. C'est elle qui etablit le portail -- deux de ses quatre criteres en
# sortent -- et montrer la selection sur les donnees deja selectionnees n'aurait
# aucun sens. Les variables `*_brut` lui sont reservees.
one_cycles_brut = one_cycles.copy()
conditions_brut = conditions.copy()
bins_brut = bins.copy()

_retenus = gate.loc[gate["retenu"], ["slug", "one_cycle"]]
_slugs_ok = set(gate.loc[gate["garde_condition"], "slug"])


def _filtrer(df):
    if "one_cycle" in df.columns:
        return df.merge(_retenus, on=["slug", "one_cycle"], how="inner")
    return df[df["slug"].isin(_slugs_ok)]


one_cycles = _filtrer(one_cycles)
bins = _filtrer(bins)
regions = _filtrer(regions)
conditions = _filtrer(conditions)

ok = conditions[conditions["status"] == "ok"]
SIGMAS = sorted(bins["sigma"].unique())
CAS = [c for c in ("local", "global") if c in set(bins["cas"])]
REGIONS = [r for r in ORDRE_REGIONS if r in set(regions["region"])]

RESUME.update({
    "n_conditions": len(ok),
    "n_one_cycles": len(one_cycles),
    "n_recalages": len(bins),
    "one_cycles_par_condition_med": float(ok["n_one_cycles_ok"].median()),
    "crop_rows_med": float(ok["crop_rows"].median()),
    "duree_lot_h": float(ok["t_total_s"].sum() / 3600),
})


# =========================================================================== #
# PAGE 1 — Pulse comparison: mask vs intensities
# =========================================================================== #
# Le pouls est extrait deux fois de la meme video, par deux voies independantes :
# l'EPAISSEUR de la choroide segmentee (voie « masque ») et une combinaison de
# composantes SVD des INTENSITES (voie « intensites », methode 1_fir du lot).
# Les deux subissent ensuite le meme passe-bande FIR. S'ils mesurent la meme
# pulsation, ils doivent se correler.
# Donnees BRUTES : cette page etablit le portail, elle ne peut pas montrer
# la selection sur les donnees deja selectionnees.
oc = one_cycles_brut.copy()
oc["p_temps"] = p_de_r(oc["r_temps"], oc["n_temps"])
oc["p_bins"] = p_de_r(oc["r_bins"], oc["n_bins_used"])
for col in ("p_temps", "p_bins"):
    oc[f"mlog10_{col}"] = -np.log10(np.clip(oc[col], 1e-300, 1.0))

_n_t = int(oc["n_temps"].median())
_n_b = int(oc["n_bins_used"].median())
SUPPORTS = [("temps", f"support temporel (CT par frame, {_n_t} points médians)", "#2a78d6"),
            ("bins", f"support bins repliés ({_n_b} points)", "#eb6834")]

fig = go.Figure()
for cle, label, couleur in SUPPORTS:
    v = oc[f"mlog10_p_{cle}"].dropna()
    fig.add_trace(go.Histogram(
        x=v, name=label, marker={"color": couleur, "line": {"width": 0}},
        opacity=0.62, xbins={"start": 0, "size": 0.25},
        hovertemplate=label + "<br>−log10(p) %{x}<br>%{y} one-cycles<extra></extra>",
    ))
    RESUME[f"p1_mlogp_{cle}_med"] = float(v.median())
    RESUME[f"p1_frac_p05_{cle}"] = float((oc[f"p_{cle}"] < 0.05).mean())
    RESUME[f"p1_frac_p001_{cle}"] = float((oc[f"p_{cle}"] < 0.001).mean())
    RESUME[f"p1_n_{cle}_med"] = float(oc[("n_temps" if cle == "temps"
                                          else "n_bins_used")].median())
fig.add_vline(x=-np.log10(0.05), line={"color": C_SEUIL, "dash": "dash", "width": 1.5},
              annotation={"text": "p = 0,05", "font": {"size": 10, "color": C_SEUIL}})
fig.add_vline(x=-np.log10(0.001), line={"color": C_SEUIL, "dash": "dot", "width": 1},
              annotation={"text": "p = 0,001", "font": {"size": 10, "color": C_SEUIL}})
fig.update_layout(barmode="overlay", bargap=0.02)
fig.update_xaxes(title="−log10(p) de la corrélation masque / intensités", range=[0, 12])
fig.update_yaxes(title="nombre de one-cycles")
mise_en_page(fig, f"Corrélation entre les deux voies d'extraction — "
                  f"{len(oc)} one-cycles, 104 conditions confondues", hauteur=470)
enregistrer(fig, "p1_hist_logp")


# --- Les deux pouls superposes, toutes conditions --------------------------
# Chaque courbe est CENTREE-REDUITE : le pouls d'intensite est sans unite (c'est
# une combinaison de composantes SVD) et le CT est en micrometres. Les
# superposer bruts ne dirait rien ; reduits, la question devient lisible -- les
# deux courbes montent-elles ensemble ?
#
# Le signe du pouls d'intensite est fixe arbitrairement par le lot d'extraction.
# Il est donc RETOURNE ici quand la correlation avec le CT est negative, sinon la
# moitie des conditions apparaitrait en opposition de phase pour une raison qui
# n'est pas physiologique. Le retournement est compte et rapporte.
slugs = [str(s) for s in traces["slugs"]]
r_par_cond = one_cycles_brut.groupby("slug")["r_temps"].median()

fig = make_subplots(rows=2, cols=1, row_heights=[0.62, 0.38], vertical_spacing=0.13,
                    subplot_titles=("les 104 conditions superposées",
                                    "une condition, en clair"))
n_retourne = 0
for slug in slugs:
    u = np.asarray(traces[f"{slug}__u_time"], dtype=float)
    core = np.asarray(traces[f"{slug}__core"], dtype=bool)
    ct = np.asarray(traces[f"{slug}__ct_fir"], dtype=float)
    pu = np.asarray(traces[f"{slug}__pulse_fir"], dtype=float)
    if core.sum() < 10:
        continue
    signe = -1.0 if float(r_par_cond.get(slug, 0.0)) < 0 else 1.0
    n_retourne += int(signe < 0)

    def z(y):
        y = y[core]
        return (y - y.mean()) / (y.std() + 1e-12)

    fig.add_trace(go.Scattergl(
        x=u[core], y=z(pu), mode="lines", showlegend=False, legendgroup="pouls",
        line={"color": "rgba(42,120,214,0.13)", "width": 0.7}, hoverinfo="skip"),
        row=1, col=1)
    fig.add_trace(go.Scattergl(
        x=u[core], y=z(ct), mode="lines", showlegend=False, legendgroup="ct",
        line={"color": "rgba(235,104,52,0.13)", "width": 0.7}, hoverinfo="skip"),
        row=1, col=1)

# Deux traces fantomes, seulement pour la legende (les 208 courbes ci-dessus
# sont muettes : 208 entrees de legende seraient illisibles).
for nom, couleur in (("pouls d'intensité (FIR)", "#2a78d6"),
                     ("épaisseur choroïdienne CT (FIR)", "#eb6834")):
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=nom,
                             line={"color": couleur, "width": 2.5}), row=1, col=1)

# La condition detaillee : celle dont le |r| median est le plus proche de la
# mediane de la cohorte -- la plus BANALE, pas la plus belle.
cible = float(one_cycles_brut["r_temps"].abs().median())
slug_ref = (r_par_cond.abs() - cible).abs().idxmin()
u = np.asarray(traces[f"{slug_ref}__u_time"], dtype=float)
core = np.asarray(traces[f"{slug_ref}__core"], dtype=bool)
signe = -1.0 if float(r_par_cond.get(slug_ref, 0.0)) < 0 else 1.0
for cle, nom, couleur, mult in (("pulse_fir", "pouls d'intensité", "#2a78d6", signe),
                                ("ct_fir", "CT", "#eb6834", 1.0)):
    y = np.asarray(traces[f"{slug_ref}__{cle}"], dtype=float)[core] * mult
    y = (y - y.mean()) / (y.std() + 1e-12)
    fig.add_trace(go.Scatter(x=u[core], y=y, mode="lines", showlegend=False,
                             line={"color": couleur, "width": 1.6},
                             hovertemplate=nom + "<br>t %{x:.1f} s<br>%{y:+.2f} σ<extra></extra>"),
                  row=2, col=1)
fig.update_xaxes(title="temps (s)", row=2, col=1)
fig.update_yaxes(title="écart-type", row=1, col=1)
fig.update_yaxes(title="écart-type", row=2, col=1)
fig.layout.annotations[1].text = (f"{slug_ref} — r = {r_par_cond[slug_ref]:+.2f}, "
                                  f"la condition la plus banale de la cohorte")
mise_en_page(fig, "Les deux voies d'extraction superposées, après le même FIR "
                  f"(signe du pouls retourné sur {n_retourne} conditions sur "
                  f"{len(slugs)})", hauteur=720)
enregistrer(fig, "p1_overlay")

RESUME.update({
    "p1_slug_ref": slug_ref,
    "p1_r_slug_ref": float(r_par_cond[slug_ref]),
    "p1_n_retournes": n_retourne,
    "p1_r_temps_absmed": float(one_cycles_brut["r_temps"].abs().median()),
    "p1_r_bins_absmed": float(one_cycles_brut["r_bins"].abs().median()),
})


# --- Le portail : ce qu'il retire, et a qui ---------------------------------
# Trois lectures dans une figure : le rendement de chaque critere, la
# distribution de la fraction retenue PAR VIDEO, et le nombre de criteres
# echoues par one-cycle (qui dit si les criteres se recouvrent ou se cumulent).
CRITERES = [("ok_pulsatilite", "pulsatilité < 5 %", "#2a78d6"),
            ("ok_phase", "phase valide (p < 0,05)", "#eb6834"),
            ("ok_frames", "≥ 5 frames / bin", "#27a567"),
            ("ok_jacobien", "jacobien non replié", "#9a9a9a")]

par_video = (gate.groupby("slug")
             .agg(n_total=("one_cycle", "size"),
                  n_retenus=("garde_one_cycle", "sum"),
                  garde=("garde_condition", "first")).reset_index())
par_video["frac_retirés"] = 100 * (1 - par_video["n_retenus"] / par_video["n_total"])

fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.085,
                    column_widths=[0.32, 0.38, 0.30],
                    subplot_titles=("rendement de chaque critère",
                                    "one-cycles retirés par vidéo",
                                    "critères échoués par one-cycle"))
noms = [lab for _, lab, _ in CRITERES] + ["<b>les quatre</b>"]
vals = [100 * gate[c].mean() for c, _, _ in CRITERES] + [100 * gate["garde_one_cycle"].mean()]
coul = [c for _, _, c in CRITERES] + ["#c2453f"]
fig.add_trace(go.Bar(y=noms[::-1], x=vals[::-1], orientation="h",
                     marker={"color": coul[::-1]}, showlegend=False,
                     text=[f"{v:.1f} %" for v in vals[::-1]], textposition="auto",
                     hovertemplate="%{y}<br>garde %{x:.1f} %<extra></extra>"),
              row=1, col=1)
fig.add_trace(go.Histogram(
    x=par_video["frac_retirés"], xbins={"start": 0, "end": 100, "size": 10},
    marker={"color": "#2a78d6"}, showlegend=False,
    hovertemplate="%{x} % retirés<br>%{y} vidéos<extra></extra>"), row=1, col=2)
_ech = gate["n_criteres_echoues"].value_counts().sort_index()
fig.add_trace(go.Bar(x=_ech.index.astype(str), y=_ech.to_numpy(),
                     marker={"color": ["#27a567", "#d0a72b", "#eb6834", "#c2453f", "#8e5bd0"][:len(_ech)]},
                     showlegend=False, text=_ech.to_numpy(), textposition="auto",
                     hovertemplate="%{x} critère(s) échoué(s)<br>%{y} one-cycles<extra></extra>"),
              row=1, col=3)
fig.update_xaxes(title="% des one-cycles gardés", range=[0, 105], row=1, col=1)
fig.update_xaxes(title="% de one-cycles retirés", row=1, col=2)
fig.update_yaxes(title="vidéos", row=1, col=2)
fig.update_xaxes(title="nombre de critères échoués", row=1, col=3)
fig.update_yaxes(title="one-cycles", row=1, col=3)
mise_en_page(fig, f"Le portail de validité — {int(gate['garde_one_cycle'].sum())} "
                  f"one-cycles retenus sur {len(gate)}, "
                  f"{int(par_video['garde'].sum())} vidéos sur {len(par_video)}",
             hauteur=470, legende=False)
enregistrer(fig, "p1_portail")

_n_retire_video = par_video.loc[~par_video["garde"]]
RESUME.update({
    "portail_n_one_cycles": int(len(gate)),
    "portail_n_retenus": int(gate["garde_one_cycle"].sum()),
    "portail_frac_retenus": float(gate["garde_one_cycle"].mean()),
    "portail_n_videos": int(len(par_video)),
    "portail_n_videos_gardees": int(par_video["garde"].sum()),
    "portail_frac_videos_gardees": float(par_video["garde"].mean()),
    "portail_frac_retires_par_video_med": float(par_video["frac_retirés"].median()),
    "portail_frac_retires_par_video_q1": float(par_video["frac_retirés"].quantile(0.25)),
    "portail_frac_retires_par_video_q3": float(par_video["frac_retirés"].quantile(0.75)),
    "portail_n_videos_tout_retire": int((par_video["n_retenus"] == 0).sum()),
    "portail_n_videos_rien_retire": int((par_video["n_retenus"] == par_video["n_total"]).sum()),
    "portail_one_cycles_apres_plancher": int(gate["retenu"].sum()),
})
for c, lab, _ in CRITERES:
    RESUME[f"portail_garde_{c}"] = float(gate[c].mean())
for k, v in gate["n_criteres_echoues"].value_counts().sort_index().items():
    RESUME[f"portail_n_echecs_{k}"] = int(v)


def table_portail() -> str:
    lignes = ["| critère | one-cycles gardés | seul motif d'échec |",
              "|:--|--:|--:|"]
    for c, lab, _ in CRITERES:
        seul = int((~gate[c] & (gate["n_criteres_echoues"] == 1)).sum())
        lignes.append(f"| {lab} | {100 * gate[c].mean():.1f} % | {seul} |")
    lignes.append(f"| **les quatre réunis** | **{100 * gate['garde_one_cycle'].mean():.1f} %** | — |")
    return chr(10).join(lignes)


def table_portail_videos() -> str:
    q1 = par_video["frac_retirés"].quantile(0.25)
    q3 = par_video["frac_retirés"].quantile(0.75)
    lignes = ["| | valeur |", "|:--|--:|",
              f"| vidéos analysées | {len(par_video)} |",
              f"| vidéos conservées (≥ 3 one-cycles valides) | "
              f"**{int(par_video['garde'].sum())}** ({100 * par_video['garde'].mean():.1f} %) |",
              f"| vidéos écartées | {int((~par_video['garde']).sum())} |",
              f"| … dont aucun one-cycle ne passe | {int((par_video['n_retenus'] == 0).sum())} |",
              f"| one-cycles retirés par vidéo, médiane | "
              f"{par_video['frac_retirés'].median():.0f} % |",
              f"| … intervalle interquartile | {q1:.0f} – {q3:.0f} % |",
              f"| vidéos dont rien n'est retiré | "
              f"{int((par_video['n_retenus'] == par_video['n_total']).sum())} |",
              f"| one-cycles retenus au total | "
              f"{int(gate['retenu'].sum())} / {len(gate)} "
              f"({100 * gate['retenu'].mean():.1f} %) |"]
    return chr(10).join(lignes)


ecrire_table(table_portail(), "tab_portail")
ecrire_table(table_portail_videos(), "tab_portail_videos")


# =========================================================================== #
# PAGE 2 — Stress-strain
# =========================================================================== #
# 2.a Le controle prealable : ce que la regularisation fait au champ.
fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.085,
                    subplot_titles=("amplitude du champ |u|max (µm)",
                                    "jacobien négatif (% des pixels)",
                                    "gain de RMS sur le tissu"))
b_gain = bins.assign(gain_rms=bins["rms_avant"] - bins["rms_apres"])
for j, col in enumerate(("u_max_um", "jac_neg_pct", "gain_rms"), start=1):
    for cas in CAS:
        for sigma in SIGMAS:
            s = b_gain[(b_gain["cas"] == cas) & (b_gain["sigma"] == sigma)]
            fig.add_trace(go.Box(
                y=s[col], x=[f"σ {sigma:g}"] * len(s),
                name=f"référence {COURT_CAS[cas]}", legendgroup=cas,
                showlegend=bool(j == 1 and sigma == SIGMAS[0]),
                offsetgroup=cas, boxpoints=False,
                marker={"color": COULEUR_CAS[cas], "size": 2},
                line={"width": 1, "color": COULEUR_CAS[cas]},
                hovertemplate=f"{cas}, σ {sigma:g}<br>%{{y:.3g}}<extra></extra>",
            ), row=1, col=j)
fig.update_layout(boxmode="group")
fig.update_yaxes(title="µm", row=1, col=1)
fig.update_yaxes(title="%", row=1, col=2)
fig.update_yaxes(title="niveaux", row=1, col=3)
mise_en_page(fig, "Contrôle préalable — effet de displacement_sigma sur le champ "
                  f"({len(bins)} recalages)", hauteur=440)
enregistrer(fig, "p2_sigma_control")

for sigma in SIGMAS:
    s = bins[bins["sigma"] == sigma]
    RESUME[f"p2_umax_sigma{sigma:g}"] = med_iqr(s["u_max_um"], "{:.1f}")
    RESUME[f"p2_jacneg_sigma{sigma:g}"] = float(s["jac_neg_pct"].max())


# 2.b LA validation : le strain choroidien retrouve-t-il la dilatation connue ?
#
# ATTENTION -- cette figure montre les donnees AVANT et APRES le portail, et
# c'est tout son interet. Le portail ecarte les one-cycles dont la pulsatilite
# depasse 5 % du CT ; ce faisant il retire exactement les points de plus grand
# dCT, qui sont ceux qui portaient la relation. La comparaison des deux nuages
# est le seul moyen de savoir laquelle des deux lectures est la bonne :
#   - soit la relation etait reelle et le portail l'attenue (restriction
#     d'etendue) -- mais une restriction d'etendue rapproche r de zero, elle
#     n'en INVERSE pas le signe ;
#   - soit la relation vivait dans les one-cycles ecartes, ou une erreur de
#     segmentation gonfle A LA FOIS le dCT mesure et la deformation que les
#     demons ont a ajuster : une correlation d'erreur partagee, pas de
#     physiologie.
# Le signe s'inverse. C'est la seconde lecture.
bl_brut = bins_brut[bins_brut["cas"] == "local"]
_keep = gate.loc[gate["retenu"], ["slug", "one_cycle"]]
bl_ret = bl_brut.merge(_keep, on=["slug", "one_cycle"], how="inner")
_cles_ret = set(map(tuple, _keep.to_numpy()))
bl_exc = bl_brut[~bl_brut[["slug", "one_cycle"]].apply(tuple, axis=1).isin(_cles_ret)]

fig = make_subplots(rows=1, cols=len(SIGMAS), shared_yaxes=True,
                    horizontal_spacing=0.022,
                    subplot_titles=[f"σ = {sg:g}" for sg in SIGMAS])
for j, sigma in enumerate(SIGMAS, start=1):
    for jeu, nom, couleur, taille in (
            (bl_exc, "écartés par le portail", "#9a9a9a", 3),
            (bl_ret, "retenus", "#2a78d6", 3.5)):
        g = jeu[jeu["sigma"] == sigma].dropna(subset=["strain_choroide_med", "dCT_um"])
        if g.empty:
            continue
        ech = g.sample(min(len(g), 2500), random_state=0)
        fig.add_trace(go.Scattergl(
            x=ech["dCT_um"], y=ech["strain_choroide_med"], mode="markers", name=nom,
            legendgroup=nom, showlegend=bool(j == 1),
            marker={"color": couleur, "size": taille, "opacity": 0.4},
            hovertemplate=f"{nom}<br>dCT %{{x:.1f}} µm<br>e_yy %{{y:.4f}}<extra></extra>"),
            row=1, col=j)
    for g, couleur, cle in ((bl_brut[bl_brut["sigma"] == sigma], "#c2453f", "brut"),
                            (bl_ret[bl_ret["sigma"] == sigma], "#2a78d6", "filtre")):
        g = g.dropna(subset=["strain_choroide_med", "dCT_um"])
        if len(g) < 5:
            continue
        r = float(np.corrcoef(g["dCT_um"], g["strain_choroide_med"])[0, 1])
        a, b0 = np.polyfit(g["dCT_um"], g["strain_choroide_med"], 1)
        xs = np.linspace(0, float(bl_brut["dCT_um"].quantile(0.99)), 2)
        fig.add_trace(go.Scatter(
            x=xs, y=a * xs + b0, mode="lines",
            name=("ajustement, tous" if cle == "brut" else "ajustement, retenus"),
            legendgroup="fit_" + cle, showlegend=bool(j == 1),
            line={"color": couleur, "width": 2.5,
                  "dash": "dot" if cle == "brut" else "solid"}), row=1, col=j)
        RESUME[f"p2_valid_r_{cle}_sigma{sigma:g}"] = r
        RESUME[f"p2_valid_pente_{cle}_sigma{sigma:g}"] = float(a)
    fig.add_hline(y=0.0, row=1, col=j,
                  line={"color": C_REF, "dash": "dot", "width": 1})

ct_med = float(one_cycles["CT_moyen_um"].median())
xs = np.linspace(0, float(bl_brut["dCT_um"].quantile(0.99)), 2)
for j in range(1, len(SIGMAS) + 1):
    fig.add_trace(go.Scatter(x=xs, y=xs / ct_med, mode="lines",
                             name=f"attendu : dCT / CT ({ct_med:.0f} µm)",
                             legendgroup="attendu", showlegend=bool(j == 1),
                             line={"color": "#8e5bd0", "width": 2, "dash": "dash"}),
                  row=1, col=j)
fig.update_xaxes(title="dCT (µm)", range=[0, float(bl_brut["dCT_um"].quantile(0.99))])
fig.update_yaxes(range=[float(bl_brut["strain_choroide_med"].quantile(0.01)),
                        float(bl_brut["strain_choroide_med"].quantile(0.99))])
fig.update_yaxes(title="strain choroïdien médian e_yy", row=1, col=1)
mise_en_page(fig, "Validation — le strain choroïdien contre la dilatation connue, "
                  "avant et après le portail (référence locale)", hauteur=520)
enregistrer(fig, "p2_validation")

RESUME["p2_ct_median_um"] = ct_med
RESUME["p2_dct_median_um"] = float(bl_ret["dCT_um"].median())
RESUME["p2_pente_attendue"] = 1.0 / ct_med
RESUME["p2_dct_etendue_brut"] = float(bl_brut["dCT_um"].max())
RESUME["p2_dct_etendue_retenu"] = float(bl_ret["dCT_um"].max())
RESUME["p2_dct_sd_brut"] = float(bl_brut["dCT_um"].std())
RESUME["p2_dct_sd_retenu"] = float(bl_ret["dCT_um"].std())


# 2.c Strain par region contre CT et contre dCT -- une grille sigma x reference.
# Un point par (condition, one-cycle, bin) serait illisible a 192 000 points :
# les bins d'un one-cycle sont moyennes, ce qui laisse un point par
# (condition, one-cycle, region) et respecte l'independance des observations.
def _strain_vs(xcol, ycol, titre, nom):
    par_cycle = (regions.groupby(["slug", "one_cycle", "cas", "sigma", "region"],
                                 sort=False)[[xcol, ycol]].mean().reset_index())
    fig = make_subplots(
        rows=len(SIGMAS), cols=len(CAS), shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=0.055, vertical_spacing=0.045,
        subplot_titles=[f"σ = {sg:g} · référence {COURT_CAS[c]}"
                        for sg in SIGMAS for c in CAS])
    lo = float(par_cycle[ycol].quantile(0.01))
    hi = float(par_cycle[ycol].quantile(0.99))
    for i, sigma in enumerate(SIGMAS, start=1):
        for j, cas in enumerate(CAS, start=1):
            sub = par_cycle[(par_cycle["sigma"] == sigma) & (par_cycle["cas"] == cas)]
            for region in REGIONS:
                g = sub[sub["region"] == region]
                if g.empty:
                    continue
                fig.add_trace(go.Scattergl(
                    x=g[xcol], y=g[ycol], mode="markers", name=region,
                    legendgroup=region,
                    showlegend=bool(i == 1 and j == 1),
                    marker={"color": COULEUR_REGION[region], "size": 3.5,
                            "opacity": 0.42},
                    hovertemplate=(region + "<br>%{x:.0f} µm<br>e_yy %{y:.4f}"
                                   "<extra></extra>")), row=i, col=j)
                if len(g) >= 5 and g[xcol].std() > 0:
                    a, b0 = np.polyfit(g[xcol], g[ycol], 1)
                    xs = np.linspace(g[xcol].quantile(0.02), g[xcol].quantile(0.98), 2)
                    fig.add_trace(go.Scatter(
                        x=xs, y=a * xs + b0, mode="lines", showlegend=False,
                        legendgroup=region,
                        line={"color": COULEUR_REGION[region], "width": 2}),
                        row=i, col=j)
            fig.add_hline(y=0.0, line={"color": C_REF, "dash": "dot", "width": 1},
                          row=i, col=j)
    fig.update_yaxes(range=[lo, hi])
    for j in range(1, len(CAS) + 1):
        fig.update_xaxes(title=xcol, row=len(SIGMAS), col=j)
    for i in range(1, len(SIGMAS) + 1):
        fig.update_yaxes(title="e_yy", row=i, col=1)
    mise_en_page(fig, titre, hauteur=1180)
    enregistrer(fig, nom)


for _col, _lab, _sfx, _ in AGREGATS:
    _strain_vs("CT_region_um", _col,
               f"Strain rétinien ({_lab}) contre l'épaisseur choroïdienne de la "
               "bande — un point par (condition, one-cycle, bande)",
               f"p2_strain_vs_ct_{_sfx}")
    _strain_vs("dCT_region_um", _col,
               f"Strain rétinien ({_lab}) contre la DILATATION de la bande — "
               "un point par (condition, one-cycle, bande)",
               f"p2_strain_vs_dct_{_sfx}")


# 2.d Volcano : -log10(p) contre la pente, un point par (condition, bande).
# Les ajustements sont ceux de `pooled_slopes.csv` : une pente par condition,
# calculee sur TOUS les couples (one-cycle, bin) de cette condition.
def _volcano(xcol, ycol, titre, nom):
    """``ycol`` est la colonne d'AGREGAT du strain (``tissu`` dans le CSV, qui
    porte ce nom depuis l'epoque ou retine et choroide y coexistaient)."""
    sub_all = pooled[(pooled["abscisse"] == xcol) & (pooled["tissu"] == ycol)].copy()
    sub_all["mlogp"] = -np.log10(np.clip(sub_all["p"], 1e-300, 1.0))
    fig = make_subplots(
        rows=len(SIGMAS), cols=len(CAS), shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=0.055, vertical_spacing=0.045,
        subplot_titles=[f"σ = {sg:g} · référence {COURT_CAS[c]}"
                        for sg in SIGMAS for c in CAS])
    xlim = float(np.nanquantile(np.abs(sub_all["pente_par_um"]), 0.99))
    for i, sigma in enumerate(SIGMAS, start=1):
        for j, cas in enumerate(CAS, start=1):
            sub = sub_all[(sub_all["sigma"] == sigma) & (sub_all["cas"] == cas)]
            for region in REGIONS:
                g = sub[sub["region"] == region]
                if g.empty:
                    continue
                fig.add_trace(go.Scattergl(
                    x=g["pente_par_um"], y=g["mlogp"], mode="markers", name=region,
                    legendgroup=region, showlegend=bool(i == 1 and j == 1),
                    marker={"color": COULEUR_REGION[region], "size": 5,
                            "opacity": 0.7},
                    hovertext=g["slug"],
                    hovertemplate=("%{hovertext}<br>" + region +
                                   "<br>pente %{x:.2e}<br>−log10(p) %{y:.2f}"
                                   "<extra></extra>")), row=i, col=j)
            fig.add_hline(y=-np.log10(0.05), row=i, col=j,
                          line={"color": C_SEUIL, "dash": "dash", "width": 1})
            fig.add_vline(x=0.0, row=i, col=j,
                          line={"color": C_REF, "dash": "dot", "width": 1})
    fig.update_xaxes(range=[-xlim, xlim])
    for j in range(1, len(CAS) + 1):
        fig.update_xaxes(title="pente e_yy / µm", row=len(SIGMAS), col=j)
    for i in range(1, len(SIGMAS) + 1):
        fig.update_yaxes(title="−log10(p)", row=i, col=1)
    mise_en_page(fig, titre, hauteur=1180)
    enregistrer(fig, nom)


for _col, _lab, _sfx, _ in AGREGATS:
    _volcano("CT_region_um", _col,
             f"Volcano — pente du strain rétinien ({_lab}) contre l'ÉPAISSEUR, "
             "un point par (condition, bande) ; la ligne marque p = 0,05",
             f"p2_volcano_ct_{_sfx}")
    _volcano("dCT_region_um", _col,
             f"Volcano — pente du strain rétinien ({_lab}) contre la DILATATION, "
             "un point par (condition, bande) ; la ligne marque p = 0,05",
             f"p2_volcano_dct_{_sfx}")

for _col, _lab, _sfx, _ in AGREGATS:
    for xcol in ("CT_region_um", "dCT_region_um"):
        g = pooled[(pooled["abscisse"] == xcol) & (pooled["tissu"] == _col)]
        cle = f"p2_pool_{xcol}_{_sfx}"
        RESUME[f"{cle}_pente_med"] = med_iqr(g["pente_par_um"], "{:.2e}")
        RESUME[f"{cle}_frac_pos"] = float((g["pente_par_um"] > 0).mean())
        RESUME[f"{cle}_frac_p05"] = float((g["p"] < 0.05).mean())
        RESUME[f"{cle}_frac_p05_pos"] = float(
            ((g["p"] < 0.05) & (g["pente_par_um"] > 0)).mean())
        RESUME[f"{cle}_n_ajustements"] = int(len(g))


# =========================================================================== #
# PAGE 3 — Repeatability
# =========================================================================== #
# Une boite par (sigma, style de reference), sur les predicteurs issus des
# demons UNIQUEMENT -- les familles `pouls` et `rigidite` n'ont ni sigma ni
# reference, les meler brouillerait la comparaison que la figure sert a faire.
FAMILLES_DEMONS = ("strain", "pente", "pente_pool")
rep = repeat[repeat["famille"].isin(FAMILLES_DEMONS)].copy()
rep = rep[rep["sigma"].notna() & rep["cas"].isin(CAS)]
# Le strain CHOROIDIEN ne fait plus partie du panel (les lots ne le produisent
# plus qu'au niveau du bin, pour la validation) : ce filtre n'est donc qu'un
# garde-fou, pour qu'une table d'avant la refonte ne reintroduise pas
# silencieusement des predicteurs choroidiens. L'EPAISSEUR choroidienne (CT),
# elle, reste partout : c'est l'abscisse de toute l'analyse, pas un strain.
rep = rep[~rep["predicteur"].str.contains("choroide", na=False)].copy()
# L'agregat du strain, lisible sur toutes les familles : `strain_retine_p95`
# comme `poolpente_CT_um_strain_retine_p95`.
rep["agregat"] = rep["predicteur"].map(agregat_de)

icc = rep[(rep["icc_type"] == "ICC(A,1)") & (rep["pair"] == "1v2")].dropna(subset=["icc"])
cv = rep[rep["pair"] == "toutes"].dropna(subset=["cv_intra_pct"])

fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.11,
                    subplot_titles=("ICC(A,1) — accord absolu, réplicats 1 vs 2",
                                    "CV intra-sujet (%, tronqué à 500)"))
for cas in CAS:
    for sigma in SIGMAS:
        a = icc[(icc["cas"] == cas) & (icc["sigma"] == sigma)]["icc"]
        c = cv[(cv["cas"] == cas) & (cv["sigma"] == sigma)]["cv_intra_pct"].clip(upper=500)
        commun = {"offsetgroup": cas, "boxpoints": False,
                  "marker": {"color": COULEUR_CAS[cas], "size": 2},
                  "line": {"width": 1, "color": COULEUR_CAS[cas]}}
        fig.add_trace(go.Box(
            y=a, x=[f"σ {sigma:g}"] * len(a), name=f"référence {COURT_CAS[cas]}",
            legendgroup=cas, showlegend=bool(sigma == SIGMAS[0]), **commun,
            hovertemplate=f"{cas}, σ {sigma:g}<br>ICC %{{y:+.3f}}<extra></extra>"),
            row=1, col=1)
        fig.add_trace(go.Box(
            y=c, x=[f"σ {sigma:g}"] * len(c), name=f"référence {COURT_CAS[cas]}",
            legendgroup=cas, showlegend=False, **commun,
            hovertemplate=f"{cas}, σ {sigma:g}<br>CV %{{y:.0f}} %<extra></extra>"),
            row=1, col=2)
for seuil, texte in ((0.5, "médiocre"), (0.75, "bon"), (0.9, "excellent")):
    fig.add_hline(y=seuil, row=1, col=1, line={"color": C_REF, "dash": "dot", "width": 1},
                  annotation={"text": texte, "font": {"size": 9, "color": C_REF}})
fig.add_hline(y=0.0, row=1, col=1, line={"color": C_SEUIL, "dash": "dash", "width": 1})
fig.add_hline(y=100.0, row=1, col=2, line={"color": C_SEUIL, "dash": "dash", "width": 1},
              annotation={"text": "CV = 100 %", "font": {"size": 9, "color": C_SEUIL}})
fig.update_layout(boxmode="group")
fig.update_yaxes(title="ICC(A,1)", range=[-1.05, 1.05], row=1, col=1)
fig.update_yaxes(title="CV intra (%)", row=1, col=2)
mise_en_page(fig, "Répétabilité des paramètres issus des démons — une boîte par "
                  f"(σ, style de référence), {icc['predicteur'].nunique()} "
                  f"prédicteurs distincts", hauteur=540)
enregistrer(fig, "p3_icc_cv")

for cas in CAS:
    for sigma in SIGMAS:
        a = icc[(icc["cas"] == cas) & (icc["sigma"] == sigma)]["icc"]
        c = cv[(cv["cas"] == cas) & (cv["sigma"] == sigma)]["cv_intra_pct"]
        RESUME[f"p3_icc_{cas}_s{sigma:g}"] = med_iqr(a, "{:+.2f}")
        RESUME[f"p3_cv_{cas}_s{sigma:g}"] = med_iqr(c, "{:.0f}")
        RESUME[f"p3_icc_frac075_{cas}_s{sigma:g}"] = float((a > 0.75).mean()) if len(a) else np.nan
RESUME["p3_icc_med_global"] = float(icc["icc"].median())
RESUME["p3_cv_med_global"] = float(cv["cv_intra_pct"].median())
RESUME["p3_var_inter_neg"] = float((cv["var_inter"] < 0).mean())
RESUME["p3_n_sujets_med"] = float(icc["n_subjects"].median())

# Reference de lecture : les memes chiffres pour les predicteurs qui ne passent
# PAS par les demons. C'est l'echelle a laquelle comparer les boites ci-dessus.
autre = repeat[repeat["famille"].isin(("ct_pouls", "pouls", "rigidite"))]
a2 = autre[(autre["icc_type"] == "ICC(A,1)") & (autre["pair"] == "1v2")]["icc"].dropna()
RESUME["p3_icc_hors_demons"] = med_iqr(a2, "{:+.2f}")
RESUME["p3_icc_hors_demons_frac075"] = float((a2 > 0.75).mean()) if len(a2) else np.nan


# --- Le meme decoupage, AGREGAT PAR AGREGAT ----------------------------------
# La question que cette figure tranche : le p95 -- une mesure de QUEUE -- se
# mesure-t-il mieux que la moyenne et la mediane, qui sont deux mesures de
# CENTRE ? Le centre du strain est domine par le biais negatif du lissage ; rien
# ne dit a priori que la queue en souffre autant, ni qu'elle en souffre moins.
#
# Elle porte sur TOUS les predicteurs issus des demons (strain par bande, pentes
# par one-cycle et pentes poolees), chacun rattache a son agregat.
_ag = icc.dropna(subset=["agregat"])
_ag_cv = cv.dropna(subset=["agregat"])
fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.11,
                    subplot_titles=("ICC(A,1) — accord absolu, réplicats 1 vs 2",
                                    "CV intra-sujet (%, tronqué à 500)"))
for agregat in ORDRE_AGREGATS:
    for sigma in SIGMAS:
        a = _ag[(_ag["agregat"] == agregat) & (_ag["sigma"] == sigma)]["icc"]
        c = (_ag_cv[(_ag_cv["agregat"] == agregat) & (_ag_cv["sigma"] == sigma)]
             ["cv_intra_pct"].clip(upper=500))
        commun = {"offsetgroup": agregat, "boxpoints": False,
                  "marker": {"color": COULEUR_AGREGAT[agregat], "size": 2},
                  "line": {"width": 1, "color": COULEUR_AGREGAT[agregat]}}
        fig.add_trace(go.Box(
            y=a, x=[f"σ {sigma:g}"] * len(a), name=agregat, legendgroup=agregat,
            showlegend=bool(sigma == SIGMAS[0]), **commun,
            hovertemplate=f"{agregat}, σ {sigma:g}<br>ICC %{{y:+.3f}}<extra></extra>"),
            row=1, col=1)
        fig.add_trace(go.Box(
            y=c, x=[f"σ {sigma:g}"] * len(c), name=agregat, legendgroup=agregat,
            showlegend=False, **commun,
            hovertemplate=f"{agregat}, σ {sigma:g}<br>CV %{{y:.0f}} %<extra></extra>"),
            row=1, col=2)
for seuil, texte in ((0.5, "médiocre"), (0.75, "bon"), (0.9, "excellent")):
    fig.add_hline(y=seuil, row=1, col=1, line={"color": C_REF, "dash": "dot", "width": 1},
                  annotation={"text": texte, "font": {"size": 9, "color": C_REF}})
fig.add_hline(y=0.0, row=1, col=1, line={"color": C_SEUIL, "dash": "dash", "width": 1})
fig.add_hline(y=100.0, row=1, col=2, line={"color": C_SEUIL, "dash": "dash", "width": 1},
              annotation={"text": "CV = 100 %", "font": {"size": 9, "color": C_SEUIL}})
fig.update_layout(boxmode="group")
fig.update_yaxes(title="ICC(A,1)", range=[-1.05, 1.05], row=1, col=1)
fig.update_yaxes(title="CV intra (%)", row=1, col=2)
mise_en_page(fig, "Répétabilité par AGRÉGAT du strain — deux mesures de centre "
                  "(moyenne, médiane) et une mesure de queue (p95 signé), "
                  f"{_ag['predicteur'].nunique()} prédicteurs distincts", hauteur=540)
enregistrer(fig, "p3_agregats")

for _col, _lab, _sfx, _ in AGREGATS:
    a = _ag[_ag["agregat"] == _lab]["icc"]
    c = _ag_cv[_ag_cv["agregat"] == _lab]
    RESUME[f"p3_agregat_icc_{_sfx}"] = med_iqr(a, "{:+.2f}")
    RESUME[f"p3_agregat_cv_{_sfx}"] = med_iqr(c["cv_intra_pct"], "{:.0f}")
    RESUME[f"p3_agregat_varneg_{_sfx}"] = float((c["var_inter"] < 0).mean()) if len(c) else np.nan
    RESUME[f"p3_agregat_frac075_{_sfx}"] = float((a > 0.75).mean()) if len(a) else np.nan
    RESUME[f"p3_agregat_n_{_sfx}"] = int(a.size)


# --- Le meme decoupage, REGION PAR REGION ------------------------------------
# La region est jusqu'ici une dimension CACHEE : chaque boite de la figure
# precedente melange les cinq bandes. Or rien n'oblige les cinq bandes a se
# repeter aussi bien -- la bande la plus a droite est deja signalee comme
# anormale par le carnet (choroide la plus fine, deltaY de signe oppose aux
# autres). C'est donc une decomposition a faire, pas une precaution de style.
def _boites_par_region(source, colonne, titre, ytitre, nom, yrange=None, clip=None):
    d = source.dropna(subset=[colonne])
    fig = make_subplots(
        rows=len(SIGMAS), cols=len(CAS), shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=0.055, vertical_spacing=0.045,
        subplot_titles=[f"σ = {sg:g} · référence {COURT_CAS[c]}"
                        for sg in SIGMAS for c in CAS])
    for i, sigma in enumerate(SIGMAS, start=1):
        for j, cas in enumerate(CAS, start=1):
            sub = d[(d["sigma"] == sigma) & (d["cas"] == cas)]
            for region in REGIONS:
                g = sub[sub["region"] == region]
                y = g[colonne]
                if clip is not None:
                    y = y.clip(upper=clip)
                fig.add_trace(go.Box(
                    y=y, x=[region] * len(y), name=region, legendgroup=region,
                    showlegend=bool(i == 1 and j == 1), boxpoints=False,
                    marker={"color": COULEUR_REGION[region], "size": 2},
                    line={"width": 1, "color": COULEUR_REGION[region]},
                    hovertemplate=(region + f"<br>{cas}, σ {sigma:g}"
                                   + "<br>%{y:.3g}<extra></extra>"),
                ), row=i, col=j)
    if yrange is not None:
        fig.update_yaxes(range=yrange)
    for i in range(1, len(SIGMAS) + 1):
        fig.update_yaxes(title=ytitre, row=i, col=1)
    mise_en_page(fig, titre, hauteur=1150)
    enregistrer(fig, nom)


_boites_par_region(
    icc, "icc",
    "Répétabilité bande par bande — chaque boîte réunit les prédicteurs issus des "
    "démons pour cette bande (indice croissant vers la GAUCHE)",
    "ICC(A,1)", "p3_regions_icc", yrange=[-1.05, 1.05])

_boites_par_region(
    cv, "cv_intra_pct",
    "Répétabilité absolue bande par bande — CV intra-sujet (%, tronqué à 700)",
    "CV intra (%)", "p3_regions_cv", clip=700)

for region in REGIONS:
    a = icc[icc["region"] == region]["icc"]
    c = cv[cv["region"] == region]
    cle = region.replace(" ", "_").replace("(", "").replace(")", "")
    RESUME[f"p3_region_icc_{cle}"] = med_iqr(a, "{:+.2f}")
    RESUME[f"p3_region_cv_{cle}"] = med_iqr(c["cv_intra_pct"], "{:.0f}")
    RESUME[f"p3_region_varneg_{cle}"] = float((c["var_inter"] < 0).mean())
    for cas in CAS:
        aa = icc[(icc["region"] == region) & (icc["cas"] == cas)]["icc"]
        RESUME[f"p3_region_icc_{cle}_{cas}"] = float(aa.median()) if len(aa) else np.nan


# --- Le meme decoupage, PENTE PAR PENTE -------------------------------------
# La figure precedente melange tout ce que les demons produisent. Celle-ci
# separe les pentes selon ce contre quoi le strain est ajuste, parce que rien ne
# garantit a priori que la pente contre l'epaisseur et la pente contre la
# dilatation se repetent aussi bien -- et parce que ce sont ces pentes-la qui
# servent de marqueurs a la page suivante.
ABSCISSES = ["CT_region_um", "dCT_region_um", "CT_um", "dCT_um"]
ABSCISSE_LABEL = {"CT_region_um": "CT de la bande", "dCT_region_um": "ΔCT de la bande",
                  "CT_um": "CT global", "dCT_um": "ΔCT global"}
# Le tissu n'est plus une dimension : tout est retinien. Ce qui la remplace est
# l'AGREGAT du strain, qui se lit dans le meme suffixe de nom de predicteur.


def _decompose_pente(nom):
    """``poolpente_dCT_region_um_strain_retine_p95`` ->
    ``("dCT_region_um", "p95 signé", "poolée")``."""
    if nom.startswith("poolpente_"):
        reste, estimateur = nom[len("poolpente_"):], "poolée"
    elif nom.startswith("pente_"):
        reste, estimateur = nom[len("pente_"):], "par one-cycle"
    else:
        return None
    # Du suffixe le PLUS LONG au plus court : sinon `..._strain_retine_med`
    # serait decoupe comme la moyenne, et l'agregat serait faux.
    for col, lab, _, _ in sorted(AGREGATS, key=lambda a: -len(a[0])):
        if reste.endswith("_" + col):
            return reste[: -len(col) - 1], lab, estimateur
    return None


def _table_pentes(df, colonne):
    """Ajoute abscisse / agregat / estimateur, et ne garde que les PENTES."""
    d = df.copy()
    parts = d["predicteur"].map(_decompose_pente)
    d = d[parts.notna()].copy()
    parts = parts[parts.notna()]
    d["abscisse"] = [p[0] for p in parts]
    d["agregat"] = [p[1] for p in parts]
    d["estimateur"] = [p[2] for p in parts]
    return d.dropna(subset=[colonne])


def _boites_par_pente(source, colonne, titre, ytitre, nom, yrange=None, clip=None):
    """Une grille abscisse x BANDE, et dans chaque case les boites sigma x
    reference -- l'identite de boite du reste de la page.

    Chaque boite ne reunit que 12 valeurs (2 moments x 2 estimateurs x 3
    agregats), ce qui est trop peu pour qu'un quartile veuille dire quelque
    chose : les points sont donc traces EN CLAIR par-dessus, et c'est eux qu'il
    faut lire. La boite n'est la que pour guider l'oeil d'une case a l'autre.

    L'agregat du strain se lit AU SURVOL, pas au symbole : melanger les trois
    dans la meme boite est voulu, la figure repondant a « la bande et l'abscisse
    changent-elles quelque chose », pas « quel agregat choisir » -- c'est
    `p3_agregats` et le tableau des eta2 qui repondent a celle-la.
    """
    d = _table_pentes(source, colonne)
    fig = make_subplots(
        rows=len(ABSCISSES), cols=len(REGIONS), shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=0.028, vertical_spacing=0.05,
        subplot_titles=[f"{ABSCISSE_LABEL[a]}<br>{r}"
                        for a in ABSCISSES for r in REGIONS])
    for i, abscisse in enumerate(ABSCISSES, start=1):
        for j, region in enumerate(REGIONS, start=1):
            sub = d[(d["abscisse"] == abscisse) & (d["region"] == region)]
            for cas in CAS:
                for sigma in SIGMAS:
                    g = sub[(sub["cas"] == cas) & (sub["sigma"] == sigma)]
                    y = g[colonne]
                    if clip is not None:
                        y = y.clip(upper=clip)
                    fig.add_trace(go.Box(
                        y=y, x=[f"σ {sigma:g}"] * len(y),
                        name=f"référence {COURT_CAS[cas]}", legendgroup=cas,
                        showlegend=bool(i == 1 and j == 1 and sigma == SIGMAS[0]),
                        offsetgroup=cas, boxpoints="all", jitter=0.6, pointpos=0,
                        # `go.Box.marker.symbol` n'est PAS array-ok (contrairement
                        # a celui de `go.Scatter`) : l'agregat ne peut pas se lire
                        # au symbole ici, il se lit au survol.
                        marker={"color": COULEUR_CAS[cas], "size": 5, "opacity": 0.85},
                        line={"width": 0.8, "color": COULEUR_CAS[cas]},
                        fillcolor="rgba(0,0,0,0)",
                        hovertext=(g["agregat"] + " · " + g["estimateur"]
                                   + " · " + g["moment"].astype(str)),
                        hovertemplate=("%{hovertext}<br>" + f"{region}, {cas}, σ {sigma:g}"
                                       + "<br>%{y:.3g}<extra></extra>"),
                    ), row=i, col=j)
    fig.update_layout(boxmode="group")
    if yrange is not None:
        fig.update_yaxes(range=yrange)
    for i in range(1, len(ABSCISSES) + 1):
        fig.update_yaxes(title=ytitre, row=i, col=1)
    fig.update_xaxes(tickangle=-45, tickfont={"size": 8})
    mise_en_page(fig, titre, hauteur=1450)
    for a in fig.layout.annotations or ():
        if a.font is not None:
            a.font.size = 9
    enregistrer(fig, nom)


_boites_par_pente(
    icc, "icc",
    "Répétabilité des PENTES du strain rétinien — une grille abscisse × bande, "
    "boîtes σ × référence ; 12 points par boîte (2 moments × 2 estimateurs × "
    "3 agrégats, lisibles au survol)",
    "ICC(A,1)", "p3_pentes_icc", yrange=[-1.05, 1.05])

_boites_par_pente(
    cv, "cv_intra_pct",
    "Répétabilité absolue des PENTES du strain rétinien — CV intra-sujet "
    "(%, tronqué à 500), même grille", "CV intra (%)", "p3_pentes_cv", clip=500)

# Chiffres de la prose : par abscisse, par agregat, par estimateur.
_icc_p = _table_pentes(icc, "icc")
_cv_p = _table_pentes(cv, "cv_intra_pct")
for abscisse in ABSCISSES:
    a = _icc_p[_icc_p["abscisse"] == abscisse]["icc"]
    c = _cv_p[_cv_p["abscisse"] == abscisse]["cv_intra_pct"]
    RESUME[f"p3_pente_icc_{abscisse}"] = med_iqr(a, "{:+.2f}")
    RESUME[f"p3_pente_cv_{abscisse}"] = med_iqr(c, "{:.0f}")
    for region in REGIONS:
        cle = f"{abscisse}_{region.replace(' ', '_').replace('(', '').replace(')', '')}"
        aa = _icc_p[(_icc_p["abscisse"] == abscisse) & (_icc_p["region"] == region)]["icc"]
        cc = _cv_p[(_cv_p["abscisse"] == abscisse) & (_cv_p["region"] == region)]["cv_intra_pct"]
        RESUME[f"p3_pente_icc_{cle}"] = float(aa.median()) if len(aa) else np.nan
        RESUME[f"p3_pente_cv_{cle}"] = float(cc.median()) if len(cc) else np.nan
for estimateur in ("par one-cycle", "poolée"):
    a = _icc_p[_icc_p["estimateur"] == estimateur]["icc"]
    c = _cv_p[_cv_p["estimateur"] == estimateur]["cv_intra_pct"]
    RESUME[f"p3_pente_icc_{estimateur.replace(' ', '_')}"] = med_iqr(a, "{:+.2f}")
    RESUME[f"p3_pente_cv_{estimateur.replace(' ', '_')}"] = med_iqr(c, "{:.0f}")
for _col, _lab, _sfx, _ in AGREGATS:
    a = _icc_p[_icc_p["agregat"] == _lab]["icc"]
    c = _cv_p[_cv_p["agregat"] == _lab]["cv_intra_pct"]
    RESUME[f"p3_pente_icc_{_sfx}"] = med_iqr(a, "{:+.2f}")
    RESUME[f"p3_pente_cv_{_sfx}"] = med_iqr(c, "{:.0f}")
RESUME["p3_pente_icc_toutes"] = med_iqr(_icc_p["icc"], "{:+.2f}")
RESUME["p3_pente_n_predicteurs"] = int(_icc_p["predicteur"].nunique())


# --- L'EFFET GLOBAL de chaque reglage sur l'ICC des pentes -------------------
# La grille de 20 panneaux montre TOUT, ce qui empeche de voir ce qui compte.
# Cette figure repond a la seule question qui decide d'un reglage : de combien
# l'ICC bouge-t-il quand on change ce reglage, tous les autres confondus ?
#
# Deux mesures complementaires, parce qu'elles ne disent pas la meme chose :
#   - l'AMPLITUDE, ecart entre la mediane du meilleur niveau et celle du pire :
#     c'est le gain qu'on peut esperer en choisissant bien ;
#   - eta^2, la part de la variance des ICC que le facteur explique : c'est la
#     place qu'il occupe FACE AU BRUIT. Un facteur peut avoir une amplitude
#     honorable et un eta^2 minuscule si la dispersion interne l'ecrase.
_eff = _table_pentes(icc, "icc").copy()
_eff["derivee"] = np.where(_eff["abscisse"].str.startswith("dCT"), "ΔCT", "CT")
_eff["portee"] = np.where(_eff["abscisse"].str.contains("region"), "de la bande", "global")
_eff["reference"] = _eff["cas"].map(COURT_CAS)
_eff["sigma_lab"] = "σ " + _eff["sigma"].map(lambda v: f"{v:g}")

FACTEURS = [
    ("agregat", "agrégat du strain", ORDRE_AGREGATS, "#d0a72b"),
    ("reference", "référence", ["locale", "globale"], "#eb6834"),
    ("region", "bande de rétine", REGIONS, "#2a78d6"),
    ("derivee", "CT ou ΔCT", ["CT", "ΔCT"], "#27a567"),
    ("portee", "portée de l'abscisse", ["de la bande", "global"], "#8e5bd0"),
    ("estimateur", "estimateur (comparateur)", ["par one-cycle", "poolée"], "#9a9a9a"),
    ("sigma_lab", "régularisation (comparateur)",
     [f"σ {v:g}" for v in SIGMAS], "#9a9a9a"),
]


def _eta2(col):
    """Part de la variance des ICC expliquee par ce facteur seul (one-way)."""
    grand = _eff["icc"].mean()
    sst = float(((_eff["icc"] - grand) ** 2).sum())
    ssb = float(sum(len(g) * (g["icc"].mean() - grand) ** 2
                    for _, g in _eff.groupby(col)))
    return 100.0 * ssb / sst if sst > 0 else np.nan


fig = make_subplots(rows=1, cols=2, column_widths=[0.72, 0.28],
                    horizontal_spacing=0.10,
                    subplot_titles=("ICC(A,1) par niveau de chaque réglage",
                                    "part de la variance expliquée"))
etiquettes, etas, couleurs = [], [], []
for col, label, niveaux, couleur in FACTEURS:
    for niveau in niveaux:
        g = _eff[_eff[col] == niveau]["icc"]
        if g.empty:
            continue
        fig.add_trace(go.Box(
            y=g, x=[f"{label}<br>{niveau}"] * len(g), name=label,
            showlegend=False, boxpoints=False,
            marker={"color": couleur, "size": 2},
            line={"width": 1.2, "color": couleur},
            hovertemplate=(f"{label} = {niveau}<br>médiane %{{median:+.3f}}"
                           f"<br>n = {len(g)}<extra></extra>"),
        ), row=1, col=1)
    etiquettes.append(label)
    etas.append(_eta2(col))
    couleurs.append(couleur)
    RESUME[f"p3_eff_eta2_{col}"] = _eta2(col)
    med = _eff.groupby(col)["icc"].median()
    RESUME[f"p3_eff_amplitude_{col}"] = float(med.max() - med.min())
    for niveau in niveaux:
        if niveau in med.index:
            RESUME[f"p3_eff_med_{col}_{niveau}"] = float(med[niveau])

fig.add_hline(y=float(_eff["icc"].median()), row=1, col=1,
              line={"color": C_REF, "dash": "dot", "width": 1},
              annotation={"text": "médiane générale",
                          "font": {"size": 9, "color": C_REF}})
fig.add_trace(go.Bar(y=etiquettes[::-1], x=etas[::-1], orientation="h",
                     marker={"color": couleurs[::-1]}, showlegend=False,
                     hovertemplate="%{y}<br>η² = %{x:.2f} %<extra></extra>"),
              row=1, col=2)
fig.update_xaxes(tickangle=-40, tickfont={"size": 9}, row=1, col=1)
fig.update_yaxes(title="ICC(A,1)", range=[-0.8, 0.9], row=1, col=1)
fig.update_xaxes(title="η² (% de la variance des ICC)", row=1, col=2)
fig.update_yaxes(tickfont={"size": 9}, row=1, col=2)

# Modele additif : ce que les SEPT reglages expliquent ENSEMBLE.
_X = pd.get_dummies(_eff[["agregat", "reference", "region", "derivee", "portee",
                          "estimateur", "sigma_lab"]], drop_first=True).astype(float)
_X.insert(0, "const", 1.0)
_y = _eff["icc"].to_numpy(dtype=float)
_beta, *_ = np.linalg.lstsq(_X.to_numpy(), _y, rcond=None)
_res = _y - _X.to_numpy() @ _beta
_r2 = 100.0 * (1 - float((_res ** 2).sum()) / float(((_y - _y.mean()) ** 2).sum()))
RESUME["p3_eff_r2_modele_additif"] = _r2
RESUME["p3_eff_n_icc"] = int(len(_eff))

mise_en_page(fig, f"Effet global de chaque réglage sur la répétabilité des pentes — "
                  f"{len(_eff)} valeurs d'ICC ; les sept réglages réunis n'expliquent "
                  f"que {_r2:.0f} % de leur variance", hauteur=560, legende=False)
enregistrer(fig, "p3_pentes_effets")


# --- L'AUTRE definition du replicat : les one-cycles d'une meme video --------
# Table `repeatability_one_cycles.csv`. La cible n'est plus le sujet mais la
# VIDEO, et la replique n'est plus une acquisition repetee mais un BATTEMENT
# replie. La question devient « la valeur publiee pour une video est-elle stable
# d'un battement a l'autre ? ».
#
# Ce n'est PAS la meme question que les sections precedentes, et il ne faut pas
# lire les deux ICC comme deux versions plus ou moins bruitees l'une de l'autre :
# entre deux videos il y a l'oeil, la seance et la derive de l'acquisition ;
# entre deux one-cycles, il n'y a que le battement. Cet ICC-ci est un PLANCHER --
# un parametre qui echoue deja ici ne peut rien reussir plus loin.
#
# `pente_pool`, `pouls` et `rigidite` sont structurellement absentes : elles
# n'existent qu'a l'echelle de la video (voir la docstring du lot).
_oc = repeat_oc[~repeat_oc["predicteur"].str.contains("choroide", na=False)].copy()
_oc_icc = _oc[(_oc["icc_type"] == "ICC(A,1)") & (_oc["pair"] == "1v2")].dropna(subset=["icc"])
_oc_vc = _oc[_oc["pair"] == "toutes"].dropna(subset=["cv_intra_pct"])
_oc_dem = _oc_icc[_oc_icc["famille"].isin(("strain", "pente"))
                  & _oc_icc["sigma"].notna() & _oc_icc["cas"].isin(CAS)]
_oc_dem_vc = _oc_vc[_oc_vc["famille"].isin(("strain", "pente"))
                    & _oc_vc["sigma"].notna() & _oc_vc["cas"].isin(CAS)]

fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.11,
                    subplot_titles=("ICC(A,1) — one-cycles 1 vs 2 d'une même vidéo",
                                    "CV intra-vidéo (%, tronqué à 500)"))
for cas in CAS:
    for sigma in SIGMAS:
        a = _oc_dem[(_oc_dem["cas"] == cas) & (_oc_dem["sigma"] == sigma)]["icc"]
        c = (_oc_dem_vc[(_oc_dem_vc["cas"] == cas) & (_oc_dem_vc["sigma"] == sigma)]
             ["cv_intra_pct"].clip(upper=500))
        commun = {"offsetgroup": cas, "boxpoints": False,
                  "marker": {"color": COULEUR_CAS[cas], "size": 2},
                  "line": {"width": 1, "color": COULEUR_CAS[cas]}}
        fig.add_trace(go.Box(
            y=a, x=[f"σ {sigma:g}"] * len(a), name=f"référence {COURT_CAS[cas]}",
            legendgroup=cas, showlegend=bool(sigma == SIGMAS[0]), **commun,
            hovertemplate=f"{cas}, σ {sigma:g}<br>ICC %{{y:+.3f}}<extra></extra>"),
            row=1, col=1)
        fig.add_trace(go.Box(
            y=c, x=[f"σ {sigma:g}"] * len(c), name=f"référence {COURT_CAS[cas]}",
            legendgroup=cas, showlegend=False, **commun,
            hovertemplate=f"{cas}, σ {sigma:g}<br>CV %{{y:.0f}} %<extra></extra>"),
            row=1, col=2)
for seuil, texte in ((0.5, "médiocre"), (0.75, "bon"), (0.9, "excellent")):
    fig.add_hline(y=seuil, row=1, col=1, line={"color": C_REF, "dash": "dot", "width": 1},
                  annotation={"text": texte, "font": {"size": 9, "color": C_REF}})
fig.add_hline(y=0.0, row=1, col=1, line={"color": C_SEUIL, "dash": "dash", "width": 1})
fig.add_hline(y=100.0, row=1, col=2, line={"color": C_SEUIL, "dash": "dash", "width": 1},
              annotation={"text": "CV = 100 %", "font": {"size": 9, "color": C_SEUIL}})
fig.update_layout(boxmode="group")
fig.update_yaxes(title="ICC(A,1)", range=[-1.05, 1.05], row=1, col=1)
fig.update_yaxes(title="CV intra (%)", row=1, col=2)
mise_en_page(fig, "Répétabilité d'un battement à l'autre — mêmes boîtes que plus "
                  "haut, mais la cible est la VIDÉO et le réplicat le one-cycle ; "
                  f"{int(_oc_icc['n_subjects'].median())} vidéos par ICC contre "
                  f"{int(icc['n_subjects'].median())} sujets pour les réplicats "
                  "d'acquisition", hauteur=540)
enregistrer(fig, "p3_oc_icc_cv")

# --- L'ICC contre l'ECART entre les deux battements compares ----------------
# La paire 1v2 compare deux battements CONSECUTIFS, la paire 1v5 deux battements
# separes par trois autres. Si la valeur d'une video etait stable, l'ICC ne
# dependrait pas de cet ecart. S'il s'effondre -- ou pire, s'il devient negatif
# --, c'est qu'il n'y a pas de valeur de video : il y a une DERIVE au fil de
# l'enregistrement, et deux battements voisins se ressemblent seulement parce
# qu'ils sont voisins.
#
# C'est la figure qui justifie de ne pas titrer sur la paire 1v2, et qui explique
# l'ecart entre celle-ci et l'ICC(1) calcule sur TOUS les one-cycles a la fois.
_PAIRES_OC = [p for p in sorted(set(_oc["pair"])) if p != "toutes"]
_oc_paires = _oc[(_oc["icc_type"] == "ICC(A,1)") & _oc["pair"].isin(_PAIRES_OC)]
_st_paires = _oc_paires[(_oc_paires["famille"] == "strain")
                        & (_oc_paires["predicteur"] != "CT_region_um")]

fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.10,
                    subplot_titles=("Toutes les familles",
                                    "Strain rétinien, par style de référence"))
for famille in [f for f in ("strain", "pente", "ct_pouls", "qc")
                if f in set(_oc_paires["famille"])]:
    g = _oc_paires[_oc_paires["famille"] == famille]
    for pair in _PAIRES_OC:
        y = g[g["pair"] == pair]["icc"].dropna()
        fig.add_trace(go.Box(
            y=y, x=[pair] * len(y), name=famille, legendgroup=famille,
            offsetgroup=famille, showlegend=bool(pair == _PAIRES_OC[0]),
            boxpoints=False, marker={"color": COULEUR_FAMILLE.get(famille, C_REF)},
            line={"width": 1, "color": COULEUR_FAMILLE.get(famille, C_REF)},
            hovertemplate=f"{famille}, %{{x}}<br>ICC %{{y:+.3f}}<extra></extra>"),
            row=1, col=1)
for cas in CAS:
    g = _st_paires[_st_paires["cas"] == cas]
    for pair in _PAIRES_OC:
        y = g[g["pair"] == pair]["icc"].dropna()
        fig.add_trace(go.Box(
            y=y, x=[pair] * len(y), name=f"référence {COURT_CAS[cas]}",
            legendgroup=cas, offsetgroup=cas,
            showlegend=bool(pair == _PAIRES_OC[0]), boxpoints=False,
            marker={"color": COULEUR_CAS[cas]},
            line={"width": 1, "color": COULEUR_CAS[cas]},
            hovertemplate=f"{cas}, %{{x}}<br>ICC %{{y:+.3f}}<extra></extra>"),
            row=1, col=2)
# Pas de titre d'axe en x : la legende commune se place sous les panneaux et le
# recouvrirait. Les etiquettes 1v2 ... 1v5 se lisent seules, et le titre de la
# figure dit ce qu'elles comparent.
for col in (1, 2):
    fig.add_hline(y=0.0, row=1, col=col,
                  line={"color": C_SEUIL, "dash": "dash", "width": 1})
    fig.add_hline(y=0.5, row=1, col=col,
                  line={"color": C_REF, "dash": "dot", "width": 1})
fig.update_layout(boxmode="group")
fig.update_yaxes(title="ICC(A,1)", range=[-1.05, 1.05], row=1, col=1)
fig.update_yaxes(range=[-1.05, 1.05], row=1, col=2)
_n_par_paire = (_oc_paires[_oc_paires["famille"] == "strain"]
                .groupby("pair")["n_subjects"].median())
mise_en_page(fig, "L'accord entre deux battements s'effondre dès qu'ils ne sont "
                  "plus voisins — 1v2 compare deux one-cycles consécutifs, 1v5 "
                  "deux one-cycles séparés par trois autres ; "
                  + ", ".join(f"{p} : {int(v)} vidéos"
                              for p, v in _n_par_paire.items()), hauteur=560)
enregistrer(fig, "p3_oc_paires")

_med_paire = _oc_paires.pivot_table(index="famille", columns="pair",
                                    values="icc", aggfunc="median")
_med_paire_st = _st_paires.pivot_table(index="cas", columns="pair",
                                       values="icc", aggfunc="median")
RESUME.update({
    "p3_oc_icc_med_strain_sans_ct": float(
        _oc_icc[(_oc_icc["famille"] == "strain")
                & (_oc_icc["predicteur"] != "CT_region_um")]["icc"].median()),
    "p3_oc_paires_disponibles": ", ".join(_PAIRES_OC),
    "p3_oc_n_videos_par_paire": ", ".join(f"{p} : {int(v)}"
                                          for p, v in _n_par_paire.items()),
})
for fam in _med_paire.index:
    RESUME["p3_oc_paires_" + str(fam)] = ", ".join(
        f"{p} {_med_paire.loc[fam, p]:+.2f}" for p in _PAIRES_OC
        if p in _med_paire.columns)
for cas in _med_paire_st.index:
    RESUME["p3_oc_paires_strain_" + str(cas)] = ", ".join(
        f"{p} {_med_paire_st.loc[cas, p]:+.2f}" for p in _PAIRES_OC
        if p in _med_paire_st.columns)
RESUME["p3_oc_icc1_strain_local"] = float(
    _oc_vc[(_oc_vc["famille"] == "strain") & (_oc_vc["cas"] == "local")
           & (_oc_vc["predicteur"] != "CT_region_um")]["icc1_desequilibre"].median())
RESUME["p3_oc_icc1_strain_global"] = float(
    _oc_vc[(_oc_vc["famille"] == "strain") & (_oc_vc["cas"] == "global")
           & (_oc_vc["predicteur"] != "CT_region_um")]["icc1_desequilibre"].median())
RESUME["p3_oc_icc1_pente"] = float(
    _oc_vc[_oc_vc["famille"] == "pente"]["icc1_desequilibre"].median())
RESUME["p3_oc_frac_pente_sup05"] = float(
    (_oc_icc[_oc_icc["famille"] == "pente"]["icc"] > 0.5).mean())


# --- Le croisement des deux definitions : est-ce la MEME reponse ? ----------
# Un point par predicteur : en abscisse son ICC sur les repliques d'acquisition
# (vue « before », celle que le reste du site lit), en ordonnee son ICC sur les
# one-cycles. La diagonale dit « les deux definitions concordent ».
_cles = ["famille", "predicteur", "cas", "sigma", "region"]
_col_icc = _cles + ["icc", "n_subjects"]
_ren = {"icc": "icc_repliques", "n_subjects": "n_sujets"}
# `icc` (defini plus haut) est restreint aux familles des demons ET aux sigmas
# renseignes : il perdrait `CT_region_um`, `ct_pouls` et `qc`. On relit donc
# `repeat` directement. Les cles de jointure contenant des NaN (sigma des
# predicteurs qui n'en dependent pas), pandas les apparie entre elles, ce qui
# est bien le comportement voulu ici.
_gauche = repeat[(repeat["icc_type"] == "ICC(A,1)") & (repeat["pair"] == "1v2")
                 & (repeat["moment"] == "before")][_col_icc].rename(columns=_ren)
_crois = (_oc_icc[_col_icc].rename(columns={"icc": "icc_one_cycles",
                                            "n_subjects": "n_videos"})
          .merge(_gauche, on=_cles, how="inner")
          .dropna(subset=["icc_repliques", "icc_one_cycles"]))

fig = go.Figure()
for famille in [f for f in ("strain", "pente", "ct_pouls", "qc")
                if f in set(_crois["famille"])]:
    g = _crois[_crois["famille"] == famille]
    fig.add_trace(go.Scatter(
        x=g["icc_repliques"], y=g["icc_one_cycles"], mode="markers", name=famille,
        marker={"color": COULEUR_FAMILLE.get(famille, C_REF), "size": 6,
                "opacity": 0.65, "line": {"width": 0}},
        customdata=g[["predicteur", "region", "sigma", "cas"]].to_numpy(),
        hovertemplate=("%{customdata[0]}<br>%{customdata[1]} · σ %{customdata[2]}"
                       " · réf. %{customdata[3]}<br>ICC répliques %{x:+.2f}"
                       "<br>ICC one-cycles %{y:+.2f}<extra></extra>")))
fig.add_shape(type="line", x0=-1, y0=-1, x1=1, y1=1,
              line={"color": C_REF, "dash": "dash", "width": 1})
fig.add_hline(y=0.0, line={"color": C_REF, "dash": "dot", "width": 1})
fig.add_vline(x=0.0, line={"color": C_REF, "dash": "dot", "width": 1})
fig.update_xaxes(title="ICC(A,1), réplicats d'acquisition (vue « avant »)",
                 range=[-1.05, 1.05])
fig.update_yaxes(title="ICC(A,1), one-cycles d'une même vidéo", range=[-1.05, 1.05])
_r_deux_icc = (float(np.corrcoef(_crois["icc_repliques"], _crois["icc_one_cycles"])[0, 1])
               if len(_crois) > 2 else float("nan"))
mise_en_page(fig, "Les deux définitions du réplicat donnent-elles la même réponse ? "
                  f"{len(_crois)} prédicteurs communs — r = {_r_deux_icc:+.3f}. "
                  "Ligne tiretée : les deux ICC égaux", hauteur=620)
enregistrer(fig, "p3_oc_vs_repliques")


def table_repeat_oc() -> str:
    """Une ligne par famille : ce que l'ICC « one-cycles » donne, a cote de
    l'ICC « repliques » des memes predicteurs."""
    tick = chr(96)
    lignes = ["| famille | prédicteurs | ICC(A,1) one-cycles | ICC(1), tous les "
              "one-cycles | CV intra | var. inter < 0 | ICC(A,1) réplicats |",
              "|:--|--:|--:|--:|--:|--:|--:|"]
    for famille in [f for f in ("strain", "pente", "ct_pouls", "qc")
                    if f in set(_oc_icc["famille"])]:
        a = _oc_icc[_oc_icc["famille"] == famille]
        v = _oc_vc[_oc_vc["famille"] == famille]
        ref = _crois[_crois["famille"] == famille]["icc_repliques"]
        lignes.append(
            f"| {tick}{famille}{tick} | {a['predicteur'].nunique()} "
            f"({len(a)} variantes) | {med_iqr(a['icc'], '{:+.2f}')} "
            f"| {med_iqr(v['icc1_desequilibre'], '{:+.2f}')} "
            f"| {med_iqr(v['cv_intra_pct'], '{:.0f}')} % "
            f"| {100 * (v['var_inter'] < 0).mean():.0f} % "
            f"| {med_iqr(ref, '{:+.2f}')} |")
    return chr(10).join(lignes)


ecrire_table(table_repeat_oc(), "tab_repeat_oc")

RESUME.update({
    "p3_oc_n_predicteurs": int(_oc_icc["predicteur"].nunique()),
    "p3_oc_n_variantes": int(len(_oc_icc)),
    "p3_oc_n_videos_med": float(_oc_icc["n_subjects"].median()),
    "p3_oc_icc_med": float(_oc_icc["icc"].median()),
    "p3_oc_icc_iqr": med_iqr(_oc_icc["icc"], "{:+.2f}"),
    "p3_oc_icc_frac05": float((_oc_icc["icc"] > 0.5).mean()),
    "p3_oc_icc_frac075": float((_oc_icc["icc"] > 0.75).mean()),
    "p3_oc_icc_neg": float((_oc_icc["icc"] < 0).mean()),
    "p3_oc_icc1_med": float(_oc_vc["icc1_desequilibre"].median()),
    "p3_oc_cv_med": float(_oc_vc["cv_intra_pct"].median()),
    "p3_oc_var_inter_neg": float((_oc_vc["var_inter"] < 0).mean()),
    "p3_oc_icc_med_strain": float(_oc_icc[_oc_icc["famille"] == "strain"]["icc"].median()),
    "p3_oc_icc_med_pente": float(_oc_icc[_oc_icc["famille"] == "pente"]["icc"].median()),
    "p3_oc_icc_med_ct": float(_oc_icc[_oc_icc["predicteur"] == "CT_region_um"]["icc"].median()),
    "p3_oc_cv_med_ct": float(_oc_vc[_oc_vc["predicteur"] == "CT_region_um"]["cv_intra_pct"].median()),
    "p3_oc_r_deux_icc": _r_deux_icc,
    "p3_oc_n_croises": int(len(_crois)),
    "p3_oc_med_repliques": float(_crois["icc_repliques"].median()),
    "p3_oc_med_one_cycles": float(_crois["icc_one_cycles"].median()),
    "p3_oc_frac_oc_sup": float((_crois["icc_one_cycles"] > _crois["icc_repliques"]).mean()),
})


# --- Stabilite AVANT / APRES le vol -----------------------------------------
# Une seconde repetabilite, a une tout autre echelle de temps : le meme oeil du
# meme astronaute, mesure avant puis apres le vol -- des mois d'intervalle, avec
# un vol au milieu. L'ICC des sections precedentes porte sur des repliques d'une
# MEME seance ; ici l'intervalle est de plusieurs mois.
#
# Attention a ce que cela mesure : un parametre de TRAIT (une propriete stable
# de l'oeil) doit rester correle, tandis qu'un parametre qui repond reellement
# au vol doit se decorreler. Une correlation nulle ne tranche donc pas entre
# « c'est du bruit » et « ca a change avec le vol » -- c'est l'ICC a court terme
# qui separe les deux, et c'est pourquoi les deux figures se lisent ensemble.
_st = stab[~stab["predicteur"].str.contains("choroide", na=False)].copy()
_st["reference"] = _st["cas"].map(lambda c: COURT_CAS.get(c, "—"))
_st["sigma_lab"] = _st["sigma"].map(lambda v: "—" if pd.isna(v) else f"σ {v:g}")
FAMILLE_ORDRE = [x for x in ("strain", "pente", "pente_pool", "ct_pouls",
                             "pouls", "rigidite", "qc")
                 if x in set(_st["famille"])]
DECOUPES = [
    ("famille", "par famille de paramètres", FAMILLE_ORDRE,
     lambda n: COULEUR_FAMILLE.get(n, C_REF)),
    ("sigma_lab", "par régularisation", [f"σ {v:g}" for v in SIGMAS] + ["—"],
     lambda n: COULEUR_SIGMA.get(float(n.split()[-1]), C_REF) if n != "—" else C_REF),
    ("reference", "par référence", ["locale", "globale", "—"],
     lambda n: COULEUR_CAS.get({"locale": "local", "globale": "global"}.get(n), C_REF)),
    ("region", "par bande de rétine", REGIONS + ["toutes"],
     lambda n: COULEUR_REGION.get(n, C_REF)),
]
MESURES = [("pearson_r", "Pearson r"), ("spearman_rho", "Spearman ρ")]

fig = make_subplots(rows=len(DECOUPES), cols=2, shared_yaxes=True,
                    horizontal_spacing=0.06, vertical_spacing=0.055,
                    subplot_titles=[f"{lab} — {m}" for _, lab, _, _ in DECOUPES
                                    for _, m in MESURES])
for i, (col, _lab, niveaux, couleur) in enumerate(DECOUPES, start=1):
    for j, (mes, _mlab) in enumerate(MESURES, start=1):
        for niveau in niveaux:
            g = _st[_st[col].astype(str) == str(niveau)][mes]
            if g.empty:
                continue
            fig.add_trace(go.Box(
                y=g, x=[str(niveau)] * len(g), name=str(niveau), showlegend=False,
                boxpoints=False, marker={"color": couleur(str(niveau)), "size": 2},
                line={"width": 1.2, "color": couleur(str(niveau))},
                hovertemplate=(f"{niveau}<br>médiane %{{median:+.3f}}"
                               f"<br>n = {len(g)}<extra></extra>"),
            ), row=i, col=j)
        fig.add_hline(y=0.0, row=i, col=j,
                      line={"color": C_REF, "dash": "dot", "width": 1})
_n_st = int(_st["n_sujets"].median())
_r_crit_st = float(stats.t.ppf(0.975, _n_st - 2))
_r_crit_st = _r_crit_st / np.sqrt(_r_crit_st ** 2 + _n_st - 2)
for i in range(1, len(DECOUPES) + 1):
    for j in (1, 2):
        for signe in (1, -1):
            fig.add_hline(y=signe * _r_crit_st, row=i, col=j,
                          line={"color": C_SEUIL, "dash": "dash", "width": 1})
    fig.update_yaxes(title="corrélation avant / après", range=[-1.02, 1.02], row=i, col=1)
fig.update_xaxes(tickangle=-35, tickfont={"size": 9})
mise_en_page(fig, f"Stabilité avant / après le vol — même œil, même astronaute, "
                  f"{_n_st} sujets. Traits rouges : seuil de significativité "
                  f"(|r| = {_r_crit_st:.2f} à p = 0,05)", hauteur=1250, legende=False)
enregistrer(fig, "p3_stabilite")


# --- Stabilite longue contre repetabilite courte -----------------------------
# Les deux mesures sur les memes predicteurs : l'ICC dit si deux acquisitions
# d'une meme seance concordent, la correlation avant/apres si la valeur survit a
# plusieurs mois et a un vol. Un parametre qui n'a NI l'un NI l'autre est du
# bruit ; un parametre qui a le premier sans le second a bel et bien change.
#
# UN PANNEAU PAR CATEGORIE de parametre, et la COULEUR porte la regularisation.
# Les familles de service -- `qc` (metriques de controle du recalage), `pouls` et
# `ct_pouls` (qualite de l'extraction du pouls) -- sont ecartees : ce sont des
# diagnostics de chaine, pas des candidats marqueurs, et leurs 50 points
# encombraient un nuage qui en compte deux mille. Elles restent dans la figure
# `p3_stabilite` et dans `tab_stabilite`, ou elles servent de comparaison.
#
# La couleur ne redit donc plus la famille (elle est deja le panneau) mais σ, le
# seul reglage des demons qu'on puisse encore choisir a ce stade. Deux familles
# echappent aux demons -- `rigidite` en entier, et `CT_region_um` range dans
# `strain` par heritage de nommage -- et forment le niveau gris « sans σ ».
FAMILLES_CROISEES = [f for f in ("strain", "pente", "pente_pool", "rigidite")
                     if f in set(_st["famille"])]
# Chaque niveau est (libelle, valeur de sigma ou None, couleur). `None` capte les
# predicteurs sans sigma : `np.isclose` rend False sur NaN, ils seraient sinon
# perdus par tous les autres niveaux.
SIGMA_NIVEAUX = ([(f"σ {v:g}", float(v), COULEUR_SIGMA.get(float(v), C_REF))
                  for v in SIGMAS]
                 + [("sans σ (hors démons)", None, C_REF)])

_icc_j = repeat[(repeat["icc_type"] == "ICC(A,1)") & (repeat["pair"] == "1v2")
                & (repeat["moment"] == "before")
                & ~repeat["predicteur"].str.contains("choroide", na=False)]
_icc_j = _icc_j.rename(columns={"icc": "icc_court"})
_j = _st.merge(_icc_j[["famille", "predicteur", "cas", "sigma", "region", "icc_court"]],
               on=["famille", "predicteur", "cas", "sigma", "region"], how="inner")
_j = _j.dropna(subset=["icc_court", "pearson_r"])
_j = _j[_j["famille"].isin(FAMILLES_CROISEES)]


def _r_deux_mesures(g):
    """Correlation entre les deux repetabilites, ou NaN si elle n'a pas de sens.

    `rigidite` ne compte que quatre predicteurs : le coefficient est calculable
    mais ne vaut rien. Il est affiche quand meme -- avec son effectif a cote, qui
    dit comment le lire.
    """
    if len(g) < 3:
        return np.nan
    return float(np.corrcoef(g["icc_court"], g["pearson_r"])[0, 1])


if not _j.empty:
    _titres = []
    for famille in FAMILLES_CROISEES:
        g = _j[_j["famille"] == famille]
        _r_f = _r_deux_mesures(g)
        _titres.append(f"{famille} — {len(g)} prédicteurs"
                       + (f", r = {_r_f:+.2f}" if np.isfinite(_r_f) else ""))
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=True,
                        horizontal_spacing=0.05, vertical_spacing=0.075,
                        subplot_titles=_titres)
    _vus = set()
    for k, famille in enumerate(FAMILLES_CROISEES):
        i, j_col = k // 2 + 1, k % 2 + 1
        sub = _j[_j["famille"] == famille]
        for libelle, valeur, couleur in SIGMA_NIVEAUX:
            g = (sub[sub["sigma"].isna()] if valeur is None
                 else sub[np.isclose(sub["sigma"].astype(float), valeur)])
            if g.empty:
                continue
            fig.add_trace(go.Scattergl(
                x=g["icc_court"], y=g["pearson_r"], mode="markers", name=libelle,
                legendgroup=libelle, showlegend=libelle not in _vus,
                marker={"color": couleur, "size": 5, "opacity": 0.6},
                hovertext=g["predicteur"] + " · " + g["region"].astype(str),
                hovertemplate=("%{hovertext}<br>" + libelle + "<br>ICC %{x:+.2f}"
                               "<br>avant/après %{y:+.2f}<extra></extra>"),
            ), row=i, col=j_col)
            _vus.add(libelle)
        fig.add_hline(y=0.0, row=i, col=j_col,
                      line={"color": C_REF, "dash": "dot", "width": 1})
        fig.add_vline(x=0.0, row=i, col=j_col,
                      line={"color": C_REF, "dash": "dot", "width": 1})
        for signe in (1, -1):
            fig.add_hline(y=signe * _r_crit_st, row=i, col=j_col,
                          line={"color": C_SEUIL, "dash": "dash", "width": 1})
        fig.add_vline(x=0.75, row=i, col=j_col,
                      line={"color": C_REF, "dash": "dot", "width": 1})
    fig.update_xaxes(range=[-0.85, 1.0])
    fig.update_yaxes(range=[-1.02, 1.02])
    for j_col in (1, 2):
        fig.update_xaxes(title="ICC(A,1) — répétabilité à court terme "
                               "(réplicats d'une séance)", row=2, col=j_col)
    for i in (1, 2):
        fig.update_yaxes(title="Pearson r — stabilité avant / après le vol",
                         row=i, col=1)
    _r_deux = _r_deux_mesures(_j)
    mise_en_page(fig, f"Les deux répétabilités, sur les mêmes {len(_j)} prédicteurs — "
                      f"elles se correspondent à r = {_r_deux:+.2f} toutes catégories "
                      f"confondues. Couleur : régularisation des démons. Traits "
                      f"rouges : |r| = {_r_crit_st:.2f}", hauteur=900)
    enregistrer(fig, "p3_stabilite_vs_icc")
    RESUME["p3_stab_r_avec_icc"] = _r_deux
    RESUME["p3_stab_n_joints"] = int(len(_j))
    RESUME["p3_stab_frac_ni_ni"] = float(((_j["icc_court"] < 0.5)
                                          & (_j["pearson_r"].abs() < _r_crit_st)).mean())
    for famille in FAMILLES_CROISEES:
        g = _j[_j["famille"] == famille]
        RESUME[f"p3_stab_n_{famille}"] = int(len(g))
        RESUME[f"p3_stab_r_avec_icc_{famille}"] = _r_deux_mesures(g)
        RESUME[f"p3_stab_icc_med_{famille}"] = float(g["icc_court"].median())
        RESUME[f"p3_stab_frac_ni_ni_{famille}"] = float(
            ((g["icc_court"] < 0.5) & (g["pearson_r"].abs() < _r_crit_st)).mean())

RESUME.update({
    "p3_stab_n": int(len(_st)),
    "p3_stab_n_sujets": _n_st,
    "p3_stab_r_crit": _r_crit_st,
    "p3_stab_r_med": float(_st["pearson_r"].median()),
    "p3_stab_rho_med": float(_st["spearman_rho"].median()),
    "p3_stab_frac_p05": float((_st["pearson_p"] < 0.05).mean()),
    "p3_stab_frac_q05": float((_st["pearson_q"] < 0.05).mean()),
    "p3_stab_r_max": float(_st["pearson_r"].max()),
})
for famille in FAMILLE_ORDRE:
    g = _st[_st["famille"] == famille]
    RESUME[f"p3_stab_r_{famille}"] = float(g["pearson_r"].median())
    RESUME[f"p3_stab_rho_{famille}"] = float(g["spearman_rho"].median())
    RESUME[f"p3_stab_frac_p05_{famille}"] = float((g["pearson_p"] < 0.05).mean())


def table_stabilite() -> str:
    """Stabilite avant/apres par famille, Pearson ET Spearman.

    L'ecart entre les deux est la colonne a surveiller : sur 26 sujets, un seul
    point extreme suffit a fabriquer un Pearson eleve que le Spearman ne suit
    pas.
    """
    lignes = ["| famille | prédicteurs | $n$ | Pearson $r$ médian "
              "| Spearman $" + BS_RHO + "$ médian | $p < 0{,}05$ | $q < 0{,}05$ |",
              "|:--|--:|--:|--:|--:|--:|--:|"]
    for famille in FAMILLE_ORDRE + ["TOUTES"]:
        g = _st if famille == "TOUTES" else _st[_st["famille"] == famille]
        if g.empty:
            continue
        nom = "**toutes**" if famille == "TOUTES" else famille
        lignes.append(
            f"| {nom} | {len(g)} | {int(g['n_sujets'].median())} "
            f"| {fmt_num(float(g['pearson_r'].median()), 3, signe=True)} "
            f"| {fmt_num(float(g['spearman_rho'].median()), 3, signe=True)} "
            f"| {100 * (g['pearson_p'] < 0.05).mean():.0f} % "
            f"| {100 * (g['pearson_q'] < 0.05).mean():.0f} % |")
    return chr(10).join(lignes)


# =========================================================================== #
# PAGES 4 — Prediction of SANS, une page par CATEGORIE de marqueurs
# =========================================================================== #
# Le panel se lit en trois questions distinctes, et les melanger avait deux
# couts : une carte de chaleur ou l'epaisseur choroidienne voisinait avec des
# pentes de strain sans qu'on puisse les comparer, et une correction de
# Benjamini-Hochberg qui noyait les quelques marqueurs d'epaisseur dans un
# millier de variantes de pente. Trois categories, trois pages, trois familles
# de correction :
#
#   rigidite  -- l'EPAISSEUR choroidienne et sa pulsation : le k de Sayah et ses
#                entrees, les metriques CT / pouls, et l'epaisseur de chaque
#                bande. Rien n'y passe par le champ de deformation.
#   strain    -- les trois agregats du strain retinien, bande par bande.
#   viscosite -- les pentes du strain contre l'epaisseur et contre la
#                dilatation, dans leurs DEUX estimateurs (par one-cycle et
#                poolee), et les coefficients r de ces ajustements.
#
# La liste ci-dessous est DUPLIQUEE dans ``compute_sans_predictors.py``
# (fonction ``categorie_de``), qui l'utilise comme perimetre de la q-valeur
# ``pearson_q_categorie``. Les deux doivent bouger ensemble, sinon la q publiee
# ne porte plus sur les cases dessinees ici.
#
# SIX issues, et la sixieme n'est PAS un TRT : ``delta_AL_mm`` est la variation
# de longueur axiale entre les deux visites, c'est-a-dire l'APLATISSEMENT DU
# GLOBE -- l'autre signe de SANS. Meme convention de signe que les TRT (apres -
# avant), donc negatif quand le globe s'aplatit ; ne pas la retourner, la couleur
# d'une case de carte de chaleur doit vouloir dire la meme chose sur les six
# colonnes. La grille des nuages ci-dessous est deja dimensionnee pour six
# panneaux (2 x 3) : le sixieme, jusqu'ici vide, se remplit.
ISSUES = [c for c in ("delta_TRT_S", "delta_TRT_I", "delta_TRT_N", "delta_TRT_T",
                      "delta_TRT_moyen", "delta_AL_mm") if c in set(corr["issue"])]
ISSUE_LABEL = {"delta_TRT_S": "ΔTRT S", "delta_TRT_I": "ΔTRT I",
               "delta_TRT_N": "ΔTRT N", "delta_TRT_T": "ΔTRT T",
               "delta_TRT_moyen": "ΔTRT moy.", "delta_AL_mm": "ΔAL"}
ISSUE_AL = "delta_AL_mm"
BS_RHO = chr(92) + "rho"
VUE = "before"  # la vue « peut-on predire AVANT le vol », celle qui interesse

# Seuils du reperage vert des tables. Ils ne designent PAS un resultat :
# |r| >= 0,2 sur 12 a 18 sujets est tres loin du seuil de significativite
# (0,58 a n = 12). C'est l'ICC qui fait le tri. Le vert veut donc dire « assez
# reproductible pour qu'on regarde sa correlation », pas « correle ».
VERT_R_MIN = 0.20
VERT_ICC_MIN = 0.45
VERT_CADRE = "#27a567"  # le vert des palettes ; `.marqueur-retenu` (#1e8a55)
# est trop sombre sur les cases bleues du theme sombre.

_MARQ_RIG = ([("rigidite", "k_sayah", "k de Sayah"),
              ("rigidite", "delta_V_mm3", "ΔV du one-cycle"),
              ("rigidite", "CT_mm", "épaisseur choroïdienne (chaîne k)"),
              ("rigidite", "delta_CT_mm", "ΔCT (chaîne k)"),
              ("ct_pouls", "CT_moyen_um", "épaisseur moyenne du one-cycle"),
              ("ct_pouls", "deltaCT_one_cycle_um", "ΔCT du one-cycle")]
             + [("ct_pouls", f"{p}_CT_pouls_{s}", f"{lab} CT / pouls ({lab_s})")
                for p, lab in (("r", "r"), ("absr", "|r|"), ("cov", "couverture"))
                for s, lab_s in (("temps", "temporel"), ("bins", "bins"))]
             + [("strain", "CT_region_um", "épaisseur de la bande")])

_MARQ_STR = [("strain", col, f"strain rétinien ({lab_ag})")
             for col, lab_ag, _, _ in AGREGATS]

_ABSCISSES_VIS = (("CT_region_um", "CT"), ("dCT_region_um", "ΔCT"))
_MARQ_VIS_PENTES = [
    (famille, modele.format(abs=abs_col, col=col),
     f"pente {lab_abs} → strain, {lab_est} ({lab_ag})")
    for famille, modele, lab_est in (
        ("pente", "pente_{abs}_{col}", "par one-cycle"),
        ("pente_pool", "poolpente_{abs}_{col}", "poolée"))
    for abs_col, lab_abs in _ABSCISSES_VIS
    for col, lab_ag, _, _ in AGREGATS]
_MARQ_VIS_R = [
    (famille, modele.format(abs=abs_col, col=col),
     f"r de l'ajustement {lab_abs}, {lab_est} ({lab_ag})")
    for famille, modele, lab_est in (
        ("pente", "r_{abs}_{col}", "par one-cycle"),
        ("pente_pool", "poolr_{abs}_{col}", "poolée"))
    for abs_col, lab_abs in _ABSCISSES_VIS
    for col, lab_ag, _, _ in AGREGATS]

CATEGORIES_P4 = [
    {"cle": "rig", "categorie": "rigidite", "nom": "rigidité",
     "article": "la rigidité",
     "marqueurs": _MARQ_RIG, "tables": _MARQ_RIG, "grille": False,
     "cartes": [("", "", _MARQ_RIG)]},
    {"cle": "str", "categorie": "strain", "nom": "strain",
     "article": "le strain",
     "marqueurs": _MARQ_STR, "tables": _MARQ_STR, "grille": True,
     "cartes": [("", "", _MARQ_STR)]},
    {"cle": "vis", "categorie": "viscosite", "nom": "viscosité",
     "article": "la viscosité",
     "marqueurs": _MARQ_VIS_PENTES + _MARQ_VIS_R, "tables": _MARQ_VIS_PENTES,
     "grille": True,
     "cartes": [("", " — les pentes", _MARQ_VIS_PENTES),
                ("_r", " — les coefficients r", _MARQ_VIS_R)]},
]

# La q-valeur par categorie est calculee par le lot et publiee dans
# `sans_correlations.csv`. On la refait ici quand la table n'a pas encore ete
# regeneree, sur exactement le meme perimetre : BH dans (categorie, vue, issue).
if "categorie" not in corr.columns:
    corr["categorie"] = ""
    for _c in CATEGORIES_P4:
        _m = corr.set_index(["famille", "predicteur"]).index.isin(
            [(f, p) for f, p, _ in _c["marqueurs"]])
        corr.loc[_m, "categorie"] = _c["categorie"]
    # `categorie_de` du lot classe TOUTE la famille, pas seulement les marqueurs
    # affiches : sans cela le repli ne porterait pas sur le meme perimetre.
    corr.loc[corr["famille"].isin(("pente", "pente_pool")), "categorie"] = "viscosite"
    corr.loc[corr["famille"] == "strain", "categorie"] = "strain"
    corr.loc[corr["famille"].isin(("rigidite", "ct_pouls")), "categorie"] = "rigidite"
    corr.loc[corr["predicteur"] == "CT_region_um", "categorie"] = "rigidite"
if "pearson_q_categorie" not in corr.columns or corr["pearson_q_categorie"].isna().all():
    print("  ! pearson_q_categorie absent de sans_correlations.csv -- recalcul local "
          "(relancer Astronauts/compute_sans_predictors.py pour que la table le porte)")
    _avec = corr["categorie"] != ""
    for _src, _dst in (("pearson_p", "pearson_q_categorie"),
                       ("spearman_p", "spearman_q_categorie")):
        corr[_dst] = np.nan
        corr.loc[_avec, _dst] = (corr.loc[_avec]
                                 .groupby(["categorie", "vue", "issue"])[_src]
                                 .transform(lambda s: bh_fdr(s.to_numpy())))


# La repetabilite de chaque marqueur, jointe a ses correlations. L'ICC et le CV
# ne dependent PAS de l'issue : ils qualifient le marqueur seul. Ils sont donc
# constants le long d'une ligne de table -- c'est voulu, et dit dans l'en-tete.
# La repetabilite retenue est celle du moment « before », le meme que la vue
# correlee : mesurer la reproductibilite sur les repliques d'avant le vol et
# correler celles d'apres n'aurait pas de sens.
_CLES_REP = ["famille", "predicteur", "cas", "sigma", "region"]


def _corr_avec_repetabilite(marqueurs, vue=VUE):
    noms = {(f, p) for f, p, _ in marqueurs}
    est_marqueur = np.array([(f, p) in noms
                             for f, p in zip(corr["famille"], corr["predicteur"])])
    c = corr[(corr["vue"] == vue).to_numpy() & est_marqueur]
    a = repeat[(repeat["icc_type"] == "ICC(A,1)") & (repeat["pair"] == "1v2")
               & (repeat["moment"] == vue)][_CLES_REP + ["icc"]]
    v = repeat[(repeat["pair"] == "toutes")
               & (repeat["moment"] == vue)][_CLES_REP + ["cv_intra_pct"]]
    return c.merge(a, on=_CLES_REP, how="left").merge(v, on=_CLES_REP, how="left")


def _etiquettes(df, marqueurs, grille):
    """Le libelle de ligne d'une carte ou d'une table.

    Avec la grille (sigma x reference), les panneaux portent deja sigma et le
    style de reference : la ligne n'a besoin que du marqueur et de la bande.
    Sans grille, la ligne doit tout porter -- certains marqueurs de la page
    rigidite existent par bande ET par style de reference, d'autres ni l'un ni
    l'autre.
    """
    lab = {(f, p): l for f, p, l in marqueurs}
    noms = df["predicteur"].map(lambda p: p)
    base = [lab.get((f, p), p) for f, p in zip(df["famille"], df["predicteur"])]
    out = []
    for b, region, cas in zip(base, df["region"], df["cas"]):
        txt = b
        if region != "toutes":
            txt += " · " + str(region)
        if not grille and isinstance(cas, str) and cas:
            txt += " · réf. " + COURT_CAS.get(cas, cas)
        out.append(txt)
    return pd.Series(out, index=df.index)


def _ordre_etiquettes(marqueurs, grille, presentes):
    """L'ordre des lignes : marqueur-majeur, puis bande, puis reference."""
    ordre = []
    for f, p, lab in marqueurs:
        for region in REGIONS + ["toutes"]:
            # Le "" final est indispensable : sans grille, les marqueurs qui
            # n'ont PAS de style de reference (le k de Sayah, les metriques
            # CT / pouls) portent un libelle sans suffixe, et l'oublier les
            # faisait disparaitre de la carte.
            for cas in (CAS + [""] if not grille else [""]):
                txt = lab
                if region != "toutes":
                    txt += " · " + str(region)
                if not grille and cas:
                    txt += " · réf. " + COURT_CAS.get(cas, cas)
                if txt in presentes and txt not in ordre:
                    ordre.append(txt)
    return ordre


def carte_prediction(cat, suffixe, titre_sup, marqueurs, merge):
    """La carte de chaleur : une case par (marqueur, issue), une grille par
    (sigma, reference) quand les marqueurs en dependent.

    Couleur = r de Pearson, asterisque = q < 0,05 DANS LA CATEGORIE, survol =
    ICC(A,1) du marqueur, cadre vert = marqueur assez repetable atteignant
    |r| >= 0,20 sur cette issue.
    """
    _noms = {(a, b) for a, b, _ in marqueurs}
    sel = merge[np.array([(f, p) in _noms
                          for f, p in zip(merge["famille"], merge["predicteur"])])].copy()
    sel["etiquette"] = _etiquettes(sel, marqueurs, cat["grille"])
    if cat["grille"]:
        panneaux = [(sg, cs) for sg in SIGMAS for cs in CAS]
        n_lignes, n_cols = len(SIGMAS), len(CAS)
        titres = [f"σ = {sg:g} · référence {COURT_CAS[cs]}" for sg, cs in panneaux]
    else:
        panneaux = [(None, None)]
        n_lignes, n_cols = 1, 1
        titres = [""]
    fig = make_subplots(rows=n_lignes, cols=n_cols, shared_xaxes=True,
                        shared_yaxes=True, horizontal_spacing=0.05,
                        vertical_spacing=0.04, subplot_titles=titres)
    n_cases = n_vertes = 0
    n_max_lignes = 1
    _libelles = []
    for k, (sigma, cas) in enumerate(panneaux):
        i, j = divmod(k, n_cols)
        i, j = i + 1, j + 1
        sub = sel
        if cat["grille"]:
            sub = sel[(sel["cas"] == cas)
                      & np.isclose(sel["sigma"].astype(float), float(sigma))]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="etiquette", columns="issue", values="pearson_r")
        ordre = _ordre_etiquettes(marqueurs, cat["grille"], set(piv.index))
        piv = piv.reindex(index=ordre, columns=[c for c in ISSUES if c in piv.columns])
        pq = sub.pivot_table(index="etiquette", columns="issue",
                             values="pearson_q_categorie").reindex_like(piv)
        picc = sub.pivot_table(index="etiquette", columns="issue",
                               values="icc").reindex_like(piv)
        n_max_lignes = max(n_max_lignes, len(piv.index))
        _libelles.extend(piv.index)
        fig.add_trace(go.Heatmap(
            z=piv.to_numpy(), x=[ISSUE_LABEL[c] for c in piv.columns],
            y=list(piv.index), zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
            showscale=bool(k == 0),
            colorbar={"title": "r", "thickness": 12, "len": 0.35, "y": 0.86},
            text=np.where(pq.to_numpy() < 0.05, "*", ""), texttemplate="%{text}",
            customdata=picc.to_numpy(),
            hovertemplate="%{y}<br>%{x}<br>r = %{z:+.2f}"
                          "<br>ICC %{customdata:+.2f}<extra></extra>"), row=i, col=j)
        vert = ((picc.to_numpy() >= VERT_ICC_MIN)
                & (np.abs(piv.to_numpy()) >= VERT_R_MIN))
        n_cases += int(np.isfinite(piv.to_numpy()).sum())
        n_vertes += int(vert.sum())
        # Les axes sont CATEGORIELS : une coordonnee numerique y designe l'indice
        # de categorie, d'ou les bords a +/- 0,5 autour de la case.
        for iy, jx in zip(*np.nonzero(vert)):
            fig.add_shape(type="rect", row=i, col=j,
                          x0=jx - 0.5, x1=jx + 0.5, y0=iy - 0.5, y1=iy + 0.5,
                          line={"color": VERT_CADRE, "width": 1.6},
                          fillcolor="rgba(0,0,0,0)", layer="above")
    # 17 px par ligne et par rangee de panneaux : la densite de l'ancienne carte
    # (45 lignes x 4 sigmas en 2 900 px). A 40 px la viscosite atteignait
    # 9 660 px, illisible et lourd.
    hauteur = max(420, int(90 + n_max_lignes * n_lignes * 17))
    mise_en_page(fig, f"Prédiction du SANS par {cat['article']}{titre_sup}, vue "
                      "« avant le vol » — corrélation de Pearson entre chaque "
                      "marqueur et chaque issue ; l'astérisque marque q < 0,05 "
                      f"dans la catégorie, le cadre vert un marqueur répétable "
                      f"(ICC ≥ {VERT_ICC_MIN:.2f}) atteignant |r| ≥ {VERT_R_MIN:.1f} "
                      f"sur cette issue — {n_vertes} cases sur {n_cases}",
                 hauteur=hauteur, legende=False)
    # Les libelles de ligne sont longs (jusqu'a 65 caracteres pour la viscosite)
    # et la marge gauche commune de 70 px les tronquerait hors du cadre. Environ
    # 6 px par caractere a 11 px de fonte, mesure dans le navigateur.
    _long = max((len(str(y)) for y in _libelles), default=20)
    fig.update_layout(margin={"l": int(min(460, max(120, 6.0 * _long + 20))),
                              "r": 20, "t": 70, "b": 50})
    enregistrer(fig, f"p4{cat['cle']}_heatmap{suffixe}")
    return n_cases, n_vertes


def nuage_q(cat, merge):
    """q de Benjamini-Hochberg en abscisse, corrélation en ordonnée, un panneau
    par issue. La q-valeur est celle de la CATEGORIE a issue fixee."""
    q = merge.dropna(subset=["pearson_q_categorie", "pearson_r"]).copy()
    q["pearson_q_categorie"] = q["pearson_q_categorie"].round(4)
    q["pearson_r"] = q["pearson_r"].round(4)
    # Poids du fragment : la categorie viscosite compte 960 marqueurs affiches
    # par panneau. Le libelle -- la chaine la plus longue du survol -- sort du
    # `customdata` en devenant le critere de decoupage des traces.
    q["etiquette"] = _etiquettes(q, cat["marqueurs"], False)
    q["sigma_lab"] = q["sigma"].map(lambda v: "—" if pd.isna(v) else f"{float(v):g}")
    q["spearman_rho"] = q["spearman_rho"].round(2)
    cols = ["sigma_lab", "pearson_p", "spearman_rho", "n_sujets"]
    survol = ("σ %{customdata[1]}"
              "<br>q = %{x:.3f}  ·  p = %{customdata[2]:.3g}"
              "<br>r = %{y:+.2f}  ·  ρ = %{customdata[3]:+.2f}"
              "  ·  n = %{customdata[4]:d}")
    fig = make_subplots(rows=2, cols=3, shared_yaxes=True,
                        horizontal_spacing=0.045, vertical_spacing=0.13,
                        subplot_titles=[ISSUE_LABEL[i] for i in ISSUES]
                                       + [""] * (6 - len(ISSUES)))
    bas = {}
    for k, issue in enumerate(ISSUES):
        i, j = divmod(k, 3)
        i, j = i + 1, j + 1
        bas[j] = max(bas.get(j, 0), i)
        sub = q[q["issue"] == issue]
        for famille in [f for f in FAMILLE_ORDRE_P4 if f in set(sub["famille"])]:
            g = sub[sub["famille"] == famille]
            fig.add_trace(go.Scatter(
                x=g["pearson_q_categorie"], y=g["pearson_r"], mode="markers",
                name=famille, legendgroup=famille, showlegend=bool(k == 0),
                marker={"color": COULEUR_FAMILLE.get(famille, C_REF), "size": 6,
                        "opacity": 0.7, "line": {"width": 0}},
                customdata=g[["etiquette"] + cols].to_numpy(),
                hovertemplate="%{customdata[0]}<br>" + survol + "<extra></extra>"),
                row=i, col=j)
        fig.add_vline(x=0.05, row=i, col=j, line={"color": C_SEUIL, "width": 1.2})
        for signe in (1, -1):
            fig.add_hline(y=signe * _r_crit, row=i, col=j,
                          line={"color": C_SEUIL, "dash": "dot", "width": 1})
        fig.add_hline(y=0.0, row=i, col=j, line={"color": C_REF, "dash": "dot", "width": 1})
    fig.update_xaxes(range=[0, 1.02], dtick=0.25)
    fig.update_yaxes(range=[-1, 1], dtick=0.5)
    for j, i in bas.items():
        fig.update_xaxes(title="q de Benjamini-Hochberg", row=i, col=j)
    for i in (1, 2):
        fig.update_yaxes(title="r de Pearson", row=i, col=1)
    fig.add_annotation(x=0.05, y=0.93, xref="x", yref="y", text="q = 0,05",
                       showarrow=False, xanchor="left", xshift=4,
                       font={"size": 10, "color": C_SEUIL})
    n_par_issue = int(q.groupby("issue").size().median()) if len(q) else 0
    q_min = float(q["pearson_q_categorie"].min()) if len(q) else float("nan")
    n_q05 = int((q["pearson_q_categorie"] < 0.05).sum())
    mise_en_page(fig, "Ce que le taux de fausses découvertes laisse — q de "
                      f"Benjamini-Hochberg dans la catégorie {cat['nom']} à issue "
                      "fixée, contre la corrélation obtenue. Trait rouge plein : "
                      "q = 0,05 ; pointillés : |r| significatif à p = 0,05 pour "
                      f"n = {_n_suj}. q minimum {q_min:.3f}, "
                      f"{n_q05} tentative(s) sous 0,05", hauteur=780)
    enregistrer(fig, f"p4{cat['cle']}_q_vs_r")
    return {"n_par_issue": n_par_issue, "q_min": q_min, "n_q05": n_q05,
            "n_points": int(len(q)), "q_med": float(q["pearson_q_categorie"].median())
            if len(q) else float("nan")}


def nuage_icc(cat, merge):
    """r contre l'ICC des repliques d'acquisition. Meme grille que les cartes
    quand les marqueurs dependent de sigma et de la reference ; un seul panneau
    sinon."""
    m = merge.dropna(subset=["icc", "pearson_r"]).copy()
    m["etiquette"] = _etiquettes(m, cat["marqueurs"], False)
    if cat["grille"]:
        panneaux = [(sg, cs) for sg in SIGMAS for cs in CAS]
        n_lignes, n_cols = len(SIGMAS), len(CAS)
        titres = [f"σ = {sg:g} · référence {COURT_CAS[cs]}" for sg, cs in panneaux]
        hauteur = 1150
    else:
        panneaux = [(None, None)]
        n_lignes, n_cols, titres, hauteur = 1, 1, [""], 620
    fig = make_subplots(rows=n_lignes, cols=n_cols, shared_xaxes=True,
                        shared_yaxes=True, horizontal_spacing=0.05,
                        vertical_spacing=0.045, subplot_titles=titres)
    for k, (sigma, cas) in enumerate(panneaux):
        i, j = divmod(k, n_cols)
        i, j = i + 1, j + 1
        sub = m
        if cat["grille"]:
            sub = m[(m["cas"] == cas)
                    & np.isclose(m["sigma"].astype(float), float(sigma))]
        # La couleur porte la bande, mais le decoupage des traces porte le
        # MARQUEUR : c'est lui dont le libelle est long, et le sortir du survol
        # de chaque point divise le poids du fragment par deux. Les entrees de
        # legende sont donc des traces vides, une par bande.
        for region in REGIONS + ["toutes"]:
            g = sub[sub["region"] == region]
            if g.empty:
                continue
            fig.add_trace(go.Scatter(
                x=g["icc"].round(3), y=g["pearson_r"].round(3), mode="markers",
                name=region, legendgroup=region, showlegend=bool(k == 0),
                marker={"color": COULEUR_REGION.get(region, C_REF), "size": 6,
                        "opacity": 0.75},
                hovertext=g["etiquette"] + " · " + g["issue"].map(ISSUE_LABEL),
                hovertemplate="%{hovertext}<br>ICC %{x:+.2f}<br>r %{y:+.2f}"
                              "<extra></extra>"), row=i, col=j)
        for signe in (1, -1):
            fig.add_hline(y=signe * _r_crit, row=i, col=j,
                          line={"color": C_SEUIL, "dash": "dash", "width": 1})
        fig.add_hline(y=0.0, row=i, col=j, line={"color": C_REF, "dash": "dot", "width": 1})
        fig.add_vline(x=0.75, row=i, col=j, line={"color": C_REF, "dash": "dot", "width": 1})
    fig.update_xaxes(range=[-0.85, 1.0])
    fig.update_yaxes(range=[-1.0, 1.0])
    for j in range(1, n_cols + 1):
        fig.update_xaxes(title="ICC(A,1) du marqueur", row=n_lignes, col=j)
    for i in range(1, n_lignes + 1):
        fig.update_yaxes(title="r de Pearson", row=i, col=1)
    r_icc = (float(np.corrcoef(m["pearson_r"].abs(), m["icc"])[0, 1])
             if len(m) > 2 else float("nan"))
    mise_en_page(fig, "La répétabilité achète-t-elle de la corrélation ? "
                      f"{len(m)} couples (marqueur, issue) de la catégorie "
                      f"{cat['nom']} — r(|r|, ICC) = {r_icc:+.3f}. Traits rouges : "
                      f"|r| significatif à p = 0,05 pour n = {_n_suj}",
                 hauteur=hauteur)
    enregistrer(fig, f"p4{cat['cle']}_r_vs_icc")
    return {"r_icc": r_icc, "n": int(len(m)),
            "icc_max": float(m["icc"].max()) if len(m) else float("nan"),
            "icc_med": float(m["icc"].median()) if len(m) else float("nan"),
            "frac_icc_075": float((m["icc"] > 0.75).mean()) if len(m) else float("nan"),
            "frac_absr_crit": float((m["pearson_r"].abs() > _r_crit).mean()) if len(m) else float("nan"),
            "frac_les_deux": float(((m["icc"] > 0.5)
                                    & (m["pearson_r"].abs() > _r_crit)).mean()) if len(m) else float("nan")}


def nuage_icc_oc(cat):
    """La meme question avec l'ICC PAR ONE-CYCLE (cible = la video), un panneau
    par issue. Tous les marqueurs de la categorie n'en ont pas un : les pentes
    poolees, `pouls` et `rigidite` n'existent pas a l'echelle du battement."""
    noms = {(f, p) for f, p, _ in cat["marqueurs"]}
    est_marqueur = np.array([(f, p) in noms
                             for f, p in zip(corr["famille"], corr["predicteur"])])
    c = corr[(corr["vue"] == VUE).to_numpy() & est_marqueur]
    a1 = (repeat_oc[(repeat_oc["icc_type"] == "ICC(A,1)") & (repeat_oc["pair"] == "1v2")]
          [_CLES_REP + ["icc"]].rename(columns={"icc": "icc_oc"}))
    i1 = (repeat_oc[repeat_oc["pair"] == "toutes"][_CLES_REP + ["icc1_desequilibre"]]
          .rename(columns={"icc1_desequilibre": "icc1_oc"}))
    d = (c.merge(a1, on=_CLES_REP, how="inner").merge(i1, on=_CLES_REP, how="left")
         .dropna(subset=["icc_oc", "pearson_r"]).copy())
    if d.empty:
        return None
    d["icc_oc"] = d["icc_oc"].round(3)
    d["pearson_r"] = d["pearson_r"].round(3)
    d["icc1_oc"] = d["icc1_oc"].round(2)
    d["spearman_rho"] = d["spearman_rho"].round(2)
    d["etiquette"] = _etiquettes(d, cat["marqueurs"], False)
    d["sigma_lab"] = d["sigma"].map(lambda v: "—" if pd.isna(v) else f"{float(v):g}")
    cols = ["etiquette", "sigma_lab", "icc1_oc", "spearman_rho", "n_sujets"]
    survol = ("%{customdata[0]}<br>σ %{customdata[1]}"
              "<br>ICC(A,1) 1v2 %{x:+.2f}  ·  ICC(1) tous %{customdata[2]:+.2f}"
              "<br>r %{y:+.2f}  ·  ρ %{customdata[3]:+.2f}  ·  n %{customdata[4]:d}")
    fig = make_subplots(rows=2, cols=3, shared_yaxes=True,
                        horizontal_spacing=0.045, vertical_spacing=0.13,
                        subplot_titles=[ISSUE_LABEL[i] for i in ISSUES]
                                       + [""] * (6 - len(ISSUES)))
    bas = {}
    for k, issue in enumerate(ISSUES):
        i, j = divmod(k, 3)
        i, j = i + 1, j + 1
        bas[j] = max(bas.get(j, 0), i)
        sub = d[d["issue"] == issue]
        for famille in [f for f in FAMILLE_ORDRE_P4 if f in set(sub["famille"])]:
            g = sub[sub["famille"] == famille]
            fig.add_trace(go.Scatter(
                x=g["icc_oc"], y=g["pearson_r"], mode="markers", name=famille,
                legendgroup=famille, showlegend=bool(k == 0),
                marker={"color": COULEUR_FAMILLE.get(famille, C_REF), "size": 5,
                        "opacity": 0.65, "line": {"width": 0}},
                customdata=g[cols].to_numpy(),
                hovertemplate=survol + "<extra></extra>"), row=i, col=j)
        for signe in (1, -1):
            fig.add_hline(y=signe * _r_crit, row=i, col=j,
                          line={"color": C_SEUIL, "dash": "dash", "width": 1})
        fig.add_hline(y=0.0, row=i, col=j, line={"color": C_REF, "dash": "dot", "width": 1})
        fig.add_vline(x=0.5, row=i, col=j, line={"color": C_REF, "dash": "dot", "width": 1})
    fig.update_xaxes(range=[-1.05, 1.05], dtick=0.5)
    fig.update_yaxes(range=[-1.05, 1.05], dtick=0.5)
    for j, i in bas.items():
        fig.update_xaxes(title="ICC(A,1) entre one-cycles", row=i, col=j)
    for i in (1, 2):
        fig.update_yaxes(title="r de Pearson", row=i, col=1)
    r_oc = float(np.corrcoef(d["pearson_r"].abs(), d["icc_oc"])[0, 1]) if len(d) > 2 else float("nan")
    mise_en_page(fig, "La répétabilité d'un battement à l'autre achète-t-elle de "
                      f"la corrélation ? {int(d.groupby('issue').size().median())} "
                      f"marqueurs de la catégorie {cat['nom']} par issue — "
                      f"r(|r|, ICC) = {r_oc:+.3f}. Pointillés rouges : |r| "
                      f"significatif à p = 0,05 pour n = {_n_suj} ; pointillé "
                      "vertical : ICC = 0,5", hauteur=800)
    enregistrer(fig, f"p4{cat['cle']}_r_vs_icc_oc")
    quad = (d["icc_oc"] > 0.5) & (d["pearson_r"].abs() > _r_crit)
    return {"r_oc": r_oc, "n_points": int(len(d)),
            "n_par_issue": int(d.groupby("issue").size().median()),
            "frac_icc_05": float((d["icc_oc"] > 0.5).mean()),
            "frac_absr_crit": float((d["pearson_r"].abs() > _r_crit).mean()),
            "n_quadrant": int(quad.sum()), "frac_quadrant": float(quad.mean()),
            "familles": ", ".join(sorted(set(d["famille"])))}


def table_prediction(cat, merge, cas=None, sigma=None) -> str:
    """Une table par (reference, sigma) : chaque cellule porte
    ``r de Pearson / rho de Spearman / ICC(A,1) / CV intra``.

    Les deux dernieres valeurs qualifient le MARQUEUR, pas le couple
    (marqueur, issue) : elles sont donc identiques dans les cinq colonnes d'une
    meme ligne. C'est la lecture voulue -- elles disent avec quelle confiance il
    faut prendre les correlations qui les precedent.
    """
    marqueurs = cat["tables"]
    _noms = {(a, b) for a, b, _ in marqueurs}
    sel = merge[np.array([(f, p) in _noms
                          for f, p in zip(merge["famille"], merge["predicteur"])])]
    if cas is not None:
        sel = sel[(sel["cas"] == cas)
                  & np.isclose(sel["sigma"].astype(float), float(sigma))]
    lignes = ["| marker | band | " + " | ".join(ISSUE_LABEL[i] for i in ISSUES) + " |",
              "|:--|:--|" + "--:|" * len(ISSUES)]
    n_vus = n_verts = n_lignes = 0
    for famille, predicteur, label in marqueurs:
        for region in REGIONS + ["toutes"]:
            g0 = sel[(sel["famille"] == famille) & (sel["predicteur"] == predicteur)
                     & (sel["region"] == region)]
            if g0.empty:
                continue
            # `cas` vaut NaN pour les marqueurs qui n'ont pas de style de
            # reference (k de Sayah, metriques CT / pouls) : NaN == NaN etant
            # faux, comparer sans normaliser faisait disparaitre ces lignes.
            cas_norm = g0["cas"].fillna("")
            for cas_l in (sorted(set(cas_norm)) if cas is None else [cas]):
                g = g0[cas_norm == cas_l]
                if g.empty:
                    continue
                n_lignes += 1
                icc_v = g["icc"].iloc[0]
                cv_v = g["cv_intra_pct"].iloc[0]
                r_max = float(g["pearson_r"].abs().max())
                vert = bool(pd.notna(icc_v) and icc_v >= VERT_ICC_MIN
                            and pd.notna(r_max) and r_max >= VERT_R_MIN)
                suffixe = (f" / {fmt_num(icc_v, 2, signe=True)}"
                           if pd.notna(icc_v) else " / —")
                suffixe += f" / {cv_v:.0f} %" if pd.notna(cv_v) else " / —"
                cells = []
                for issue in ISSUES:
                    h = g[g["issue"] == issue]
                    if h.empty:
                        cells.append("—")
                        continue
                    r = h["pearson_r"].iloc[0]
                    rho = h["spearman_rho"].iloc[0]
                    q = h["pearson_q_categorie"].iloc[0]
                    txt = f"{fmt_num(r, 2, signe=True)} / {fmt_num(rho, 2, signe=True)}"
                    cells.append((f"**{txt}**" if pd.notna(q) and q < 0.05 else txt)
                                 + suffixe)
                    n_vus += 1
                etq_m = cellule(label if cas is not None or not cas_l
                                else f"{label} (réf. {COURT_CAS.get(cas_l, cas_l)})")
                etq_r = cellule(region)
                if vert:
                    etq_m = "[" + etq_m + "]{.marqueur-retenu}"
                    etq_r = "[" + etq_r + "]{.marqueur-retenu}"
                    n_verts += 1
                lignes.append(f"| {etq_m} | {etq_r} | " + " | ".join(cells) + " |")
    if n_vus == 0:
        return "_Aucune corrélation calculable pour cette combinaison._"
    quoi = (f"**Référence {COURT_CAS[cas]}, σ = {sigma:g}**" if cas is not None
            else f"**Catégorie {cat['nom']}**")
    entete = (f"{quoi} — dans chaque cellule : "
              f"Pearson $r$ / Spearman ${BS_RHO}$ / **ICC(A,1)** / **CV intra**, "
              f"sur les {int(sel['n_sujets'].median())} sujets appariés. "
              f"L'ICC et le CV qualifient le marqueur seul : ils ne dépendent pas "
              f"de l'issue et se répètent donc à l'identique sur la ligne. "
              f"En gras : $q < 0{{,}}05$ après Benjamini-Hochberg dans la "
              f"catégorie. [En vert]{{.marqueur-retenu}} : les marqueurs dont "
              f"l'ICC atteint {VERT_ICC_MIN:.2f} **et** qui touchent "
              f"$|r| ≥ {VERT_R_MIN:.1f}$ contre au moins une issue — {n_verts} sur "
              f"{n_lignes} ici."
              + chr(10) + chr(10) + "::: {.table-scroll}" + chr(10) + chr(10))
    return entete + chr(10).join(lignes) + chr(10) + chr(10) + ":::"


FAMILLE_ORDRE_P4 = ["rigidite", "ct_pouls", "strain", "pente", "pente_pool"]
_n_suj = int(corr[corr["vue"] == VUE]["n_sujets"].median())
_r_crit = float(stats.t.ppf(0.975, _n_suj - 2))
_r_crit = _r_crit / np.sqrt(_r_crit ** 2 + _n_suj - 2)
RESUME["p4_r_critique_p05"] = _r_crit
RESUME["p4_n_sujets"] = _n_suj

for _cat in CATEGORIES_P4:
    _k = _cat["cle"]
    _merge = _corr_avec_repetabilite(_cat["marqueurs"])
    _n_cases = _n_vertes = 0
    for _sfx, _titre, _marq in _cat["cartes"]:
        _a, _b = carte_prediction(_cat, _sfx, _titre, _marq, _merge)
        _n_cases += _a
        _n_vertes += _b
    _q = nuage_q(_cat, _merge)
    _icc = nuage_icc(_cat, _merge)
    _oc = nuage_icc_oc(_cat)
    if _cat["grille"]:
        for _cas in CAS:
            for _sigma in SIGMAS:
                ecrire_table(table_prediction(_cat, _merge, _cas, _sigma),
                             f"p4{_k}_tab_{_cas}_s{_sigma:g}")
    else:
        ecrire_table(table_prediction(_cat, _merge), f"p4{_k}_tab")

    _tous = corr[(corr["categorie"] == _cat["categorie"]) & (corr["vue"] == VUE)]
    _lignes = _merge.groupby(["cas", "sigma", "predicteur", "region"], dropna=False).agg(
        icc=("icc", "first"), r_max=("pearson_r", lambda s: s.abs().max()))
    RESUME.update({
        f"p4{_k}_n_cellules": _n_cases,
        f"p4{_k}_n_cellules_vertes": _n_vertes,
        f"p4{_k}_frac_cellules_vertes": (_n_vertes / _n_cases) if _n_cases else float("nan"),
        f"p4{_k}_n_lignes": int(len(_lignes)),
        f"p4{_k}_n_lignes_vertes": int(((_lignes["icc"] >= VERT_ICC_MIN)
                                        & (_lignes["r_max"] >= VERT_R_MIN)).sum()),
        f"p4{_k}_n_marqueurs": int(len(_cat["marqueurs"])),
        f"p4{_k}_n_tests_categorie": int(len(_tous)),
        f"p4{_k}_n_tests_par_issue": int(_tous.groupby("issue").size().median()) if len(_tous) else 0,
        f"p4{_k}_n_affiches": int(len(_merge)),
        f"p4{_k}_n_p05": int((_merge["pearson_p"] < 0.05).sum()),
        f"p4{_k}_attendu_p05": float(0.05 * len(_merge)),
        f"p4{_k}_n_q05_categorie": int((_tous["pearson_q_categorie"] < 0.05).sum()),
        f"p4{_k}_n_q05_famille": int((_merge["pearson_q_famille"] < 0.05).sum()),
        f"p4{_k}_n_q05_global": int((_merge["pearson_q"] < 0.05).sum()),
        f"p4{_k}_r_abs_max": float(_merge["pearson_r"].abs().max()),
        f"p4{_k}_q_min": _q["q_min"],
        f"p4{_k}_q_med": _q["q_med"],
        f"p4{_k}_q_n_sous_005": _q["n_q05"],
        f"p4{_k}_q_n_par_issue": _q["n_par_issue"],
        f"p4{_k}_icc_r": _icc["r_icc"],
        f"p4{_k}_icc_n": _icc["n"],
        f"p4{_k}_icc_max": _icc["icc_max"],
        f"p4{_k}_icc_med": _icc["icc_med"],
        f"p4{_k}_icc_frac075": _icc["frac_icc_075"],
        f"p4{_k}_frac_absr_crit": _icc["frac_absr_crit"],
        f"p4{_k}_frac_les_deux": _icc["frac_les_deux"],
        f"p4{_k}_q_min_par_issue": ", ".join(
            f"{ISSUE_LABEL[i]} {v:.3f}" for i, v in
            _tous.groupby("issue")["pearson_q_categorie"].min().reindex(ISSUES).items()),
    })
    if _oc:
        RESUME.update({
            f"p4{_k}_oc_r": _oc["r_oc"],
            f"p4{_k}_oc_n_points": _oc["n_points"],
            f"p4{_k}_oc_n_par_issue": _oc["n_par_issue"],
            f"p4{_k}_oc_frac_icc05": _oc["frac_icc_05"],
            f"p4{_k}_oc_frac_absr_crit": _oc["frac_absr_crit"],
            f"p4{_k}_oc_n_quadrant": _oc["n_quadrant"],
            f"p4{_k}_oc_frac_quadrant": _oc["frac_quadrant"],
            f"p4{_k}_oc_familles": _oc["familles"],
        })
    # L'issue « aplatissement du globe » a sa propre section sur chaque page, et
    # elle a besoin de ses propres chiffres : sur les cinq delta_TRT le decompte
    # est domine par le quadrant nasal, et une moyenne sur les six issues
    # masquerait exactement ce que la section veut montrer.
    _dal = _tous[_tous["issue"] == ISSUE_AL]
    if len(_dal):
        _fort = _dal.reindex(_dal["pearson_r"].abs().sort_values(ascending=False).index)
        RESUME.update({
            f"p4{_k}_dAL_n_tests": int(len(_dal)),
            f"p4{_k}_dAL_n_sujets": int(_dal["n_sujets"].median()),
            f"p4{_k}_dAL_r_abs_max": float(_dal["pearson_r"].abs().max()),
            f"p4{_k}_dAL_n_p05": int((_dal["pearson_p"] < 0.05).sum()),
            f"p4{_k}_dAL_attendu_p05": float(0.05 * len(_dal)),
            f"p4{_k}_dAL_q_min": float(_dal["pearson_q_categorie"].min()),
            f"p4{_k}_dAL_n_q05": int((_dal["pearson_q_categorie"] < 0.05).sum()),
            f"p4{_k}_dAL_top": " ; ".join(
                f"{r['predicteur']}"
                + (f" ({r['region']})" if r["region"] != "toutes" else "")
                + (f" σ={r['sigma']:g}" if pd.notna(r["sigma"]) else "")
                + f" r={r['pearson_r']:+.2f} p={r['pearson_p']:.3f}"
                  f" q={r['pearson_q_categorie']:.3f} n={int(r['n_sujets'])}"
                for _, r in _fort.head(4).iterrows()),
        })

    # Les marqueurs qui passent la q-valeur de leur categorie : c'est ce que
    # chaque page cite nommement.
    _surv = _tous[_tous["pearson_q_categorie"] < 0.05]
    RESUME[f"p4{_k}_survivants"] = " ; ".join(
        f"{r['predicteur']}"
        + (f" ({r['region']})" if r["region"] != "toutes" else "")
        + f" vs {ISSUE_LABEL[r['issue']]} r={r['pearson_r']:+.2f} q={r['pearson_q_categorie']:.3f}"
        for _, r in _surv.sort_values("pearson_q_categorie").head(8).iterrows()) or "aucun"

RESUME.update({
    "p4_n_correlations_total": int(len(corr)),
    "p4_n_q05_total": int((corr["pearson_q"] < 0.05).sum()),
    "p4_n_p05_total": int((corr["pearson_p"] < 0.05).sum()),
    "p4_n_tests_sans_categorie": int((corr["categorie"] == "").sum()),
})


# =========================================================================== #
# Les DEUX marqueurs de SANS, l'un contre l'autre
# =========================================================================== #
def nuage_issues():
    """L'aplatissement du globe contre le gonflement retinien, par sujet.

    Aucun predicteur OCT ici : ce sont les deux ISSUES cliniques, tracees l'une
    contre l'autre. La figure repond a la question prealable a tout le reste --
    ajouter delta_AL apporte-t-il quelque chose, ou est-ce le meme signal sous un
    autre nom ? Un nuage aligne dirait qu'une seule issue suffisait ; un nuage
    disperse justifie de tester les deux.

    Un panneau par delta_TRT, plus la moyenne. La droite des moindres carres
    n'est tracee que la ou p < 0,05 : sur une quinzaine de sujets, une droite
    dessinee sur un nuage rond se lit comme un resultat qu'elle n'est pas.
    """
    cibles = [c for c in ISSUES if c != ISSUE_AL]
    fig = make_subplots(rows=2, cols=3, shared_yaxes=True,
                        horizontal_spacing=0.045, vertical_spacing=0.14,
                        subplot_titles=[ISSUE_LABEL[c] for c in cibles]
                                       + [""] * (6 - len(cibles)))
    stats_par_issue = {}
    for k, issue in enumerate(cibles):
        i, j = divmod(k, 3)
        i, j = i + 1, j + 1
        sub = outcomes[[issue, ISSUE_AL, "subject"]].dropna()
        n = len(sub)
        if n < 4:
            continue
        x = sub[issue].to_numpy(dtype=float)
        y = sub[ISSUE_AL].to_numpy(dtype=float)
        r, pval = stats.pearsonr(x, y)
        stats_par_issue[issue] = (float(r), float(pval), n)
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers", showlegend=False,
            marker={"color": COULEUR_FAMILLE["rigidite"], "size": 8,
                    "opacity": 0.8, "line": {"width": 0}},
            customdata=sub["subject"].to_numpy(),
            hovertemplate="%{customdata}<br>" + ISSUE_LABEL[issue]
                          + " = %{x:+.1f} µm<br>ΔAL = %{y:+.3f} mm"
                            "<extra></extra>"), row=i, col=j)
        if pval < 0.05:
            xs = np.array([x.min(), x.max()])
            a, b = np.polyfit(x, y, 1)
            fig.add_trace(go.Scatter(x=xs, y=a * xs + b, mode="lines",
                                     showlegend=False, hoverinfo="skip",
                                     line={"color": C_SEUIL, "width": 1.4}),
                          row=i, col=j)
        # row/col plutot qu'un identifiant d'axe construit a la main : plotly
        # resout lui-meme « x domain » vers l'axe du sous-graphe, et « x1 » n'est
        # pas un alias fiable pour le premier panneau.
        # Virgule decimale : tout le reste du site en utilise une, et plotly
        # n'a pas de reglage de locale pour le texte d'une annotation.
        _txt = (f"r = {r:+.2f}  ·  p = {pval:.3f}  ·  n = {n}").replace(".", ",")
        fig.add_annotation(text=_txt,
                           xref="x domain", yref="y domain", row=i, col=j,
                           x=0.02, y=0.06, showarrow=False, xanchor="left",
                           font={"size": 10, "color": COULEUR_TEXTE})
        fig.add_hline(y=0.0, row=i, col=j,
                      line={"color": C_REF, "dash": "dot", "width": 1})
        fig.add_vline(x=0.0, row=i, col=j,
                      line={"color": C_REF, "dash": "dot", "width": 1})
        fig.update_xaxes(title="µm", row=i, col=j)
    for i in (1, 2):
        fig.update_yaxes(title="ΔAL (mm)", row=i, col=1)
    # Cinq issues pour six cases : le sixieme panneau resterait une boite d'axes
    # vide avec ses graduations. `make_subplots` l'a deja cree, on le masque.
    for reste in range(len(cibles), 6):
        i, j = divmod(reste, 3)
        fig.update_xaxes(visible=False, row=i + 1, col=j + 1)
        fig.update_yaxes(visible=False, row=i + 1, col=j + 1)
    n_sujets = int(outcomes[[c for c in cibles] + [ISSUE_AL]].dropna().shape[0])
    mise_en_page(fig, "Les deux marqueurs de SANS, l'un contre l'autre — "
                      "aplatissement du globe (ΔAL, longueur axiale après − "
                      "avant) contre gonflement rétinien (ΔTRT, après − avant). "
                      f"{n_sujets} sujets ont les deux ; droite tracée seulement "
                      "où p < 0,05", hauteur=620, legende=False)
    enregistrer(fig, "p4_issues")
    return stats_par_issue


_st_issues = nuage_issues()
_dal_seul = outcomes[ISSUE_AL].dropna()
_trt_seul = outcomes["delta_TRT_moyen"].dropna()
RESUME.update({
    "issues_n_sujets_dAL": int(len(_dal_seul)),
    "issues_n_sujets_dTRT": int(len(_trt_seul)),
    "issues_n_sujets_les_deux": int(outcomes[[ISSUE_AL, "delta_TRT_moyen"]]
                                    .dropna().shape[0]),
    "issues_dAL_med": float(_dal_seul.median()) if len(_dal_seul) else float("nan"),
    "issues_dAL_min": float(_dal_seul.min()) if len(_dal_seul) else float("nan"),
    "issues_dAL_max": float(_dal_seul.max()) if len(_dal_seul) else float("nan"),
    "issues_dAL_frac_negatif": float((_dal_seul < 0).mean()) if len(_dal_seul) else float("nan"),
    "issues_dAL_vs_dTRT": " ; ".join(
        f"{ISSUE_LABEL[i]} r={r:+.2f} p={pv:.3f} n={n}"
        for i, (r, pv, n) in _st_issues.items()),
})


# =========================================================================== #
# PAGE 4 strain — boxplots par CLASSE DE GONFLEMENT RETINIEN
# =========================================================================== #
# Les cartes de chaleur de la page traitent le delta_TRT comme une variable
# CONTINUE et en tirent un r. Ici il est decoupe en CLASSES et l'on compare les
# moyennes du strain d'une classe a l'autre -- meme question, autre estimateur,
# et surtout des points visibles : avec douze yeux, un r de 0,5 peut n'etre
# qu'un sujet mal place, ce qu'un nuage montre et qu'un coefficient cache.
#
# TROIS CHOSES A SAVOIR AVANT DE LIRE CETTE SECTION.
#
# 1. LA TROISIEME CLASSE EST VIDE, ET C'EST UN RESULTAT. Les seuils demandes
#    sont 20 et 50 um. Or le delta_TRT maximal de toute la cohorte vaut 35,0 um
#    (quadrant S) et 26,5 um en moyenne des quadrants : PERSONNE n'atteint
#    50 um. La classe est tracee quand meme, vide, plutot que retiree -- la
#    montrer vide dit quelque chose sur la cohorte, la supprimer laisserait
#    croire qu'on a compare trois groupes.
# 2. LE SEXE EST DESEQUILIBRE ENTRE LES CLASSES, MAIS PAS AU POINT DE BLOQUER LA
#    CORRECTION. Sur la vue « avant le vol » et le delta_TRT nasal, la classe
#    < 20 um compte 6 hommes pour 2 femmes et la classe 20-50 um 1 homme pour
#    3 femmes -- de quoi craindre une colinearite qui ferait exploser
#    l'ecart-type du coefficient de classe. Mesure, ce n'est pas le cas :
#    |r(classe, sexe)| plafonne a 0,53 et le facteur d'inflation de la variance
#    reste entre 1,0 et 1,4. La colonne ajustee teste donc bien quelque chose.
#    Le VIF est publie a cote de chaque test pour que ce soit verifiable et non
#    a croire -- et pour qu'il alerte si la cohorte change.
# 3. LES DEUX YEUX D'UN MEME ASTRONAUTE NE SONT PAS INDEPENDANTS. Les tests les
#    comptent quand meme comme deux observations (c'est la convention de tout le
#    panel, `subject = astro + oeil`). Une version par ASTRONAUTE -- moyenne des
#    deux yeux -- est calculee en parallele et son decompte publie : si un
#    resultat ne survit pas au passage a l'astronaute, il ne tenait qu'a la
#    duplication des yeux.
# Les seuils sont en MICROMETRES : ils ne s'appliquent qu'aux cinq delta_TRT.
# Le delta_AL est en millimetres (etendue -0,21 a 0,00) et tomberait en entier
# dans la classe « < 20 », ce qui ne comparerait rien. Il est donc hors de cette
# section -- et pas « oublie » : c'est une question d'unite, pas de resultat.
BOX_ISSUES = [i for i in ISSUES if i.startswith("delta_TRT")]
BOX_SIGMA = 10.0          # une seule regularisation : 30 boxplots restent lisibles
BOX_VUE = "before"        # la vue « peut-on predire », celle de toute la page
BOX_BORNES = [("< 20 µm", -np.inf, 20.0),
              ("20 – 50 µm", 20.0, 50.0),
              ("≥ 50 µm", 50.0, np.inf)]
COULEUR_CLASSE = {"< 20 µm": "#9a9a9a", "20 – 50 µm": "#d0a72b", "≥ 50 µm": "#c2453f"}
COULEUR_SEXE = {"homme": "#2a78d6", "femme": "#e2649f"}


def _strain_par_sujet():
    """Valeur de strain par (sujet, moment, reference, bande) a sigma = 10.

    Refait, a partir de `regions.csv` deja filtre par le portail, exactement la
    chaine d'agregation du lot (`compute_sans_predictors`) :

        mediane sur les (one-cycle, bin) d'une condition   -> valeur de condition
        moyenne sur les conditions repliquees d'un sujet   -> valeur de sujet

    Passer par `regions.csv` plutot que par `subject_values.csv` n'est pas un
    detour : cette derniere table pese 245 Mo parce que son pivot produit le
    PRODUIT CARTESIEN des cles (des lignes `famille=ct_pouls` /
    `predicteur=CT_mm` / `sigma=5` qui n'existent pas). Ces lignes sont
    integralement NaN et n'ont jamais fausse un resultat -- elles sont ecartees
    par le `dropna` de chaque test -- mais les relire ici couterait des
    gigaoctets pour rien.
    """
    cols = [c for c, _, _, _ in AGREGATS]
    reg = regions[np.isclose(regions["sigma"].astype(float), BOX_SIGMA)]
    par_condition = (reg.groupby(["slug", "cas", "region"], sort=False)[cols]
                     .median().reset_index())
    ident = (one_cycles[["slug", "astro", "moment", "condition"]]
             .drop_duplicates("slug"))
    d = par_condition.merge(ident, on="slug", how="left")
    d["eye"] = d["condition"].str.extract(r"(OD|OS)" + chr(92) + "d*$")[0]
    d["subject"] = d["astro"] + "_" + d["eye"]
    # La colonne `moment` de `one_cycles.csv` porte le NOM DU DOSSIER
    # (`210830001before_rigidity`), pas `before` / `after` : le lot la traduit
    # avec `parse_moment` au moment de construire son identite. La meme
    # traduction ici, faute de quoi le filtre sur la vue ne trouve rien et la
    # section sort vide sans erreur.
    bas = d["moment"].str.lower()
    d["vue"] = np.where(bas.str.contains("before"), "before",
                        np.where(bas.str.contains("post|after", regex=True), "after",
                                 np.where(bas.str.contains("during"), "during", "")))
    return (d.groupby(["subject", "vue", "cas", "region"], sort=False)[cols]
            .mean().reset_index())


def _table_boxplots():
    """Une ligne par (sujet, reference, bande, agregat) sur la vue retenue,
    jointe aux six issues et au sexe."""
    d = _strain_par_sujet()
    d = d[d["vue"] == BOX_VUE]
    long = d.melt(id_vars=["subject", "cas", "region"],
                  value_vars=[c for c, _, _, _ in AGREGATS],
                  var_name="predicteur", value_name="valeur").dropna(subset=["valeur"])
    long["agregat"] = long["predicteur"].map(
        {c: lab for c, lab, _, _ in AGREGATS})
    garde = ["subject", "sexe", "astro"] + ISSUES
    return long.merge(outcomes[garde], on="subject", how="inner")


def _classe(serie):
    """Etiquette de classe d'une serie d'issues, NaN hors bornes."""
    out = pd.Series(index=serie.index, dtype=object)
    for lab, lo, hi in BOX_BORNES:
        out[(serie >= lo) & (serie < hi)] = lab
    return out


BOX = _table_boxplots()
if "sexe" not in BOX.columns or BOX["sexe"].eq("").all():
    raise SystemExit("outcomes.csv sans colonne `sexe` -- relancer "
                     "Astronauts/compute_sans_predictors.py")


SEUILS_ETOILES = ((0.001, "***"), (0.01, "**"), (0.05, "*"))


def etoiles(p):
    """Convention usuelle : *** p < 0,001, ** p < 0,01, * p < 0,05, rien sinon.

    Les etoiles portent la p BRUTE du t-test de la paire, PAS la q de
    Benjamini-Hochberg -- c'est ce qu'une etoile veut dire partout ailleurs, et
    l'inverse serait un piege pour le lecteur. Comme aucune des 150 comparaisons
    ne survit a la correction, la legende de chaque figure le dit en toutes
    lettres : une etoile ici ne signale pas un resultat, seulement une paire qui
    passerait le seuil nominal si elle avait ete la seule testee.
    """
    if p is None or not np.isfinite(p):
        return ""
    for seuil, glyphe in SEUILS_ETOILES:
        if p < seuil:
            return glyphe
    return ""


def boxplots_classes(issue, tests):
    """Six panneaux (3 agregats x 2 references), 5 bandes x 3 classes chacun.

    Les points sont portes par des boites INVISIBLES qui partagent
    l'`offsetgroup` de la vraie boite : c'est le seul moyen d'aligner
    exactement des points colores par sexe sur une boite groupee, plotly ne
    sachant pas colorer point par point a l'interieur d'une trace `box`.

    Une etoile au-dessus d'une bande resume le t-test de Welch entre les DEUX
    classes peuplees de cette bande. Elle est posee au centre de la bande et non
    en pont entre les deux boites : les abscisses exactes des boites groupees
    dependent de `boxgap` / `boxgroupgap` et d'un decompte d'`offsetgroup` que
    plotly calcule lui-meme, les recalculer ici casserait au premier changement
    de mise en page. Le centre de la bande, lui, est l'indice de categorie -- une
    coordonnee stable.
    """
    d = BOX[BOX[issue].notna()].copy()
    d["classe"] = _classe(d[issue])
    d = d[d["classe"].notna()]
    titres = [f"{lab} · référence {COURT_CAS[cs]}"
              for _, lab, _, _ in AGREGATS for cs in CAS]
    # Axes y INDEPENDANTS, y compris sur une meme ligne. Partager l'axe entre les
    # deux references parait naturel -- c'est le meme strain mesure deux fois --
    # mais la reference globale est environ dix fois plus etalee que la locale :
    # sur un axe commun, le panneau « locale » se reduit a un trait et l'on ne
    # voit plus rien de ce que la figure est censee montrer. La comparaison qui
    # compte est entre CLASSES a l'interieur d'un panneau, pas entre panneaux.
    fig = make_subplots(rows=len(AGREGATS), cols=len(CAS), shared_xaxes=True,
                        shared_yaxes=False, horizontal_spacing=0.09,
                        vertical_spacing=0.07, subplot_titles=titres)
    vu_classe, vu_sexe = set(), set()
    for i, (_, lab_ag, _, _) in enumerate(AGREGATS, start=1):
        for j, cs in enumerate(CAS, start=1):
            sub = d[(d["agregat"] == lab_ag) & (d["cas"] == cs)]
            for lab_cl, _, _ in BOX_BORNES:
                g = sub[sub["classe"] == lab_cl]
                fig.add_trace(go.Box(
                    x=g["region"], y=g["valeur"], name=lab_cl, legendgroup=lab_cl,
                    offsetgroup=lab_cl, alignmentgroup="bandes",
                    showlegend=lab_cl not in vu_classe, boxpoints=False,
                    marker={"color": COULEUR_CLASSE[lab_cl]},
                    line={"width": 1.3}, fillcolor="rgba(0,0,0,0)",
                    hovertemplate="%{x}<br>" + lab_cl
                                  + "<br>médiane %{median:.4f}<extra></extra>"),
                    row=i, col=j)
                vu_classe.add(lab_cl)
                for sexe, coul in COULEUR_SEXE.items():
                    h = g[g["sexe"] == sexe]
                    fig.add_trace(go.Box(
                        x=h["region"], y=h["valeur"], name=sexe, legendgroup=sexe,
                        offsetgroup=lab_cl, alignmentgroup="bandes",
                        showlegend=sexe not in vu_sexe,
                        boxpoints="all", pointpos=0, jitter=0.7,
                        fillcolor="rgba(0,0,0,0)", line={"color": "rgba(0,0,0,0)"},
                        marker={"color": coul, "size": 6, "opacity": 0.95,
                                "line": {"width": 0}},
                        hoveron="points",
                        customdata=h[["subject", "classe"]].to_numpy(),
                        hovertemplate="%{customdata[0]}<br>%{customdata[1]}"
                                      "<br>" + sexe + " · %{y:.4f}<extra></extra>"),
                        row=i, col=j)
                    vu_sexe.add(sexe)
    # --- etoiles, un passage par panneau ------------------------------------
    t_iss = tests[tests["issue"] == issue] if len(tests) else tests
    n_etoiles = 0
    for i, (_, lab_ag, _, _) in enumerate(AGREGATS, start=1):
        for j, cs in enumerate(CAS, start=1):
            sub = d[(d["agregat"] == lab_ag) & (d["cas"] == cs)]
            if sub.empty:
                continue
            lo, hi = float(sub["valeur"].min()), float(sub["valeur"].max())
            span = (hi - lo) or 1.0
            for bande in REGIONS:
                sb = sub[sub["region"] == bande]
                ligne = t_iss[(t_iss["agregat"] == lab_ag) & (t_iss["cas"] == cs)
                              & (t_iss["region"] == bande)]
                if sb.empty or ligne.empty:
                    continue
                glyphe = etoiles(float(ligne["p_welch"].iloc[0]))
                if not glyphe:
                    continue
                n_etoiles += 1
                fig.add_annotation(
                    x=bande, y=float(sb["valeur"].max()) + 0.07 * span,
                    text=glyphe, showarrow=False, xanchor="center",
                    yanchor="bottom", row=i, col=j,
                    hovertext=f"Welch p = {float(ligne['p_welch'].iloc[0]):.4f}",
                    font={"size": 15, "color": C_SEUIL})
            # De la place pour l'etoile : sans marge haute, le glyphe sort du
            # cadre et plotly le rogne sans prevenir.
            fig.update_yaxes(range=[lo - 0.10 * span, hi + 0.010 * span + 0.22 * span],
                             row=i, col=j)
    fig.update_xaxes(categoryorder="array", categoryarray=REGIONS)
    for i in range(1, len(AGREGATS) + 1):
        for j in range(1, len(CAS) + 1):
            fig.update_yaxes(title="strain" if j == 1 else None, row=i, col=j)
    n_par_classe = d.drop_duplicates("subject")["classe"].value_counts()
    detail = " · ".join(f"{lab} : {int(n_par_classe.get(lab, 0))} yeux"
                        for lab, _, _ in BOX_BORNES)
    fig.update_layout(boxmode="group", boxgroupgap=0.25, boxgap=0.2)
    mise_en_page(fig, f"Strain rétinien avant le vol par classe de "
                      f"{ISSUE_LABEL[issue]} — σ = {BOX_SIGMA:g}, points "
                      f"<span style='color:{COULEUR_SEXE['homme']}'>bleus</span> "
                      f"= hommes, "
                      f"<span style='color:{COULEUR_SEXE['femme']}'>roses</span> "
                      f"= femmes. {detail}. Chaque panneau a sa propre "
                      f"échelle. ✳ = t de Welch entre « < 20 µm » et "
                      f"« 20 – 50 µm », p NON corrigée "
                      f"(*** &lt; 0,001, ** &lt; 0,01, * &lt; 0,05) — "
                      f"{n_etoiles} sur {len(REGIONS) * len(AGREGATS) * len(CAS)} "
                      f"paires, et aucune ne survit à Benjamini-Hochberg",
                 hauteur=1150)
    enregistrer(fig, f"p4str_box_{issue}")
    return ({lab: int(n_par_classe.get(lab, 0)) for lab, _, _ in BOX_BORNES},
            n_etoiles)


# --------------------------------------------------------------------------- #
# Comparaison des moyennes, avec et sans correction du sexe
# --------------------------------------------------------------------------- #
def _mco(y, X):
    """Moindres carres ordinaires : coefficients, t et p de CHAQUE colonne.

    Ecrit a la main plutot qu'avec statsmodels, absent de l'environnement (meme
    raison que `bh_fdr`). `pinv` plutot que `inv` : quand le sexe est presque
    colineaire a la classe -- ce qui est le cas ici -- la matrice normale est
    mal conditionnee et `inv` renverrait du bruit numerique sans prevenir.
    """
    n, k = X.shape
    ddl = n - k
    if ddl < 1:
        return None
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = float(resid @ resid) / ddl
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(s2 * np.diag(XtX_inv), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
    p = 2 * stats.t.sf(np.abs(t), ddl)
    return {"beta": beta, "se": se, "t": t, "p": p, "ddl": ddl}


def _welch(a, b):
    """t de Welch (variances inegales) et p bilateral."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan, np.nan
    va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    se2 = va / na + vb / nb
    if se2 <= 0:
        return np.nan, np.nan
    t = (float(np.mean(a)) - float(np.mean(b))) / np.sqrt(se2)
    ddl = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return float(t), float(2 * stats.t.sf(abs(t), ddl))


def _un_test(g):
    """Les deux tests d'une combinaison (issue, agregat, bande, reference).

    La classe entre comme un RANG (0, 1, 2 dans l'ordre des bornes) et non comme
    des indicatrices : avec deux classes peuplees c'est strictement le contraste
    des deux moyennes, et le jour ou la troisieme se remplira ce sera une
    tendance monotone -- ce qu'on veut tester d'un gonflement croissant.

    Les deux colonnes publiees viennent du MEME estimateur, a un terme pres :
    `y ~ 1 + classe` puis `y ~ 1 + classe + sexe`. Comparer un Welch a un modele
    lineaire melangerait deux differences a la fois. Le Welch est calcule quand
    meme, mais seulement pour verifier que l'hypothese de variance commune ne
    porte pas le resultat.
    """
    rang = {lab: i for i, (lab, _, _) in enumerate(BOX_BORNES)}
    g = g.dropna(subset=["valeur", "classe"])
    g = g[g["sexe"].isin(COULEUR_SEXE)]
    classes = [lab for lab, _, _ in BOX_BORNES if (g["classe"] == lab).any()]
    if len(classes) < 2:
        return None
    y = g["valeur"].to_numpy(dtype=float)
    cl = g["classe"].map(rang).to_numpy(dtype=float)
    sx = (g["sexe"] == "femme").to_numpy(dtype=float)
    n = len(y)
    if n < 4 or np.std(cl) == 0:
        return None

    brut = _mco(y, np.column_stack([np.ones(n), cl]))
    if brut is None:
        return None
    out = {"n": n, "n_classes": len(classes),
           "diff": float(brut["beta"][1]), "t_brut": float(brut["t"][1]),
           "p_brut": float(brut["p"][1])}
    for lab in [l for l, _, _ in BOX_BORNES]:
        v = g.loc[g["classe"] == lab, "valeur"]
        out[f"moy[{lab}]"] = float(v.mean()) if len(v) else np.nan
        out[f"n[{lab}]"] = int(len(v))

    a = g.loc[g["classe"] == classes[0], "valeur"].to_numpy(dtype=float)
    b = g.loc[g["classe"] == classes[1], "valeur"].to_numpy(dtype=float)
    out["t_welch"], out["p_welch"] = _welch(a, b)

    # --- avec le sexe -------------------------------------------------------
    if np.std(sx) == 0:
        out.update({"diff_aj": np.nan, "t_aj": np.nan, "p_aj": np.nan,
                    "vif": np.nan, "beta_sexe": np.nan, "p_sexe": np.nan})
        return out
    aj = _mco(y, np.column_stack([np.ones(n), cl, sx]))
    if aj is None:
        out.update({"diff_aj": np.nan, "t_aj": np.nan, "p_aj": np.nan,
                    "vif": np.nan, "beta_sexe": np.nan, "p_sexe": np.nan})
        return out
    # Facteur d'inflation de la variance du coefficient de classe : 1/(1-R2) de
    # la regression de la classe sur le sexe. C'est le prix exact de la
    # correction, et il se lit avant les p ajustees.
    r = float(np.corrcoef(cl, sx)[0, 1])
    vif = 1.0 / max(1.0 - r ** 2, 1e-12)
    out.update({"diff_aj": float(aj["beta"][1]), "t_aj": float(aj["t"][1]),
                "p_aj": float(aj["p"][1]), "vif": float(vif),
                "beta_sexe": float(aj["beta"][2]), "p_sexe": float(aj["p"][2]),
                "r_classe_sexe": r})
    return out


def tests_classes():
    """Les 150 comparaisons : 5 issues x 3 agregats x 5 bandes x 2 references.

    Deux familles de Benjamini-Hochberg, une par colonne de p : corriger les
    deux ensemble melangerait deux modeles differents sur les memes donnees.

    La version par ASTRONAUTE (moyenne des deux yeux) tourne sur exactement les
    memes combinaisons ; seul son decompte est publie, comme garde-fou.
    """
    lignes = []
    for issue in BOX_ISSUES:
        d = BOX[BOX[issue].notna()].copy()
        d["classe"] = _classe(d[issue])
        # Meme table, une ligne par ASTRONAUTE. La moyenne des deux yeux est
        # prise SUR L'ISSUE AUSSI, et la classe deduite de cette moyenne : classer
        # chaque oeil puis regrouper ferait apparaitre deux fois l'astronaute
        # dont les deux yeux tombent de part et d'autre d'un seuil.
        par_astro = (d.groupby(["astro", "agregat", "cas", "region", "sexe"],
                               sort=False)
                     .agg(valeur=("valeur", "mean"), issue_moy=(issue, "mean"))
                     .reset_index())
        par_astro["classe"] = _classe(par_astro["issue_moy"])
        for (ag, cs, reg_), g in d.groupby(["agregat", "cas", "region"], sort=False):
            res = _un_test(g)
            if res is None:
                continue
            ga = par_astro[(par_astro["agregat"] == ag) & (par_astro["cas"] == cs)
                           & (par_astro["region"] == reg_)]
            res_a = _un_test(ga)
            lignes.append({"issue": issue, "agregat": ag, "cas": cs,
                           "region": reg_, **res,
                           "n_astro": (res_a or {}).get("n", np.nan),
                           "p_brut_astro": (res_a or {}).get("p_brut", np.nan),
                           "p_aj_astro": (res_a or {}).get("p_aj", np.nan)})
    t = pd.DataFrame(lignes)
    if t.empty:
        return t
    # TROIS familles de Benjamini-Hochberg, une par colonne de p : corriger
    # ensemble des p issues de modeles differents sur les memes donnees
    # melangerait les hypotheses. `q_welch` sert a une affirmation precise de la
    # page -- « aucune etoile ne survit a la correction » --, qui ne peut pas
    # s'appuyer sur la q d'un autre test que celui qui porte les etoiles.
    for src, dst in (("p_brut", "q_brut"), ("p_aj", "q_aj"),
                     ("p_welch", "q_welch")):
        t[dst] = bh_fdr(t[src].to_numpy())
    return t


def table_tests_classes(t, issue):
    """La table d'une issue : moyennes par classe, puis les deux tests."""
    g = t[t["issue"] == issue]
    if g.empty:
        return "_Aucun test calculable pour cette issue._"
    labs = [lab for lab, _, _ in BOX_BORNES]
    lignes = ["| agrégat | bande | réf. | "
              + " | ".join(f"moyenne {l}" for l in labs)
              + " | pente / classe | $t$ Welch | $p$ Welch | $q$ Welch | "
                "| $p$ | $q$ | $p$ ajusté | $q$ ajusté | VIF |",
              "|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|:-:|--:|--:|--:|--:|--:|"]
    for _, r in g.iterrows():
        moys = " | ".join(
            (f"{r[f'moy[{l}]']:+.4f} ({int(r[f'n[{l}]'])})"
             if pd.notna(r[f"moy[{l}]"]) else "—") for l in labs)
        def gras(p, q):
            txt = fmt_num(p, 3) if pd.notna(p) else "—"
            return f"**{txt}**" if pd.notna(q) and q < 0.05 else txt
        lignes.append(
            f"| {r['agregat']} | {cellule(r['region'])} | {COURT_CAS[r['cas']]} "
            f"| {moys} | {fmt_sci(r['diff'], signe=True)} "
            f"| {fmt_num(r['t_welch'], 2, signe=True) if pd.notna(r['t_welch']) else '—'} "
            f"| {fmt_num(r['p_welch'], 4) if pd.notna(r['p_welch']) else '—'} "
            f"| {fmt_num(r['q_welch'], 3) if pd.notna(r['q_welch']) else '—'} "
            f"| {etoiles(r['p_welch'])} "
            f"| {gras(r['p_brut'], r['q_brut'])} "
            f"| {fmt_num(r['q_brut'], 3) if pd.notna(r['q_brut']) else '—'} "
            f"| {gras(r['p_aj'], r['q_aj'])} "
            f"| {fmt_num(r['q_aj'], 3) if pd.notna(r['q_aj']) else '—'} "
            f"| {fmt_num(r['vif'], 1) if pd.notna(r['vif']) else '—'} |")
    entete = (f"**{ISSUE_LABEL[issue]}** — moyenne du strain par classe (effectif "
              f"entre parenthèses), puis le **t de Welch** de la paire "
              f"« < 20 µm » contre « 20 – 50 µm » avec ses étoiles (p NON "
              f"corrigée : *** &lt; 0,001, ** &lt; 0,01, * &lt; 0,05), et enfin "
              f"la pente par classe et ses deux tests : "
              f"$p$ sans correction (`y ~ 1 + classe`) et $p$ ajusté "
              f"(`y ~ 1 + classe + sexe`). En gras : $q < 0{{,}}05$ après "
              f"Benjamini-Hochberg sur les {len(t)} tests de la colonne. **VIF** "
              f"= facteur d'inflation de la variance du coefficient de classe dû "
              f"au sexe — au-delà de 5, la colonne ajustée ne teste plus "
              f"grand-chose."
              + chr(10) + chr(10) + "::: {.table-scroll}" + chr(10) + chr(10))
    return entete + chr(10).join(lignes) + chr(10) + chr(10) + ":::"


# Les tests d'abord : la figure a besoin de leurs p pour poser les etoiles.
_TESTS = tests_classes()
_BOX_N, _BOX_ETOILES = {}, {}
for _issue in BOX_ISSUES:
    _BOX_N[_issue], _BOX_ETOILES[_issue] = boxplots_classes(_issue, _TESTS)
for _issue in BOX_ISSUES:
    ecrire_table(table_tests_classes(_TESTS, _issue), f"p4str_box_tab_{_issue}")

_sujets = BOX.drop_duplicates("subject")
_cl_moy = _classe(_sujets["delta_TRT_moyen"])
RESUME.update({
    "box_sigma": BOX_SIGMA,
    "box_vue": BOX_VUE,
    "box_n_sujets": int(len(_sujets)),
    "box_n_astronautes": int(_sujets["astro"].nunique()),
    "box_n_variantes": int(BOX.groupby(["agregat", "cas", "region"]).ngroups),
    "box_effectifs": " ; ".join(
        f"{iss} " + "/".join(str(_BOX_N[iss][lab]) for lab, _, _ in BOX_BORNES)
        for iss in BOX_ISSUES),
    "box_sexe_par_classe": " ; ".join(
        f"{lab} {int(((_cl_moy == lab) & (_sujets['sexe'] == 'homme')).sum())} H / "
        f"{int(((_cl_moy == lab) & (_sujets['sexe'] == 'femme')).sum())} F"
        for lab, _, _ in BOX_BORNES),
})
if len(_TESTS):
    _v = _TESTS["vif"].dropna()
    RESUME.update({
        "box_n_tests": int(len(_TESTS)),
        "box_n_p05_brut": int((_TESTS["p_brut"] < 0.05).sum()),
        "box_n_p05_aj": int((_TESTS["p_aj"] < 0.05).sum()),
        "box_attendu_p05": float(0.05 * len(_TESTS)),
        "box_n_q05_brut": int((_TESTS["q_brut"] < 0.05).sum()),
        "box_n_q05_aj": int((_TESTS["q_aj"] < 0.05).sum()),
        "box_q_min_brut": float(_TESTS["q_brut"].min()),
        "box_q_min_aj": float(_TESTS["q_aj"].min()),
        "box_p_min_brut": float(_TESTS["p_brut"].min()),
        "box_p_min_aj": float(_TESTS["p_aj"].min()),
        "box_vif_med": float(_v.median()) if len(_v) else float("nan"),
        "box_vif_max": float(_v.max()) if len(_v) else float("nan"),
        "box_r_classe_sexe": float(_TESTS["r_classe_sexe"].abs().max())
        if "r_classe_sexe" in _TESTS else float("nan"),
        "box_n_p05_welch": int((_TESTS["p_welch"] < 0.05).sum()),
        "box_n_q05_welch": int((_TESTS["q_welch"] < 0.05).sum()),
        "box_q_min_welch": float(_TESTS["q_welch"].min()),
        "box_n_etoiles": int(sum(_BOX_ETOILES.values())),
        "box_etoiles_par_issue": " ; ".join(
            f"{ISSUE_LABEL[i]} {_BOX_ETOILES[i]}/30" for i in BOX_ISSUES),
        "box_n_etoiles_1": int(((_TESTS["p_welch"] < 0.05)
                                & (_TESTS["p_welch"] >= 0.01)).sum()),
        "box_n_etoiles_2": int(((_TESTS["p_welch"] < 0.01)
                                & (_TESTS["p_welch"] >= 0.001)).sum()),
        "box_n_etoiles_3": int((_TESTS["p_welch"] < 0.001).sum()),
        "box_p_min_welch": float(_TESTS["p_welch"].min()),
        "box_top_welch": " ; ".join(
            f"{r['agregat']} {r['region']} réf. {COURT_CAS[r['cas']]} vs "
            f"{ISSUE_LABEL[r['issue']]} t={r['t_welch']:+.2f} p={r['p_welch']:.4f}"
            for _, r in _TESTS.nsmallest(4, "p_welch").iterrows()),
        "box_n_desaccord_welch": int(((_TESTS["p_welch"] < 0.05)
                                      != (_TESTS["p_brut"] < 0.05)).sum()),
        "box_n_p05_astro": int((_TESTS["p_brut_astro"] < 0.05).sum()),
        "box_n_p05_aj_astro": int((_TESTS["p_aj_astro"] < 0.05).sum()),
        "box_n_astro_med": float(_TESTS["n_astro"].median()),
        "box_top_brut": " ; ".join(
            f"{r['agregat']} {r['region']} réf. {COURT_CAS[r['cas']]} vs "
            f"{ISSUE_LABEL[r['issue']]} p={r['p_brut']:.4f} q={r['q_brut']:.3f}"
            for _, r in _TESTS.nsmallest(4, "p_brut").iterrows()),
        "box_top_aj": " ; ".join(
            f"{r['agregat']} {r['region']} réf. {COURT_CAS[r['cas']]} vs "
            f"{ISSUE_LABEL[r['issue']]} p={r['p_aj']:.4f} q={r['q_aj']:.3f}"
            for _, r in _TESTS.nsmallest(4, "p_aj").iterrows()),
    })


# =========================================================================== #
# PAGE 5 — Marker relationships : les issues entre elles, et ce qu'on possede
# =========================================================================== #
# Les pages de prediction correlent des MARQUEURS OCT a des issues cliniques.
# Cette page-ci ne regarde que les ISSUES : comment elles se tiennent entre
# elles, comment elles se repartissent, et lesquelles manquent. Aucun strain,
# aucune segmentation -- rien qui vienne de la chaine de traitement.
#
# C'est la premiere chose a lire de la section, et elle a ete ecrite en dernier :
# les quatre quadrants du TRT ont ete traites comme quatre issues independantes
# tout du long, et la matrice ci-dessous montre qu'ils ne le sont pas.
P5_ISSUES = [i for i in ISSUES]          # les cinq delta_TRT + le delta_AL
P5_QUADRANTS_LAB = ["S", "I", "N", "T"]
P5_QUADRANTS = [f"delta_TRT_{q}" for q in P5_QUADRANTS_LAB]
P5_N_MIN = 4             # sous 4 paires completes, un r ne veut rien dire

# Les tranches demandees sont 0-20, 20-50 et > 50 um. Une QUATRIEME est ajoutee
# en tete, « < 0 um », et elle n'est pas decorative : le TRT DIMINUE chez
# certains astronautes -- une fois sur le quadrant superieur, trois fois sur le
# nasal, trois fois sur le temporal. Sans ce bac, ces yeux disparaitraient de
# l'histogramme sans que rien ne le signale, et les effectifs affiches ne
# sommeraient plus au nombre de sujets.
P5_TRANCHES = [("< 0 µm", -np.inf, 0.0), ("0 – 20 µm", 0.0, 20.0),
               ("20 – 50 µm", 20.0, 50.0), ("≥ 50 µm", 50.0, np.inf)]
COULEUR_TRANCHE = {"< 0 µm": "#8e5bd0", "0 – 20 µm": "#9a9a9a",
                   "20 – 50 µm": "#d0a72b", "≥ 50 µm": "#c2453f"}


def correlations_issues():
    """Matrice de correlation des six issues, et la table qui la detaille.

    Pearson en couleur, Spearman et n au survol. Les paires ne sont PAS toutes
    calculees sur le meme effectif : 18 yeux entre deux delta_TRT, 16 des que le
    delta_AL entre en jeu -- deux astronautes ont un TRT sans longueur axiale.
    L'effectif est donc affiche dans chaque case, faute de quoi on comparerait
    des r qui ne reposent pas sur les memes sujets.
    """
    o = outcomes
    n_iss = len(P5_ISSUES)
    r_mat = np.full((n_iss, n_iss), np.nan)
    n_mat = np.zeros((n_iss, n_iss), dtype=int)
    rho_mat = np.full((n_iss, n_iss), np.nan)
    p_mat = np.full((n_iss, n_iss), np.nan)
    lignes_tab = []
    for a in range(n_iss):
        for b in range(n_iss):
            sub = o[[P5_ISSUES[a], P5_ISSUES[b]]].dropna()
            n_mat[a, b] = len(sub)
            if a == b:
                r_mat[a, b] = 1.0
                rho_mat[a, b] = 1.0
                continue
            if len(sub) < P5_N_MIN:
                continue
            x, y = sub.iloc[:, 0].to_numpy(), sub.iloc[:, 1].to_numpy()
            if np.std(x) == 0 or np.std(y) == 0:
                continue
            r, p = stats.pearsonr(x, y)
            rho, p_rho = stats.spearmanr(x, y)
            r_mat[a, b] = float(r)
            rho_mat[a, b] = float(rho)
            p_mat[a, b] = float(p)
            if a < b:
                lignes_tab.append({"a": P5_ISSUES[a], "b": P5_ISSUES[b],
                                   "n": len(sub), "r": float(r), "p": float(p),
                                   "rho": float(rho), "p_rho": float(p_rho)})
    tab = pd.DataFrame(lignes_tab)
    if len(tab):
        # BH sur les 15 paires : la matrice en compte 15 distinctes, et les lire
        # toutes puis ne citer que la plus forte serait le meme travers que les
        # pages de prediction corrigent partout ailleurs.
        tab["q"] = bh_fdr(tab["p"].to_numpy())
        tab["q_rho"] = bh_fdr(tab["p_rho"].to_numpy())

    etiq = [ISSUE_LABEL[i] for i in P5_ISSUES]
    # Texte de case construit a la main : `np.char.mod` sur un tableau
    # contenant des NaN ecrirait « +nan » au lieu de laisser la case vide.
    # Virgule decimale, comme partout ailleurs sur le site.
    texte = [[f"{r_mat[a][b]:+.2f}".replace(".", ",") + f"<br>n={n_mat[a][b]}"
              if np.isfinite(r_mat[a][b]) else ""
              for b in range(n_iss)] for a in range(n_iss)]
    fig = go.Figure(go.Heatmap(
        z=r_mat, x=etiq, y=etiq, zmin=-1, zmax=1, colorscale="RdBu",
        reversescale=True, colorbar={"title": "r", "thickness": 12, "len": 0.7},
        text=texte, texttemplate="%{text}", textfont={"size": 10},
        customdata=np.dstack([rho_mat, p_mat, n_mat]),
        hovertemplate="%{y} × %{x}<br>r = %{z:+.3f}"
                      "<br>ρ = %{customdata[0]:+.3f}"
                      "<br>p = %{customdata[1]:.4f}"
                      "<br>n = %{customdata[2]:d}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    mise_en_page(fig, "Les six issues les unes contre les autres — corrélation "
                      "de Pearson, effectif dans chaque case. Les quadrants du "
                      "TRT ne sont pas indépendants : la page les traite pourtant "
                      "comme cinq issues distinctes", hauteur=560, legende=False)
    fig.update_layout(margin={"l": 110, "r": 20, "t": 70, "b": 60})
    enregistrer(fig, "p5_corr_issues")
    return tab


def table_correlations_issues(tab):
    """Les 15 paires, triees par q croissante."""
    if tab.empty:
        return "_Aucune paire calculable._"
    lignes = ["| paire | $n$ | Pearson $r$ | $p$ | $q$ | Spearman $"
              + BS_RHO + "$ | $p$ |", "|:--|--:|--:|--:|--:|--:|--:|"]
    for _, r in tab.sort_values("q").iterrows():
        gras = "**" if r["q"] < 0.05 else ""
        lignes.append(
            f"| {ISSUE_LABEL[r['a']]} × {ISSUE_LABEL[r['b']]} | {int(r['n'])} "
            f"| {gras}{fmt_num(r['r'], 3, signe=True)}{gras} "
            f"| {fmt_num(r['p'], 4)} | {fmt_num(r['q'], 3)} "
            f"| {fmt_num(r['rho'], 3, signe=True)} | {fmt_num(r['p_rho'], 4)} |")
    entete = ("Les **15 paires** de la matrice, triées par $q$ croissante. "
              "En gras : $q < 0{,}05$ après Benjamini-Hochberg sur les 15. "
              "Les paires impliquant le ΔAL portent sur 16 yeux, les autres "
              "sur 18." + chr(10) + chr(10))
    return entete + chr(10).join(lignes)


def histogrammes_tranches():
    """Effectif de chaque delta_TRT par tranche.

    Barres groupees plutot que cinq panneaux : la question est « les quadrants
    se repartissent-ils pareil », et cinq axes separes obligeraient a comparer de
    memoire d'un panneau a l'autre.
    """
    o = outcomes
    lignes = []
    for issue in P5_QUADRANTS + ["delta_TRT_moyen"]:
        v = o[issue].dropna()
        for lab, lo, hi in P5_TRANCHES:
            lignes.append({"issue": issue, "tranche": lab,
                           "n": int(((v >= lo) & (v < hi)).sum()),
                           "total": len(v)})
    h = pd.DataFrame(lignes)
    fig = go.Figure()
    for lab, _, _ in P5_TRANCHES:
        g = h[h["tranche"] == lab]
        fig.add_trace(go.Bar(
            x=[ISSUE_LABEL[i] for i in g["issue"]], y=g["n"], name=lab,
            marker={"color": COULEUR_TRANCHE[lab]},
            text=[str(v) if v else "" for v in g["n"]], textposition="outside",
            hovertemplate="%{x}<br>" + lab + "<br>%{y} yeux<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig.update_yaxes(title="yeux", rangemode="tozero")
    n_neg = int(h[h["tranche"] == "< 0 µm"]["n"].sum())
    n_haut = int(h[h["tranche"] == "≥ 50 µm"]["n"].sum())
    mise_en_page(fig, "Répartition de chaque ΔTRT par tranche — 18 yeux par "
                      f"issue. La tranche « ≥ 50 µm » est vide ({n_haut} yeux "
                      f"sur les cinq issues) ; la tranche « < 0 µm » compte "
                      f"{n_neg} yeux, où la rétine s'est AMINCIE après le vol",
                 hauteur=480)
    enregistrer(fig, "p5_hist_tranches")
    return h


def table_tranches(h):
    """Le meme decompte en chiffres, pour pouvoir le citer."""
    labs = [lab for lab, _, _ in P5_TRANCHES]
    lignes = ["| issue | " + " | ".join(labs) + " | total |",
              "|:--|" + "--:|" * (len(labs) + 1)]
    for issue in P5_QUADRANTS + ["delta_TRT_moyen"]:
        g = h[h["issue"] == issue].set_index("tranche")
        lignes.append(f"| {ISSUE_LABEL[issue]} | "
                      + " | ".join(str(int(g.loc[l, "n"])) for l in labs)
                      + f" | {int(g.iloc[0]['total'])} |")
    return chr(10).join(lignes)


def table_donnees_par_patient():
    """Une ligne par ASTRONAUTE : ce qu'on possede, et ce qui manque en rouge.

    Une ligne par astronaute et non par oeil, parce que le manque est TOUJOURS
    au niveau de l'astronaute : sur les quatorze, aucun n'a un oeil renseigne et
    l'autre pas (verifie). Les deux yeux tiennent donc dans la meme case, sous
    la forme « OD / OS », sans rien perdre.

    La couleur est portee par la classe `.donnee-manquante` de `styles.css` :
    ecrire du HTML avec un style en ligne dans une cellule de table pipe passe
    aussi, mais casse des que Bootstrap pose sa propre couleur sur la cellule --
    meme piege que `.marqueur-retenu`.
    """
    o = outcomes.copy()
    o["id"] = o["astro"].str.split("_").str[0].astype(int)
    mesures = ([(f"TRT250_{q}_before", f"TRT{q} avant") for q in P5_QUADRANTS_LAB]
               + [(f"TRT250_{q}_after", f"TRT{q} après") for q in P5_QUADRANTS_LAB]
               + [("AL_before_mm", "AL avant"), ("AL_after_mm", "AL après")])
    lignes = ["| astronaute | sexe | " + " | ".join(lab for _, lab in mesures) + " |",
              "|:--|:--|" + "--:|" * len(mesures)]
    n_cases = n_manquantes = 0
    par_astro = []
    for id_, g in o.groupby("id", sort=True):
        g = g.set_index("eye")
        sexe = g["sexe"].iloc[0] or "—"
        cells, manque_ici = [], 0
        for col, _ in mesures:
            vals = []
            for oeil in ("OD", "OS"):
                v = g[col].get(oeil, np.nan) if oeil in g.index else np.nan
                vals.append(v)
            n_cases += 1
            if all(pd.isna(v) for v in vals):
                cells.append("[manquant]{.donnee-manquante}")
                n_manquantes += 1
                manque_ici += 1
            else:
                fmt = "{:.2f}" if col.startswith("AL") else "{:.0f}"
                cells.append(" / ".join(fmt.format(v) if pd.notna(v) else "—"
                                        for v in vals))
        lignes.append(f"| {id_:02d} | {sexe} | " + " | ".join(cells) + " |")
        par_astro.append({"id": id_, "manquants": manque_ici,
                          "total": len(mesures)})
    entete = (f"Une ligne par astronaute, deux yeux par cellule (**OD / OS**). "
              f"Le manque est toujours au niveau de l'astronaute — aucun n'a un "
              f"œil renseigné et l'autre pas — d'où une seule case pour les "
              f"deux. [En rouge]{{.donnee-manquante}} : la mesure est absente de "
              f"`sansori_db.db`. TRT en µm, AL en mm. "
              f"**{n_manquantes} cases manquantes sur {n_cases}**."
              + chr(10) + chr(10) + "::: {.table-scroll}" + chr(10) + chr(10))
    return (entete + chr(10).join(lignes) + chr(10) + chr(10) + ":::",
            pd.DataFrame(par_astro))


_P5_CORR = correlations_issues()
ecrire_table(table_correlations_issues(_P5_CORR), "p5_tab_corr")
_P5_HIST = histogrammes_tranches()
ecrire_table(table_tranches(_P5_HIST), "p5_tab_tranches")
_P5_TAB, _P5_MANQ = table_donnees_par_patient()
ecrire_table(_P5_TAB, "p5_tab_donnees")

_paires_trt = _P5_CORR[~_P5_CORR["a"].eq(ISSUE_AL) & ~_P5_CORR["b"].eq(ISSUE_AL)]
_paires_quad = _paires_trt[_paires_trt["a"].isin(P5_QUADRANTS)
                           & _paires_trt["b"].isin(P5_QUADRANTS)]
RESUME.update({
    "p5_n_paires": int(len(_P5_CORR)),
    "p5_n_q05": int((_P5_CORR["q"] < 0.05).sum()),
    "p5_r_quadrants_med": float(_paires_quad["r"].median()),
    "p5_r_quadrants_min": float(_paires_quad["r"].min()),
    "p5_r_quadrants_max": float(_paires_quad["r"].max()),
    "p5_paires_quadrants": " ; ".join(
        f"{ISSUE_LABEL[r['a']]}×{ISSUE_LABEL[r['b']]} r={r['r']:+.2f} q={r['q']:.3f}"
        for _, r in _paires_quad.sort_values("q").iterrows()),
    "p5_paires_al": " ; ".join(
        f"{ISSUE_LABEL[r['a'] if r['b'] == ISSUE_AL else r['b']]} r={r['r']:+.2f} "
        f"p={r['p']:.3f} q={r['q']:.3f}"
        for _, r in _P5_CORR[(_P5_CORR["a"] == ISSUE_AL) | (_P5_CORR["b"] == ISSUE_AL)]
        .sort_values("q").iterrows()),
    "p5_n_neg": int(_P5_HIST[_P5_HIST["tranche"] == "< 0 µm"]["n"].sum()),
    "p5_n_haut": int(_P5_HIST[_P5_HIST["tranche"] == "≥ 50 µm"]["n"].sum()),
    "p5_tranches": " ; ".join(
        f"{ISSUE_LABEL[i]} " + "/".join(
            str(int(_P5_HIST[(_P5_HIST["issue"] == i)
                             & (_P5_HIST["tranche"] == lab)]["n"].iloc[0]))
            for lab, _, _ in P5_TRANCHES)
        for i in P5_QUADRANTS + ["delta_TRT_moyen"]),
    "p5_n_astronautes": int(len(_P5_MANQ)),
    "p5_n_complets": int((_P5_MANQ["manquants"] == 0).sum()),
    "p5_n_vides": int((_P5_MANQ["manquants"] == _P5_MANQ["total"]).sum()),
    "p5_n_partiels": int(((_P5_MANQ["manquants"] > 0)
                          & (_P5_MANQ["manquants"] < _P5_MANQ["total"])).sum()),
    "p5_detail_manquants": " ; ".join(
        f"{int(r['id']):02d} : {int(r['manquants'])}/{int(r['total'])}"
        for _, r in _P5_MANQ[_P5_MANQ["manquants"] > 0].iterrows()),
})


# =========================================================================== #
# Tables communes et sortie
# =========================================================================== #
def table_cohorte() -> str:
    lignes = ["| | médiane | IQR | étendue |", "|:--|--:|--:|--:|"]

    def ligne(lab, serie, fmt="{:.1f}"):
        s = pd.Series(serie).dropna()
        if s.empty:
            lignes.append(f"| {cellule(lab)} | n/a | n/a | n/a |")
            return
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        lignes.append(f"| {cellule(lab)} | {fmt.format(s.median())} | "
                      f"{fmt.format(q1)} – {fmt.format(q3)} | "
                      f"{fmt.format(s.min())} – {fmt.format(s.max())} |")

    ligne("one-cycles par condition", ok["n_one_cycles_ok"], "{:.0f}")
    ligne("recalages par condition", ok["n_recalages"], "{:.0f}")
    ligne("hauteur du crop (lignes, sur 496)", ok["crop_rows"], "{:.0f}")
    ligne("échelle axiale (µm/px)", ok["um_per_px_y"], "{:.3f}")
    ligne("épaisseur choroïdienne (µm)", one_cycles["CT_moyen_um"], "{:.0f}")
    ligne("ΔCT dans un one-cycle (µm)", one_cycles["deltaCT_um"], "{:.1f}")
    ligne("|u|max (µm)", bins["u_max_um"], "{:.1f}")
    ligne("jacobien négatif (% des pixels)", bins["jac_neg_pct"], "{:.3f}")
    ligne("frames dans le bin le plus pauvre", one_cycles["n_frames_min_bin"], "{:.0f}")
    # Les deux issues cliniques, en queue de table : ce ne sont pas des mesures
    # OCT et elles se comptent par SUJET, pas par condition -- d'ou la separation
    # visuelle. Le ΔAL est en mm, les ΔTRT en µm.
    lignes.append("| | | | |")
    ligne("ΔTRT moyen (µm, issue)", outcomes["delta_TRT_moyen"], "{:+.1f}")
    ligne("ΔAL (mm, issue)", outcomes["delta_AL_mm"], "{:+.3f}")
    return chr(10).join(lignes)


def table_validation() -> str:
    """Le meme ajustement avant et apres le portail, cote a cote."""
    pente_att = 1.0 / ct_med
    eps = "\varepsilon_{yy}"
    lignes = [f"| $\sigma$ | $r$ (tous) | $r$ (retenus) | pente (tous) "
              "| pente (retenus) | % de la pente attendue, retenus |",
              "|--:|--:|--:|--:|--:|--:|"]
    for sigma in SIGMAS:
        rb = RESUME.get(f"p2_valid_r_brut_sigma{sigma:g}")
        rf = RESUME.get(f"p2_valid_r_filtre_sigma{sigma:g}")
        pb = RESUME.get(f"p2_valid_pente_brut_sigma{sigma:g}")
        pf = RESUME.get(f"p2_valid_pente_filtre_sigma{sigma:g}")
        if rb is None or rf is None:
            continue
        lignes.append(f"| {sigma:g} | {fmt_num(rb, 3, signe=True)} "
                      f"| {fmt_num(rf, 3, signe=True)} | {fmt_sci(pb, signe=True)} "
                      f"| {fmt_sci(pf, signe=True)} "
                      f"| {fmt_num(100 * pf / pente_att, 1)} % |")
    return chr(10).join(lignes)


def table_pentes_pool() -> str:
    """Une ligne par (abscisse, agregat du strain).

    La pente attendue par la geometrie est la MEME pour les trois agregats
    (1/CT) : les trois lignes d'une meme abscisse se lisent donc les unes contre
    les autres, et contre cette valeur.
    """
    lignes = ["| abscisse | agrégat | pente médiane (µm⁻¹) | fraction positive "
              "| fraction à $p < 0{,}05$ | ajustements |",
              "|:--|:--|--:|--:|--:|--:|"]
    for xcol, lab in (("CT_region_um", "épaisseur de la bande (CT)"),
                      ("dCT_region_um", "dilatation de la bande (ΔCT)")):
        premier = True
        for col, lab_ag, _, _ in AGREGATS:
            g = pooled[(pooled["abscisse"] == xcol) & (pooled["tissu"] == col)]
            if g.empty:
                continue
            etq = lab if premier else ""
            lignes.append(f"| {etq} | {lab_ag} "
                          f"| {fmt_sci(float(g['pente_par_um'].median()))} "
                          f"| {100 * (g['pente_par_um'] > 0).mean():.0f} % "
                          f"| {100 * (g['p'] < 0.05).mean():.0f} % | {len(g)} |")
            premier = False
    return chr(10).join(lignes)


def table_repeat() -> str:
    lignes = [f"| référence | $\\sigma$ | ICC(A,1) médian | ICC > 0,75 "
              "| CV intra médian | prédicteurs |",
              "|:--|--:|--:|--:|--:|--:|"]
    for cas in CAS:
        for sigma in SIGMAS:
            a = icc[(icc["cas"] == cas) & (icc["sigma"] == sigma)]["icc"]
            c = cv[(cv["cas"] == cas) & (cv["sigma"] == sigma)]["cv_intra_pct"]
            if a.empty:
                continue
            lignes.append(f"| {COURT_CAS[cas]} | {sigma:g} | {fmt_num(float(a.median()), 2, signe=True)} "
                          f"| {100 * (a > 0.75).mean():.0f} % "
                          f"| {c.median():.0f} % | {len(a)} |")
    return chr(10).join(lignes)


ecrire_table(table_cohorte(), "tab_cohort")
ecrire_table(table_validation(), "tab_validation")
ecrire_table(table_pentes_pool(), "tab_pentes_pool")
def table_repeat_pentes() -> str:
    """ICC et CV par (abscisse, bande, estimateur, agregat), strain RETINIEN.

    La distinction ``par one-cycle`` / ``poolee`` est ici explicite, alors que
    les boites la melangent : c'est le plus grand ecart de toute la page et il
    ne doit pas rester cache. L'AGREGAT l'est aussi, pour la meme raison -- les
    douze points d'une boite en melangent trois.
    """
    a = _table_pentes(icc, "icc")
    c = _table_pentes(cv, "cv_intra_pct")
    lignes = ["| abscisse | bande | estimateur | agrégat "
              "| ICC(A,1) médian | CV intra médian |",
              "|:--|:--|:--|:--|--:|--:|"]
    for abscisse in ABSCISSES:
        for region in REGIONS:
            for estimateur in ("par one-cycle", "poolée"):
                for _, lab_ag, _, _ in AGREGATS:
                    ga = a[(a["abscisse"] == abscisse) & (a["region"] == region)
                           & (a["estimateur"] == estimateur)
                           & (a["agregat"] == lab_ag)]["icc"]
                    gc = c[(c["abscisse"] == abscisse) & (c["region"] == region)
                           & (c["estimateur"] == estimateur)
                           & (c["agregat"] == lab_ag)]["cv_intra_pct"]
                    if ga.empty:
                        continue
                    lignes.append(
                        f"| {ABSCISSE_LABEL[abscisse]} | {region} | {estimateur} "
                        f"| {lab_ag} | {fmt_num(float(ga.median()), 2, signe=True)} "
                        f"| {gc.median():.0f} % |")
    return chr(10).join(lignes)


def table_repeat_regions() -> str:
    """ICC, CV et fraction de variances inter-sujets negatives, bande par bande.

    La derniere colonne est la plus parlante : une variance inter-sujets negative
    veut dire que la dispersion entre sujets est plus petite que le bruit de
    mesure, donc que la bande ne separe pas les sujets du tout.
    """
    lignes = ["| bande | ICC(A,1) médian | ICC local | ICC global | ICC > 0,75 "
              "| CV intra médian | variance inter < 0 |",
              "|:--|--:|--:|--:|--:|--:|--:|"]
    for region in REGIONS:
        a = icc[icc["region"] == region]["icc"]
        c = cv[cv["region"] == region]
        if a.empty:
            continue
        cellules = []
        for cas in CAS:
            aa = icc[(icc["region"] == region) & (icc["cas"] == cas)]["icc"]
            cellules.append(fmt_num(float(aa.median()), 2, signe=True) if len(aa) else "—")
        lignes.append(
            f"| {region} | {fmt_num(float(a.median()), 2, signe=True)} "
            f"| {cellules[0]} | {cellules[1] if len(cellules) > 1 else '—'} "
            f"| {100 * (a > 0.75).mean():.1f} % ".replace(".", ",")
            + f"| {c['cv_intra_pct'].median():.0f} % "
            + f"| {100 * (c['var_inter'] < 0).mean():.1f} % |".replace(".", ","))
    return chr(10).join(lignes)


ecrire_table(table_repeat_pentes(), "tab_repeat_pentes")
def table_pentes_effets() -> str:
    """Effet global de chaque reglage sur l'ICC des pentes : mediane par niveau,
    amplitude du facteur, et part de variance expliquee."""
    lignes = ["| réglage | niveau | ICC(A,1) médian | amplitude du réglage | $\eta^2$ |",
              "|:--|:--|--:|--:|--:|"]
    for col, label, niveaux, _ in FACTEURS:
        med = _eff.groupby(col)["icc"].median()
        amp = float(med.max() - med.min())
        eta = _eta2(col)
        premiers = True
        for niveau in niveaux:
            if niveau not in med.index:
                continue
            lignes.append(
                f"| {label if premiers else ''} | {niveau} "
                f"| {fmt_num(float(med[niveau]), 3, signe=True)} "
                f"| {fmt_num(amp, 3, signe=True) if premiers else ''} "
                f"| {fmt_num(eta, 2) + ' %' if premiers else ''} |")
            premiers = False
    return chr(10).join(lignes)


ecrire_table(table_repeat_regions(), "tab_repeat_regions")
ecrire_table(table_pentes_effets(), "tab_pentes_effets")
ecrire_table(table_stabilite(), "tab_stabilite")

(SORTIE / "plotlyjs.html").write_text(
    '<script type="text/javascript">' + chr(10) + get_plotlyjs() + chr(10) + "</script>" + chr(10),
    encoding="utf-8")
print("  plotlyjs.html")

(SORTIE / "resume.txt").write_text(
    chr(10).join(f"{k} = {v}" for k, v in RESUME.items()) + chr(10), encoding="utf-8")
print("  resume.txt")
print(f"{chr(10)}{len(FICHIERS)} fragment(s) écrit(s) dans {SORTIE}")
