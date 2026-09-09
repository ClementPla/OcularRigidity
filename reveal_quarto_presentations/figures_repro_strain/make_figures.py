# -*- coding: utf-8 -*-
"""
Figures et tables -- page « Strain repeatability » (section Reproducibility).

Ce script NE LIT QUE des tables deja ecrites ; il ne recalcule ni pouls, ni
repliement, ni strain. Toute correction de fond se fait dans le lot qui a
produit la table, pas ici.

Entrees
-------
    E:/NASA_Rigidity/Reproducibility/SegmentationVariations/<variante>/
        pulse_from_data/conditions.csv     HR, fs, methode retenue
        pulse_from_data/traces/<slug>.npz  pouls brut et FIR, spectres, axe BPM
        demons_strain/one_cycles.csv       CT et accord CT/pouls par one-cycle
        demons_strain/bins.csv             frames et CT par bin, strain retine
        demons_strain/regions.csv          strain par bande laterale
        demons_strain/ct_pulse_traces.npz  courbes CT et pouls, brutes et FIR
        repeatability/markers.csv          LE marqueur par acquisition
        repeatability/icc.csv              ICC par configuration
        repeatability/variance.csv         composantes de variance
        repeatability/pairs.csv            paires de replicats

Sorties (fragments Quarto, inclus par ``repro-strain-repeatability.qmd``)
------------------------------------------------------------------------
    fig_hr_ancrage.qmd    frequence cardiaque avant / apres ancrage
    fig_ct_pulse.qmd      pouls SVD et epaisseur choroidienne, brut et FIR
    tab_ct_pulse.qmd      correlation et covariance, par support
    fig_spectre.qmd       spectres et frequence cardiaque retenue
    fig_bins.qmd          frames par bin et CT par bin
    fig_strain.qmd        le strain par bande, sigma et agregat
    fig_icc.qmd           ICC par configuration
    fig_variance.qmd      ou passe la variance
    fig_paires.qmd        replicats deux a deux (accord et Bland-Altman)
    tab_icc.qmd           le tableau de synthese
    plotlyjs.html         plotly.js, insere une seule fois par page
    resume.txt            les chiffres cites dans la prose

Lancer DEPUIS LA RACINE DU DEPOT (ce dossier est une JONCTION vers E:) :
    python reveal_quarto_presentations/figures_repro_strain/make_figures.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------- #
# Entrees / sorties
# --------------------------------------------------------------------------- #
# ``OR_REPRO_ROOT`` sert a pointer le script sur un jeu d'essai sans toucher au
# fichier -- c'est ainsi que les figures sont exercees avant que le lot de
# plusieurs heures ait fini de produire les vraies tables.
VARIANT_ROOT = Path(os.environ.get(
    "OR_REPRO_ROOT",
    "E:/NASA_Rigidity/Reproducibility/SegmentationVariations"
    "/model1_scale_1.0_flatten_choroid_xcorr"))
PULSE_DIR = VARIANT_ROOT / "pulse_from_data"
# Les memes conditions AVANT ancrage de la frequence cardiaque : c'est la seule
# facon de chiffrer ce que l'ancrage a change, et c'est un resultat de l'etude.
PULSE_DIR_BRUT = VARIANT_ROOT / "pulse_from_data_sans_ancrage"
STRAIN_DIR = VARIANT_ROOT / "demons_strain"
REPEAT_DIR = VARIANT_ROOT / "repeatability"

# `.absolute()` et surtout PAS `.resolve()` : ce dossier est une JONCTION vers E:.
SORTIE = Path(__file__).absolute().parent

COULEUR_TEXTE = "#c9c9c9"
GRILLE = "rgba(150,150,150,0.18)"
C_POULS = "#2a78d6"
C_CT = "#e08a1e"
C_REF = "#c2453f"
C_OK = "#27a567"
C_SIGMA = {5.0: "#2a78d6", 10.0: "#27a567", 12.0: "#c98b1a", 15.0: "#9b59b6"}
C_AGG = {"moyenne": "#2a78d6", "mediane": "#27a567", "p95_signe": "#c98b1a"}

# La bande physiologique attendue pour cette cohorte (adultes au repos).
BPM_ATTENDU = (50.0, 80.0)

CONFIG = {"displaylogo": False, "responsive": True,
          "modeBarButtonsToRemove": ["select2d", "lasso2d"]}

FICHIERS: list[str] = []
RESUME: dict = {}


# --------------------------------------------------------------------------- #
# Habillage commun
# --------------------------------------------------------------------------- #
def mise_en_page(fig, titre, hauteur=440, legende=True):
    fig.update_layout(
        title={"text": titre, "font": {"size": 14}, "x": 0.01, "xanchor": "left"},
        template="none",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COULEUR_TEXTE, "size": 11},
        height=hauteur,
        margin={"l": 60, "r": 20, "t": 70, "b": 80 if legende else 50},
        showlegend=legende,
        legend={"font": {"size": 10}, "orientation": "h", "yanchor": "top",
                "y": -0.10, "xanchor": "left", "x": 0},
        hoverlabel={"font": {"size": 11}},
        bargap=0.06,
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
    entete = ("<!-- Genere par figures_repro_strain/make_figures.py"
              " -- ne pas editer a la main. -->" + chr(10) + chr(10))
    (SORTIE / f"{nom}.qmd").write_text(entete + texte + chr(10), encoding="utf-8")
    FICHIERS.append(f"{nom}.qmd")
    print(f"  {nom}.qmd")


def absente(nom, pourquoi):
    """Ecrit quand meme le fragment, en disant ce qui manque.

    Un ``{{< include >}}`` sur un fichier absent ne degrade pas la page : il
    fait ECHOUER le rendu en entier. Une figure qui ne peut pas etre produite
    doit donc laisser une trace lisible a sa place, et non un trou.
    """
    ecrire_table(f"::: {{.callout-warning}}{chr(10)}"
                 f"## Figure indisponible{chr(10)}{chr(10)}{pourquoi}{chr(10)}:::",
                 nom)


def fr(x, chiffres=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{{:.{chiffres}f}}".format(x).replace(".", ",")


def med_iqr(s, chiffres=2):
    s = pd.Series(s).dropna()
    if s.empty:
        return "—"
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return f"{fr(s.median(), chiffres)} [{fr(q1, chiffres)} – {fr(q3, chiffres)}]"


def essaim(v, largeur=0.34, n_bins=36):
    """Decalages horizontaux : les valeurs proches s'ecartent au lieu de se
    superposer. Meme fonction que sur la page « Data exploration » -- a ces
    effectifs, un point cache derriere un autre est un effectif perdu."""
    v = np.asarray(v, dtype=float)
    x = np.zeros(len(v))
    fini = np.isfinite(v)
    if not fini.any():
        return x
    lo, hi = float(np.min(v[fini])), float(np.max(v[fini]))
    pas = (hi - lo) / n_bins if hi > lo else 1.0
    cle = np.full(len(v), np.nan)
    cle[fini] = np.round((v[fini] - lo) / pas)
    for c in np.unique(cle[fini]):
        idx = np.flatnonzero(cle == c)
        n = len(idx)
        if n > 1:
            x[idx] = np.linspace(-1.0, 1.0, n) * largeur * min(1.0, (n - 1) / 8.0)
    return x


# --------------------------------------------------------------------------- #
# L'ancrage de la frequence cardiaque
# --------------------------------------------------------------------------- #
# Cette figure est produite AVANT le chargement des tables de strain : elle ne
# lit que les deux tables de pouls, disponibles des la deuxieme etape de la
# chaine. La placer apres la rendrait indisponible pendant les heures que dure
# le lot des demons, alors qu'elle est deja calculable.
RE_SLUG_FIG = __import__("re").compile(
    r"^(?P<participant>.+)_(?P<eye>OD|OS)(?P<replicate>\d+)$")


def _par_oeil(df, col):
    x = df["slug"].str.extract(RE_SLUG_FIG)
    d = df.copy()
    d["eye_id"] = x["participant"] + "_" + x["eye"]
    d["replicate"] = pd.to_numeric(x["replicate"], errors="coerce")
    d["v"] = pd.to_numeric(d[col], errors="coerce")
    return d[d["status"] == "ok"] if "status" in d else d


def fig_hr_ancrage():
    """Ce que l'ancrage a change, mesure sur une frequence qu'il n'impose PAS.

    La colonne ``hr_BPM`` est inutilisable pour ce controle : apres ancrage elle
    vaut le consensus par construction, et tous les replicats d'un participant y
    portent le meme nombre. Constater qu'ils « s'accordent » serait une
    tautologie.

    ``dmd_hr_BPM`` est en revanche estimee par decomposition en modes
    dynamiques, sans qu'on la lui souffle. Elle reste cherchee dans la bande
    FC +/- 20 % -- donc l'ancrage la contraint -- mais l'accord obtenu est dix
    fois plus serre que la largeur de cette bande, ce qu'une simple contrainte
    n'expliquerait pas.
    """
    f_brut = PULSE_DIR_BRUT / "conditions.csv"
    if not f_brut.exists():
        absente("fig_hr_ancrage",
                "`pulse_from_data_sans_ancrage/conditions.csv` est absent : la "
                "comparaison avant / apres ancrage n'est pas calculable.")
        return
    av = _par_oeil(pd.read_csv(f_brut), "dmd_hr_BPM")
    ap = _par_oeil(pd.read_csv(PULSE_DIR / "conditions.csv"), "dmd_hr_BPM")
    yeux = sorted(set(ap["eye_id"]) & set(av["eye_id"]))

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.62, 0.38], horizontal_spacing=0.11,
        subplot_titles=("Fréquence DMD par acquisition, œil par œil",
                        "Étendue au sein d'un œil"))
    for j, (d, nom, coul) in enumerate([(av, "sans ancrage", C_REF),
                                        (ap, "ancrée", C_OK)]):
        xs, ys, txt = [], [], []
        for k, e in enumerate(yeux):
            g = d[d["eye_id"] == e]
            xs += list(k + essaim(g["v"].to_numpy(), largeur=0.28))
            ys += list(g["v"])
            txt += list(g["slug"])
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", name=nom, legendgroup=nom,
            marker={"color": coul, "size": 6, "opacity": 0.8},
            text=txt, hovertemplate="%{text}<br>%{y:.1f} BPM<extra></extra>"),
            1, 1)
    fig.update_xaxes(tickvals=list(range(len(yeux))),
                     ticktext=[e.replace("_", " ") for e in yeux],
                     tickangle=-55, tickfont={"size": 8}, row=1, col=1)
    fig.update_yaxes(title_text="fréquence DMD (BPM)", row=1, col=1)

    ea = av.groupby("eye_id")["v"].agg(lambda s: s.max() - s.min())
    eb = ap.groupby("eye_id")["v"].agg(lambda s: s.max() - s.min())
    for nom, e, coul in (("sans ancrage", ea, C_REF), ("ancrée", eb, C_OK)):
        fig.add_trace(go.Scatter(
            x=essaim(e.to_numpy(), largeur=0.3) + (0 if nom == "sans ancrage" else 1),
            y=e, mode="markers", name=nom, legendgroup=nom, showlegend=False,
            marker={"color": coul, "size": 8, "opacity": 0.8},
            text=list(e.index),
            hovertemplate="%{text}<br>étendue %{y:.1f} BPM<extra></extra>"),
            1, 2)
    fig.update_xaxes(tickvals=[0, 1], ticktext=["sans", "ancrée"],
                     range=[-0.6, 1.6], row=1, col=2)
    fig.update_yaxes(title_text="étendue intra-œil (BPM)", type="log",
                     row=1, col=2)

    RESUME["hr_etendue_avant"] = med_iqr(ea, 1)
    RESUME["hr_etendue_apres"] = med_iqr(eb, 1)
    RESUME["hr_yeux_sup10_avant"] = f"{int((ea > 10).sum())} / {len(ea)}"
    RESUME["hr_yeux_sup10_apres"] = f"{int((eb > 10).sum())} / {len(eb)}"

    fig = mise_en_page(
        fig, "L'ancrage de la fréquence cardiaque, vu par une estimation "
             "qu'il n'impose pas", hauteur=470)
    enregistrer(fig, "fig_hr_ancrage")


fig_hr_ancrage()


# --------------------------------------------------------------------------- #
# Donnees
# --------------------------------------------------------------------------- #
def lire():
    manquants = []
    tables = {}
    for cle, chemin in [
        ("pulse", PULSE_DIR / "conditions.csv"),
        ("one_cycles", STRAIN_DIR / "one_cycles.csv"),
        ("bins", STRAIN_DIR / "bins.csv"),
        ("regions", STRAIN_DIR / "regions.csv"),
        ("markers", REPEAT_DIR / "markers.csv"),
        ("icc", REPEAT_DIR / "icc.csv"),
        ("variance", REPEAT_DIR / "variance.csv"),
        ("pairs", REPEAT_DIR / "pairs.csv"),
    ]:
        if chemin.exists():
            tables[cle] = pd.read_csv(chemin)
        else:
            manquants.append(str(chemin))
    if manquants:
        # Ne PAS s'arreter la : la page est enregistree dans ``_quarto.yml``,
        # et un ``{{< include >}}`` sur un fichier absent fait echouer le rendu
        # du SITE entier, pas seulement de cette page. On ecrit donc, pour
        # chaque fragment attendu, un encadre qui dit ce qui manque et comment
        # le produire -- que la prochaine execution reussie remplacera.
        nl = chr(10)
        pourquoi = (
            "Le calcul n'a pas encore produit ces tables :" + nl + nl
            + nl.join(f"- `{Path(m).name}`" for m in manquants)
            + nl + nl + "Lancer :" + nl + nl + "```bash" + nl
            + "python Reproducibility/run_batch.py all" + nl
            + "python Reproducibility/compute_repeatability.py" + nl
            + "python reveal_quarto_presentations/figures_repro_strain"
              "/make_figures.py" + nl + "```")
        for nom in ("fig_ct_pulse", "fig_accord", "tab_ct_pulse", "fig_spectre",
                    "fig_bins", "fig_strain", "fig_icc", "tab_icc",
                    "fig_variance", "fig_paires"):
            absente(nom, pourquoi)
        (SORTIE / "plotlyjs.html").write_text("", encoding="utf-8")
        # Conserver ce que RESUME contient DEJA : les figures calculables sans
        # les tables de strain (l'ancrage de la FC) ont pu s'executer avant, et
        # leurs chiffres sont cites dans la prose. Les ecraser rendrait la page
        # fausse le temps que le lot se termine.
        RESUME["etat"] = "en attente du calcul"
        (SORTIE / "resume.txt").write_text(
            nl.join(f"{k} = {v}" for k, v in RESUME.items()) + nl,
            encoding="utf-8")
        print(f"{chr(10)}Tables absentes : {len(FICHIERS)} fragment(s) "
              f"d'attente ecrits, rien d'autre.")
        raise SystemExit(0)
    npz = STRAIN_DIR / "ct_pulse_traces.npz"
    tables["traces"] = np.load(npz, allow_pickle=True) if npz.exists() else None
    return tables


T = lire()
pulse, one_cycles = T["pulse"], T["one_cycles"]
bins, regions = T["bins"], T["regions"]
markers, icc, variance, pairs = (T["markers"], T["icc"], T["variance"],
                                 T["pairs"])
traces = T["traces"]

RESUME["n_acquisitions"] = int(markers["slug"].nunique())
RESUME["n_participants"] = int(markers["participant"].nunique())
RESUME["n_yeux"] = int(markers["eye_id"].nunique())
RESUME["n_one_cycles"] = int(len(one_cycles))
RESUME["one_cycles_par_acq"] = med_iqr(one_cycles.groupby("slug").size(), 0)


# --------------------------------------------------------------------------- #
# 0. L'ancrage de la frequence cardiaque
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# 1. Le pouls SVD contre l'epaisseur choroidienne
# --------------------------------------------------------------------------- #
def fig_ct_pulse():
    """Une acquisition representative : les deux signaux, bruts puis filtres.

    Ce sont DEUX mesures independantes du meme phenomene -- l'un tire des
    INTENSITES par SVD, l'autre de la geometrie des MASQUES segmentes. Rien
    n'impose qu'ils s'accordent : c'est justement ce que la figure montre.

    L'acquisition affichee est celle dont la correlation est la plus proche de
    la MEDIANE de la cohorte, et non la meilleure : une figure choisie sur son
    maximum ne dit rien de ce qu'on obtient en general.
    """
    if traces is None:
        absente("fig_ct_pulse",
                "`demons_strain/ct_pulse_traces.npz` est absent : lancer "
                "`python Reproducibility/run_batch.py ct_pulse`.")
        return
    slugs = [str(s) for s in traces["slugs"]]
    r_par_slug = one_cycles.groupby("slug")["r_temps"].median()
    dispo = [s for s in slugs if s in r_par_slug.index]
    if not dispo:
        return
    cible = float(r_par_slug.loc[dispo].abs().median())
    slug = min(dispo, key=lambda s: abs(abs(r_par_slug.loc[s]) - cible))
    RESUME["exemple_slug"] = slug
    RESUME["exemple_r"] = fr(float(r_par_slug.loc[slug]), 2)

    # Deux bases de temps, et il ne faut pas les confondre : les signaux BRUTS
    # vivent sur les horodatages reels (``t``, non uniformes -- c'est tout le
    # probleme de cette acquisition), les signaux FILTRES sur la grille
    # reguliere (``u_time``) ou le FIR a pu etre applique.
    u = traces[f"{slug}__u_time"]
    t_reel = traces[f"{slug}__t"]
    ct_brut = traces[f"{slug}__ct_brut_um"]
    ct_fir = traces[f"{slug}__ct_fir"]
    pulse_fir = traces[f"{slug}__pulse_fir"]
    npz_slug = PULSE_DIR / "traces" / f"{slug}.npz"
    pulse_brut = None
    if npz_slug.exists():
        with np.load(npz_slug, allow_pickle=True) as d:
            if "pulse_0_sans_filtre" in d.files:
                pulse_brut = np.asarray(d["pulse_0_sans_filtre"], dtype=float)

    def norm(a):
        """Centre-reduit : les deux signaux n'ont ni la meme unite (um contre
        intensite arbitraire) ni la meme echelle. Seule leur FORME se compare."""
        a = np.asarray(a, dtype=float)
        s = np.nanstd(a)
        return (a - np.nanmean(a)) / s if s > 0 else a - np.nanmean(a)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Signaux BRUTS (centrés-réduits)",
                                        "Après le même filtre FIR (FC ± 20 %)"),
                        vertical_spacing=0.13)
    for rang, (xp, p, xc, c) in enumerate(
            [(u, pulse_brut, t_reel, ct_brut), (u, pulse_fir, u, ct_fir)]):
        if p is not None and len(p) == len(xp):
            fig.add_trace(go.Scatter(
                x=xp, y=norm(p), mode="lines", name="pouls SVD (intensités)",
                legendgroup="p", showlegend=(rang == 0),
                line={"color": C_POULS, "width": 1.4},
                hovertemplate="%{x:.2f} s — %{y:.2f}<extra>pouls</extra>"),
                rang + 1, 1)
        if len(c) == len(xc):
            fig.add_trace(go.Scatter(
                x=xc, y=norm(c), mode="lines",
                name="épaisseur choroïdienne (masques)",
                legendgroup="c", showlegend=(rang == 0),
                line={"color": C_CT, "width": 1.4},
                hovertemplate="%{x:.2f} s — %{y:.2f}<extra>CT</extra>"),
                rang + 1, 1)
    fig.update_xaxes(title_text="temps (s)", row=2, col=1)
    fig.update_yaxes(title_text="écart-type", row=1, col=1)
    fig.update_yaxes(title_text="écart-type", row=2, col=1)
    fig = mise_en_page(
        fig, f"Deux mesures du même pouls — {slug} "
             f"(r = {RESUME['exemple_r']}, condition médiane de la cohorte)",
        hauteur=520)
    enregistrer(fig, "fig_ct_pulse")


fig_ct_pulse()


def fig_accord():
    """Correlation ET covariance, sur les deux supports, toute la cohorte.

    La correlation dit si les deux signaux ont la meme FORME ; la covariance
    dit si l'accord porte sur une amplitude qui compte. Les deux sont demandees
    parce qu'une correlation forte sur un signal minuscule ne vaut rien.

    Les valeurs sont SIGNEES et non prises en valeur absolue : le signe du
    pouls SVD est fixe par une convention interne au lot, pas par la
    physiologie, et une distribution centree sur zero -- si c'est ce qu'on voit
    -- est donc un resultat sur cette convention autant que sur l'accord.
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Corrélation, support temporel",
                        "Corrélation, support des bins",
                        "Covariance (support temporel)"),
        horizontal_spacing=0.09)
    for j, (col, titre) in enumerate([("r_temps", "r"), ("r_bins", "r"),
                                      ("cov_temps", "cov")]):
        v = pd.to_numeric(one_cycles[col], errors="coerce").dropna()
        fig.add_trace(go.Histogram(
            x=v, nbinsx=45, marker_color=C_POULS, showlegend=False,
            hovertemplate="%{x:.2f} — %{y} one-cycle(s)<extra></extra>"),
            1, j + 1)
        fig.add_vline(x=float(v.median()), line_color=C_REF, line_dash="dash",
                      row=1, col=j + 1,
                      annotation_text=f"médiane {fr(v.median(), 2)}",
                      annotation_font_color=C_REF, annotation_font_size=10)
        fig.update_xaxes(title_text=titre, row=1, col=j + 1)
        RESUME[f"accord_{col}"] = med_iqr(v, 2)
        RESUME[f"accord_{col}_absmed"] = fr(float(v.abs().median()), 2)
    fig.update_yaxes(title_text="one-cycles", row=1, col=1)
    fig = mise_en_page(fig, "Accord entre le pouls SVD et l'épaisseur choroïdienne",
                       hauteur=400, legende=False)
    enregistrer(fig, "fig_accord")

    lignes = ["| mesure | support | médiane [IQR] | \\|médiane\\| |",
              "|:--|:--|--:|--:|"]
    for col, mesure, support in [
            ("r_temps", "corrélation", "toute la fenêtre du one-cycle"),
            ("r_bins", "corrélation", "les 6 bins repliés"),
            ("cov_temps", "covariance", "toute la fenêtre du one-cycle"),
            ("cov_bins", "covariance", "les 6 bins repliés")]:
        if col not in one_cycles.columns:
            continue
        v = pd.to_numeric(one_cycles[col], errors="coerce").dropna()
        lignes.append(f"| {mesure} | {support} | {med_iqr(v, 3)} | "
                      f"{fr(float(v.abs().median()), 3)} |")
    ecrire_table(chr(10).join(lignes), "tab_ct_pulse")


fig_accord()


# --------------------------------------------------------------------------- #
# 2. Le spectre et la frequence cardiaque
# --------------------------------------------------------------------------- #
def fig_spectre():
    """Spectres empiles + la FC retenue par acquisition, contre la bande attendue.

    La bande 50-80 BPM est un PRIOR physiologique (adulte au repos), pas un
    filtre : le lot cherche le pic sur 20-240 BPM. Une FC hors bande n'est donc
    pas corrigee, elle est visible -- et c'est ce qu'on veut voir.
    """
    # L'axe des BPM est PROPRE a chaque condition : sa resolution depend de la
    # duree de l'enregistrement, qui va de 18 a 78 s. Le lire une fois et le
    # reutiliser pour toutes ferait correspondre des puissances a des
    # frequences qui ne sont pas les leurs.
    lignes_spec = []
    for slug in sorted(pulse.loc[pulse["status"] == "ok", "slug"]):
        f = PULSE_DIR / "traces" / f"{slug}.npz"
        if not f.exists():
            continue
        with np.load(f, allow_pickle=True) as d:
            if "spec_1_fir" not in d.files or "bpm_axis" not in d.files:
                continue
            axe = np.asarray(d["bpm_axis"], dtype=float)
            spec = np.asarray(d["spec_1_fir"], dtype=float)
            if axe.shape != spec.shape:
                continue
            m = np.nanmax(spec)
            lignes_spec.append((slug, axe, spec / m if m > 0 else spec))
    if not lignes_spec:
        absente("fig_spectre",
                "Aucun spectre lisible dans `pulse_from_data/traces/` : lancer "
                "`python Reproducibility/run_batch.py pulse`.")
        return

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.58, 0.42], horizontal_spacing=0.10,
        subplot_titles=(f"Spectres normalisés ({len(lignes_spec)} acquisitions)",
                        "Fréquence cardiaque retenue"))
    for slug, axe, spec in lignes_spec:
        garder = (axe >= 20) & (axe <= 200)
        fig.add_trace(go.Scatter(
            x=axe[garder], y=spec[garder], mode="lines", name=slug,
            showlegend=False, opacity=0.45,
            line={"color": C_POULS, "width": 0.9},
            hovertemplate=f"{slug}<br>%{{x:.0f}} BPM<extra></extra>"), 1, 1)
    for x in BPM_ATTENDU:
        fig.add_vline(x=x, line_color=C_REF, line_dash="dash", row=1, col=1)
    fig.add_vrect(x0=BPM_ATTENDU[0], x1=BPM_ATTENDU[1], row=1, col=1,
                  fillcolor=C_REF, opacity=0.07, line_width=0,
                  annotation_text="attendu 50–80", annotation_font_size=10,
                  annotation_font_color=C_REF)
    fig.update_xaxes(title_text="fréquence (BPM)", row=1, col=1)
    fig.update_yaxes(title_text="puissance (normalisée)", row=1, col=1)

    hr = pd.to_numeric(pulse.loc[pulse["status"] == "ok", "hr_BPM"],
                       errors="coerce").dropna()
    fig.add_trace(go.Scatter(
        x=essaim(hr.to_numpy()), y=hr, mode="markers", showlegend=False,
        marker={"color": C_POULS, "size": 7, "opacity": 0.8},
        text=list(pulse.loc[pulse["status"] == "ok", "slug"]),
        hovertemplate="%{text}<br>%{y:.1f} BPM<extra></extra>"), 1, 2)
    fig.add_hrect(y0=BPM_ATTENDU[0], y1=BPM_ATTENDU[1], row=1, col=2,
                  fillcolor=C_REF, opacity=0.07, line_width=0)
    for y in BPM_ATTENDU:
        fig.add_hline(y=y, line_color=C_REF, line_dash="dash", row=1, col=2)
    fig.update_xaxes(showticklabels=False, range=[-0.8, 0.8], row=1, col=2)
    fig.update_yaxes(title_text="FC (BPM)", row=1, col=2)

    dans = int(((hr >= BPM_ATTENDU[0]) & (hr <= BPM_ATTENDU[1])).sum())
    RESUME["hr"] = med_iqr(hr, 1)
    RESUME["hr_dans_bande"] = f"{dans} / {len(hr)}"
    RESUME["hr_min"] = fr(hr.min(), 1)
    RESUME["hr_max"] = fr(hr.max(), 1)
    fig = mise_en_page(fig, "Le spectre, et où tombe la fréquence cardiaque",
                       hauteur=430, legende=False)
    enregistrer(fig, "fig_spectre")


fig_spectre()


# --------------------------------------------------------------------------- #
# 3. Le repliement : frames par bin et CT par bin
# --------------------------------------------------------------------------- #
def fig_bins():
    """Ce que le repliement a reellement mis dans chaque bin.

    Un bin peu peuple n'est pas un bin moyenne : c'est une image. La figure
    donne donc l'effectif AVANT la grandeur, parce que la seconde ne se lit
    qu'a la lumiere de la premiere.

    Le CT est celui des masques REPLIES puis resegmentes -- c'est-a-dire la
    grandeur telle que le recalage des demons la voit, et non l'epaisseur
    frame par frame de la figure precedente.
    """
    b = bins[bins["sigma"] == bins["sigma"].min()]  # le CT ne depend pas de sigma
    b = b[b["cas"] == "local"] if "local" in set(b["cas"]) else b

    fig = make_subplots(
        rows=1, cols=3, horizontal_spacing=0.09,
        subplot_titles=("Frames par bin", "CT par bin (replié, resegmenté)",
                        "Amplitude CT du cycle (max − min)"))

    n = pd.to_numeric(b["n_frames_bin"], errors="coerce").dropna()
    fig.add_trace(go.Histogram(
        x=n, nbinsx=40, marker_color=C_POULS, showlegend=False,
        hovertemplate="%{x:.0f} frames — %{y} bin(s)<extra></extra>"), 1, 1)
    fig.update_xaxes(title_text="frames", row=1, col=1)
    fig.update_yaxes(title_text="bins", row=1, col=1)
    RESUME["frames_par_bin"] = med_iqr(n, 0)
    RESUME["frames_par_bin_min"] = fr(n.min(), 0)
    RESUME["bins_sous_4"] = f"{int((n < 4).sum())} / {len(n)}"

    # CT par bin, une ligne par one-cycle : la FORME du cycle, pas seulement sa
    # dispersion. Les cycles sont centres sur leur propre moyenne, sinon
    # l'epaisseur absolue (250 a 400 um selon le sujet) ecraserait la pulsation
    # (quelques um), qui est la seule chose que le bin fait varier.
    for (slug, oc), g in b.groupby(["slug", "one_cycle"]):
        g = g.sort_values("bin")
        y = pd.to_numeric(g["CT_um"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(y).any():
            continue
        fig.add_trace(go.Scatter(
            x=g["bin"], y=y - np.nanmean(y), mode="lines", opacity=0.25,
            line={"color": C_CT, "width": 0.8}, showlegend=False,
            hovertemplate=f"{slug} #{oc}<br>bin %{{x}} — %{{y:+.2f}} µm"
                          "<extra></extra>"), 1, 2)
    fig.update_xaxes(title_text="bin", dtick=1, row=1, col=2)
    fig.update_yaxes(title_text="CT − moyenne du cycle (µm)", row=1, col=2)

    amp = pd.to_numeric(one_cycles["deltaCT_um"], errors="coerce").dropna()
    fig.add_trace(go.Histogram(
        x=amp, nbinsx=40, marker_color=C_OK, showlegend=False,
        hovertemplate="%{x:.1f} µm — %{y} one-cycle(s)<extra></extra>"), 1, 3)
    fig.update_xaxes(title_text="ΔCT (µm)", row=1, col=3)
    RESUME["deltaCT"] = med_iqr(amp, 2)

    fig = mise_en_page(fig, "Le repliement, bin par bin", hauteur=420,
                       legende=False)
    enregistrer(fig, "fig_bins")


fig_bins()


# --------------------------------------------------------------------------- #
# 4. Le strain
# --------------------------------------------------------------------------- #
ORDRE_REGIONS = ["retine entiere", "colonne -2", "colonne -1",
                 "colonne 0 (centre)", "colonne +1", "colonne +2"]


def _regions_presentes(df):
    return [r for r in ORDRE_REGIONS if r in set(df["region"])]


def fig_strain():
    """Le marqueur lui-meme, avant toute question de repetabilite.

    Une ligne de panneaux par agregat, un panneau par sigma : c'est le plan de
    l'experience, et il faut le voir avant les ICC -- un marqueur dont la
    dispersion inter-sujets est nulle ne peut pas avoir un bon ICC, quelle que
    soit la qualite de la mesure.
    """
    m = markers[markers["cas"] == "local"] if "local" in set(markers["cas"]) \
        else markers
    regs = _regions_presentes(m)
    sigmas = sorted(m["sigma"].unique())
    aggs = [a for a in ("moyenne", "mediane", "p95_signe")
            if a in set(m["agregat"])]

    fig = make_subplots(
        rows=len(aggs), cols=len(sigmas),
        subplot_titles=[f"σ = {s:g}" for s in sigmas] * len(aggs),
        horizontal_spacing=0.045, vertical_spacing=0.10,
        shared_yaxes=True)
    for i, agg in enumerate(aggs):
        for j, s in enumerate(sigmas):
            sub = m[(m["agregat"] == agg) & (m["sigma"] == s)]
            for k, reg in enumerate(regs):
                v = pd.to_numeric(sub.loc[sub["region"] == reg, "valeur"],
                                  errors="coerce")
                if v.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=k + essaim(v.to_numpy(), largeur=0.30),
                    y=v, mode="markers", showlegend=False,
                    marker={"color": C_AGG.get(agg, C_POULS), "size": 4.5,
                            "opacity": 0.7},
                    text=list(sub.loc[sub["region"] == reg, "slug"]),
                    hovertemplate="%{text}<br>%{y:.2e}<extra></extra>"),
                    i + 1, j + 1)
                med = float(v.median())
                fig.add_trace(go.Scatter(
                    x=[k - 0.38, k + 0.38], y=[med, med], mode="lines",
                    showlegend=False, line={"color": C_REF, "width": 1.6},
                    hovertemplate=f"médiane {med:.2e}<extra></extra>"),
                    i + 1, j + 1)
            fig.update_xaxes(tickvals=list(range(len(regs))),
                             ticktext=[r.replace("colonne ", "")
                                       .replace("retine entiere", "tout")
                                       .replace(" (centre)", "")
                                       for r in regs],
                             range=[-0.6, len(regs) - 0.4], row=i + 1, col=j + 1)
        fig.update_yaxes(title_text=agg, row=i + 1, col=1)

    fig = mise_en_page(fig, "Le strain rétinien, bin le plus mince → bin le plus épais",
                       hauteur=260 * len(aggs) + 90, legende=False)
    enregistrer(fig, "fig_strain")


fig_strain()


# --------------------------------------------------------------------------- #
# 5. ICC
# --------------------------------------------------------------------------- #
def fig_icc():
    """L'ICC de chaque configuration, avec son intervalle de confiance.

    L'intervalle est affiche parce qu'il est LARGE : quatorze yeux et trois
    replicats ne suffisent pas a distinguer un ICC de 0,4 d'un ICC de 0,8, et
    lire les points sans leurs barres donnerait une precision qui n'existe pas.
    """
    d = icc[icc["icc_type"] == "ICC(1,1)"].dropna(subset=["icc"])
    if d.empty:
        absente("fig_icc", "Aucun ICC exploitable : trop peu d'yeux ayant "
                           "leurs trois réplicats.")
        return
    d = d[d["cas"] == "local"] if "local" in set(d["cas"]) else d
    regs = _regions_presentes(d)
    sigmas = sorted(d["sigma"].unique())
    aggs = [a for a in ("moyenne", "mediane", "p95_signe")
            if a in set(d["agregat"])]

    fig = make_subplots(rows=1, cols=len(aggs),
                        subplot_titles=aggs, horizontal_spacing=0.06,
                        shared_yaxes=True)
    for i, agg in enumerate(aggs):
        for s in sigmas:
            sub = d[(d["agregat"] == agg) & (d["sigma"] == s)]
            sub = sub.set_index("region").reindex(regs).reset_index()
            fig.add_trace(go.Scatter(
                x=list(range(len(regs))), y=sub["icc"], mode="markers+lines",
                name=f"σ = {s:g}", legendgroup=f"{s:g}", showlegend=(i == 0),
                line={"color": C_SIGMA.get(s, C_POULS), "width": 1},
                marker={"color": C_SIGMA.get(s, C_POULS), "size": 7},
                error_y={"type": "data", "symmetric": False,
                         "array": (sub["icc_ci95_high"] - sub["icc"]).abs(),
                         "arrayminus": (sub["icc"] - sub["icc_ci95_low"]).abs(),
                         "color": C_SIGMA.get(s, C_POULS), "thickness": 1,
                         "width": 2},
                text=sub["region"],
                hovertemplate="%{text}<br>ICC %{y:.2f}<extra></extra>"),
                1, i + 1)
        fig.update_xaxes(tickvals=list(range(len(regs))),
                         ticktext=[r.replace("colonne ", "")
                                   .replace("retine entiere", "tout")
                                   .replace(" (centre)", "") for r in regs],
                         row=1, col=i + 1)
    # Les seuils usuels de lecture d'un ICC (Koo & Li 2016).
    for y, txt in ((0.5, "médiocre / passable"), (0.75, "passable / bon"),
                   (0.9, "bon / excellent")):
        fig.add_hline(y=y, line_color=C_REF, line_dash="dot", opacity=0.6,
                      annotation_text=txt, annotation_font_size=9,
                      annotation_font_color=C_REF)
    fig.update_yaxes(title_text="ICC(1,1)", range=[-0.4, 1.05], row=1, col=1)
    fig = mise_en_page(fig, "Répétabilité du strain : ICC(1,1) entre réplicats",
                       hauteur=460)
    enregistrer(fig, "fig_icc")

    a = icc[icc["icc_type"] == "ICC(1,1)"].dropna(subset=["icc"])
    RESUME["icc_median"] = fr(float(a["icc"].median()), 3)
    RESUME["icc_max"] = fr(float(a["icc"].max()), 3)
    RESUME["icc_min"] = fr(float(a["icc"].min()), 3)
    RESUME["icc_n_sup_075"] = f"{int((a['icc'] >= 0.75).sum())} / {len(a)}"
    RESUME["icc_n_sup_050"] = f"{int((a['icc'] >= 0.50).sum())} / {len(a)}"
    RESUME["icc_n_neg"] = f"{int((a['icc'] < 0).sum())} / {len(a)}"
    best = a.loc[a["icc"].idxmax()]
    RESUME["icc_best"] = (f"{fr(best['icc'], 2)} (cas {best['cas']}, "
                          f"σ = {best['sigma']:g}, {best['region']}, "
                          f"{best['agregat']})")
    RESUME["icc_n_yeux"] = int(a["n_yeux"].iloc[0])


fig_icc()


def tab_icc():
    """Le meilleur ICC par agregat et par sigma, avec l'unite de l'erreur.

    Un ICC seul est un rapport ; la colonne ``sd_intra`` dit dans quelle unite
    on se trompe, et c'est elle qu'on emporte pour dimensionner une etude.
    """
    d = icc[icc["icc_type"] == "ICC(1,1)"].dropna(subset=["icc"])
    if d.empty:
        ecrire_table("_Aucun ICC exploitable._", "tab_icc")
        return
    v = variance.set_index(["cas", "sigma", "region", "agregat"])
    lignes = ["| agrégat | σ | meilleure région | ICC(1,1) [IC 95 %] "
              "| écart-type intra | coeff. de répétabilité |",
              "|:--|--:|:--|--:|--:|--:|"]
    for agg in ("moyenne", "mediane", "p95_signe"):
        for s in sorted(d["sigma"].unique()):
            sub = d[(d["agregat"] == agg) & (d["sigma"] == s)]
            if sub.empty:
                continue
            r = sub.loc[sub["icc"].idxmax()]
            cle = (r["cas"], r["sigma"], r["region"], r["agregat"])
            sd = rc = np.nan
            if cle in v.index:
                sd = float(v.loc[cle, "sd_intra"])
                rc = float(v.loc[cle, "rc"])
            lignes.append(
                f"| {agg} | {s:g} | {r['region']} | {fr(r['icc'], 2)} "
                f"[{fr(r['icc_ci95_low'], 2)} – {fr(r['icc_ci95_high'], 2)}] "
                f"| {sd:.2e} | {rc:.2e} |".replace("e-0", "e−0"))
    ecrire_table(chr(10).join(lignes), "tab_icc")


tab_icc()


# --------------------------------------------------------------------------- #
# 6. Ou passe la variance
# --------------------------------------------------------------------------- #
def fig_variance():
    """La question posee : l'ecart entre individus depasse-t-il l'ecart intra ?

    Barres empilees a 100 % : sujet / oeil dans le sujet / replicat. Un
    marqueur utilisable a une barre dominee par le SUJET ; une barre dominee
    par le residu dit que la mesure ne distingue pas deux personnes.
    """
    if variance.empty:
        absente("fig_variance", "`variance.csv` est vide : aucun participant "
                                "n'a ses deux yeux avec trois réplicats.")
        return
    v = variance[variance["cas"] == "local"] if "local" in set(variance["cas"]) \
        else variance
    aggs = [a for a in ("moyenne", "mediane", "p95_signe")
            if a in set(v["agregat"])]
    sigmas = sorted(v["sigma"].unique())
    regs = _regions_presentes(v)

    fig = make_subplots(rows=1, cols=len(aggs), subplot_titles=aggs,
                        horizontal_spacing=0.05, shared_yaxes=True)
    x_labels = [f"{r.replace('colonne ', '').replace(' (centre)', '').replace('retine entiere', 'tout')} · σ{s:g}"
                for s in sigmas for r in regs]
    for i, agg in enumerate(aggs):
        for nom, col, coul in (("participant", "pct_sujet", C_OK),
                               ("œil du participant", "pct_oeil", C_POULS),
                               ("réplicat (bruit)", "pct_residuelle", C_REF)):
            ys = []
            for s in sigmas:
                for r in regs:
                    sub = v[(v["agregat"] == agg) & (v["sigma"] == s)
                            & (v["region"] == r)]
                    ys.append(float(sub[col].iloc[0]) if len(sub) else np.nan)
            fig.add_trace(go.Bar(
                x=x_labels, y=ys, name=nom, legendgroup=nom,
                showlegend=(i == 0), marker_color=coul,
                hovertemplate="%{x}<br>" + nom + " : %{y:.0f} %<extra></extra>"),
                1, i + 1)
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="part de la variance (%)", range=[0, 100],
                     row=1, col=1)
    fig.update_xaxes(tickangle=-60, tickfont={"size": 8})
    fig = mise_en_page(fig, "Où passe la variance : entre sujets, entre yeux, "
                            "entre réplicats", hauteur=520)
    enregistrer(fig, "fig_variance")

    RESUME["var_pct_sujet"] = med_iqr(variance["pct_sujet"], 1)
    RESUME["var_pct_oeil"] = med_iqr(variance["pct_oeil"], 1)
    RESUME["var_pct_res"] = med_iqr(variance["pct_residuelle"], 1)
    RESUME["var_n_sujet_domine"] = (
        f"{int((variance['pct_sujet'] > variance['pct_residuelle']).sum())}"
        f" / {len(variance)}")
    RESUME["var_n_comp_negative"] = int(
        (variance["composante_negative"].fillna("") != "").sum())


fig_variance()


# --------------------------------------------------------------------------- #
# 7. Les replicats deux a deux
# --------------------------------------------------------------------------- #
def fig_paires():
    """Accord brut (gauche) et Bland-Altman (droite), configuration la meilleure.

    Le nuage repond a « deux mesures du meme oeil tombent-elles au meme
    endroit », le Bland-Altman a « l'ecart depend-il du niveau ». Ils sont
    traces sur la configuration de meilleur ICC : c'est le cas le plus
    favorable, et le dire evite de le lire comme le cas general.
    """
    if pairs.empty or icc.empty:
        absente("fig_paires", "`pairs.csv` ou `icc.csv` est vide.")
        return
    a = icc[icc["icc_type"] == "ICC(1,1)"].dropna(subset=["icc"])
    if a.empty:
        absente("fig_paires", "Aucun ICC exploitable pour choisir une "
                              "configuration à tracer.")
        return
    best = a.loc[a["icc"].idxmax()]
    p = pairs[(pairs["cas"] == best["cas"]) & (pairs["sigma"] == best["sigma"])
              & (pairs["region"] == best["region"])
              & (pairs["agregat"] == best["agregat"])]
    if p.empty:
        absente("fig_paires", "Aucune paire de réplicats dans la meilleure "
                              "configuration.")
        return

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.11,
                        subplot_titles=("Réplicat b contre réplicat a",
                                        "Bland-Altman"))
    couleurs = {pid: c for pid, c in zip(
        sorted(p["participant"].unique()),
        ["#2a78d6", "#27a567", "#c98b1a", "#9b59b6", "#c2453f", "#17a2b8",
         "#e08a1e", "#7f8c8d"] * 3)}
    for pid, g in p.groupby("participant"):
        fig.add_trace(go.Scatter(
            x=g["valeur_a"], y=g["valeur_b"], mode="markers", name=pid,
            legendgroup=pid, marker={"color": couleurs[pid], "size": 7,
                                     "opacity": 0.8},
            text=g["eye_id"] + " (" + g["rep_a"].astype(str) + "↔"
                 + g["rep_b"].astype(str) + ")",
            hovertemplate="%{text}<br>%{x:.2e} → %{y:.2e}<extra></extra>"), 1, 1)
        fig.add_trace(go.Scatter(
            x=g["moyenne"], y=g["ecart"], mode="markers", name=pid,
            legendgroup=pid, showlegend=False,
            marker={"color": couleurs[pid], "size": 7, "opacity": 0.8},
            text=g["eye_id"],
            hovertemplate="%{text}<br>moy %{x:.2e}, écart %{y:.2e}"
                          "<extra></extra>"), 1, 2)
    lo = float(min(p["valeur_a"].min(), p["valeur_b"].min()))
    hi = float(max(p["valeur_a"].max(), p["valeur_b"].max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             showlegend=False, hoverinfo="skip",
                             line={"color": C_REF, "dash": "dash", "width": 1}),
                  1, 1)
    biais = float(p["ecart"].mean())
    sd = float(p["ecart"].std(ddof=1))
    for y, dash in ((biais, "solid"), (biais + 1.96 * sd, "dash"),
                    (biais - 1.96 * sd, "dash")):
        fig.add_hline(y=y, line_color=C_REF, line_dash=dash, opacity=0.7,
                      row=1, col=2)
    fig.update_xaxes(title_text="réplicat a", row=1, col=1)
    fig.update_yaxes(title_text="réplicat b", row=1, col=1)
    fig.update_xaxes(title_text="moyenne des deux", row=1, col=2)
    fig.update_yaxes(title_text="écart (b − a)", row=1, col=2)
    fig = mise_en_page(
        fig, f"Réplicats deux à deux — meilleure configuration "
             f"(σ = {best['sigma']:g}, {best['region']}, {best['agregat']}, "
             f"ICC = {fr(best['icc'], 2)})", hauteur=470)
    enregistrer(fig, "fig_paires")

    RESUME["ba_biais"] = f"{biais:.2e}"
    RESUME["ba_loa"] = f"{biais - 1.96 * sd:.2e} … {biais + 1.96 * sd:.2e}"
    RESUME["n_paires"] = int(len(p))


fig_paires()


# --------------------------------------------------------------------------- #
# Sortie
# --------------------------------------------------------------------------- #
(SORTIE / "plotlyjs.html").write_text(
    '<script type="text/javascript">' + chr(10) + get_plotlyjs() + chr(10)
    + "</script>" + chr(10), encoding="utf-8")
print("  plotlyjs.html")

(SORTIE / "resume.txt").write_text(
    chr(10).join(f"{k} = {v}" for k, v in RESUME.items()) + chr(10),
    encoding="utf-8")
print("  resume.txt")
print(f"{chr(10)}{len(FICHIERS)} fragment(s) écrit(s) dans {SORTIE}")
