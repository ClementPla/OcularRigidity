# -*- coding: utf-8 -*-
"""
Figures et tables -- page « Valider HR des réplicats » (section Reproducibility).

Ce script NE LIT QUE les sorties de ``Reproducibility/compute_hr_bandpass.py`` ;
il ne recalcule ni SVD, ni filtre, ni spectre. Toute correction de fond se fait
dans le lot, pas ici.

Entrees
-------
    E:/NASA_Rigidity/Reproducibility/SegmentationVariations/<variante>/
        hr_bandpass_<lo>_<hi>/conditions.csv   FC par acquisition, deux signaux
        hr_bandpass_<lo>_<hi>/traces.npz       traces brutes / filtrees, spectres
    (bande : ``OR_BANDE_BPM``, 50,85 par defaut -- la meme que le lot)

Sorties (fragments Quarto, inclus par ``repro-hr-validation.qmd``)
-----------------------------------------------------------------
    fig_traces.qmd       traces brutes et filtrees, menu sur les acquisitions
    fig_spectres.qmd     spectres filtres, pics et a priori, meme menu
    fig_boxplot_hr.qmd   FC par participant, un point = une acquisition
    tab_hr.qmd           synthese par participant
    plotlyjs.html        plotly.js, insere une seule fois par page
    resume.txt           les chiffres cites dans la prose

Lancer DEPUIS LA RACINE DU DEPOT (ce dossier est une JONCTION vers E:) :
    python reveal_quarto_presentations/figures_repro_hr/make_figures.py
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
VARIANT_ROOT = Path(os.environ.get(
    "OR_REPRO_ROOT",
    "E:/NASA_Rigidity/Reproducibility/SegmentationVariations"
    "/model1_scale_1.0_flatten_choroid_xcorr"))
# Meme convention que le lot : la bande choisit le dossier lu.
_BANDE = [float(v) for v in os.environ.get("OR_BANDE_BPM", "50,85").split(",")]
HR_DIR = VARIANT_ROOT / f"hr_bandpass_{_BANDE[0]:.0f}_{_BANDE[1]:.0f}"

# `.absolute()` et surtout PAS `.resolve()` : ce dossier est une JONCTION vers E:.
SORTIE = Path(__file__).absolute().parent

COULEUR_TEXTE = "#c9c9c9"
GRILLE = "rgba(150,150,150,0.18)"
C_INT = "#2a78d6"
C_CT = "#e08a1e"
C_REF = "#c2453f"
SYMBOLE_OEIL = {"OD": "circle", "OS": "diamond"}

# Tolerances de lecture, et non de calcul : elles ne servent qu'a compter.
TOL_PRIOR = 0.10  # |FC - a priori| <= 10 % de l'a priori
TOL_ACCORD_BPM = 5.0  # intensite et epaisseur « d'accord »
MARGE_BORD_BPM = 1.5  # pic colle a une borne de la bande

CONFIG = {"displaylogo": False, "responsive": True,
          "modeBarButtonsToRemove": ["select2d", "lasso2d"]}

FICHIERS: list[str] = []
RESUME: dict = {}


# --------------------------------------------------------------------------- #
# Habillage commun (meme habillage que figures_repro_strain)
# --------------------------------------------------------------------------- #
def mise_en_page(fig, titre, hauteur=440, legende=True):
    fig.update_layout(
        title={"text": titre, "font": {"size": 14}, "x": 0.01, "xanchor": "left"},
        template="none",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COULEUR_TEXTE, "size": 11},
        height=hauteur,
        margin={"l": 60, "r": 20, "t": 110, "b": 80 if legende else 50},
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
    entete = ("<!-- Genere par figures_repro_hr/make_figures.py"
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
    return f"{{:.{chiffres}f}}".format(x).replace(".", ",")


def med_iqr(s, chiffres=1):
    s = pd.Series(s).dropna()
    if s.empty:
        return "—"
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return f"{fr(s.median(), chiffres)} [{fr(q1, chiffres)} – {fr(q3, chiffres)}]"


def essaim(v, largeur=0.12, n_bins=30):
    """Decalages horizontaux : les valeurs proches s'ecartent au lieu de se
    superposer (meme fonction que les autres pages de la section)."""
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
            x[idx] = np.linspace(-1.0, 1.0, n) * largeur * min(1.0, (n - 1) / 4.0)
    return x


def norm(a):
    """Centre-reduit : intensite (u.a.) et epaisseur (µm) n'ont ni la meme unite
    ni la meme echelle ; seule leur FORME, donc leur periode, se compare."""
    a = np.asarray(a, dtype=float)
    s = np.nanstd(a)
    return (a - np.nanmean(a)) / s if s > 0 else a - np.nanmean(a)


def nom_participant(p):
    return str(p).replace("_", " ")


# --------------------------------------------------------------------------- #
# Donnees
# --------------------------------------------------------------------------- #
NOMS = ("fig_traces", "fig_spectres", "fig_boxplot_hr", "tab_hr")
csv, npz = HR_DIR / "conditions.csv", HR_DIR / "traces.npz"
if not (csv.exists() and npz.exists()):
    nl = chr(10)
    pourquoi = ("Les sorties du lot sont absentes :" + nl + nl
                + f"- `{csv.name}`" + nl + f"- `{npz.name}`" + nl + nl
                + "Lancer :" + nl + nl + "```bash" + nl
                + "python Reproducibility/compute_hr_bandpass.py" + nl
                + "python reveal_quarto_presentations/figures_repro_hr/make_figures.py"
                + nl + "```")
    for nom in NOMS:
        absente(nom, pourquoi)
    (SORTIE / "plotlyjs.html").write_text("", encoding="utf-8")
    raise SystemExit(0)

D = pd.read_csv(csv)
D = D[D["status"] == "ok"].sort_values(["participant", "eye", "replicate"])
D = D.reset_index(drop=True)
TR = np.load(npz, allow_pickle=True)
SLUGS = [s for s in D["slug"] if f"{s}__u_time" in TR.files]
D = D[D["slug"].isin(SLUGS)].reset_index(drop=True)
LO, HI = float(D["bande_lo_BPM"].iloc[0]), float(D["bande_hi_BPM"].iloc[0])
PARTICIPANTS = sorted(D["participant"].unique())

RESUME["n_acquisitions"] = len(D)
RESUME["n_participants"] = len(PARTICIPANTS)
RESUME["bande"] = f"{LO:.0f}-{HI:.0f} BPM"


def titre_slug(r):
    return (f"{r['slug']}<br>intensité {fr(r['hr_intensite_BPM'])} BPM · "
            f"épaisseur {fr(r['hr_epaisseur_BPM'])} BPM · "
            f"a priori {fr(r['hr_prior_BPM'])} BPM")


def menu(n_par_slug, titres, formes=None):
    """Un bouton par acquisition : ne rend visibles que SES traces."""
    n = len(SLUGS)
    boutons = []
    for i, s in enumerate(SLUGS):
        vis = [False] * (n * n_par_slug)
        vis[i * n_par_slug:(i + 1) * n_par_slug] = [True] * n_par_slug
        layout = {"title.text": titres[i]}
        if formes is not None:
            layout["shapes"] = formes[i]
        boutons.append({"label": s, "method": "update",
                        "args": [{"visible": vis}, layout]})
    # Menu a DROITE, au-dessus de la zone de trace : a gauche il chevauchait le
    # titre du premier sous-graphe.
    return [{"buttons": boutons, "direction": "down", "showactive": True,
             "x": 1.0, "xanchor": "right", "y": 1.0, "yanchor": "bottom",
             "pad": {"b": 28}, "bgcolor": "rgba(120,120,120,0.25)",
             "bordercolor": GRILLE, "font": {"size": 11, "color": "#222"}}]


# --------------------------------------------------------------------------- #
# 1. Traces brutes et filtrees
# --------------------------------------------------------------------------- #
def fig_traces():
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        subplot_titles=("Signaux BRUTS (centrés-réduits)",
                                        f"Après le FIR {LO:.0f}–{HI:.0f} BPM "
                                        "(centrés-réduits)"))
    titres, formes = [], []
    for i, s in enumerate(SLUGS):
        r = D[D["slug"] == s].iloc[0]
        t, u = TR[f"{s}__t"], TR[f"{s}__u_time"]
        core = TR[f"{s}__core"].astype(bool)
        premier = i == 0
        for rang, (x, y_int, y_ct) in enumerate(
                [(t, TR[f"{s}__int_brut"], TR[f"{s}__ct_brut_um"]),
                 (u, TR[f"{s}__int_filt"], TR[f"{s}__ct_filt_um"])]):
            fig.add_trace(go.Scatter(
                x=x, y=norm(y_int), mode="lines", visible=premier,
                name=f"intensité (SVD, comp. {int(r['svd_comp'])})",
                legendgroup="int", showlegend=(rang == 0),
                line={"color": C_INT, "width": 1.3},
                hovertemplate="%{x:.2f} s — %{y:.2f}<extra>intensité</extra>"),
                rang + 1, 1)
            fig.add_trace(go.Scatter(
                x=x, y=norm(y_ct), mode="lines", visible=premier,
                name="épaisseur choroïdienne (masques)",
                legendgroup="ct", showlegend=(rang == 0),
                line={"color": C_CT, "width": 1.3},
                hovertemplate="%{x:.2f} s — %{y:.2f}<extra>épaisseur</extra>"),
                rang + 1, 1)
        # Les bords exclus du pic (transitoire du FIR), grises sur le panneau filtre.
        u0, u1 = float(u[core][0]), float(u[core][-1])
        grise = {"type": "rect", "xref": "x2", "yref": "y2 domain", "y0": 0, "y1": 1,
                 "fillcolor": "rgba(150,150,150,0.15)", "line": {"width": 0},
                 "layer": "below"}
        formes.append([{**grise, "x0": float(u[0]), "x1": u0},
                       {**grise, "x0": u1, "x1": float(u[-1])}])
        titres.append(titre_slug(r))
    fig.update_xaxes(title_text="temps (s)", row=2, col=1)
    fig.update_yaxes(title_text="écart-type", row=1, col=1)
    fig.update_yaxes(title_text="écart-type", row=2, col=1)
    fig = mise_en_page(fig, titres[0], hauteur=600)
    fig.update_layout(shapes=formes[0], updatemenus=menu(4, titres, formes))
    enregistrer(fig, "fig_traces")


fig_traces()


# --------------------------------------------------------------------------- #
# 2. Spectres filtres
# --------------------------------------------------------------------------- #
def fig_spectres():
    fig = go.Figure()
    titres, formes = [], []
    for i, s in enumerate(SLUGS):
        r = D[D["slug"] == s].iloc[0]
        bpm = TR[f"{s}__bpm_bande"]
        for cle, nom, coul in (("spec_int", "intensité", C_INT),
                               ("spec_ct", "épaisseur", C_CT)):
            p = np.asarray(TR[f"{s}__{cle}"], dtype=float)
            fig.add_trace(go.Scatter(
                x=bpm, y=p / p.max() if p.max() > 0 else p, mode="lines",
                visible=(i == 0), name=nom, legendgroup=nom,
                line={"color": coul, "width": 1.6},
                hovertemplate="%{x:.1f} BPM — %{y:.2f}<extra>" + nom + "</extra>"))

        def vline(x, coul, dash):
            return {"type": "line", "xref": "x", "yref": "paper", "x0": x, "x1": x,
                    "y0": 0, "y1": 1, "line": {"color": coul, "width": 1.5,
                                               "dash": dash}}
        f = [vline(float(r["hr_intensite_BPM"]), C_INT, "dot"),
             vline(float(r["hr_epaisseur_BPM"]), C_CT, "dot")]
        if np.isfinite(r["hr_prior_BPM"]):
            f.append(vline(float(r["hr_prior_BPM"]), C_REF, "dash"))
            # La sous-harmonique de l'a priori tombe dans la bande des qu'il
            # depasse 60 BPM : c'est le premier piege a reconnaitre.
            if r["hr_prior_BPM"] / 2 >= LO:
                f.append(vline(float(r["hr_prior_BPM"]) / 2, C_REF, "dashdot"))
        formes.append(f)
        titres.append(titre_slug(r))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="a priori (consensus)",
                             line={"color": C_REF, "dash": "dash"}))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="a priori / 2",
                             line={"color": C_REF, "dash": "dashdot"}))
    fig.update_xaxes(title_text="fréquence (BPM)", range=[LO, HI])
    fig.update_yaxes(title_text="puissance Lomb-Scargle (normalisée au max)")
    fig = mise_en_page(fig, titres[0], hauteur=460)
    # Les deux traces de legende finales restent visibles pour tous les boutons.
    m = menu(2, titres, formes)
    for b in m[0]["buttons"]:
        b["args"][0]["visible"] = b["args"][0]["visible"] + [True, True]
    fig.update_layout(shapes=formes[0], updatemenus=m)
    enregistrer(fig, "fig_spectres")


fig_spectres()


# --------------------------------------------------------------------------- #
# 3. Boxplot par participant
# --------------------------------------------------------------------------- #
def fig_boxplot_hr():
    """Une personne = deux boites (intensite, epaisseur) ; un point = une
    acquisition. L'axe x est NUMERIQUE pour pouvoir poser les points a la main :
    les boites Plotly n'acceptent qu'un seul symbole par trace, et l'oeil doit
    se lire sur le point."""
    fig = go.Figure()
    dx = 0.2
    for cle, nom, coul, dec in (("hr_intensite_BPM", "intensité (SVD)", C_INT, -dx),
                                ("hr_epaisseur_BPM", "épaisseur (masques)", C_CT, +dx)):
        xs = D["participant"].map({p: k for k, p in enumerate(PARTICIPANTS)}) + dec
        fig.add_trace(go.Box(
            x=xs, y=D[cle], name=nom, legendgroup=nom, width=0.34,
            marker={"color": coul}, line={"color": coul, "width": 1.4},
            fillcolor="rgba(0,0,0,0)", boxpoints=False, hoverinfo="skip"))
        for oeil, symb in SYMBOLE_OEIL.items():
            g = D[D["eye"] == oeil]
            if g.empty:
                continue
            x = np.empty(len(g))
            for p in g["participant"].unique():
                sel = (g["participant"] == p).to_numpy()
                x[sel] = PARTICIPANTS.index(p) + dec + essaim(g.loc[sel, cle].to_numpy())
            fig.add_trace(go.Scatter(
                x=x, y=g[cle], mode="markers", legendgroup=nom, showlegend=False,
                marker={"color": coul, "symbol": symb, "size": 8, "opacity": 0.85,
                        "line": {"color": "rgba(20,20,20,0.6)", "width": 0.6}},
                customdata=np.stack([g["slug"], g["replicate"],
                                     g["hr_prior_BPM"].round(1)], axis=1),
                hovertemplate=("%{customdata[0]}<br>" + nom + " : %{y:.1f} BPM"
                               "<br>a priori %{customdata[2]} BPM<extra></extra>")))
    # Symboles de legende pour l'oeil.
    for oeil, symb in SYMBOLE_OEIL.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=oeil,
                                 marker={"color": COULEUR_TEXTE, "symbol": symb,
                                         "size": 8}))
    # L'a priori consensus : un tiret par participant.
    xs, ys = [], []
    for k, p in enumerate(PARTICIPANTS):
        v = D.loc[D["participant"] == p, "hr_prior_BPM"].median()
        if np.isfinite(v):
            xs += [k - 0.45, k + 0.45, None]
            ys += [v, v, None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="a priori (consensus)",
                             line={"color": C_REF, "width": 2, "dash": "dash"},
                             hovertemplate="a priori %{y:.1f} BPM<extra></extra>"))
    for b in (LO, HI):
        fig.add_hline(y=b, line={"color": GRILLE.replace("0.18", "0.6"), "width": 1,
                                 "dash": "dot"})
    # Nom et prenom sur deux lignes, sans inclinaison : inclines, le premier nom
    # debordait a gauche de la figure et disparaissait.
    fig.update_xaxes(tickvals=list(range(len(PARTICIPANTS))),
                     ticktext=[nom_participant(p).replace(" ", "<br>", 1)
                               for p in PARTICIPANTS],
                     range=[-0.6, len(PARTICIPANTS) - 0.4], tickangle=0,
                     tickfont={"size": 10})
    # L'axe couvre la bande ET les a priori : un a priori hors bande (MCNEIL,
    # ~102 BPM) doit rester visible, c'est justement ce qu'il faut voir.
    y_max = np.nanmax([HI, D["hr_prior_BPM"].max()])
    y_min = np.nanmin([LO, D["hr_prior_BPM"].min()])
    fig.update_yaxes(title_text="fréquence cardiaque trouvée (BPM)",
                     range=[y_min - 5, y_max + 5])
    fig = mise_en_page(
        fig, f"FC trouvée par participant, sans a priori (FIR {LO:.0f}–{HI:.0f} BPM)"
             "<br>un point = une acquisition", hauteur=560)
    # Legende sous les noms de participants : a y = -0.10 elle les recouvrait.
    fig.update_layout(margin={"t": 80, "b": 130},
                      legend={"y": -0.20, "yanchor": "top"})
    enregistrer(fig, "fig_boxplot_hr")


fig_boxplot_hr()


# --------------------------------------------------------------------------- #
# 4. Synthese
# --------------------------------------------------------------------------- #
def proche(v, ref, tol=TOL_PRIOR):
    return (np.abs(v - ref) <= tol * ref) & np.isfinite(ref)


# --- Reference « au hasard » ------------------------------------------------ #
# Le pic est cherche DANS la bande : un signal sans pouls y rend quand meme une
# valeur. Plus la bande est etroite, plus les estimations se ressemblent et
# ressemblent a l'a priori, sans que rien ne soit mesure. Chaque accord est donc
# compare a ce que donneraient des pics tires UNIFORMEMENT dans la bande.
LARGEUR = HI - LO


def etendue_hasard(n):
    """Esperance de l'etendue de n tirages uniformes sur la bande."""
    return LARGEUR * (n - 1) / (n + 1)


def p_proche_hasard(ref, tol=TOL_PRIOR):
    """P(un tirage uniforme tombe a +/- tol x ref de ref)."""
    if not np.isfinite(ref):
        return 0.0
    return max(0.0, min((1 + tol) * ref, HI) - max((1 - tol) * ref, LO)) / LARGEUR


# P(|X - Y| <= d) pour deux tirages uniformes independants sur la bande.
P_ACCORD_HASARD = 1.0 - (1.0 - min(TOL_ACCORD_BPM / LARGEUR, 1.0)) ** 2


def tab_hr():
    lignes = ["| participant | n | FC intensité (BPM) | étendue | FC épaisseur (BPM)"
              " | étendue | étendue au hasard | a priori | accord int./ép. |",
              "|:--|--:|:--|--:|:--|--:|--:|--:|--:|"]
    for p in PARTICIPANTS:
        g = D[D["participant"] == p]
        acc = (np.abs(g["ecart_int_ct_BPM"]) <= TOL_ACCORD_BPM).sum()
        lignes.append(
            f"| {nom_participant(p)} | {len(g)} "
            f"| {med_iqr(g['hr_intensite_BPM'])} "
            f"| {fr(g['hr_intensite_BPM'].max() - g['hr_intensite_BPM'].min())} "
            f"| {med_iqr(g['hr_epaisseur_BPM'])} "
            f"| {fr(g['hr_epaisseur_BPM'].max() - g['hr_epaisseur_BPM'].min())} "
            f"| {fr(etendue_hasard(len(g)))} "
            f"| {fr(g['hr_prior_BPM'].median())} | {acc} / {len(g)} |")
    ecrire_table(chr(10).join(lignes) + chr(10) + chr(10)
                 + f": FC par participant. Médiane [IQR] sur les acquisitions ; "
                   f"« étendue au hasard » = étendue attendue si les n pics étaient "
                   f"tirés uniformément dans {LO:.0f}–{HI:.0f} BPM ; "
                   f"« accord » = intensité et épaisseur à ± {TOL_ACCORD_BPM:.0f} BPM "
                   f"l'une de l'autre. {{.striped}}", "tab_hr")


tab_hr()

prior = D["hr_prior_BPM"].to_numpy(float)
for cle, court in (("hr_intensite_BPM", "int"), ("hr_epaisseur_BPM", "ct")):
    v = D[cle].to_numpy(float)
    RESUME[f"{court}_proche_prior"] = f"{int(proche(v, prior).sum())} / {len(v)}"
    RESUME[f"{court}_sous_harmonique"] = (
        f"{int((proche(v, prior / 2) & ~proche(v, prior)).sum())} / {len(v)}")
    RESUME[f"{court}_bord_bande"] = (
        f"{int(((v <= LO + MARGE_BORD_BPM) | (v >= HI - MARGE_BORD_BPM)).sum())}"
        f" / {len(v)}")
    RESUME[f"{court}_ecart_prior_med"] = med_iqr(np.abs(v - prior))
    etendue = D.groupby("participant")[cle].agg(lambda s: s.max() - s.min())
    RESUME[f"{court}_etendue_participant"] = med_iqr(etendue)
    etendue_oeil = D.groupby(["participant", "eye"])[cle].agg(lambda s: s.max() - s.min())
    RESUME[f"{court}_etendue_oeil"] = med_iqr(etendue_oeil)
RESUME["accord_int_ct"] = (
    f"{int((np.abs(D['ecart_int_ct_BPM']) <= TOL_ACCORD_BPM).sum())} / {len(D)}")
RESUME["hasard_accord_int_ct"] = fr(P_ACCORD_HASARD * len(D), 1)
RESUME["hasard_proche_prior"] = fr(sum(p_proche_hasard(p) for p in prior), 1)
RESUME["hasard_etendue_participant"] = med_iqr(
    [etendue_hasard(n) for n in D.groupby("participant").size()])
RESUME["hasard_etendue_oeil"] = med_iqr(
    [etendue_hasard(n) for n in D.groupby(["participant", "eye"]).size()])
RESUME["svd_comp"] = med_iqr(D["svd_comp"], 0)
RESUME["svd_comp_sup_20"] = f"{int((D['svd_comp'] >= 20).sum())} / {len(D)}"
RESUME["svd_frac_bande"] = med_iqr(D["svd_frac_bande"], 2)
RESUME["pic_frac_int"] = med_iqr(D["pic_frac_intensite"], 3)
RESUME["pic_frac_ct"] = med_iqr(D["pic_frac_epaisseur"], 3)
RESUME["ct_proche_prior_par_participant"] = "; ".join(
    f"{p}: {int(proche(g['hr_epaisseur_BPM'].to_numpy(float), g['hr_prior_BPM'].to_numpy(float)).sum())}/{len(g)}"
    for p, g in D.groupby("participant"))
RESUME["int_proche_prior_par_participant"] = "; ".join(
    f"{p}: {int(proche(g['hr_intensite_BPM'].to_numpy(float), g['hr_prior_BPM'].to_numpy(float)).sum())}/{len(g)}"
    for p, g in D.groupby("participant"))

(SORTIE / "plotlyjs.html").write_text(
    '<script type="text/javascript">' + chr(10) + get_plotlyjs() + chr(10)
    + "</script>" + chr(10), encoding="utf-8")
print("  plotlyjs.html")
(SORTIE / "resume.txt").write_text(
    chr(10).join(f"{k} = {v}" for k, v in RESUME.items()) + chr(10), encoding="utf-8")
print("  resume.txt")
