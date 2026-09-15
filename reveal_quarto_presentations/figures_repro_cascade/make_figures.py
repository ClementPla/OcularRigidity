# -*- coding: utf-8 -*-
"""
Figures et tables -- page « Recalage appris cascade_v9 » (section Reproducibility).

Ce script NE LIT QUE des tables deja ecrites ; il ne recale rien, ne recalcule
aucune FC. Toute correction de fond se fait dans le lot qui a produit la table.

Entrees
-------
    E:/NASA_Rigidity/Reproducibility/SegmentationVariations/
        cascade_v9_1536x1024/registration_summary.csv      NCC cascade + identite
        cascade_v9_1536x1024/hr_bandpass_<lo>_<hi>/        FC sur le recalage appris
        model1_scale_1.0_flatten_choroid_xcorr/hr_bandpass_<lo>_<hi>/   FC de reference
    E:/NASA_Rigidity/Reproducibility/registration_comparison/metrics.csv
                                                          NCC des variantes existantes

Sorties (fragments Quarto, inclus par ``repro-cascade-v9.qmd``)
--------------------------------------------------------------
    fig_qualite_recalage.qmd   NCC par variante, et differences appariees
    tab_qualite.qmd            synthese par variante
    fig_boxplot_hr.qmd         FC par participant, recalage de reference vs cascade_v9
    fig_hr_paires.qmd          FC cascade_v9 contre FC de reference, par acquisition
    tab_hr.qmd                 indicateurs de FC, deux recalages et hasard
    fig_traces.qmd             traces brutes et filtrees sur le recalage appris
    plotlyjs.html              plotly.js, insere une seule fois par page
    resume.txt                 les chiffres cites dans la prose

Lancer DEPUIS LA RACINE DU DEPOT (ce dossier est une JONCTION vers E:) :
    python reveal_quarto_presentations/figures_repro_cascade/make_figures.py
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
SEGVAR = Path("E:/NASA_Rigidity/Reproducibility/SegmentationVariations")
_BANDE = [float(v) for v in os.environ.get("OR_BANDE_BPM", "50,85").split(",")]
HR_SUB = f"hr_bandpass_{_BANDE[0]:.0f}_{_BANDE[1]:.0f}"
CAS_ROOT = SEGVAR / "cascade_v9_1536x1024"
REF_ROOT = SEGVAR / "model1_scale_1.0_flatten_choroid_xcorr"
CAS_SUMMARY = CAS_ROOT / "registration_summary.csv"
METRICS = Path("E:/NASA_Rigidity/Reproducibility/registration_comparison/metrics.csv")
HR_CAS = CAS_ROOT / HR_SUB
HR_REF = REF_ROOT / HR_SUB

# `.absolute()` et surtout PAS `.resolve()` : ce dossier est une JONCTION vers E:.
SORTIE = Path(__file__).absolute().parent

COULEUR_TEXTE = "#c9c9c9"
GRILLE = "rgba(150,150,150,0.18)"
C_INT = "#2a78d6"
C_CT = "#e08a1e"
C_REF = "#c2453f"
C_REG_REF = "#8a94a6"
C_CAS = "#27a567"
SYMBOLE_OEIL = {"OD": "circle", "OS": "diamond"}

VARIANTES = [
    ("identite", "aucun recalage", "#6c757d"),
    ("registered_model1_fullframe", "U-Net + fullframe", "#c98b1a"),
    ("registered_model1_flatten_choroid_xcorr", "U-Net + chaîne actuelle", "#2a78d6"),
    ("registered_segformer2_flatten_choroid_xcorr", "SegFormer + chaîne actuelle", "#9b59b6"),
    ("cascade_v9", "cascade_v9 agrandi", C_CAS),
]
LIBELLE = {k: v for k, v, _ in VARIANTES}
COULEUR = {k: c for k, _, c in VARIANTES}

# Tolerances de lecture, et non de calcul : elles ne servent qu'a compter.
TOL_PRIOR = 0.10
TOL_ACCORD_BPM = 5.0
MARGE_BORD_BPM = 1.5

CONFIG = {"displaylogo": False, "responsive": True,
          "modeBarButtonsToRemove": ["select2d", "lasso2d"]}

FICHIERS: list[str] = []
RESUME: dict = {}
NOMS = ("fig_qualite_recalage", "tab_qualite", "fig_boxplot_hr", "fig_hr_paires",
        "tab_hr", "fig_traces")


# --------------------------------------------------------------------------- #
# Habillage commun (le meme que figures_repro_hr)
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
    entete = ("<!-- Genere par figures_repro_cascade/make_figures.py"
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
    s = pd.Series(s, dtype=float).dropna()
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
    a = np.asarray(a, dtype=float)
    s = np.nanstd(a)
    return (a - np.nanmean(a)) / s if s > 0 else a - np.nanmean(a)


def nom_participant(p):
    return str(p).replace("_", " ")


def absentes_toutes(pourquoi):
    for nom in NOMS:
        absente(nom, pourquoi)
    (SORTIE / "plotlyjs.html").write_text("", encoding="utf-8")
    raise SystemExit(0)


# --------------------------------------------------------------------------- #
# Donnees
# --------------------------------------------------------------------------- #
nl = chr(10)
manquants = [p for p in (CAS_SUMMARY, METRICS, HR_CAS / "conditions.csv",
                         HR_CAS / "traces.npz", HR_REF / "conditions.csv") if not p.exists()]
if manquants:
    absentes_toutes(
        "Le calcul n'a pas encore produit ces tables :" + nl + nl
        + nl.join(f"- `{p}`" for p in manquants) + nl + nl + "Lancer, dans l'ordre :"
        + nl + nl + "```bash" + nl
        + "python Reproducibility/register_newmodel.py --all --resize 1536x1024" + nl
        + "python Reproducibility/compute_registration_metrics.py" + nl
        + "OR_VARIANT=cascade_v9_1536x1024 OR_DATA_SUBDIR=registered_cascade_v9_1536x1024 "
          "python Reproducibility/compute_hr_bandpass.py" + nl
        + "python reveal_quarto_presentations/figures_repro_cascade/make_figures.py"
        + nl + "```")

SUM = pd.read_csv(CAS_SUMMARY)
MET = pd.read_csv(METRICS)
HC = pd.read_csv(HR_CAS / "conditions.csv")
HR = pd.read_csv(HR_REF / "conditions.csv")
HC = HC[HC["status"] == "ok"].copy()
HR = HR[HR["status"] == "ok"].copy()
TR = np.load(HR_CAS / "traces.npz", allow_pickle=True)
LO, HI = _BANDE
LARGEUR = HI - LO

# --- Qualite : une ligne par (acquisition, variante), meme critere partout -----
# Par dictionnaires et non par itertuples, qui renomme les colonnes contenant
# un point (``ncc_reg_n_lt_0.5``).
lignes = []
for r in SUM.to_dict("records"):
    ok = r.get("status") == "ok"
    lignes.append({"slug": r["slug"], "variante": "cascade_v9", "status": r.get("status"),
                   "ncc": r.get("ncc_reg_median") if ok else np.nan,
                   "n_lt": r.get("ncc_reg_n_lt_0.5") if ok else np.nan,
                   "p5": r.get("ncc_reg_p5") if ok else np.nan})
    lignes.append({"slug": r["slug"], "variante": "identite", "status": r.get("status"),
                   "ncc": r.get("ncc_identity_median") if ok else np.nan,
                   "n_lt": r.get("ncc_identity_n_lt_0.5") if ok else np.nan,
                   "p5": r.get("ncc_identity_p5") if ok else np.nan})
for r in MET.to_dict("records"):
    ok = r.get("status") == "ok"
    lignes.append({"slug": r["slug"], "variante": r["variant"], "status": r.get("status"),
                   "ncc": r.get("ncc_reg_median") if ok else np.nan,
                   "n_lt": r.get("ncc_reg_n_lt_0.5") if ok else np.nan,
                   "p5": r.get("ncc_reg_p5") if ok else np.nan})
Q = pd.DataFrame(lignes)
Q = Q[Q["variante"].isin(LIBELLE)]
QP = Q.pivot_table(index="slug", columns="variante", values="ncc")

RESUME["bande"] = f"{LO:.0f}-{HI:.0f} BPM"
RESUME["n_cascade_ok"] = f"{int((SUM['status'] == 'ok').sum())} / {len(SUM)}"
RESUME["cascade_echecs"] = "; ".join(
    f"{r['slug']} ({r['reason']})" for r in SUM[SUM["status"] != "ok"].to_dict("records")) or "aucun"


# --------------------------------------------------------------------------- #
# 1. Qualite du recalage
# --------------------------------------------------------------------------- #
def fig_qualite_recalage():
    fig = make_subplots(rows=1, cols=2, column_widths=[0.58, 0.42], horizontal_spacing=0.1,
                        subplot_titles=("NCC médiane par acquisition, par variante",
                                        "Différences appariées (cascade_v9 − autre)"))
    for k, (cle, lib, coul) in enumerate(VARIANTES):
        g = Q[(Q["variante"] == cle) & Q["ncc"].notna()]
        if g.empty:
            continue
        fig.add_trace(go.Box(x=[k] * len(g), y=g["ncc"], name=lib, marker={"color": coul},
                             line={"color": coul}, fillcolor="rgba(0,0,0,0)", width=0.5,
                             boxpoints=False, hoverinfo="skip", showlegend=False), 1, 1)
        fig.add_trace(go.Scatter(
            x=k + essaim(g["ncc"].to_numpy(), largeur=0.2), y=g["ncc"], mode="markers",
            marker={"color": coul, "size": 6, "opacity": 0.8}, showlegend=False,
            text=g["slug"], hovertemplate="%{text}<br>NCC %{y:.3f}<extra>" + lib + "</extra>"),
            1, 1)
    fig.update_xaxes(tickvals=list(range(len(VARIANTES))),
                     ticktext=[lib.replace(" + ", "<br>+ ") for _, lib, _ in VARIANTES],
                     tickfont={"size": 9}, row=1, col=1)
    fig.update_yaxes(title_text="NCC frame / médiane (médiane par acquisition)", row=1, col=1)

    autres = [c for c in ("identite", "registered_model1_fullframe",
                          "registered_model1_flatten_choroid_xcorr") if c in QP.columns]
    for k, cle in enumerate(autres):
        if "cascade_v9" not in QP.columns:
            break
        d = (QP["cascade_v9"] - QP[cle]).dropna()
        fig.add_trace(go.Box(x=[k] * len(d), y=d, marker={"color": COULEUR[cle]},
                             line={"color": COULEUR[cle]}, fillcolor="rgba(0,0,0,0)",
                             width=0.5, boxpoints=False, hoverinfo="skip",
                             showlegend=False), 1, 2)
        fig.add_trace(go.Scatter(
            x=k + essaim(d.to_numpy(), largeur=0.2), y=d, mode="markers",
            marker={"color": COULEUR[cle], "size": 6, "opacity": 0.8}, showlegend=False,
            text=d.index, hovertemplate="%{text}<br>Δ NCC %{y:+.3f}<extra></extra>"), 1, 2)
        RESUME[f"delta_vs_{cle}"] = (f"{med_iqr(d, 3)} ; cascade meilleure sur "
                                     f"{int((d > 0).sum())} / {len(d)}")
    fig.add_hline(y=0, line={"color": "rgba(150,150,150,0.6)", "dash": "dot"}, row=1, col=2)
    fig.update_xaxes(tickvals=list(range(len(autres))),
                     ticktext=[f"− {LIBELLE[c]}".replace(" + ", "<br>+ ") for c in autres],
                     tickfont={"size": 9}, row=1, col=2)
    fig.update_yaxes(title_text="Δ NCC médiane", row=1, col=2)
    fig = mise_en_page(fig, "Qualité du recalage : même critère pour toutes les variantes",
                       hauteur=520, legende=False)
    fig.update_layout(margin={"b": 90})
    enregistrer(fig, "fig_qualite_recalage")


def tab_qualite():
    lignes_t = ["| variante | acquisitions ok | NCC médiane [IQR] | p5 médian "
                "| acquisitions avec ≥ 1 frame < 0,5 | frames < 0,5 (total) |",
                "|:--|--:|:--|--:|--:|--:|"]
    for cle, lib, _ in VARIANTES:
        g = Q[Q["variante"] == cle]
        ok = g[g["status"] == "ok"]
        lignes_t.append(
            f"| {lib} | {len(ok)} / {len(g)} | {med_iqr(ok['ncc'], 3)} "
            f"| {fr(ok['p5'].median(), 3)} | {int((ok['n_lt'] > 0).sum())} "
            f"| {int(ok['n_lt'].sum())} |")
        RESUME[f"ncc_{cle}"] = f"{med_iqr(ok['ncc'], 3)} (n = {len(ok)})"
    ecrire_table(nl.join(lignes_t) + nl + nl
                 + ": NCC de chaque frame recalée à la médiane temporelle, sur la bande "
                   "de choroïde, dans la géométrie propre à chaque variante ; puis "
                   "médiane par acquisition. « aucun recalage » : frames natives et masque "
                   "SegFormer, même filtre de colonnes que cascade_v9. {.striped}",
                 "tab_qualite")


fig_qualite_recalage()
tab_qualite()


# --------------------------------------------------------------------------- #
# 2. La FC, sur les deux recalages
# --------------------------------------------------------------------------- #
COMMUNS = sorted(set(HC["slug"]) & set(HR["slug"]))
HC_c = HC[HC["slug"].isin(COMMUNS)].sort_values("slug").reset_index(drop=True)
HR_c = HR[HR["slug"].isin(COMMUNS)].sort_values("slug").reset_index(drop=True)
PARTICIPANTS = sorted(HC_c["participant"].unique())
RESUME["n_fc_communs"] = len(COMMUNS)
RESUME["n_fc_participants"] = len(PARTICIPANTS)


def etendue_hasard(n):
    return LARGEUR * (n - 1) / (n + 1)


def p_proche_hasard(ref, tol=TOL_PRIOR):
    if not np.isfinite(ref):
        return 0.0
    return max(0.0, min((1 + tol) * ref, HI) - max((1 - tol) * ref, LO)) / LARGEUR


P_ACCORD_HASARD = 1.0 - (1.0 - min(TOL_ACCORD_BPM / LARGEUR, 1.0)) ** 2


def proche(v, ref, tol=TOL_PRIOR):
    return (np.abs(v - ref) <= tol * ref) & np.isfinite(ref)


def fig_boxplot_hr():
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("FC tirée des intensités (SVD)",
                                        "FC tirée de l'épaisseur choroïdienne"))
    dx = 0.2
    for rang, cle in ((1, "hr_intensite_BPM"), (2, "hr_epaisseur_BPM")):
        for D, lib, coul, dec in ((HR_c, "recalage de référence", C_REG_REF, -dx),
                                  (HC_c, "cascade_v9 agrandi", C_CAS, +dx)):
            xs = D["participant"].map({p: k for k, p in enumerate(PARTICIPANTS)}) + dec
            fig.add_trace(go.Box(x=xs, y=D[cle], name=lib, legendgroup=lib,
                                 showlegend=(rang == 1), width=0.34,
                                 marker={"color": coul}, line={"color": coul, "width": 1.4},
                                 fillcolor="rgba(0,0,0,0)", boxpoints=False,
                                 hoverinfo="skip"), rang, 1)
            for oeil, symb in SYMBOLE_OEIL.items():
                g = D[D["eye"] == oeil]
                if g.empty:
                    continue
                x = np.empty(len(g))
                for p in g["participant"].unique():
                    sel = (g["participant"] == p).to_numpy()
                    x[sel] = PARTICIPANTS.index(p) + dec + essaim(g.loc[sel, cle].to_numpy())
                fig.add_trace(go.Scatter(
                    x=x, y=g[cle], mode="markers", legendgroup=lib, showlegend=False,
                    marker={"color": coul, "symbol": symb, "size": 7, "opacity": 0.85,
                            "line": {"color": "rgba(20,20,20,0.6)", "width": 0.6}},
                    text=g["slug"],
                    hovertemplate="%{text}<br>" + lib + " : %{y:.1f} BPM<extra></extra>"),
                    rang, 1)
        xs_p, ys_p = [], []
        for k, p in enumerate(PARTICIPANTS):
            v = HC_c.loc[HC_c["participant"] == p, "hr_prior_BPM"].median()
            if np.isfinite(v):
                xs_p += [k - 0.45, k + 0.45, None]
                ys_p += [v, v, None]
        fig.add_trace(go.Scatter(x=xs_p, y=ys_p, mode="lines", name="a priori (consensus)",
                                 legendgroup="prior", showlegend=(rang == 1),
                                 line={"color": C_REF, "width": 2, "dash": "dash"},
                                 hovertemplate="a priori %{y:.1f} BPM<extra></extra>"),
                      rang, 1)
        for b in (LO, HI):
            fig.add_hline(y=b, line={"color": "rgba(150,150,150,0.6)", "width": 1,
                                     "dash": "dot"}, row=rang, col=1)
        fig.update_yaxes(title_text="FC (BPM)", row=rang, col=1,
                         range=[min(LO, HC_c["hr_prior_BPM"].min()) - 5,
                                max(HI, HC_c["hr_prior_BPM"].max()) + 5])
    for oeil, symb in SYMBOLE_OEIL.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=oeil,
                                 marker={"color": COULEUR_TEXTE, "symbol": symb, "size": 8}))
    fig.update_xaxes(tickvals=list(range(len(PARTICIPANTS))),
                     ticktext=[nom_participant(p).replace(" ", "<br>", 1) for p in PARTICIPANTS],
                     range=[-0.6, len(PARTICIPANTS) - 0.4], tickangle=0,
                     tickfont={"size": 10}, row=2, col=1)
    fig = mise_en_page(fig, f"FC par participant, sans a priori (FIR {LO:.0f}–{HI:.0f} BPM)"
                            "<br>un point = une acquisition", hauteur=820)
    fig.update_layout(margin={"t": 90, "b": 110}, legend={"y": -0.09})
    enregistrer(fig, "fig_boxplot_hr")


def fig_hr_paires():
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=("Intensités (SVD)", "Épaisseur choroïdienne"))
    M = HR_c.merge(HC_c, on="slug", suffixes=("_ref", "_cas"))
    for col, (cle, coul) in enumerate((("hr_intensite_BPM", C_INT),
                                        ("hr_epaisseur_BPM", C_CT)), start=1):
        x, y = M[f"{cle}_ref"], M[f"{cle}_cas"]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers", showlegend=False,
            marker={"color": coul, "size": 7, "opacity": 0.8},
            text=M["slug"],
            hovertemplate="%{text}<br>référence %{x:.1f} BPM<br>cascade_v9 %{y:.1f} BPM"
                          "<extra></extra>"), 1, col)
        fig.add_trace(go.Scatter(x=[LO, HI], y=[LO, HI], mode="lines", showlegend=False,
                                 line={"color": "rgba(150,150,150,0.6)", "dash": "dot"},
                                 hoverinfo="skip"), 1, col)
        fig.update_xaxes(title_text="FC, recalage de référence (BPM)", range=[LO - 2, HI + 2],
                         row=1, col=col)
        fig.update_yaxes(title_text="FC, cascade_v9 agrandi (BPM)", range=[LO - 2, HI + 2],
                         row=1, col=col)
        d = np.abs(y - x)
        court = "int" if "intensite" in cle else "ct"
        RESUME[f"fc_{court}_identique_5bpm"] = f"{int((d <= TOL_ACCORD_BPM).sum())} / {len(d)}"
        RESUME[f"fc_{court}_ecart_abs"] = med_iqr(d)
    fig = mise_en_page(fig, "La même acquisition, deux recalages : la FC trouvée change-t-elle ?",
                       hauteur=460, legende=False)
    enregistrer(fig, "fig_hr_paires")


def indicateurs(D):
    prior = D["hr_prior_BPM"].to_numpy(float)
    out = {}
    for cle, court in (("hr_intensite_BPM", "int"), ("hr_epaisseur_BPM", "ct")):
        v = D[cle].to_numpy(float)
        out[f"{court}_proche"] = int(proche(v, prior).sum())
        out[f"{court}_etendue_part"] = D.groupby("participant")[cle].agg(
            lambda s: s.max() - s.min()).median()
        out[f"{court}_etendue_oeil"] = D.groupby(["participant", "eye"])[cle].agg(
            lambda s: s.max() - s.min()).median()
        out[f"{court}_bord"] = int(((v <= LO + MARGE_BORD_BPM) | (v >= HI - MARGE_BORD_BPM)).sum())
    out["accord"] = int((np.abs(D["hr_intensite_BPM"] - D["hr_epaisseur_BPM"])
                         <= TOL_ACCORD_BPM).sum())
    return out


def tab_hr():
    n = len(COMMUNS)
    R, C = indicateurs(HR_c), indicateurs(HC_c)
    prior = HC_c["hr_prior_BPM"].to_numpy(float)
    h_proche = sum(p_proche_hasard(p) for p in prior)
    h_part = np.median([etendue_hasard(k) for k in HC_c.groupby("participant").size()])
    h_oeil = np.median([etendue_hasard(k) for k in HC_c.groupby(["participant", "eye"]).size()])
    t = ["| | référence, intensité | référence, épaisseur | cascade_v9, intensité "
         "| cascade_v9, épaisseur | au hasard |",
         "|:--|--:|--:|--:|--:|--:|",
         f"| à ± 10 % de l'a priori | {R['int_proche']} / {n} | {R['ct_proche']} / {n} "
         f"| {C['int_proche']} / {n} | {C['ct_proche']} / {n} | {fr(h_proche)} / {n} |",
         f"| étendue par participant (BPM, médiane) | {fr(R['int_etendue_part'])} "
         f"| {fr(R['ct_etendue_part'])} | {fr(C['int_etendue_part'])} "
         f"| {fr(C['ct_etendue_part'])} | {fr(h_part)} |",
         f"| étendue par œil (BPM, médiane) | {fr(R['int_etendue_oeil'])} "
         f"| {fr(R['ct_etendue_oeil'])} | {fr(C['int_etendue_oeil'])} "
         f"| {fr(C['ct_etendue_oeil'])} | {fr(h_oeil)} |",
         f"| pic collé à une borne | {R['int_bord']} / {n} | {R['ct_bord']} / {n} "
         f"| {C['int_bord']} / {n} | {C['ct_bord']} / {n} | — |",
         f"| intensité et épaisseur à ± 5 BPM | {R['accord']} / {n} | | {C['accord']} / {n} "
         f"| | {fr(P_ACCORD_HASARD * n)} / {n} |"]
    ecrire_table(nl.join(t) + nl + nl
                 + f": FC sans a priori, FIR fixe {LO:.0f}–{HI:.0f} BPM, sur les {n} "
                   "acquisitions communes aux deux recalages. « au hasard » : pics tirés "
                   "uniformément dans la bande. {.striped}", "tab_hr")
    for nom, d in (("ref", R), ("cas", C)):
        for k, v in d.items():
            RESUME[f"hr_{nom}_{k}"] = fr(v) if isinstance(v, float) else v
    RESUME["hr_hasard_proche"] = fr(h_proche)
    RESUME["hr_hasard_etendue_part"] = fr(h_part)
    RESUME["hr_hasard_etendue_oeil"] = fr(h_oeil)
    RESUME["hr_hasard_accord"] = fr(P_ACCORD_HASARD * n)


def fig_traces():
    slugs = [s for s in HC_c["slug"] if f"{s}__u_time" in TR.files]
    if not slugs:
        absente("fig_traces", "`traces.npz` ne contient aucune acquisition.")
        return
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        subplot_titles=("Signaux BRUTS (centrés-réduits)",
                                        f"Après le FIR {LO:.0f}–{HI:.0f} BPM (centrés-réduits)"))
    titres, formes = [], []
    for i, s in enumerate(slugs):
        r = HC_c[HC_c["slug"] == s].iloc[0]
        t, u = TR[f"{s}__t"], TR[f"{s}__u_time"]
        core = TR[f"{s}__core"].astype(bool)
        for rang, (x, y_int, y_ct) in enumerate(
                [(t, TR[f"{s}__int_brut"], TR[f"{s}__ct_brut_um"]),
                 (u, TR[f"{s}__int_filt"], TR[f"{s}__ct_filt_um"])]):
            fig.add_trace(go.Scatter(
                x=x, y=norm(y_int), mode="lines", visible=(i == 0),
                name="intensité (SVD)", legendgroup="int", showlegend=(rang == 0),
                line={"color": C_INT, "width": 1.3},
                hovertemplate="%{x:.2f} s — %{y:.2f}<extra>intensité</extra>"), rang + 1, 1)
            fig.add_trace(go.Scatter(
                x=x, y=norm(y_ct), mode="lines", visible=(i == 0),
                name="épaisseur choroïdienne", legendgroup="ct", showlegend=(rang == 0),
                line={"color": C_CT, "width": 1.3},
                hovertemplate="%{x:.2f} s — %{y:.2f}<extra>épaisseur</extra>"), rang + 1, 1)
        u0, u1 = float(u[core][0]), float(u[core][-1])
        grise = {"type": "rect", "xref": "x2", "yref": "y2 domain", "y0": 0, "y1": 1,
                 "fillcolor": "rgba(150,150,150,0.15)", "line": {"width": 0}, "layer": "below"}
        formes.append([{**grise, "x0": float(u[0]), "x1": u0},
                       {**grise, "x0": u1, "x1": float(u[-1])}])
        titres.append(f"{s}<br>intensité {fr(r['hr_intensite_BPM'])} BPM · "
                      f"épaisseur {fr(r['hr_epaisseur_BPM'])} BPM · "
                      f"a priori {fr(r['hr_prior_BPM'])} BPM")
    boutons = []
    for i, s in enumerate(slugs):
        vis = [False] * (4 * len(slugs))
        vis[4 * i:4 * i + 4] = [True] * 4
        boutons.append({"label": s, "method": "update",
                        "args": [{"visible": vis}, {"title.text": titres[i],
                                                    "shapes": formes[i]}]})
    fig.update_xaxes(title_text="temps (s)", row=2, col=1)
    fig.update_yaxes(title_text="écart-type", row=1, col=1)
    fig.update_yaxes(title_text="écart-type", row=2, col=1)
    fig = mise_en_page(fig, titres[0], hauteur=600)
    fig.update_layout(
        shapes=formes[0], margin={"t": 110},
        updatemenus=[{"buttons": boutons, "direction": "down", "showactive": True,
                      "x": 1.0, "xanchor": "right", "y": 1.0, "yanchor": "bottom",
                      "pad": {"b": 28}, "bgcolor": "rgba(120,120,120,0.25)",
                      "bordercolor": GRILLE, "font": {"size": 11, "color": "#222"}}])
    enregistrer(fig, "fig_traces")


fig_boxplot_hr()
fig_hr_paires()
tab_hr()
fig_traces()

(SORTIE / "plotlyjs.html").write_text(
    '<script type="text/javascript">' + nl + get_plotlyjs() + nl + "</script>" + nl,
    encoding="utf-8")
print("  plotlyjs.html")
(SORTIE / "resume.txt").write_text(
    nl.join(f"{k} = {v}" for k, v in RESUME.items()) + nl, encoding="utf-8")
print("  resume.txt")
