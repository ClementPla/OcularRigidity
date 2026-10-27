# -*- coding: utf-8 -*-
"""
Figures -- page « Analyse de la pulsatilité » (section SANS).

Le k de Sayah et ses entrées, et leurs corrélations avec les paramètres de SANS.
Ce script ne recalcule NI k NI les issues : il relit les tables du lot
``Astronauts/compute_sans_predictors.py`` et n'en tire que des statistiques de
lecture (moyennes par oeil, r, p, q).

Entrees
-------
    E:/NASA_Rigidity/SegmentationVariations/model1_scale_1.0_flatten_choroid_xcorr/
        sans_predictors/rigidity_cache.csv     k, CT, dCT, dV, AL, IOP, OPA par acquisition
        sans_predictors/outcomes.csv           les six issues de SANS par oeil
        sans_predictors/repeatability.csv      ICC + composantes de variance
        sans_predictors/stability.csv          correlation avant / apres le vol
        sans_predictors/sans_correlations.csv  r et q publies (controle de coherence)

La moyenne par (oeil, moment) refait exactement celle du lot : chaque r des
quatre marqueurs de la famille ``rigidite`` est compare a ``sans_correlations.csv``
et le script s'arrete au moindre ecart.

Sorties : un fragment Quarto par figure ou table (``pk_*.qmd``), plotly.js une
seule fois dans ``plotlyjs.html``, et ``resume.json`` pour les chiffres de la prose.

Lancer DEPUIS LA RACINE DU DEPOT (ce dossier est une JONCTION vers E:) :
    python reveal_quarto_presentations/figures_sans_pulsatilite/make_figures.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots
from scipy import stats

VARIANT = Path("E:/NASA_Rigidity/SegmentationVariations/model1_scale_1.0_flatten_choroid_xcorr")
SANS = VARIANT / "sans_predictors"
# `.absolute()` et PAS `.resolve()` : la jonction ferait sortir SORTIE du depot.
SORTIE = Path(__file__).absolute().parent

# --- Marqueurs : la chaine de k, puis ses entrees cliniques -------------------
MARQUEURS = {
    "k_sayah": "k de Sayah (mm⁻³)",
    "delta_V_mm3": "ΔV (mm³)",
    "delta_CT_mm": "ΔCT (mm)",
    "CT_mm": "épaisseur choroïdienne CT (mm)",
    "AL_mm": "longueur axiale AL (mm)",
    "IOP_mmHg": "PIO (mmHg)",
    "OPA_mmHg": "OPA (mmHg)",
}
COURT = {"k_sayah": "k", "delta_V_mm3": "ΔV", "delta_CT_mm": "ΔCT", "CT_mm": "CT",
         "AL_mm": "AL", "IOP_mmHg": "PIO", "OPA_mmHg": "OPA"}
CHAINE_K = ["k_sayah", "delta_V_mm3", "delta_CT_mm", "CT_mm"]   # famille ``rigidite`` du lot
ISSUES = {"delta_TRT_S": "ΔTRT S", "delta_TRT_I": "ΔTRT I", "delta_TRT_N": "ΔTRT N",
          "delta_TRT_T": "ΔTRT T", "delta_TRT_moyen": "ΔTRT moy.", "delta_AL_mm": "ΔAL"}
UNITE_ISSUE = {k: "µm" for k in ISSUES} | {"delta_AL_mm": "mm"}
VUES = {"before": "avant le vol", "after": "après le vol", "delta": "après − avant"}
MOMENTS = {"before": "avant", "post": "après"}
AL_FACTOR = 0.87

# --- Habillage (memes conventions que figures_sans_predictors) ----------------
COULEUR_TEXTE = "#c9c9c9"
GRILLE = "rgba(150,150,150,0.18)"
C_POINT = "#2a78d6"
C_AJUST = "#eb6834"
C_MOMENT = {"before": "#2a78d6", "post": "#eb6834"}
C_TERME = {"ΔCT": "#2a78d6", "OPA / PIO": "#27a567", "AL + CT": "#d0a72b"}
CONFIG = {"displaylogo": False, "responsive": True,
          "modeBarButtonsToRemove": ["select2d", "lasso2d"]}
RESUME: dict = {}


def mise_en_page(fig, titre, hauteur=440, legende=True):
    fig.update_layout(
        title={"text": titre, "font": {"size": 14}, "x": 0.01, "xanchor": "left"},
        template="none", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COULEUR_TEXTE, "size": 11}, height=hauteur,
        margin={"l": 70, "r": 20, "t": 80, "b": 90 if legende else 50},
        showlegend=legende,
        legend={"font": {"size": 10}, "orientation": "h", "yanchor": "top",
                "y": -0.12, "xanchor": "left", "x": 0},
        hoverlabel={"font": {"size": 11}},
    )
    fig.update_xaxes(gridcolor=GRILLE, zerolinecolor=GRILLE, linecolor=GRILLE,
                     ticks="outside", tickcolor=GRILLE)
    fig.update_yaxes(gridcolor=GRILLE, zerolinecolor=GRILLE, linecolor=GRILLE,
                     ticks="outside", tickcolor=GRILLE)
    for a in fig.layout.annotations or ():
        if a.font is not None:
            a.font.size = 11
    return fig


def enregistrer(fig, nom):
    html = fig.to_html(full_html=False, include_plotlyjs=False, config=CONFIG,
                       div_id=f"plot-{nom}")
    (SORTIE / f"{nom}.qmd").write_text("```{=html}\n" + html + "\n```\n", encoding="utf-8")
    print(f"  {nom}.qmd")


def ecrire_table(texte, nom):
    entete = ("<!-- Genere par figures_sans_pulsatilite/make_figures.py"
              " -- ne pas editer a la main. -->\n\n")
    (SORTIE / f"{nom}.qmd").write_text(entete + texte + "\n", encoding="utf-8")
    print(f"  {nom}.qmd")


def virgule(x, chiffres=2, signe=False):
    if x is None or not np.isfinite(x):
        return "n/a"
    fmt = f"{{:+.{chiffres}f}}" if signe else f"{{:.{chiffres}f}}"
    return fmt.format(x).replace(".", ",")


def bh_fdr(p):
    """q de Benjamini-Hochberg, les p non finis ecartes du decompte."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.size, np.nan)
    fini = np.isfinite(p)
    if not fini.any():
        return out
    pf = p[fini]
    n = pf.size
    order = np.argsort(pf)
    ranked = np.minimum.accumulate((pf[order] * n / (np.arange(n) + 1))[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(ranked, 0.0, 1.0)
    out[fini] = q
    return out


def correl(x, y):
    d = pd.concat([pd.Series(x, name="x"), pd.Series(y, name="y")], axis=1).dropna()
    n = len(d)
    if n < 4 or d.x.nunique() < 2 or d.y.nunique() < 2:
        return {"n": n, "r": np.nan, "p": np.nan, "rho": np.nan, "p_rho": np.nan}
    r, p = stats.pearsonr(d.x, d.y)
    rho, p_rho = stats.spearmanr(d.x, d.y)
    return {"n": n, "r": float(r), "p": float(p), "rho": float(rho), "p_rho": float(p_rho)}


# --------------------------------------------------------------------------- #
# Chargement
# --------------------------------------------------------------------------- #
print("lecture des tables du lot ...")
ACQ = pd.read_csv(SANS / "rigidity_cache.csv").rename(columns={"k": "k_sayah", "delta_V": "delta_V_mm3"})
ids = ACQ["slug"].str.extract(
    r"^(?P<astro>\d+_\d+)__\d+(?P<moment>before|post)_rigidity_(?P<eye>O[DS])(?P<rep>\d*)$")
if ids.isna().any(axis=None):
    raise SystemExit(f"slug illisible : {ACQ.loc[ids.isna().any(axis=1), 'slug'].tolist()}")
ACQ = pd.concat([ACQ, ids], axis=1)
ACQ["subject"] = ACQ["astro"] + "_" + ACQ["eye"]
OUT = pd.read_csv(SANS / "outcomes.csv").set_index("subject")
REP = pd.read_csv(SANS / "repeatability.csv")
STAB = pd.read_csv(SANS / "stability.csv")
PUB = pd.read_csv(SANS / "sans_correlations.csv")
PUB = PUB[PUB["famille"] == "rigidite"]

# Une valeur par (oeil, moment) : moyenne des repliques, comme le lot.
PAR_OEIL = ACQ.groupby(["subject", "moment"])[list(MARQUEURS)].mean().unstack("moment")


def vue_de(marqueur, vue):
    v = PAR_OEIL[marqueur]
    if vue == "before":
        return v.get("before")
    if vue == "after":
        return v.get("post")
    return v.get("post") - v.get("before")


# --------------------------------------------------------------------------- #
# Correlations : 7 marqueurs x 6 issues x 3 vues
# --------------------------------------------------------------------------- #
rows = []
for vue in VUES:
    for m in MARQUEURS:
        x = vue_de(m, vue)
        for iss in ISSUES:
            # AL apres - AVANT EST l'issue dAL : la ligne serait une tautologie
            # (r = 1 contre dAL) et ne mesure rien contre les dTRT que la carte
            # des issues entre elles ne montre deja.
            c = (correl([], []) if (m == "AL_mm" and vue == "delta") else correl(x, OUT[iss]))
            rows.append({"marqueur": m, "vue": vue, "issue": iss, **c})
COR = pd.DataFrame(rows)
# q de la PAGE : dans chaque vue, sur ses tests effectifs (42, 36 pour apres - avant).
COR["q_page"] = COR.groupby("vue")["p"].transform(lambda p: bh_fdr(p.to_numpy()))

# Controle : la chaine de k redonne au bit pres les r publies par le lot.
ctl = COR[COR["marqueur"].isin(CHAINE_K)].merge(
    PUB[["predicteur", "vue", "issue", "pearson_r", "n_sujets", "pearson_q_categorie"]],
    left_on=["marqueur", "vue", "issue"], right_on=["predicteur", "vue", "issue"], how="left")
ecart = np.nanmax(np.abs(ctl["r"] - ctl["pearson_r"]))
if not (ecart < 1e-6 and (ctl["n"] == ctl["n_sujets"]).all()):
    raise SystemExit(f"ecart avec sans_correlations.csv : {ecart:.2e}")
COR = COR.merge(ctl[["marqueur", "vue", "issue", "pearson_q_categorie"]],
                on=["marqueur", "vue", "issue"], how="left")
print(f"  controle : r identiques a sans_correlations.csv (ecart max {ecart:.1e})")

# --------------------------------------------------------------------------- #
# Effectifs, valeurs de k
# --------------------------------------------------------------------------- #
n_k = ACQ.groupby("moment")["k_sayah"].agg(lambda s: int(s.notna().sum()))
RESUME["effectifs"] = {
    "acquisitions": int(len(ACQ)), "acquisitions_k": int(ACQ["k_sayah"].notna().sum()),
    "acq_par_moment": {MOMENTS[m]: int(v) for m, v in ACQ["moment"].value_counts().items()},
    "acq_k_par_moment": {MOMENTS[m]: int(v) for m, v in n_k.items()},
    "yeux": int(ACQ["subject"].nunique()), "astronautes": int(ACQ["astro"].nunique()),
    "yeux_sans_AL": sorted(ACQ.loc[ACQ["AL_mm"].isna(), "subject"].unique().tolist()),
}
RESUME["k_acquisitions"] = {
    MOMENTS[m]: {"mediane": float(g.median()), "q1": float(g.quantile(.25)),
                 "q3": float(g.quantile(.75)), "min": float(g.min()), "max": float(g.max())}
    for m, g in ACQ.groupby("moment")["k_sayah"]}

# --- Figure 1 : k par oeil, avant et apres, yeux relies -----------------------
k_oeil = PAR_OEIL["k_sayah"].rename(columns=MOMENTS)
pairs = k_oeil.dropna()
w = stats.wilcoxon(pairs["après"], pairs["avant"]) if len(pairs) >= 5 else None
RESUME["k_avant_apres"] = {
    "n_paires": int(len(pairs)),
    "mediane_diff": float((pairs["après"] - pairs["avant"]).median()),
    "wilcoxon_p": float(w.pvalue) if w else None,
    "n_hausse": int((pairs["après"] > pairs["avant"]).sum()),
}
fig = go.Figure()
for subj, r in k_oeil.iterrows():
    if r.notna().all():
        fig.add_trace(go.Scatter(x=["avant", "après"], y=r.values * 1e3, mode="lines",
                                 line={"color": "rgba(150,150,150,0.45)", "width": 1},
                                 hoverinfo="skip", showlegend=False))
for m, lab in MOMENTS.items():
    y = PAR_OEIL["k_sayah"].get(m).dropna()
    acq = ACQ[(ACQ["moment"] == m)]
    fig.add_trace(go.Box(x=[lab] * len(y), y=y.values * 1e3, name=f"{lab} le vol (n = {len(y)} yeux)",
                         marker={"color": C_MOMENT[m], "size": 8}, line={"color": C_MOMENT[m], "width": 2},
                         fillcolor="rgba(0,0,0,0)", boxpoints="all", jitter=0.25, pointpos=0,
                         customdata=np.stack([y.index, acq.groupby("subject").size().reindex(y.index).values], -1),
                         hovertemplate=("%{customdata[0]}<br>k = %{y:.1f} × 10⁻³ mm⁻³"
                                        "<br>%{customdata[1]} acquisition(s)<extra></extra>")))
fig.update_yaxes(title_text="k de Sayah (× 10⁻³ mm⁻³)", rangemode="tozero")
mise_en_page(fig, "k de Sayah par œil (moyenne des répliques) ; traits gris : le même œil avant et après",
             hauteur=460)
enregistrer(fig, "pk_k_avant_apres")

# --- Table : valeurs des entrees ------------------------------------------------
lignes = ["| grandeur | avant le vol | après le vol |", "|:--|--:|--:|"]
for m, lab in MARQUEURS.items():
    cel = []
    for mo in MOMENTS:
        s = ACQ.loc[ACQ["moment"] == mo, m].dropna()
        f = 1e3 if m == "k_sayah" else 1.0
        ch = 3 if m in ("CT_mm", "delta_CT_mm") else 1
        cel.append(f"{virgule(s.median() * f, ch)} [{virgule(s.quantile(.25) * f, ch)} – "
                   f"{virgule(s.quantile(.75) * f, ch)}] (n = {len(s)})")
    lab = lab.replace("(mm⁻³)", "(× 10⁻³ mm⁻³)")
    lignes.append(f"| {lab} | " + " | ".join(cel) + " |")
ecrire_table("Médiane [IQR] par **acquisition** ; n = acquisitions où la grandeur est définie.\n\n"
             + "\n".join(lignes), "pk_tab_valeurs")

# --------------------------------------------------------------------------- #
# Ce qui fait varier k : decomposition de la variance de ln k
# --------------------------------------------------------------------------- #
# ln k = ln ln(1 + OPA/PIO) - ln(pi/2) - 2 ln(0,87 AL + CT) - ln dCT : la variance
# de ln k se partage EXACTEMENT entre les trois termes, chacun comptant
# cov(terme, ln k) / var(ln k) (la somme vaut 1, termes croises compris).
ok = ACQ["k_sayah"].notna()
A = ACQ[ok]
termes = {
    "ΔCT": -np.log(A["delta_CT_mm"]),
    "OPA / PIO": np.log(np.log1p(A["OPA_mmHg"] / A["IOP_mmHg"])),
    "AL + CT": -2 * np.log(AL_FACTOR * A["AL_mm"] + A["CT_mm"]),
}
lnk = np.log(A["k_sayah"])
recon = sum(termes.values()) - np.log(np.pi / 2)
if not np.allclose(recon, lnk, atol=1e-6):
    raise SystemExit("la decomposition de ln k ne redonne pas k : formule a revoir")
part = {t: float(np.cov(v, lnk)[0, 1] / np.var(lnk, ddof=1)) for t, v in termes.items()}
RESUME["part_variance_lnk"] = part
RESUME["cv_termes"] = {t: float(np.std(v, ddof=1)) for t, v in termes.items()}
fig = make_subplots(rows=1, cols=2, column_widths=[0.38, 0.62], horizontal_spacing=0.14,
                    subplot_titles=["part de la variance de ln k", "k contre ΔCT (axes log)"])
ordre = sorted(part, key=part.get)
fig.add_trace(go.Bar(y=ordre, x=[part[t] * 100 for t in ordre], orientation="h",
                     marker={"color": [C_TERME[t] for t in ordre], "cornerradius": 4},
                     text=[f"{part[t] * 100:.0f} %" for t in ordre], textposition="outside",
                     hovertemplate="%{y} : %{x:.1f} % de var(ln k)<extra></extra>", showlegend=False),
              row=1, col=1)
for m, lab in MOMENTS.items():
    a = A[A["moment"] == m]
    fig.add_trace(go.Scatter(x=a["delta_CT_mm"] * 1e3, y=a["k_sayah"] * 1e3, mode="markers",
                             name=f"acquisitions {lab} le vol",
                             marker={"color": C_MOMENT[m], "size": 8, "line": {"width": 1, "color": "rgba(0,0,0,0)"}},
                             customdata=np.stack([a["slug"], a["OPA_mmHg"], a["IOP_mmHg"], a["AL_mm"]], -1),
                             hovertemplate=("%{customdata[0]}<br>ΔCT %{x:.1f} µm · k %{y:.1f} × 10⁻³ mm⁻³"
                                            "<br>OPA %{customdata[1]} / PIO %{customdata[2]} mmHg"
                                            " · AL %{customdata[3]} mm<extra></extra>")),
                  row=1, col=2)
xx = np.geomspace(A["delta_CT_mm"].min(), A["delta_CT_mm"].max(), 50)
k_typ = np.exp(np.median(lnk + np.log(A["delta_CT_mm"])))       # k x dCT median
fig.add_trace(go.Scatter(x=xx * 1e3, y=k_typ / xx * 1e3, mode="lines", name="k ∝ 1 / ΔCT (autres termes à leur médiane)",
                         line={"color": "#9a9a9a", "dash": "dot", "width": 2}, hoverinfo="skip"),
              row=1, col=2)
fig.update_xaxes(title_text="% de var(ln k)", range=[0, max(part.values()) * 118], row=1, col=1)
fig.update_xaxes(type="log", title_text="ΔCT (µm)", row=1, col=2)
fig.update_yaxes(type="log", title_text="k (× 10⁻³ mm⁻³)", row=1, col=2)
mise_en_page(fig, f"Ce qui fait varier k ({int(ok.sum())} acquisitions où k est défini)", hauteur=500)
fig.update_layout(legend={"y": -0.2}, margin={"b": 120})
enregistrer(fig, "pk_decomposition")

# --------------------------------------------------------------------------- #
# Repetabilite (court terme) et stabilite (avant / apres le vol)
# --------------------------------------------------------------------------- #
lignes = ["| marqueur | n yeux | ICC(A,1) [IC95] | CV intra | r avant / après (n) |",
          "|:--|--:|--:|--:|--:|"]
RESUME["repetabilite"] = {}
for m in CHAINE_K:
    r = REP[(REP["famille"] == "rigidite") & (REP["predicteur"] == m) & (REP["moment"] == "before")]
    a = r[(r["icc_type"] == "ICC(A,1)") & (r["pair"] == "1v2")].iloc[0]
    v = r[r["icc_type"] == "variance"].iloc[0]
    s = STAB[(STAB["famille"] == "rigidite") & (STAB["predicteur"] == m)].iloc[0]
    RESUME["repetabilite"][m] = {"icc_A1": float(a["icc"]), "cv_intra_pct": float(v["cv_intra_pct"]),
                                 "r_stabilite": float(s["pearson_r"]), "rho_stabilite": float(s["spearman_rho"])}
    lignes.append(f"| {MARQUEURS[m]} | {int(a['n_subjects'])} | **{virgule(a['icc'])}** "
                  f"[{virgule(a['icc_ci95_low'])} ; {virgule(a['icc_ci95_high'])}] | "
                  f"{v['cv_intra_pct']:.0f} % | {virgule(s['pearson_r'])} ({int(s['n_sujets'])}) |")
ecrire_table("\n".join(lignes), "pk_tab_repetabilite")

# --------------------------------------------------------------------------- #
# Carte des correlations
# --------------------------------------------------------------------------- #
fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.025,
                    subplot_titles=[f"vue : {v}" for v in VUES.values()])
lignes_y = [COURT[m] for m in MARQUEURS]
for c_, vue in enumerate(VUES, start=1):
    d = COR[COR["vue"] == vue]
    z = d.pivot(index="marqueur", columns="issue", values="r").reindex(index=list(MARQUEURS), columns=list(ISSUES))
    t = d.set_index(["marqueur", "issue"])
    txt, cd = [], []
    for m in MARQUEURS:
        lt, lc = [], []
        for iss in ISSUES:
            e = t.loc[(m, iss)]
            etoile = "*" if e["q_page"] < 0.05 else ""
            lt.append("" if not np.isfinite(e["r"]) else f"{e['r']:+.2f}{etoile}".replace(".", ","))
            lc.append([e["n"], e["p"], e["q_page"], e["rho"],
                       e["pearson_q_categorie"] if np.isfinite(e["pearson_q_categorie"]) else -1])
        txt.append(lt)
        cd.append(lc)
    fig.add_trace(go.Heatmap(
        z=z.values, x=list(ISSUES.values()), y=lignes_y, zmin=-1, zmax=1, zmid=0,
        colorscale="RdBu_r", showscale=(c_ == 3), xgap=2, ygap=2,
        colorbar={"title": {"text": "r"}, "len": 0.8, "thickness": 12},
        text=txt, texttemplate="%{text}", textfont={"size": 10}, customdata=np.array(cd, dtype=float),
        hovertemplate=(f"{VUES[vue]}<br>%{{y}} ~ %{{x}}<br>r = %{{z:.2f}} · ρ = %{{customdata[3]:.2f}}"
                       "<br>n = %{customdata[0]} yeux · p = %{customdata[1]:.3g}"
                       "<br>q (page, tests de la vue) = %{customdata[2]:.3g}"
                       "<br>q (catégorie du lot ; −1 : hors lot) = %{customdata[4]:.3g}<extra></extra>")),
        row=1, col=c_)
fig.update_yaxes(autorange="reversed", row=1, col=1)
fig.add_hline(y=3.5, line={"color": COULEUR_TEXTE, "width": 1, "dash": "dot"})
mise_en_page(fig, "Pearson r entre la chaîne de k (haut) / ses entrées cliniques (bas) et les issues de SANS "
                  "· * : q < 0,05 (BH, par vue)", hauteur=430, legende=False)
enregistrer(fig, "pk_heatmap")

# --- Table des correlations qui passent un seuil ------------------------------
sel = COR[(COR["p"] < 0.05)].sort_values(["vue", "p"])
lignes = ["| vue | marqueur | issue | n | r | ρ | p | q (page) | q (catégorie du lot) |",
          "|:--|:--|:--|--:|--:|--:|--:|--:|--:|"]
for _, e in sel.iterrows():
    qcat = virgule(e["pearson_q_categorie"], 3) if np.isfinite(e["pearson_q_categorie"]) else "—"
    gras = "**" if e["q_page"] < 0.05 else ""
    lignes.append(f"| {VUES[e['vue']]} | {MARQUEURS[e['marqueur']]} | {ISSUES[e['issue']]} | {int(e['n'])} | "
                  f"{gras}{virgule(e['r'], 2, True)}{gras} | {virgule(e['rho'], 2, True)} | {virgule(e['p'], 4)} | "
                  f"{gras}{virgule(e['q_page'], 3)}{gras} | {qcat} |")
ecrire_table("Toutes les corrélations à $p < 0{,}05$ (non corrigé), triées par vue puis par p. En gras : "
             "$q < 0{,}05$ sur les tests de la vue (42 ; 36 pour après − avant, sans la ligne AL).\n\n::: {.table-scroll}\n\n" + "\n".join(lignes)
             + "\n\n:::", "pk_tab_significatifs")
RESUME["n_tests"] = {VUES[v]: int(COR[COR["vue"] == v]["p"].notna().sum()) for v in VUES}
RESUME["n_p05"] = {VUES[v]: int((COR[COR["vue"] == v]["p"] < 0.05).sum()) for v in VUES}
RESUME["n_q05"] = {VUES[v]: int((COR[COR["vue"] == v]["q_page"] < 0.05).sum()) for v in VUES}
RESUME["top"] = COR.sort_values("p").head(8)[["marqueur", "vue", "issue", "n", "r", "p", "q_page",
                                               "pearson_q_categorie"]].to_dict("records")

# --------------------------------------------------------------------------- #
# Nuages : un marqueur x une vue (menu), les six issues
# --------------------------------------------------------------------------- #
combos = [(m, v) for m in MARQUEURS for v in VUES]
fig = make_subplots(rows=2, cols=3, horizontal_spacing=0.08, vertical_spacing=0.2,
                    subplot_titles=list(ISSUES.values()))
annotations_de, n_par = {}, 2 * len(ISSUES)
for i, (m, vue) in enumerate(combos):
    x_all = vue_de(m, vue)
    f = 1e3 if m == "k_sayah" else 1.0
    titres = []
    for j, iss in enumerate(ISSUES):
        d = pd.concat([x_all.rename("x"), OUT[iss].rename("y")], axis=1).dropna()
        e = COR[(COR["marqueur"] == m) & (COR["vue"] == vue) & (COR["issue"] == iss)].iloc[0]
        r_, c_ = j // 3 + 1, j % 3 + 1
        fig.add_trace(go.Scatter(x=d["x"] * f, y=d["y"], mode="markers", visible=(i == 0), showlegend=False,
                                 marker={"color": C_POINT, "size": 9, "line": {"width": 1, "color": "rgba(0,0,0,0)"}},
                                 customdata=d.index,
                                 hovertemplate=f"%{{customdata}}<br>{COURT[m]} %{{x:.4g}}<br>{ISSUES[iss]} %{{y}} "
                                               f"{UNITE_ISSUE[iss]}<extra></extra>"),
                      row=r_, col=c_)
        if len(d) >= 4:
            pente, oo = np.polyfit(d["x"] * f, d["y"], 1)
            xs = np.array([d["x"].min(), d["x"].max()]) * f
            fig.add_trace(go.Scatter(x=xs, y=pente * xs + oo, mode="lines", visible=(i == 0), showlegend=False,
                                     line={"color": C_AJUST, "width": 2}, hoverinfo="skip"), row=r_, col=c_)
        else:
            fig.add_trace(go.Scatter(x=[], y=[], visible=(i == 0), showlegend=False), row=r_, col=c_)
        q = "" if not np.isfinite(e["q_page"]) else f" · q {e['q_page']:.2f}".replace(".", ",")
        titres.append(f"{ISSUES[iss]} · r {virgule(e['r'], 2, True)} · n {int(e['n'])}{q}")
    annotations_de[(m, vue)] = titres
base_ann = list(fig.layout.annotations)
boutons = []
for i, (m, vue) in enumerate(combos):
    vis = [False] * (len(combos) * n_par)
    vis[i * n_par:(i + 1) * n_par] = [True] * n_par
    anns = [dict(a.to_plotly_json(), text=t) for a, t in zip(base_ann, annotations_de[(m, vue)])]
    unite = "× 10⁻³ mm⁻³" if m == "k_sayah" else MARQUEURS[m].split("(")[-1].rstrip(")")
    boutons.append({"label": f"{COURT[m]} · {VUES[vue]}", "method": "update",
                    "args": [{"visible": vis},
                             {"annotations": anns,
                              **{f"xaxis{'' if k == 1 else k}.title.text": f"{COURT[m]} ({unite}), {VUES[vue]}"
                                 for k in range(1, 7)}}]})
fig.update_layout(updatemenus=[{"buttons": boutons, "direction": "down", "x": 1.0, "xanchor": "right",
                                "y": 1.13, "yanchor": "bottom", "showactive": True,
                                "bgcolor": "#5a5a5a", "bordercolor": "#888", "font": {"color": "#f0f0f0"}}])
for j, iss in enumerate(ISSUES):
    fig.update_yaxes(title_text=f"{ISSUES[iss]} ({UNITE_ISSUE[iss]})", row=j // 3 + 1, col=j % 3 + 1)
    fig.update_xaxes(title_text=boutons[0]["args"][1]["xaxis.title.text"], row=j // 3 + 1, col=j % 3 + 1)
for a, t in zip(fig.layout.annotations, annotations_de[combos[0]]):
    a.text = t
mise_en_page(fig, "Un point par œil — choisir le marqueur et la vue dans le menu ; trait orange : moindres carrés",
             hauteur=640, legende=False)
fig.update_layout(margin={"t": 120})
enregistrer(fig, "pk_nuages")

# --------------------------------------------------------------------------- #
# Deux yeux par astronaute : la meme correlation a l'echelle de l'astronaute
# --------------------------------------------------------------------------- #
astro_of = pd.Series(PAR_OEIL.index.str.rsplit("_", n=1).str[0], index=PAR_OEIL.index)
OUT_A = OUT.groupby(OUT.index.str.rsplit("_", n=1).str[0])[list(ISSUES)].mean()
lignes = ["| marqueur (avant le vol) | issue | r par œil (n) | r par astronaute (n) |", "|:--|:--|--:|--:|"]
RESUME["astronautes"] = []
for m in ("k_sayah", "CT_mm", "OPA_mmHg"):
    x_a = vue_de(m, "before").groupby(astro_of).mean()
    for iss in ("delta_TRT_N", "delta_TRT_moyen", "delta_AL_mm"):
        e = COR[(COR["marqueur"] == m) & (COR["vue"] == "before") & (COR["issue"] == iss)].iloc[0]
        ca = correl(x_a, OUT_A[iss])
        RESUME["astronautes"].append({"marqueur": m, "issue": iss, "r_oeil": e["r"], "n_oeil": int(e["n"]),
                                      "r_astro": ca["r"], "n_astro": ca["n"], "p_astro": ca["p"]})
        lignes.append(f"| {MARQUEURS[m]} | {ISSUES[iss]} | {virgule(e['r'], 2, True)} ({int(e['n'])}) | "
                      f"{virgule(ca['r'], 2, True)} ({ca['n']}, p = {virgule(ca['p'], 3)}) |")
ecrire_table("\n".join(lignes), "pk_tab_astronautes")

# --------------------------------------------------------------------------- #
# Circularite : k contient AL, et AL apres entre dans dAL
# --------------------------------------------------------------------------- #
circ = {}
for nom, x in (("AL avant", OUT["AL_before_mm"]), ("AL après", OUT["AL_after_mm"])):
    circ[nom] = correl(x, OUT["delta_AL_mm"])
k_sans_al = (np.log1p(ACQ["OPA_mmHg"] / ACQ["IOP_mmHg"]) / (np.pi / 2 * ACQ["delta_CT_mm"]))
ACQ["k_sans_AL"] = k_sans_al.where(ACQ["k_sayah"].notna())
kA = ACQ.groupby(["subject", "moment"])["k_sans_AL"].mean().unstack("moment")
circ["k sans le facteur AL (après)"] = correl(kA["post"], OUT["delta_AL_mm"])
circ["k (après)"] = correl(PAR_OEIL["k_sayah"]["post"], OUT["delta_AL_mm"])
RESUME["circularite"] = circ
lignes = ["| grandeur | r avec ΔAL | n | p |", "|:--|--:|--:|--:|"]
for nom, c in circ.items():
    lignes.append(f"| {nom} | {virgule(c['r'], 2, True)} | {c['n']} | {virgule(c['p'], 3)} |")
ecrire_table("\n".join(lignes), "pk_tab_circularite")

# --------------------------------------------------------------------------- #
# Les parametres de SANS entre eux
# --------------------------------------------------------------------------- #
noms = list(ISSUES) + ["AL_before_mm"]
lib = list(ISSUES.values()) + ["AL avant"]
M = np.full((len(noms), len(noms)), np.nan)
N = np.zeros_like(M)
P = np.full_like(M, np.nan)
for a_, x in enumerate(noms):
    for b_, y in enumerate(noms):
        c = correl(OUT[x], OUT[y])
        M[a_, b_], N[a_, b_], P[a_, b_] = c["r"], c["n"], c["p"]
txt = [[("" if a_ == b_ else f"{M[a_, b_]:+.2f}".replace(".", ",")) for b_ in range(len(noms))]
       for a_ in range(len(noms))]
fig = go.Figure(go.Heatmap(z=np.where(np.eye(len(noms), dtype=bool), np.nan, M), x=lib, y=lib,
                           zmin=-1, zmax=1, zmid=0, colorscale="RdBu_r", xgap=2, ygap=2,
                           text=txt, texttemplate="%{text}", textfont={"size": 10},
                           customdata=np.stack([N, P], -1),
                           colorbar={"title": {"text": "r"}, "len": 0.8, "thickness": 12},
                           hovertemplate="%{y} ~ %{x}<br>r = %{z:.2f} · n = %{customdata[0]} yeux"
                                         " · p = %{customdata[1]:.3g}<extra></extra>"))
fig.update_yaxes(autorange="reversed")
mise_en_page(fig, "Les paramètres de SANS entre eux (Pearson r, un point par œil)", hauteur=480, legende=False)
enregistrer(fig, "pk_issues")
RESUME["issues_r"] = {f"{lib[a_]} ~ {lib[b_]}": float(M[a_, b_])
                      for a_ in range(len(noms)) for b_ in range(a_ + 1, len(noms))}

(SORTIE / "plotlyjs.html").write_text(
    '<script type="text/javascript">\n' + get_plotlyjs() + "\n</script>\n", encoding="utf-8")
(SORTIE / "resume.json").write_text(json.dumps(RESUME, ensure_ascii=False, indent=1, default=float),
                                    encoding="utf-8")
print("  plotlyjs.html, resume.json")
