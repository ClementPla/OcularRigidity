# -*- coding: utf-8 -*-
"""
Figures -- comparaison des outils d'extraction du pouls choroidien SUR DONNEES.

Contrairement aux pages `figures_gaussienne_svd/` et
`figures_deux_gaussiennes_svd/`, qui simulent l'objet, celle-ci ne simule rien :
elle ne fait que LIRE les tables produites par
`Astronauts/compute_pulse_from_data.py` sur la cohorte SANSORI complete
et les mettre en images.

Entrees (ecrites par le script de batch, jamais recalculees ici) :

    E:/NASA_Rigidity/SegmentationVariations/model1_scale_1.0/pulse_from_data/
        conditions.csv    1 ligne / condition
        methods.csv       1 ligne / (condition, methode)   <- les 7 pouls
        dmd_eigs.csv      1 ligne / (condition, variante BOPDMD, paire propre)
        ssa_sweep.csv     1 ligne / (condition, approche SSA, L)
        traces/<slug>.npz pouls, phases, spectres par condition

Deux niveaux de lecture, et les figures suivent ce decoupage :

  - UNE condition representative (`SLUG_REF`), pour voir a quoi ressemblent
    reellement les sept pouls, leurs spectres, leur frequence instantanee ;
  - la COHORTE, pour savoir si ce qu'on a vu sur une condition tient partout.

Sorties : un fragment Quarto par figure (`fN_*.qmd`), contenant le HTML Plotly
dans un bloc brut ```{=html} et SANS plotly.js -- la bibliotheque est inseree
une seule fois par page via `plotlyjs.html` (`include-in-header` dans le .qmd),
sinon les 11 figures embarqueraient 11 x 4,9 Mo. Les fragments sont inclus par
`{{< include ... >}}` cote page.

L'interactivite n'est pas decorative ici : sur les nuages par condition, le
survol donne le slug, ce qui permet d'aller voir la condition aberrante au lieu
de constater qu'elle existe.

Le fond est transparent et le texte gris : les memes figures passent sur le
theme clair et sur le theme sombre du site.

Lancer :  python make_figures.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------- #
# Entrees / sorties
# --------------------------------------------------------------------------- #
DATA = Path("E:/NASA_Rigidity/SegmentationVariations/model1_scale_1.0/pulse_from_data")
SORTIE = Path(__file__).parent

# Condition representative : celle qui minimise l'ecart median normalise sur
# (corr SSA/FIR, nb de canaux, duree, part de frequence negative sans filtre,
# ecart HR de la DMD) parmi les enregistrements de plus de 50 s. Ce n'est pas la
# plus belle condition, c'est la plus BANALE -- montrer la meilleure donnerait
# une idee fausse de ce que le pipeline voit d'habitude.
SLUG_REF = "09_210921001__210921001post_rigidity_OD1"

# Les sept methodes, dans l'ordre du raisonnement : temoin, reference, SSA, DMD.
METHODES = ["0_sans_filtre", "1_fir", "2_ssa", "3a_mssa", "3b_ssa_canal",
            "dmd_modele", "dmd_projete"]
LABELS = {
    "0_sans_filtre": "0 · no filter",
    "1_fir": "1 · FIR band-pass",
    "2_ssa": "2 · SSA on the pulse",
    "3a_mssa": "3a · M-SSA (multichannel)",
    "3b_ssa_canal": "3b · SSA per channel",
    "dmd_modele": "DMD · model",
    "dmd_projete": "DMD · projected",
}
COULEURS = {
    "0_sans_filtre": "#9a9a9a",
    "1_fir": "#2a78d6",
    "2_ssa": "#27a567",
    "3a_mssa": "#8e5bd0",
    "3b_ssa_canal": "#d0a72b",
    "dmd_modele": "#eb6834",
    "dmd_projete": "#c2453f",
}

COULEUR_TEXTE = "#c9c9c9"
C_REF = "#9a9a9a"
C_BANDE = "#2a78d6"
GRILLE = "rgba(150,150,150,0.18)"

CONFIG = {"displaylogo": False, "responsive": True,
          "modeBarButtonsToRemove": ["select2d", "lasso2d"]}

FICHIERS: list[str] = []


def mise_en_page(fig, titre, hauteur=420, legende=True):
    """Habillage commun : fond transparent, texte gris, grille discrete."""
    fig.update_layout(
        title={"text": titre, "font": {"size": 14}, "x": 0.01, "xanchor": "left"},
        template="none",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COULEUR_TEXTE, "size": 11},
        height=hauteur,
        # Legende SOUS le graphique : avec sept methodes, une legende
        # horizontale posee au-dessus passe sur deux lignes et vient recouvrir
        # le titre. En dessous, elle deborde sur la marge basse, qu'on elargit
        # en consequence.
        margin={"l": 60, "r": 20, "t": 60, "b": 95 if legende else 50},
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
    """Fragment Quarto : le HTML Plotly dans un bloc brut ```{=html}.

    Sans ce bloc, `{{< include >}}` livre le fragment au lecteur markdown de
    Pandoc, qui prend le `<div>` de Plotly pour un div natif et se plaint de ne
    pas trouver sa fermeture ("Div at line N unclosed"). Le bloc brut le fait
    passer verbatim. Pas de plotly.js ici : il est insere une fois par page via
    `plotlyjs.html` (`include-in-header` du .qmd).
    """
    html = fig.to_html(full_html=False, include_plotlyjs=False,
                       config=CONFIG, div_id=f"plot-{nom}")
    chemin = SORTIE / f"{nom}.qmd"
    chemin.write_text("```{=html}\n" + html + "\n```\n", encoding="utf-8")
    FICHIERS.append(chemin.name)
    print(f"  {chemin.name}")


# --------------------------------------------------------------------------- #
# Chargement
# --------------------------------------------------------------------------- #
conditions = pd.read_csv(DATA / "conditions.csv")
ok = conditions[conditions.status == "ok"].copy()
methods = pd.read_csv(DATA / "methods.csv")
dmd = pd.read_csv(DATA / "dmd_eigs.csv")
sweep = pd.read_csv(DATA / "ssa_sweep.csv")
z = np.load(DATA / "traces" / f"{SLUG_REF}.npz")

HR = float(z["hr"])
u_time = z["u_time"]
bpm_axis = z["bpm_axis"]
core = z["core"]
LO, HI = HR * 0.8, HR * 1.2

print(f"{len(ok)} conditions exploitables / {len(conditions)} parcourues")
print(f"condition de reference : {SLUG_REF}  ({HR:.0f} BPM)")


# =========================================================================== #
# 1. Une condition : les sept pouls
# =========================================================================== #
# Fenetre de 12 s prise au MILIEU de l'enregistrement : les 20 % de bord sont
# exclus des statistiques (transitoire du FIR), les montrer ici serait
# defavorable aux methodes filtrees pour une raison purement numerique.
t0 = u_time[core][0] + 0.5 * (u_time[core][-1] - u_time[core][0]) - 6.0
fen = (u_time >= t0) & (u_time <= t0 + 12.0)

fig = make_subplots(rows=len(METHODES), cols=1, shared_xaxes=True,
                    vertical_spacing=0.012)
for i, nom in enumerate(METHODES, start=1):
    y = z[f"pulse_{nom}"][fen].astype(float)
    y = y / (np.abs(y).max() + 1e-12)
    fig.add_trace(go.Scatter(
        x=u_time[fen] - t0, y=y, mode="lines", name=LABELS[nom],
        line={"color": COULEURS[nom], "width": 1.6},
        hovertemplate="%{x:.2f} s<br>%{y:.2f}<extra>" + LABELS[nom] + "</extra>"),
        row=i, col=1)
    fig.update_yaxes(title_text=LABELS[nom], title_font={"size": 9},
                     showticklabels=False, row=i, col=1)
# Reperes a la periode cardiaque de reference : un battement attendu par
# intervalle. C'est la seule verite terrain disponible ici.
for k in range(1, int(12.0 * HR / 60.0) + 1):
    fig.add_vline(x=k * 60.0 / HR, line={"color": C_REF, "width": 0.7,
                                         "dash": "dot"}, opacity=0.45)
fig.update_xaxes(title_text="time (s)", row=len(METHODES), col=1)
mise_en_page(fig, f"The seven pulses, same condition, same 12 s window "
                  f"(reference HR {HR:.0f} BPM; dotted lines = one cardiac period)",
             hauteur=760, legende=False)
enregistrer(fig, "f1_pulses")


# =========================================================================== #
# 2. Une condition : les spectres
# =========================================================================== #
fig = go.Figure()
fig.add_vrect(x0=LO, x1=HI, fillcolor=C_BANDE, opacity=0.10, line_width=0,
              annotation_text="scoring band", annotation_position="top left",
              annotation_font_size=10)
fig.add_vline(x=HR, line={"color": C_REF, "width": 1, "dash": "dash"})
for nom in METHODES:
    P = z[f"spec_{nom}"].astype(float)
    fig.add_trace(go.Scatter(
        x=bpm_axis, y=P / (P.max() + 1e-12), mode="lines", name=LABELS[nom],
        line={"color": COULEURS[nom], "width": 1.5},
        hovertemplate="%{x:.0f} BPM<br>%{y:.2f}<extra>" + LABELS[nom] + "</extra>"))
fig.update_xaxes(range=[20, 240], title_text="frequency (BPM)")
fig.update_yaxes(title_text="Lomb–Scargle power (normalised)")
mise_en_page(fig, "Where each pulse puts its power — shaded: scoring band (HR ± 20 %)",
             hauteur=440)
enregistrer(fig, "f2_spectra")


# =========================================================================== #
# 3. Une condition : frequence instantanee
# =========================================================================== #
fig = go.Figure()
fig.add_hrect(y0=LO, y1=HI, fillcolor=C_BANDE, opacity=0.10, line_width=0)
fig.add_hline(y=HR, line={"color": C_REF, "width": 1, "dash": "dash"},
              annotation_text="reference HR", annotation_font_size=10)
for nom in METHODES:
    fig.add_trace(go.Scatter(
        x=u_time[core], y=z[f"bpm_{nom}"][core].astype(float), mode="lines",
        name=LABELS[nom], line={"color": COULEURS[nom], "width": 1.3},
        hovertemplate="%{x:.1f} s<br>%{y:.1f} BPM<extra>" + LABELS[nom] + "</extra>"))
fig.update_xaxes(title_text="time (s)")
fig.update_yaxes(range=[0, 200], title_text="instantaneous rate (BPM)")
mise_en_page(fig, "Instantaneous rate from the Hilbert phase, median-filtered over one cycle",
             hauteur=440)
enregistrer(fig, "f3_hr_inst")


# =========================================================================== #
# 4. Une condition : ou est le pouls dans la SVD
# =========================================================================== #
P = z["P_svd"].astype(float)
P = P / (P.max(axis=0, keepdims=True) + 1e-12)
canaux = z["channels"]
poids = np.asarray(z["weights"])

# Le periodogramme complet fait 502 x 100 : on le restreint a la bande affichee
# (20-240 BPM) plutot que de l'envoyer entier dans le HTML.
vis = (bpm_axis >= 20) & (bpm_axis <= 240)
fig = make_subplots(rows=1, cols=2, column_widths=[0.66, 0.34],
                    horizontal_spacing=0.13,
                    subplot_titles=("Periodogram of every singular vector",
                                    "Weight in the optimised combination"))
fig.add_trace(go.Heatmap(
    # float32 : Plotly 6 serialise les tableaux numpy en base64 binaire, donc
    # arrondir ne change rien mais halver la largeur du type, si. 37 000 valeurs
    # dont on ne lit qu'une couleur -- la simple precision est largement assez.
    z=P[vis].astype(np.float32), x=np.arange(P.shape[1]), y=bpm_axis[vis],
    colorscale="Magma",
    showscale=False,
    hovertemplate="component %{x}<br>%{y:.0f} BPM<br>power %{z:.2f}<extra></extra>"),
    row=1, col=1)
fig.add_trace(go.Scatter(
    x=canaux, y=np.full(canaux.size, HR), mode="markers", name="kept as cardiac",
    marker={"color": "#7fd6ff", "size": 9, "symbol": "triangle-down"},
    hovertemplate="component %{x} kept<extra></extra>"), row=1, col=1)
fig.add_hline(y=HR, line={"color": "#7fd6ff", "width": 1, "dash": "dash"},
              row=1, col=1)
fig.update_xaxes(title_text="singular component index", row=1, col=1)
fig.update_yaxes(title_text="frequency (BPM)", row=1, col=1)

ordre = np.argsort(np.abs(poids))
fig.add_trace(go.Bar(
    x=np.abs(poids)[ordre], y=[f"v{c}" for c in canaux[ordre]], orientation="h",
    marker={"color": C_BANDE}, showlegend=False,
    hovertemplate="%{y}: |weight| %{x:.3f}<extra></extra>"), row=1, col=2)
fig.update_xaxes(title_text="|weight|", row=1, col=2)
fig.update_yaxes(type="category", row=1, col=2)
mise_en_page(fig, "", hauteur=460, legende=False)
fig.update_annotations(font_size=12)
enregistrer(fig, "f4_svd_map")


# =========================================================================== #
# 4b. Cohorte : a quel RANG vit le pouls
# =========================================================================== #
# La figure precedente montre, sur une condition, que les canaux cardiaques
# retenus ne sont pas les premiers vecteurs singuliers. Celle-ci verifie que ce
# n'est pas une particularite de cette condition : c'est l'argument qui justifie
# de decomposer au rang 100 plutot qu'au rang 6 ou 10.
idx_canaux = np.concatenate([np.fromstring(s, dtype=int, sep=" ")
                             for s in ok["channels"]])
fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                    subplot_titles=(
                        "The pulse does not live in the leading components",
                        "Where the cardiac subspace starts"))
fig.add_trace(go.Histogram(
    x=idx_canaux, xbins={"start": 0, "end": 100, "size": 5},
    marker={"color": C_BANDE}, showlegend=False,
    hovertemplate="components %{x}<br>%{y} kept<extra></extra>"), row=1, col=1)
fig.update_xaxes(title_text="index of a component kept as cardiac", row=1, col=1)
fig.update_yaxes(title_text=f"count ({idx_canaux.size} components)", row=1, col=1)

fig.add_trace(go.Histogram(
    x=ok["channel_min"], xbins={"start": 0, "end": 100, "size": 5},
    marker={"color": "#27a567"}, showlegend=False,
    hovertemplate="lowest index %{x}<br>%{y} conditions<extra></extra>"),
    row=1, col=2)
fig.add_vline(x=10, line={"color": C_REF, "width": 1, "dash": "dash"},
              row=1, col=2)
fig.add_annotation(
    x=0.98, y=0.95, xref="x2 domain", yref="y2 domain",
    text=f"{int((ok['channel_min'] >= 10).sum())}/{len(ok)} conditions have<br>"
         "nothing cardiac before index 10",
    showarrow=False, align="right", xanchor="right", yanchor="top",
    font={"size": 9, "color": COULEUR_TEXTE})
fig.update_xaxes(title_text="lowest cardiac component index, per condition",
                 row=1, col=2)
fig.update_yaxes(title_text="conditions", row=1, col=2)
mise_en_page(fig, "", hauteur=380, legende=False)
fig.update_annotations(font_size=12)
enregistrer(fig, "f4b_channel_index")


# =========================================================================== #
# Boites a moustaches par methode (figures 5 et 6)
# =========================================================================== #
def boites(fig, colonne, echelle, row, col):
    """Une boite par methode, tous les points visibles et survolables.

    Les points portent le slug : la dispersion ne sert a rien si on ne peut pas
    remonter a la condition qui traine.
    """
    for m in METHODES:
        s = methods[methods.methode == m]
        fig.add_trace(go.Box(
            y=s[colonne].values * echelle, name=LABELS[m],
            marker={"color": COULEURS[m], "size": 3, "opacity": 0.45},
            line={"color": COULEUR_TEXTE, "width": 1},
            fillcolor=COULEURS[m], opacity=0.75,
            boxpoints="all", jitter=0.5, pointpos=0, showlegend=False,
            text=s["slug"].values, hovertemplate="%{text}<br>%{y:.2f}<extra></extra>"),
            row=row, col=col)


fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.10,
                    subplot_titles=("Phase going backwards",
                                    "Instantaneous frequency out of band"))
boites(fig, "f_neg_frac", 100.0, 1, 1)
boites(fig, "hors_bande_frac", 100.0, 1, 2)
fig.update_yaxes(title_text="samples with negative inst. frequency (%)", row=1, col=1)
fig.update_yaxes(title_text="samples out of the cardiac band (%)", row=1, col=2)
fig.update_xaxes(tickangle=-30, tickfont_size=9)
mise_en_page(fig, f"Is the Hilbert phase interpretable at all? — {len(ok)} conditions",
             hauteur=560, legende=False)
# Les noms de methodes pivotes a -30 deg debordent sous l'axe : la marge basse
# commune (50 px) ne suffit pas pour ces deux figures.
fig.update_layout(margin={"b": 130})
fig.update_annotations(font_size=12)
enregistrer(fig, "f5_phase_legitimacy")

methods["abs_ecart"] = methods["ecart_HR_ref_BPM"].abs()
fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.10,
                    subplot_titles=("Accuracy against the ECG-derived HR",
                                    "Stability within a recording"))
boites(fig, "abs_ecart", 1.0, 1, 1)
boites(fig, "HR_IQR_BPM", 1.0, 1, 2)
fig.update_yaxes(title_text="| median rate − reference HR | (BPM)",
                 range=[0, 60], row=1, col=1)
fig.update_yaxes(title_text="IQR of the instantaneous rate (BPM)",
                 range=[0, 60], row=1, col=2)
fig.update_xaxes(tickangle=-30, tickfont_size=9)
mise_en_page(fig, f"Rate recovered by each method — {len(ok)} conditions",
             hauteur=560, legende=False)
fig.update_layout(margin={"b": 130})
fig.update_annotations(font_size=12)
enregistrer(fig, "f6_hr_accuracy")


# =========================================================================== #
# 7. Cohorte : fidelite de forme d'onde contre cout
# =========================================================================== #
# corr_fir est la correlation au pouls passe-bande, la reference de FORME. Une
# methode peut donner la bonne frequence et une forme d'onde fausse : c'est
# exactement le cas de la DMD, et c'est ce que cette figure isole.
fig = go.Figure()
for m in METHODES:
    sub = methods[methods.methode == m]
    cout = max(float(sub["cout_s"].median()), 3e-4)
    corr = sub["corr_fir"]
    fig.add_trace(go.Scatter(
        x=[cout], y=[corr.median()], mode="markers", name=LABELS[m],
        marker={"color": COULEURS[m], "size": 13,
                "line": {"color": COULEUR_TEXTE, "width": 0.5}},
        error_y={"type": "data", "symmetric": False,
                 "array": [corr.quantile(0.75) - corr.median()],
                 "arrayminus": [corr.median() - corr.quantile(0.25)],
                 "color": COULEURS[m], "thickness": 1.2, "width": 5},
        hovertemplate=(f"{LABELS[m]}<br>cost %{{x:.4f}} s<br>"
                       f"corr %{{y:.2f}} "
                       f"[{corr.quantile(0.25):.2f}–{corr.quantile(0.75):.2f}]"
                       "<extra></extra>")))
fig.add_hline(y=0, line={"color": C_REF, "width": 0.8, "dash": "dot"})
fig.update_xaxes(type="log", title_text="median cost per condition (s, log scale)")
fig.update_yaxes(title_text="correlation with the FIR band-passed pulse")
mise_en_page(fig, "Waveform fidelity vs. computational cost "
                  f"(bars: interquartile range over {len(ok)} conditions)",
             hauteur=460)
enregistrer(fig, "f7_cost_fidelity")


# =========================================================================== #
# 8. Cohorte : l'initialisation de BOPDMD
# =========================================================================== #
fig = make_subplots(rows=1, cols=2, column_widths=[0.38, 0.62],
                    horizontal_spacing=0.12,
                    subplot_titles=("Does BOPDMD find the cardiac mode?",
                                    "Rank-6 eigenvalues: fundamental and harmonics"))
noms_var = ["B1<br>default init", "B2<br>harmonic init",
            "B3<br>harmonic init<br>+ imaginary constraint"]
taux = [100.0 * ok[f"dmd_b{i}_ok"].mean() for i in (1, 2, 3)]
fig.add_trace(go.Bar(
    x=noms_var, y=taux, marker={"color": ["#c2453f", "#d0a72b", "#27a567"]},
    text=[f"{v:.0f} %" for v in taux], textposition="outside",
    textfont={"color": COULEUR_TEXTE}, showlegend=False,
    hovertemplate="%{x}<br>%{y:.1f} %<extra></extra>"), row=1, col=1)
fig.update_yaxes(title_text="conditions with a mode within HR ± 20 % (%)",
                 range=[0, 118], row=1, col=1)

# Plan des valeurs propres : frequence rapportee a la HR de la condition (sinon
# autant de HR differentes rendent le nuage illisible) contre taux de croissance.
d_ok = dmd[dmd.status == "ok"]
g = d_ok["croissance_1s"]
bas = float(np.percentile(g, 2)) - 0.5
for var, coul in zip(["B1", "B2", "B3"], ["#c2453f", "#d0a72b", "#27a567"]):
    s = d_ok[d_ok.variante == var]
    fig.add_trace(go.Scatter(
        x=s["ratio_HR"], y=s["croissance_1s"], mode="markers", name=var,
        marker={"color": coul, "size": 5, "opacity": 0.5},
        text=s["slug"], customdata=s["f_BPM"],
        hovertemplate=("%{text}<br>%{customdata:.1f} BPM = %{x:.2f}×HR"
                       "<br>growth %{y:.2f} /s<extra>" + var + "</extra>")),
        row=1, col=2)
for k in (1, 2, 3):
    fig.add_vline(x=k, line={"color": C_REF, "width": 0.8, "dash": "dash"},
                  opacity=0.6, row=1, col=2)
fig.update_xaxes(range=[0, 4.5], title_text="mode frequency / reference HR",
                 row=1, col=2)
fig.update_yaxes(range=[bas, max(float(g.max()), 0.5) * 1.15],
                 title_text="growth rate (1/s)", row=1, col=2)
fig.add_annotation(x=4.4, y=bas, xref="x2", yref="y2",
                   text=f"{int((g < bas).sum())} strongly damped modes below the axis",
                   showarrow=False, font={"size": 9, "color": C_REF},
                   xanchor="right", yanchor="bottom")
mise_en_page(fig, "", hauteur=460)
fig.update_annotations(font_size=12)
enregistrer(fig, "f8_dmd_init")


# =========================================================================== #
# 9. Cohorte : balayage de la fenetre SSA
# =========================================================================== #
s_ok = sweep[sweep.status == "ok"]
mesures = [("en_bande_frac", "in-band power (%)", 100.0),
           ("HR_IQR_BPM", "IQR of instantaneous rate (BPM)", 1.0),
           ("n_comp_groupe", "components grouped as cardiac", 1.0)]
fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.075,
                    subplot_titles=tuple(lab for _, lab, _ in mesures))
for j, (col, lab, ech) in enumerate(mesures, start=1):
    for app in ("2_ssa", "3a_mssa"):
        gr = s_ok[s_ok.approche == app].groupby("cycles")[col]
        cyc = np.array(sorted(gr.groups.keys()))
        med, q1, q3 = (gr.median().values * ech, gr.quantile(0.25).values * ech,
                       gr.quantile(0.75).values * ech)
        coul = COULEURS[app]
        rgba = ("rgba(39,165,103,0.15)" if app == "2_ssa"
                else "rgba(142,91,208,0.15)")
        fig.add_trace(go.Scatter(
            x=np.concatenate([cyc, cyc[::-1]]),
            y=np.concatenate([q3, q1[::-1]]), fill="toself", fillcolor=rgba,
            line={"width": 0}, hoverinfo="skip", showlegend=False), row=1, col=j)
        fig.add_trace(go.Scatter(
            x=cyc, y=med, mode="lines+markers", name=LABELS[app],
            line={"color": coul, "width": 2}, marker={"size": 6},
            showlegend=(j == 1),
            hovertemplate="L = %{x} cycles<br>%{y:.2f}<extra>"
                          + LABELS[app] + "</extra>"), row=1, col=j)
    fig.add_vline(x=3.0, line={"color": C_REF, "width": 1, "dash": "dash"},
                  row=1, col=j)
    fig.update_xaxes(title_text="window L (cardiac cycles)", row=1, col=j)
mise_en_page(fig, "SSA embedding window — dashed line: the value used elsewhere (3 cycles)",
             hauteur=420)
fig.update_annotations(font_size=11)
enregistrer(fig, "f9_ssa_sweep")


# =========================================================================== #
# 10. Cohorte : ce qui n'a pas abouti
# =========================================================================== #
def libelle_echec(status: str) -> str:
    """Statut brut du CSV -> etiquette lisible."""
    if "NaN" in status:
        return "all-NaN traces after normalisation"
    if "No trace" in status or "no ROI" in status:
        # Depuis le seuil sur l'aire du masque, ce cas ne veut plus dire "une
        # frame noire a vide l'intersection" mais "les masques ne se recouvrent
        # jamais assez", ce qui est un probleme de recalage, pas de garde.
        return "masks never overlap enough for a stable ROI"
    return status[:44]


echecs = conditions[conditions.status != "ok"]["status"].value_counts()
etiquettes = [f"processed ({len(ok)})"] + [
    f"{libelle_echec(s)} ({n})" for s, n in echecs.items()]
valeurs = [len(ok)] + list(echecs.values)
coul = ["#27a567", "#c2453f", "#d0a72b"][:len(valeurs)]
fig = go.Figure(go.Bar(
    x=valeurs, y=etiquettes, orientation="h", marker={"color": coul},
    text=[f"{100 * v / len(conditions):.0f} %" for v in valeurs],
    textposition="outside", textfont={"color": COULEUR_TEXTE},
    hovertemplate="%{y}<br>%{x} conditions<extra></extra>"))
fig.update_xaxes(range=[0, len(conditions) * 1.15], title_text="conditions")
fig.update_yaxes(autorange="reversed")
mise_en_page(fig, f"Outcome over the {len(conditions)} conditions of the cohort",
             hauteur=max(220, 90 + 60 * len(valeurs)), legende=False)
enregistrer(fig, "f10_outcome")


# =========================================================================== #
# plotly.js, une seule fois par page (via `include-in-header` du .qmd)
# =========================================================================== #
(SORTIE / "plotlyjs.html").write_text(
    "<script type=\"text/javascript\">\n" + get_plotlyjs() + "\n</script>\n",
    encoding="utf-8")


# =========================================================================== #
# Tables markdown -- pour que les chiffres du texte ne soient jamais recopies
# =========================================================================== #
def table_methodes() -> str:
    lignes = ["| method | peak (BPM) | in-band power | \\|rate − HR\\| (BPM) | "
              "rate IQR (BPM) | f < 0 | corr. with FIR | cost (s) |",
              "|---|---|---|---|---|---|---|---|"]
    for m in METHODES:
        s = methods[methods.methode == m]
        lignes.append(
            f"| {LABELS[m]} "
            f"| {s['pic_LS_BPM'].median():.1f} "
            f"| {100 * s['en_bande_frac'].median():.0f} % "
            f"| {s['abs_ecart'].median():.2f} "
            f"| {s['HR_IQR_BPM'].median():.1f} "
            f"| {100 * s['f_neg_frac'].median():.1f} % "
            f"| {s['corr_fir'].median():.2f} "
            f"| {s['cout_s'].median():.3f} |")
    return "\n".join(lignes)


def table_cohorte() -> str:
    def ligne(lab, serie, fmt="{:.2f}"):
        return (f"| {lab} | {fmt.format(serie.median())} | "
                f"{fmt.format(serie.quantile(0.25))} – {fmt.format(serie.quantile(0.75))} | "
                f"{fmt.format(serie.min())} – {fmt.format(serie.max())} |")

    lignes = ["| quantity | median | IQR | range |", "|---|---|---|---|",
              ligne("reference HR (BPM)", ok["hr_BPM"], "{:.0f}"),
              ligne("recording length (s)", ok["duree_s"], "{:.0f}"),
              ligne("frames", ok["n_frames"], "{:.0f}"),
              ligne("effective sampling rate (Hz)", ok["fs_Hz"]),
              ligne("pixel traces", ok["n_pixels"], "{:.0f}"),
              ligne("cardiac components kept", ok["n_channels"], "{:.0f}"),
              ligne("variance in the first triplet", ok["var_1er_triplet"]),
              ligne("corr. cost of resampling", ok["corr_interp"]),
              ligne("SSA window L (samples)", ok["L_ssa"], "{:.0f}")]
    return "\n".join(lignes)


(SORTIE / "tab_methods.qmd").write_text(table_methodes() + "\n", encoding="utf-8")
(SORTIE / "tab_cohort.qmd").write_text(table_cohorte() + "\n", encoding="utf-8")

# Chiffres cites en prose, regeneres pour qu'ils suivent les donnees.
resume = {
    "n_total": len(conditions),
    "n_ok": len(ok),
    "n_astro": int(ok.astro.nunique()),
    "b1": 100 * ok["dmd_b1_ok"].mean(),
    "b2": 100 * ok["dmd_b2_ok"].mean(),
    "b3": 100 * ok["dmd_b3_ok"].mean(),
    "dmd_ecart": np.median(np.abs(ok["dmd_ecart_HR_BPM"])),
    "conc_degrade": int((ok["min_conc_used"] < 0.15).sum()),
    "hr_svd": int((ok["hr_source"] == "svd").sum()),
    "idx_median": float(np.median(idx_canaux)),
    "idx_n": int(idx_canaux.size),
    "idx_sous_10": 100 * float((idx_canaux < 10).mean()),
    "cond_rien_avant_10": int((ok["channel_min"] >= 10).sum()),
    "n_comp_en_bande": float(ok["n_comp_pic_en_bande"].median()),
}
for m in METHODES:
    s = methods[methods.methode == m]
    resume[f"corr_{m}"] = s["corr_fir"].median()
    resume[f"fneg_{m}"] = 100 * s["f_neg_frac"].median()
    resume[f"horsbande_{m}"] = 100 * s["hors_bande_frac"].median()
    resume[f"iqr_{m}"] = s["HR_IQR_BPM"].median()
    resume[f"err_{m}"] = s["abs_ecart"].median()
    resume[f"cout_{m}"] = s["cout_s"].median()

(SORTIE / "resume.txt").write_text(
    "\n".join(f"{k} = {v:.4g}" if isinstance(v, float) else f"{k} = {v}"
              for k, v in resume.items()) + "\n", encoding="utf-8")

print(f"\n{len(FICHIERS)} figures + plotly.js + 2 tables + resume.txt -> {SORTIE}")
for k, v in resume.items():
    print(f"  {k:<26}{v:.4g}" if isinstance(v, float) else f"  {k:<26}{v}")
