# -*- coding: utf-8 -*-
"""
Figures -- videos one-cycle repliees sur deux phases concurrentes (FIR / M-SSA).

Comme `figures_pulse_from_data/`, cette page ne calcule RIEN : elle lit ce que
`Astronauts/compute_one_cycle_compare_methods.py` a ecrit sur la cohorte et le
met en images.

Entrees :
    E:/NASA_Rigidity/SegmentationVariations/model1_scale_1.0/one_cycle_compare/
        conditions.csv    1 ligne / condition
        methods.csv       1 ligne / (condition, methode)
        bins.csv          1 ligne / (condition, methode, bin)
        <astro>/<moment>/<condition>/one_cycle_compare.mp4   <- video livree
        <astro>/<moment>/<condition>/one_cycle_compare.npz   <- courbes

Sorties :
  - `fN_*.qmd`  fragments Plotly (bloc brut ```{=html}, sans plotly.js : la
    bibliotheque est inseree une seule fois par page via `plotlyjs.html`) ;
  - `tab_*.qmd` tables de synthese ;
  - `videos/<slug>.mp4` les videos de comparaison, copiees ici pour que Quarto
    les emporte dans `_site` (ce dossier est une jonction vers `E:`, donc la
    copie ne touche pas `C:`) ;
  - `../one_cycles/oc_<slug>.qmd` une page par condition -- la video, ses
    chiffres, et un trace SVG des deux courbes d'epaisseur ;
  - `_table.qmd` la table des 106 liens, incluse par la page principale.

Pourquoi du SVG en ligne, et pas du Plotly, sur les pages par condition
----------------------------------------------------------------------
`plotlyjs.html` pese 4,9 Mo et `include-in-header` l'inline dans CHAQUE page qui
le declare : 106 pages par condition couteraient un demi-gigaoctet pour tracer
dix points. Le trace d'epaisseur y est donc un SVG ecrit a la main (2 ko, sans
JS, couleurs valides sur les deux themes du site). Les figures de la page
principale, elles, sont bien interactives.

Lancer :  python make_figures.py
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots
from scipy.stats import wilcoxon

# --------------------------------------------------------------------------- #
# Entrees / sorties
# --------------------------------------------------------------------------- #
DATA = Path(
    "E:/NASA_Rigidity/SegmentationVariations/model1_scale_1.0/one_cycle_compare")
# Les pouls et phases viennent de la page precedente ; seule la HR instantanee
# (`bpm_<methode>`, filtree sur un cycle) n'est pas recopiee dans les .npz du
# repliement, on la relit donc a la source.
TRACES_POULS = DATA.parent / "pulse_from_data" / "traces"
SORTIE = Path(__file__).parent
VIDEOS = SORTIE / "videos"
PAGES = SORTIE.parent / "one_cycles"  # pages par condition (texte, dans le depot)

# Meme condition de reference que `figures_pulse_from_data/` : le lecteur arrive
# de cette page-la, autant lui montrer le meme oeil.
SLUG_REF = "09_210921001__210921001post_rigidity_OD1"

METHODES = ["1_fir", "3a_mssa"]
NUL = "null_shuffle"  # temoin negatif : la phase FIR permutee au hasard
TOUTES = METHODES + [NUL]
LABELS = {"1_fir": "FIR band-pass", "3a_mssa": "M-SSA",
          NUL: "shuffled phase (null)"}
COULEURS = {"1_fir": "#2a78d6", "3a_mssa": "#8e5bd0", NUL: "#9a9a9a"}

COULEUR_TEXTE = "#c9c9c9"
C_REF = "#9a9a9a"
GRILLE = "rgba(150,150,150,0.18)"

CONFIG = {"displaylogo": False, "responsive": True,
          "modeBarButtonsToRemove": ["select2d", "lasso2d"]}

FICHIERS: list[str] = []


def mise_en_page(fig, titre, hauteur=420, legende=True):
    """Habillage commun : fond transparent, texte gris, grille discrete --
    les memes figures passent sur le theme clair et sur le theme sombre."""
    fig.update_layout(
        title={"text": titre, "font": {"size": 14}, "x": 0.01, "xanchor": "left"},
        template="none",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COULEUR_TEXTE, "size": 11},
        height=hauteur,
        # Legende SOUS le graphique, et marge basse assez large pour qu'elle ne
        # vienne pas se poser sur le titre de l'axe des x -- avec six entrees
        # (deux methodes x trois courbes) elle passe sur deux lignes.
        margin={"l": 62, "r": 20, "t": 62, "b": 140 if legende else 50},
        showlegend=legende,
        legend={"font": {"size": 10}, "orientation": "h", "yanchor": "top",
                "y": -0.34, "xanchor": "left", "x": 0},
        hoverlabel={"font": {"size": 11}},
    )
    fig.update_xaxes(gridcolor=GRILLE, zerolinecolor=GRILLE,
                     linecolor=GRILLE, ticks="outside", tickcolor=GRILLE)
    fig.update_yaxes(gridcolor=GRILLE, zerolinecolor=GRILLE,
                     linecolor=GRILLE, ticks="outside", tickcolor=GRILLE)
    return fig


def enregistrer(fig, nom):
    """Fragment Quarto : HTML Plotly dans un bloc brut ```{=html}.

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


# --------------------------------------------------------------------------- #
# Chargement
# --------------------------------------------------------------------------- #
conditions = pd.read_csv(DATA / "conditions.csv")
ok = conditions[conditions.status == "ok"].copy().reset_index(drop=True)
methods = pd.read_csv(DATA / "methods.csv")
bins = pd.read_csv(DATA / "bins.csv")

if SLUG_REF not in set(ok.slug):
    SLUG_REF = ok.sort_values("duree_s", ascending=False).slug.iloc[0]
ref = ok[ok.slug == SLUG_REF].iloc[0]
zref = np.load(DATA / ref.out_rel / "one_cycle_compare.npz")

N_BINS = int(bins.groupby(["slug", "methode"]).size().max())
A, B = METHODES
print(f"{len(ok)} conditions exploitables / {len(conditions)} parcourues")
print(f"condition de reference : {SLUG_REF}  ({ref.hr_BPM:.0f} BPM)")


def paire(colonne):
    """(valeurs FIR, valeurs M-SSA, slugs) alignees, lignes finies seulement."""
    x = ok[f"{colonne}_{A}"].to_numpy(float)
    y = ok[f"{colonne}_{B}"].to_numpy(float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m], ok.slug.to_numpy()[m]


def diagonale(fig, lo, hi):
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines", showlegend=False, hoverinfo="skip",
        line={"color": C_REF, "width": 1, "dash": "dash"}))


# =========================================================================== #
# 1. Une condition : la courbe d'epaisseur du cycle, par methode
# =========================================================================== #
# C'est la sortie qui compte en aval : deltaY(phase). Les moities sont tracees
# en pointille -- une phase juste les superpose, une phase fausse les disperse.
phase_axis = (np.arange(N_BINS) + 0.5) / N_BINS
fig = go.Figure()
for nom in METHODES:
    th = zref[f"thickness_{nom}"].astype(float)
    fig.add_trace(go.Scatter(
        x=phase_axis, y=th, mode="lines+markers", name=LABELS[nom],
        line={"color": COULEURS[nom], "width": 2.4},
        marker={"size": 7},
        hovertemplate="phase %{x:.2f}<br>%{y:.2f} px<extra>"
                      + LABELS[nom] + "</extra>"))
    for half, dash in (("a", "dot"), ("b", "dashdot")):
        key = f"thickness_half_{half}_{nom}"
        if key not in zref.files:
            continue
        fig.add_trace(go.Scatter(
            x=phase_axis, y=zref[key].astype(float), mode="lines",
            name=f"{LABELS[nom]} — half {half.upper()}",
            line={"color": COULEURS[nom], "width": 1.1, "dash": dash},
            opacity=0.75,
            hovertemplate="phase %{x:.2f}<br>%{y:.2f} px<extra>"
                          + LABELS[nom] + f" half {half.upper()}</extra>"))
fig.update_xaxes(title_text="cardiac phase (cycle fraction)", title_standoff=6)
fig.update_yaxes(title_text="choroidal thickness (px)")
mise_en_page(fig, f"Folded choroidal thickness, {SLUG_REF} "
                  f"({ref.hr_BPM:.0f} BPM, {N_BINS} bins) — solid: whole "
                  f"recording, dotted: each half folded on its own", hauteur=500)
enregistrer(fig, "f1_thickness_ref")


# =========================================================================== #
# 2. Une condition : occupation des bins
# =========================================================================== #
fig = go.Figure()
for nom in METHODES:
    c = zref[f"counts_{nom}"].astype(float)
    fig.add_trace(go.Bar(
        x=phase_axis, y=c, name=LABELS[nom], marker_color=COULEURS[nom],
        opacity=0.85,
        hovertemplate="phase %{x:.2f}<br>%{y:.0f} frames<extra>"
                      + LABELS[nom] + "</extra>"))
fig.add_hline(y=float(np.mean(zref[f"counts_{A}"])),
              line={"color": C_REF, "width": 1, "dash": "dash"})
fig.update_xaxes(title_text="cardiac phase (cycle fraction)", title_standoff=6)
fig.update_yaxes(title_text="frames in bin")
mise_en_page(fig, f"How the {N_BINS} bins are filled, {SLUG_REF} — a phase that "
                  f"lingers leaves bins uneven (dashed: uniform)", hauteur=420)
enregistrer(fig, "f2_bins_ref")


# =========================================================================== #
# 3. Cohorte : les deux phases sont-elles la meme phase ?
# =========================================================================== #
fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.11,
                    subplot_titles=("phase offset vs its own constancy",
                                    "correlation of the two folded cycles"))
fig.add_trace(go.Scatter(
    x=ok.phase_offset_deg, y=ok.phase_plv, mode="markers",
    name="condition", marker={"size": 8, "color": COULEURS[A], "opacity": 0.7,
                              "line": {"width": 0}},
    text=ok.slug,
    hovertemplate="%{text}<br>offset %{x:+.1f}°<br>PLV %{y:.2f}<extra></extra>"),
    row=1, col=1)
fig.add_vline(x=0, line={"color": C_REF, "width": 1, "dash": "dash"}, row=1, col=1)
fig.add_trace(go.Histogram(
    x=ok.cycle_corr, nbinsx=30, marker_color=COULEURS[B], opacity=0.8,
    name="conditions",
    hovertemplate="r %{x:.2f}<br>%{y} conditions<extra></extra>"), row=1, col=2)
fig.update_xaxes(title_text="FIR − M-SSA phase offset (°)", row=1, col=1)
fig.update_yaxes(title_text="phase-locking value", range=[0, 1.02], row=1, col=1)
fig.update_xaxes(title_text="corr(FIR cycle, M-SSA cycle)", row=1, col=2)
fig.update_yaxes(title_text="conditions", row=1, col=2)
mise_en_page(fig, "Do the two phases describe the same cardiac cycle?",
             hauteur=420, legende=False)
fig.update_annotations(font_size=11)
enregistrer(fig, "f3_phase_agreement")


# =========================================================================== #
# 4. Cohorte : reproductibilite moitie/moitie
# =========================================================================== #
# Le seul critere qui ne suppose pas de connaitre la bonne reponse. Trois
# representations du meme cycle, du plus bruite (pixel) au plus reduit
# (epaisseur) : c'est leur ACCORD qui fait la conclusion, pas l'une d'elles.
noms = [("split_half_r_pix", "per pixel"),
        ("split_half_r_kymo", "depth profile"),
        ("split_half_r_thick", "thickness curve")]

# 4a. Les trois methodes CONTRE LE TEMOIN. C'est la figure qui decide : sans la
# barre grise, un r de 0,04 n'a pas d'echelle.
fig = make_subplots(rows=3, cols=1, vertical_spacing=0.10,
                    subplot_titles=[lab for _, lab in noms])
for i, (col, _) in enumerate(noms, start=1):
    for nom in TOUTES:
        s = methods[methods.methode == nom]
        fig.add_trace(go.Box(
            x=s[col], name=LABELS[nom], marker_color=COULEURS[nom],
            boxpoints="all", jitter=0.55, pointpos=-1.7, orientation="h",
            marker={"size": 3.5, "opacity": 0.45}, line={"width": 1.4},
            showlegend=False, text=s.slug,
            hovertemplate="%{text}<br>r %{x:.3f}<extra>"
                          + LABELS[nom] + "</extra>"), row=i, col=1)
    fig.add_vline(x=0, line={"color": C_REF, "width": 1, "dash": "dash"},
                  row=i, col=1)
fig.update_xaxes(title_text="split-half correlation", row=3, col=1)
mise_en_page(fig, "Split-half reproducibility against a null phase — the grey "
                  "row is the FIR phase shuffled at random, and it is the only "
                  "scale these numbers have", hauteur=760, legende=False)
# Les noms de methode servent d'etiquettes de categorie : il leur faut la place.
fig.update_layout(margin_l=150)
fig.update_annotations(font_size=11)
enregistrer(fig, "f4_split_half_null")


# 4b. Condition par condition, FIR contre M-SSA.
fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.075,
                    subplot_titles=[f"{lab}" for _, lab in noms])
for i, (col, lab) in enumerate(noms, start=1):
    x, y, slugs = paire(col)
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    pad = 0.06 * (hi - lo + 1e-9)
    fig.add_trace(go.Scatter(
        x=[lo - pad, hi + pad], y=[lo - pad, hi + pad], mode="lines",
        showlegend=False, hoverinfo="skip",
        line={"color": C_REF, "width": 1, "dash": "dash"}), row=1, col=i)
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers", showlegend=False, text=slugs,
        marker={"size": 7, "opacity": 0.65, "line": {"width": 0},
                "color": np.where(x > y, COULEURS[A], COULEURS[B])},
        hovertemplate="%{text}<br>FIR %{x:.2f}<br>M-SSA %{y:.2f}<extra></extra>"),
        row=1, col=i)
    fig.update_xaxes(title_text="FIR", range=[lo - pad, hi + pad], row=1, col=i)
    fig.update_yaxes(title_text="M-SSA" if i == 1 else None,
                     range=[lo - pad, hi + pad], row=1, col=i)
mise_en_page(fig, "Split-half reproducibility, condition by condition — above "
                  "the diagonal M-SSA wins, below it FIR does", hauteur=420,
             legende=False)
fig.update_annotations(font_size=11)
enregistrer(fig, "f4b_split_half_paired")


# =========================================================================== #
# 5. Cohorte : la consequence en aval, deltaY
# =========================================================================== #
x, y, slugs = paire("deltaY_px")
lo, hi = 0.0, max(x.max(), y.max()) * 1.05
fig = go.Figure()
diagonale(fig, lo, hi)
fig.add_trace(go.Scatter(
    x=x, y=y, mode="markers", showlegend=False, text=slugs,
    marker={"size": 8, "opacity": 0.7, "line": {"width": 0},
            "color": ok.hr_BPM[np.isfinite(ok[f"deltaY_px_{A}"])
                               & np.isfinite(ok[f"deltaY_px_{B}"])],
            "colorscale": "Viridis", "showscale": True,
            "colorbar": {"title": {"text": "HR<br>(BPM)", "font": {"size": 10}},
                         "thickness": 12, "len": 0.8}},
    hovertemplate="%{text}<br>FIR %{x:.2f} px<br>M-SSA %{y:.2f} px<extra></extra>"))
# Ce que la phase permutee obtient sur les memes donnees : une amplitude de
# repliement n'est un signal qu'au-dessus de cette ligne-la. En crete-a-crete,
# elle ne l'est pas -- le temoin est MEDIANEMENT PLUS HAUT que les deux
# methodes, parce qu'un max-moins-min sur dix bins bruites mesure surtout le
# bruit.
dy_nul = float(methods[methods.methode == NUL].deltaY_px.median())
for axe in ("v", "h"):
    kw = {"x" if axe == "v" else "y": dy_nul}
    (fig.add_vline if axe == "v" else fig.add_hline)(
        line={"color": COULEURS[NUL], "width": 1, "dash": "dot"}, **kw)
fig.add_annotation(x=dy_nul, y=hi, text="median ΔY of the shuffled-phase null",
                   showarrow=False, xanchor="left", yanchor="top",
                   font={"size": 10, "color": COULEURS[NUL]})
fig.update_xaxes(title_text="ΔY peak-to-peak, FIR phase (px)", range=[lo, hi])
fig.update_yaxes(title_text="ΔY peak-to-peak, M-SSA phase (px)", range=[lo, hi])
mise_en_page(fig, "Pulsatile thickness amplitude recovered by each phase — "
                  "peak-to-peak of the folded thickness curve", hauteur=470,
             legende=False)
enregistrer(fig, "f5_deltaY")


# =========================================================================== #
# 6. Cohorte : uniformite du remplissage des bins
# =========================================================================== #
fig = go.Figure()
for nom in TOUTES:
    s = methods[methods.methode == nom]
    fig.add_trace(go.Box(
        x=s.bin_cv, name=LABELS[nom], marker_color=COULEURS[nom],
        boxpoints="all", jitter=0.5, pointpos=-1.6, orientation="h",
        marker={"size": 4, "opacity": 0.5}, line={"width": 1.4},
        text=s.slug,
        hovertemplate="%{text}<br>CV %{x:.3f}<extra>" + LABELS[nom] + "</extra>"))
fig.update_xaxes(title_text="coefficient of variation of the bin counts")
mise_en_page(fig, f"Are the {N_BINS} bins filled evenly? "
                  "(0 = perfectly uniform)", hauteur=380, legende=False)
fig.update_layout(margin_l=150)
enregistrer(fig, "f6_bin_uniformity")


# =========================================================================== #
# 7. Cohorte : ce que le desaccord de phase coute
# =========================================================================== #
# Si les deux phases sont d'accord (PLV eleve, decalage nul), les deux videos
# sont la meme video. Le nuage dit si l'ecart de reproductibilite se creuse la
# ou les phases divergent.
m = np.isfinite(ok.phase_plv) & np.isfinite(ok.split_half_gain_fir)
fig = go.Figure()
fig.add_hline(y=0, line={"color": C_REF, "width": 1, "dash": "dash"})
fig.add_trace(go.Scatter(
    x=ok.phase_plv[m], y=ok.split_half_gain_fir[m], mode="markers",
    showlegend=False, text=ok.slug[m],
    marker={"size": 8, "opacity": 0.7, "line": {"width": 0},
            "color": np.where(ok.split_half_gain_fir[m] > 0,
                              COULEURS[A], COULEURS[B])},
    hovertemplate="%{text}<br>PLV %{x:.2f}<br>gain %{y:+.2f}<extra></extra>"))
fig.update_xaxes(title_text="phase-locking value between the two phases")
fig.update_yaxes(title_text="split-half r, FIR − M-SSA (depth profile)")
mise_en_page(fig, "Where the two phases disagree, does one of them fold better?",
             hauteur=430, legende=False)
enregistrer(fig, "f7_disagreement_cost")


# =========================================================================== #
# Tables
# =========================================================================== #
def stat(s, fmt="{:.2f}"):
    """Mediane [Q1, Q2] -- une mediane seule masquerait la dispersion, qui est
    ici du meme ordre que l'ecart entre les deux methodes."""
    s = pd.Series(s).astype(float).dropna()
    if s.empty:
        return "—"
    return (f"{fmt.format(s.median())} "
            f"[{fmt.format(s.quantile(.25))}, {fmt.format(s.quantile(.75))}]")


def table_methodes():
    lignes = ["<!-- Genere par figures_one_cycle_compare/make_figures.py"
              " -- ne pas editer a la main. -->", "",
              "| | FIR band-pass | M-SSA | shuffled phase (null) |",
              "|:--|--:|--:|--:|"]
    rangs = [
        ("split-half r · per pixel", "split_half_r_pix", "{:.3f}"),
        ("split-half r · depth profile", "split_half_r_kymo", "{:.2f}"),
        ("split-half r · thickness curve", "split_half_r_thick", "{:.2f}"),
        ("ΔY, 1-harmonic fit (px)", "deltaY_fit_px", "{:.2f}"),
        ("ΔY, peak-to-peak (px)", "deltaY_px", "{:.2f}"),
        ("modulation depth", "mod_depth", "{:.3f}"),
        ("frames folded", "n_good", "{:.0f}"),
        ("bin-count CV", "bin_cv", "{:.3f}"),
        ("smallest bin", "bin_min", "{:.0f}"),
    ]
    for label, col, fmt in rangs:
        cells = [stat(methods[methods.methode == nom][col], fmt)
                 for nom in TOUTES]
        lignes.append(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lignes += ["", f"Median [IQR] over the {len(ok)} conditions. The null "
               "column is the FIR phase randomly permuted — same frames, same "
               "bins, no timing."]
    return "\n".join(lignes)


def table_cohorte():
    def ligne(label, s, fmt="{:.2f}"):
        s = pd.Series(s).astype(float).dropna()
        return (f"| {label} | {fmt.format(s.median())} | "
                f"{fmt.format(s.quantile(.25))} – {fmt.format(s.quantile(.75))} | "
                f"{fmt.format(s.min())} – {fmt.format(s.max())} |")

    lignes = ["<!-- Genere par figures_one_cycle_compare/make_figures.py"
              " -- ne pas editer a la main. -->", "",
              "| | median | IQR | range |", "|:--|--:|--:|--:|",
              ligne("reference HR (BPM)", ok.hr_BPM, "{:.0f}"),
              ligne("frames per recording", ok.n_frames, "{:.0f}"),
              ligne("recording length (s)", ok.duree_s, "{:.0f}"),
              ligne("frames folded, FIR", ok[f"n_good_{A}"], "{:.0f}"),
              ligne("ROI pixels", ok.roi_pixels, "{:.0f}"),
              ligne("FIR − M-SSA phase offset (°)", ok.phase_offset_deg, "{:.1f}"),
              ligne("phase-locking value", ok.phase_plv, "{:.2f}"),
              ligne("corr. of the two folded cycles", ok.cycle_corr, "{:.2f}")]
    return "\n".join(lignes)


(SORTIE / "tab_methods.qmd").write_text(table_methodes() + "\n", encoding="utf-8")
(SORTIE / "tab_cohort.qmd").write_text(table_cohorte() + "\n", encoding="utf-8")
print("  tab_methods.qmd\n  tab_cohort.qmd")


# =========================================================================== #
# Videos + pages par condition + table de liens
# =========================================================================== #
def fragment(fig, div_id):
    """Figure Plotly prete a etre collee dans un bloc ```{=html} d'une page."""
    return fig.to_html(full_html=False, include_plotlyjs=False, config=CONFIG,
                       div_id=div_id)


def fig_deltaY(z, r):
    """ΔY le long du cycle replie, les trois phases superposees.

    L'axe des x est le TEMPS DANS LE CYCLE REPLIE (s) : les bins sont des bins de
    phase, mais c'est en secondes que l'amplitude se lit contre une periode
    cardiaque. La fraction de phase reste dans l'infobulle.
    """
    periode = 60.0 / float(r.hr_BPM)
    phase = (np.arange(N_BINS) + 0.5) / N_BINS
    temps = phase * periode

    fig = go.Figure()
    for nom in TOUTES:
        cle = f"thickness_{nom}"
        if cle not in z.files:
            continue
        th = np.asarray(z[cle], float)
        p2p = th.max() - th.min()
        fig.add_trace(go.Scatter(
            x=np.round(temps, 4), y=np.round(th, 4), mode="lines+markers",
            name=f"{LABELS[nom]} — ΔY {p2p:.2f} px",
            line={"color": COULEURS[nom], "width": 2.2,
                  "dash": "dot" if nom == NUL else "solid"},
            marker={"size": 6},
            customdata=np.round(phase, 3),
            hovertemplate="%{x:.3f} s (phase %{customdata:.2f})<br>"
                          "%{y:.2f} px<extra>" + LABELS[nom] + "</extra>"))
    fig.update_xaxes(title_text="time within the averaged cycle (s)",
                     title_standoff=6, range=[0, periode])
    fig.update_yaxes(title_text="choroidal thickness (px)")
    mise_en_page(fig, f"ΔY along the folded cycle — {N_BINS} bins, one cardiac "
                      f"period ({periode * 1000:.0f} ms at {r.hr_BPM:.0f} BPM)",
                 hauteur=420)
    fig.update_layout(margin_b=95, legend_y=-0.24)
    return fig


def fig_pouls(z, zp, r):
    """Pouls FIR et M-SSA superposes, leur phase, et la HR instantanee.

    Trois panneaux qui partagent l'axe du temps : c'est l'empilement qui rend
    lisible le lien entre une phase qui derape et une frequence instantanee qui
    part. Les pouls sont normalises a l'unite -- leurs amplitudes sont en unites
    arbitraires et ne se comparent pas.
    """
    t = np.asarray(z["uniform_time"], float)
    bon = np.asarray(z["good_uniform"], bool)
    hr = float(r.hr_BPM)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.055,
        subplot_titles=("pulse (normalised)", "cardiac phase",
                        "instantaneous heart rate"))
    for nom in METHODES:
        y = np.asarray(z[f"pulse_{nom}"], float)
        y = y / (np.abs(y).max() + 1e-12)
        fig.add_trace(go.Scatter(
            x=np.round(t, 3), y=np.round(y, 4), mode="lines", name=LABELS[nom],
            legendgroup=nom, line={"color": COULEURS[nom], "width": 1.5},
            hovertemplate="%{x:.2f} s<br>%{y:.2f}<extra>"
                          + LABELS[nom] + "</extra>"), row=1, col=1)
        # Phase repliee, en fraction de cycle : plus parlant que des radians, et
        # c'est l'unite dans laquelle le repliement decoupe ses bins.
        ph = np.mod(np.asarray(z[f"phase_uniform_{nom}"], float), 2 * np.pi)
        fig.add_trace(go.Scatter(
            x=np.round(t, 3), y=np.round(ph / (2 * np.pi), 4), mode="lines",
            name=LABELS[nom], legendgroup=nom, showlegend=False,
            line={"color": COULEURS[nom], "width": 1.2},
            hovertemplate="%{x:.2f} s<br>phase %{y:.2f}<extra>"
                          + LABELS[nom] + "</extra>"), row=2, col=1)
        cle = f"bpm_{nom}"
        if zp is not None and cle in zp.files:
            fig.add_trace(go.Scatter(
                x=np.round(t, 3), y=np.round(np.asarray(zp[cle], float), 2),
                mode="lines", name=LABELS[nom], legendgroup=nom,
                showlegend=False, line={"color": COULEURS[nom], "width": 1.2},
                hovertemplate="%{x:.2f} s<br>%{y:.0f} BPM<extra>"
                              + LABELS[nom] + "</extra>"), row=3, col=1)
    fig.add_hline(y=hr, line={"color": C_REF, "width": 1, "dash": "dash"},
                  row=3, col=1)

    # Les bords ecartes du repliement (un cycle de chaque cote, identique pour
    # les deux methodes) : sans eux on lit un transitoire de filtre comme une
    # anomalie physiologique.
    if not bon.all():
        for x0, x1 in ((t[0], t[bon][0]), (t[bon][-1], t[-1])):
            if x1 > x0:
                fig.add_vrect(x0=x0, x1=x1, fillcolor=C_REF, opacity=0.12,
                              line_width=0)
    fig.update_xaxes(title_text="time (s)", title_standoff=6, row=3, col=1)
    fig.update_yaxes(title_text="a.u.", row=1, col=1)
    fig.update_yaxes(title_text="cycle fraction", range=[0, 1], row=2, col=1)
    fig.update_yaxes(title_text="BPM", row=3, col=1)
    # Plotly TRONQUE un titre trop long au lieu de le replier : le <br> est
    # explicite, et la marge haute suit.
    mise_en_page(fig, "The two pulses, their phase and the instantaneous rate"
                      f"<br><sub>dashed: reference {hr:.0f} BPM · shaded: "
                      "trimmed edges, not folded</sub>", hauteur=660)
    # Trois panneaux : la legende n'a qu'un cran a descendre sous le titre de
    # l'axe du bas, pas les 34 % calibres pour une figure a un seul panneau.
    fig.update_layout(margin={"l": 62, "r": 20, "t": 78, "b": 78},
                      legend_y=-0.10)
    fig.update_annotations(font_size=11)
    return fig


VIDEOS.mkdir(parents=True, exist_ok=True)
PAGES.mkdir(parents=True, exist_ok=True)
for vieux in list(VIDEOS.glob("*.mp4")) + list(PAGES.glob("oc_*.qmd")):
    vieux.unlink()

octets = 0
liens = []
for _, r in ok.iterrows():
    src = DATA / r.out_rel / "one_cycle_compare.mp4"
    dst = VIDEOS / f"{r.slug}.mp4"
    shutil.copyfile(src, dst)
    octets += dst.stat().st_size
    z = np.load(DATA / r.out_rel / "one_cycle_compare.npz")

    zp = TRACES_POULS / f"{r.slug}.npz"
    zp = np.load(zp) if zp.exists() else None

    gain = r[f"split_half_r_kymo_{A}"] - r[f"split_half_r_kymo_{B}"]
    vainqueur = "FIR" if gain > 0 else "M-SSA"
    corps = f"""---
title: "{r.astro} · {r.condition}"
subtitle: "One-cycle video folded on two competing phases — {r.hr_BPM:.0f} BPM, {N_BINS} bins"
toc: false
include-in-header: ../figures_one_cycle_compare/plotly_src.html
---

[← Back to the comparison](../comparing-one-cycle-videos.qmd)

```{{=html}}
<video class="oc-video" src="../figures_one_cycle_compare/videos/{r.slug}.mp4"
       autoplay loop muted playsinline controls></video>
```

**Top: FIR band-pass. Bottom: M-SSA.** Same frames, same {N_BINS} bins, same
folding — only the phase differs. The file holds one averaged cardiac cycle,
written three times over so it loops smoothly.

## ΔY along the folded cycle

```{{=html}}
{fragment(fig_deltaY(z, r), f"dy-{r.slug}")}
```

The shuffled phase is the dotted grey curve. Where it swings as widely as the
two real ones, the peak-to-peak amplitude on this condition is telling you about
the number of frames per bin, not about the choroid.

## The pulses behind the folds

```{{=html}}
{fragment(fig_pouls(z, zp, r), f"pulse-{r.slug}")}
```

Both pulses come from the same rank-100 SVD of the same pixel traces; they
differ only in what was done afterwards. The phase panel is what actually sorts
the frames into bins, and the instantaneous rate is the diagnostic: a phase that
is tracking the heartbeat holds near the dashed reference line.

| | FIR band-pass | M-SSA | shuffled (null) |
|:--|--:|--:|--:|
| split-half r · per pixel | {r[f'split_half_r_pix_{A}']:.3f} | {r[f'split_half_r_pix_{B}']:.3f} | {r[f'split_half_r_pix_{NUL}']:.3f} |
| split-half r · depth profile | {r[f'split_half_r_kymo_{A}']:.2f} | {r[f'split_half_r_kymo_{B}']:.2f} | {r[f'split_half_r_kymo_{NUL}']:.2f} |
| split-half r · thickness curve | {r[f'split_half_r_thick_{A}']:.2f} | {r[f'split_half_r_thick_{B}']:.2f} | {r[f'split_half_r_thick_{NUL}']:.2f} |
| ΔY, 1-harmonic fit (px) | {r[f'deltaY_fit_px_{A}']:.2f} | {r[f'deltaY_fit_px_{B}']:.2f} | {r[f'deltaY_fit_px_{NUL}']:.2f} |
| ΔY, peak-to-peak (px) | {r[f'deltaY_px_{A}']:.2f} | {r[f'deltaY_px_{B}']:.2f} | {r[f'deltaY_px_{NUL}']:.2f} |
| frames folded | {r[f'n_good_{A}']:.0f} | {r[f'n_good_{B}']:.0f} | {r[f'n_good_{NUL}']:.0f} |

The **null** column is the FIR phase permuted at random, folded the same way: it
is what these numbers look like when the phase carries no timing at all.

The two phases sit **{r.phase_offset_deg:+.1f}°** apart, and that offset holds with a
phase-locking value of **{r.phase_plv:.2f}**; the two folded cycles correlate at
**{r.cycle_corr:.2f}**. On the split-half test read on the depth profile,
**{vainqueur}** is ahead here by {abs(gain):.2f} — one condition proves nothing on
its own, the cohort figure does.

::: {{.source-note}}
Recording: `{r.astro}/{r.moment}/{r.condition}` — {r.n_frames:.0f} frames,
{r.duree_s:.0f} s, {r.fs_Hz:.1f} Hz, reference HR {r.hr_BPM:.0f} BPM
(`{r.hr_source}`). Videos, folded curves and the `.npz` diagnostics:
`E:/NASA_Rigidity/SegmentationVariations/model1_scale_1.0/one_cycle_compare/{r.out_rel}/`.
:::
"""
    (PAGES / f"oc_{r.slug}.qmd").write_text(corps, encoding="utf-8")
    # Le nom brut de la condition repete l'identifiant du sujet
    # ("210713001before_rigidity_OD") : on n'en garde que ce qui distingue une
    # ligne d'une autre, sinon la table deborde de la colonne de texte.
    visite = "before" if "before" in str(r.moment) else "post"
    oeil = str(r.condition).split("_rigidity_")[-1]
    liens.append(
        f"| {r.astro.split('_')[0]} | {visite} | {oeil} | {r.hr_BPM:.0f} | "
        f"{r[f'split_half_r_kymo_{A}']:.2f} | {r[f'split_half_r_kymo_{B}']:.2f} | "
        f"{r[f'deltaY_px_{A}']:.2f} | {r[f'deltaY_px_{B}']:.2f} | "
        f"{r[f'deltaY_px_{NUL}']:.2f} | "
        f"{r.cycle_corr:.2f} | [watch](one_cycles/oc_{r.slug}.qmd) |")

# La table est enveloppee dans un conteneur qui defile horizontalement : avec
# dix colonnes elle depasse la largeur de la colonne de texte sur un ecran
# etroit, et sans cela elle passe SOUS la table des matieres.
entete = [
    "<!-- Genere par figures_one_cycle_compare/make_figures.py"
    " -- ne pas editer a la main. -->", "",
    "::: {.table-scroll}", "",
    "| # | visit | eye | HR | r½ FIR | r½ M-SSA | ΔY p-p FIR | ΔY p-p M-SSA "
    "| ΔY p-p null | cycle r | one-cycle |",
    "|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|:--|",
]
(SORTIE / "_table.qmd").write_text(
    "\n".join(entete + liens + ["", ":::"]) + "\n", encoding="utf-8")

# La page principale inline plotly.js (une page, un fichier de 4,9 Mo).
(SORTIE / "plotlyjs.html").write_text(
    f"<script type='text/javascript'>{get_plotlyjs()}</script>", encoding="utf-8")
# Les 106 pages par condition ne peuvent pas faire pareil : elles le CHARGENT,
# toutes depuis le meme fichier, que le navigateur met en cache une fois pour
# toutes. Le detecteur de ressources de Quarto suit les <script src>, donc ce
# .js arrive dans _site tout seul (contrairement aux .mp4, cf.
# `copy_video_resources.py`).
(SORTIE / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")
(SORTIE / "plotly_src.html").write_text(
    '<script src="../figures_one_cycle_compare/plotly.min.js"></script>\n',
    encoding="utf-8")

# Chiffres cites en prose, regeneres pour qu'ils suivent les donnees.
resume = {
    "n_total": len(conditions),
    "n_ok": len(ok),
    "n_astro": int(ok.astro.nunique()),
    "n_bins": N_BINS,
    "offset_abs_median": float(np.median(np.abs(ok.phase_offset_deg))),
    "plv_median": float(ok.phase_plv.median()),
    "plv_sous_0.9": int((ok.phase_plv < 0.9).sum()),
    "cycle_corr_median": float(ok.cycle_corr.median()),
    "videos_Mo": octets / 1e6,
}
for nom in TOUTES:
    s = methods[methods.methode == nom]
    for col in ("split_half_r_pix", "split_half_r_kymo", "split_half_r_thick",
                "deltaY_fit_px", "deltaY_px", "mod_depth", "bin_cv",
                "good_frac"):
        resume[f"{col}_{nom}"] = float(s[col].median())
for col in ("split_half_r_pix", "split_half_r_kymo", "split_half_r_thick",
            "deltaY_fit_px", "deltaY_px"):
    x, y, _ = paire(col)
    resume[f"fir_gagne_{col}"] = 100.0 * float((x > y).mean())
    resume[f"p_fir_vs_mssa_{col}"] = wilcoxon(x, y).pvalue
    # Ce qui compte avant de departager les deux : chacune bat-elle le hasard ?
    # Test APPARIE (chaque condition contre son propre temoin) : la dispersion
    # entre conditions est enorme devant l'ecart mesure, un test non apparie ne
    # verrait rien.
    for nom in METHODES:
        v = ok[f"{col}_{nom}"].to_numpy(float)
        n = ok[f"{col}_{NUL}"].to_numpy(float)
        m = np.isfinite(v) & np.isfinite(n)
        resume[f"bat_temoin_{col}_{nom}"] = 100.0 * float((v[m] > n[m]).mean())
        resume[f"p_temoin_{col}_{nom}"] = wilcoxon(v[m], n[m]).pvalue
        resume[f"med_gain_{col}_{nom}"] = float(np.median(v[m] - n[m]))

(SORTIE / "resume.txt").write_text(
    "\n".join(f"{k} = {v:.4g}" if isinstance(v, float) else f"{k} = {v}"
              for k, v in resume.items()) + "\n", encoding="utf-8")

print(f"\n{len(FICHIERS)} figures + plotly.js + 2 tables + _table.qmd")
print(f"{len(ok)} videos ({octets / 1e6:.0f} Mo) -> {VIDEOS}")
print(f"{len(ok)} pages par condition -> {PAGES}")
for k, v in resume.items():
    print(f"  {k:<30}{v:.4g}" if isinstance(v, float) else f"  {k:<30}{v}")
