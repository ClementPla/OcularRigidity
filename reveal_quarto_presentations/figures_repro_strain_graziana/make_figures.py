# -*- coding: utf-8 -*-
"""
Figures et tables -- page « Strain des réplicats — GRAZIANA OD ».

Ce script NE LIT QUE des sorties deja ecrites ; il ne recale rien.

Entrees
-------
    E:/NASA_Rigidity/Reproducibility/SegmentationVariations/
        cascade_v9_1536x1024_hr_mediane/one_cycle_6beats_6bins/demons_strain_s15/
            bins.csv  markers.csv  icc.csv  fields/<slug>.npz
    (ecrits par ``Reproducibility/compute_one_cycle_strain.py``)

Sorties (fragments inclus par ``repro-strain-graziana.qmd``)
-----------------------------------------------------------
    fig_uy.qmd, fig_strain.qmd, fig_box.qmd, tab_icc.qmd, tab_resume.qmd,
    plotlyjs.html, resume.txt

Lancer DEPUIS LA RACINE DU DEPOT (ce dossier est une JONCTION vers E:) :
    python reveal_quarto_presentations/figures_repro_strain_graziana/make_figures.py
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
SOURCE = Path("E:/NASA_Rigidity/Reproducibility/SegmentationVariations/"
              "cascade_v9_1536x1024_hr_mediane/one_cycle_6beats_6bins/demons_strain_s15")
SLUGS = ("MODICA_GRAZIANA_OD1", "MODICA_GRAZIANA_OD2", "MODICA_GRAZIANA_OD3")

# `.absolute()` et surtout PAS `.resolve()` : ce dossier est une JONCTION vers E:.
SORTIE = Path(__file__).absolute().parent

COULEUR_TEXTE = "#c9c9c9"
GRILLE = "rgba(150,150,150,0.18)"
# Trois replicats : les trois premiers slots categoriels, valides toutes paires.
C_REPLICAT = {1: "#3987e5", 2: "#eb6834", 3: "#1baf7a"}
DS_Y, DS_X = 2, 3  # sous-echantillonnage des cartes (poids du HTML)

CONFIG = {"displaylogo": False, "responsive": True,
          "modeBarButtonsToRemove": ["select2d", "lasso2d"]}
NOMS = ("fig_epaisseur", "fig_ct_retine", "fig_uy", "fig_strain", "fig_box", "tab_icc", "tab_resume")
# Figure « Epaisseur et occupation des bins » (reprise de figures_repro_one_cycles).
RAMPE_CYCLES = [[0.0, "#86b6ef"], [1.0, "#184f95"]]
C_SEG = "#eb6834"
C_FRAMES = "rgba(150,150,150,0.35)"
RESUME: dict = {}
nl = chr(10)


# --------------------------------------------------------------------------- #
# Habillage commun (le meme que figures_repro_one_cycles)
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
    (SORTIE / f"{nom}.qmd").write_text("```{=html}\n" + html + "\n```\n", encoding="utf-8")
    print(f"  {nom}.qmd")


def ecrire_table(texte, nom):
    entete = ("<!-- Genere par figures_repro_strain_graziana/make_figures.py"
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


# --------------------------------------------------------------------------- #
# Donnees
# --------------------------------------------------------------------------- #
entrees = [SOURCE / "bins.csv", SOURCE / "markers.csv", SOURCE / "icc.csv",
           SOURCE / "retina_bins.csv", SOURCE / "retina_bands.csv"] + \
          [SOURCE / "fields" / f"{s}.npz" for s in SLUGS]
manquants = [p for p in entrees if not p.exists()]
if manquants:
    pourquoi = ("Le calcul n'a pas encore produit ces fichiers :" + nl + nl
                + nl.join(f"- `{p}`" for p in manquants) + nl + nl
                + "Lancer `Reproducibility/compute_one_cycle_strain.py`.")
    for nom in NOMS:
        absente(nom, pourquoi)
    (SORTIE / "plotlyjs.html").write_text("", encoding="utf-8")
    raise SystemExit(0)

BINS = pd.read_csv(SOURCE / "bins.csv")
MARK = pd.read_csv(SOURCE / "markers.csv")
ICC = pd.read_csv(SOURCE / "icc.csv")
FIELDS = {s: dict(np.load(SOURCE / "fields" / f"{s}.npz")) for s in SLUGS}
N_CYC = max(int(FIELDS[s]["n_groups"]) for s in SLUGS)
REGIONS = (MARK[["k", "region"]].drop_duplicates().sort_values("k")["region"].tolist())


def replicat(slug):
    return int(slug[-1])


RETINE = pd.read_csv(SOURCE / "retina_bins.csv")          # choroide par bin (+ retine globale, non tracee)
RETINE_BANDES = pd.read_csv(SOURCE / "retina_bands.csv")  # retine par bin ET par bande laterale


# --------------------------------------------------------------------------- #
# 0. Epaisseur et occupation des bins, choroide ET retine, une figure par replicat
# --------------------------------------------------------------------------- #
def figure_epaisseur(slug):
    from plotly.colors import sample_colorscale

    b = RETINE[RETINE["slug"] == slug].sort_values("slot")
    n_g = int(b["cycle"].max())
    n_bins = int(b["bin"].max()) + 1
    um = float(b["um_per_px_y"].iloc[0])
    couleurs = (sample_colorscale(RAMPE_CYCLES, np.linspace(0, 1, n_g))
                if n_g > 1 else [RAMPE_CYCLES[0][1]])

    bandes = RETINE_BANDES[RETINE_BANDES["slug"] == slug]
    regions = (bandes[["k", "region"]].drop_duplicates().sort_values("k")
               .itertuples(index=False))
    regions = list(regions)
    n_rows = 1 + len(regions)

    # Un panneau par bande, a abscisse partagee : les bandes vont de ~65 a ~89 px
    # (fovea au centre), et un axe commun ecraserait des pulsations de 0,2 px.
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.035,
                        row_heights=[0.3] + [0.7 / len(regions)] * len(regions),
                        specs=[[{"secondary_y": True}]] + [[{}]] * len(regions),
                        subplot_titles=["choroïde (moyenne des colonnes exploitables)"]
                        + [f"rétine · {r.region}" for r in regions])
    fig.add_trace(go.Bar(
        x=b["slot"], y=b["n_frames"], name="frames par bin", legendgroup="frames",
        marker={"color": C_FRAMES, "line": {"width": 0}},
        customdata=np.stack([b["cycle"], b["bin"]], axis=-1),
        hovertemplate="cycle %{customdata[0]} — bin %{customdata[1]} : %{y} frames<extra></extra>"),
        row=1, col=1, secondary_y=True)
    for g in range(1, n_g + 1):
        bg = b[b["cycle"] == g]
        lib = f"cycle {g}"
        c = couleurs[g - 1]
        cd_base = np.stack([bg["bin"], bg["n_frames"]], axis=-1)
        courbes = [
            (1, "choroid_thickness_px", dict(mode="lines+markers", name=lib, showlegend=True,
                                             line={"color": c, "width": 2}, marker={"size": 8}),
             "choroïde repliée"),
            (1, "choroid_thickness_fit_px", dict(mode="lines", name=lib, showlegend=False,
                                                 line={"color": c, "width": 1.5, "dash": "dash"}),
             "choroïde, harmonique"),
        ]
        for row, r in enumerate(regions, start=2):
            bb = bandes[(bandes["k"] == r.k) & (bandes["cycle"] == g)].sort_values("slot")
            courbes += [
                (row, bb, "retina_thickness_px",
                 dict(mode="lines+markers", name=lib, showlegend=False,
                      line={"color": c, "width": 2}, marker={"size": 7, "symbol": "diamond"}),
                 f"rétine {r.region}"),
                (row, bb, "retina_thickness_fit_px",
                 dict(mode="lines", name=lib, showlegend=False,
                      line={"color": c, "width": 1.5, "dash": "dash"}),
                 f"rétine {r.region}, harmonique"),
            ]
        for item in courbes:
            if len(item) == 4:  # panneau choroide : lignes de RETINE (table par bin)
                row, col, style, quoi = item
                src = bg
            else:
                row, src, col, style, quoi = item
            cd = np.column_stack([np.stack([src["bin"], src["n_frames"]], axis=-1),
                                  src[col].to_numpy() * um])
            fig.add_trace(go.Scatter(
                x=src["slot"], y=src[col], legendgroup=lib, customdata=cd, **style,
                hovertemplate=(f"{lib} — bin de phase %{{customdata[0]}} "
                               f"(%{{customdata[1]}} frames)<br>{quoi} : %{{y:.2f}} px "
                               f"(%{{customdata[2]:.1f}} µm)<extra></extra>")),
                row=row, col=1, **({"secondary_y": False} if row == 1 else {}))
        fig.add_trace(go.Scatter(
            x=bg["slot"], y=bg["choroid_thickness_seg_px"], mode="lines+markers",
            name="segmentation du one-cycle", legendgroup="seg", showlegend=(g == 1),
            line={"color": C_SEG, "width": 1.5, "dash": "dot"},
            marker={"size": 8, "symbol": "square"},
            hovertemplate=(lib + " — choroïde resegmentée : %{y:.2f} px<extra></extra>")),
            row=1, col=1, secondary_y=False)
        if g > 1:
            for row in range(1, n_rows + 1):
                fig.add_vline(x=(g - 1) * n_bins - 0.5, row=row, col=1,
                              line={"color": "rgba(150,150,150,0.55)", "width": 1, "dash": "dot"})
    fig.update_yaxes(title_text="px", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="frames par bin", row=1, col=1, secondary_y=True,
                     showgrid=False, rangemode="tozero")
    for row in range(2, n_rows + 1):
        fig.update_yaxes(title_text="px", row=row, col=1)
    fig.update_xaxes(title_text=f"bin ({n_g} one-cycles de {n_bins} bins, bout à bout)",
                     row=n_rows, col=1)
    ptp_cho = np.nanmedian([np.ptp(b["choroid_thickness_fit_px"][b["cycle"] == g])
                            for g in range(1, n_g + 1)])
    RESUME[f"epaisseur_choroide_{slug[-3:]}"] = (
        f"{b['choroid_thickness_px'].mean():.2f} px, harm. ptp med {ptp_cho:.2f} px")
    for r in regions:
        bb = bandes[bandes["k"] == r.k]
        ptp_k = np.nanmedian([np.ptp(x["retina_thickness_fit_px"]) for _, x in bb.groupby("cycle")])
        RESUME[f"epaisseur_retine_{slug[-3:]}_{r.region}"] = (
            f"{bb['retina_thickness_px'].mean():.2f} px (sd {bb['retina_thickness_px'].std():.2f}), "
            f"harm. ptp med {ptp_k:.2f} px")
    fig = mise_en_page(
        fig, f"{slug} — {n_g} one-cycles × {n_bins} bins<br>plein : masques repliés · tirets : "
             f"harmonique · pointillé orange : segmentation du one-cycle", hauteur=1150)
    fig.update_layout(barmode="overlay", margin={"t": 100, "b": 120}, legend={"y": -0.075})
    return fig


def fig_epaisseur():
    blocs = []
    for s in SLUGS:
        blocs.append(figure_epaisseur(s).to_html(full_html=False, include_plotlyjs=False,
                                                 config=CONFIG, div_id=f"plot-epaisseur-{s[-3:]}"))
    (SORTIE / "fig_epaisseur.qmd").write_text(
        "".join("```{=html}\n" + h + "\n```\n\n" for h in blocs), encoding="utf-8")
    print("  fig_epaisseur.qmd")


fig_epaisseur()


# --------------------------------------------------------------------------- #
# 0 bis. Choroide (x) contre retine (y), par bande, un graphique par replicat
# --------------------------------------------------------------------------- #
def figure_ct_retine(slug):
    from plotly.colors import sample_colorscale

    d = RETINE_BANDES[RETINE_BANDES["slug"] == slug].sort_values(["k", "slot"])
    regions = list(d[["k", "region"]].drop_duplicates().sort_values("k").itertuples(index=False))
    n_g = int(d["cycle"].max())
    um = float(d["um_per_px_y"].iloc[0])
    couleurs = (sample_colorscale(RAMPE_CYCLES, np.linspace(0, 1, n_g))
                if n_g > 1 else [RAMPE_CYCLES[0][1]])

    # Une colonne par bande, axes PROPRES a chaque bande : la retine va de ~65 a
    # ~89 px d'une bande a l'autre, et la trajectoire d'un cycle tient en ~1 px.
    fig = make_subplots(rows=1, cols=len(regions), horizontal_spacing=0.045,
                        subplot_titles=[r.region for r in regions])
    for j, r in enumerate(regions, start=1):
        dk = d[d["k"] == r.k]
        for g in range(1, n_g + 1):
            dg = dk[dk["cycle"] == g].sort_values("bin")
            # Le cycle est ferme (dernier bin -> bin 0) : une boucle se lit comme
            # une boucle, et pas comme un segment ouvert.
            x = np.append(dg["choroid_thickness_px"].to_numpy(), dg["choroid_thickness_px"].iloc[0])
            y = np.append(dg["retina_thickness_px"].to_numpy(), dg["retina_thickness_px"].iloc[0])
            b = np.append(dg["bin"].to_numpy(), dg["bin"].iloc[0])
            nf = np.append(dg["n_frames"].to_numpy(), dg["n_frames"].iloc[0])
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines+markers", name=f"cycle {g}", legendgroup=f"cycle {g}",
                showlegend=(j == 1), line={"color": couleurs[g - 1], "width": 1.5},
                marker={"color": couleurs[g - 1],
                        "size": [11 if v == 0 else 7 for v in b],
                        "symbol": ["star" if v == 0 else "circle" for v in b]},
                customdata=np.column_stack([b, nf, x * um, y * um]),
                hovertemplate=(f"{r.region} · cycle {g} · bin %{{customdata[0]}} "
                               "(%{customdata[1]} frames)<br>choroïde %{x:.2f} px "
                               "(%{customdata[2]:.1f} µm)<br>rétine %{y:.2f} px "
                               "(%{customdata[3]:.1f} µm)<extra></extra>")),
                row=1, col=j)
        fig.update_xaxes(title_text="choroïde (px)", row=1, col=j, nticks=4)
    fig.update_yaxes(title_text="rétine (px)", row=1, col=1)
    fig = mise_en_page(fig, f"{slug} — épaisseur de la rétine contre épaisseur de la choroïde, "
                            "par bande<br>un point par bin · étoile : bin 0 (maximum du pouls) · "
                            "une couleur par one-cycle", hauteur=420)
    fig.update_layout(margin={"t": 100, "b": 130}, legend={"y": -0.3})
    return fig


def fig_ct_retine():
    blocs = [figure_ct_retine(s).to_html(full_html=False, include_plotlyjs=False, config=CONFIG,
                                         div_id=f"plot-ct-retine-{s[-3:]}") for s in SLUGS]
    (SORTIE / "fig_ct_retine.qmd").write_text(
        "".join("```{=html}\n" + h + "\n```\n\n" for h in blocs), encoding="utf-8")
    print("  fig_ct_retine.qmd")


fig_ct_retine()


# --------------------------------------------------------------------------- #
# 1-2. Cartes : 5 one-cycles x 3 replicats
# --------------------------------------------------------------------------- #
def carte(nom, cle, echelle, titre, unite, fmt):
    """``cle`` : 'u_y' (px -> um via ``echelle``) ou 'strain' (sans dimension)."""
    titres = []
    for g in range(N_CYC):
        for s in SLUGS:
            b = BINS[(BINS["slug"] == s) & (BINS["one_cycle"] == g + 1) & BINS["is_thick"]]
            if b.empty:
                titres.append(f"{s[-3:]} · cycle {g + 1} : absent")
            else:
                r = b.iloc[0]
                titres.append(f"{s[-3:]} · cycle {g + 1} · bins {int(r.ref_bin)}→{int(r.bin)} · "
                              f"ΔCT {fr(r.dCT_um)} µm")
    fig = make_subplots(rows=N_CYC, cols=len(SLUGS), subplot_titles=titres,
                        shared_xaxes=True, shared_yaxes=True,
                        horizontal_spacing=0.02, vertical_spacing=0.035)

    valeurs = []
    for s in SLUGS:
        f = FIELDS[s]
        for g in range(int(f["n_groups"])):
            if f"c{g}__{cle}" in f:
                v = f[f"c{g}__{cle}"] * (f["um_y"] if echelle else 1.0)
                valeurs.append(np.abs(v[f["tissue"]]))
    lim = float(np.percentile(np.concatenate(valeurs), 99)) if valeurs else 1.0

    for j, s in enumerate(SLUGS):
        f = FIELDS[s]
        um_y, um_x = float(f["um_y"]), float(f["um_x"])
        tissue = f["tissue"]
        h, w = tissue.shape
        y = np.arange(0, h, DS_Y) * um_y
        x = np.arange(0, w, DS_X) * um_x
        # Frontieres des bandes : premiere colonne de chaque masque de bande.
        bords = [int(np.flatnonzero(m.any(axis=0))[0]) for m in f["column_masks"][1:]]
        for g in range(N_CYC):
            if f"c{g}__{cle}" not in f:
                continue
            z = f[f"c{g}__{cle}"] * (um_y if echelle else 1.0)
            z = np.where(tissue, z, np.nan)[::DS_Y, ::DS_X].astype(np.float32)
            fig.add_trace(go.Heatmap(
                x=x, y=y, z=z, coloraxis="coloraxis",
                hovertemplate=(f"{s} · cycle {g + 1}<br>latéral %{{x:.0f}} µm · profondeur "
                               f"%{{y:.0f}} µm<br>{unite} %{{z:{fmt}}}<extra></extra>")),
                row=g + 1, col=j + 1)
            for xb in bords:
                fig.add_vline(x=xb * um_x, line={"color": "rgba(200,200,200,0.45)",
                                                 "width": 1, "dash": "dot"},
                              row=g + 1, col=j + 1)
    fig.update_yaxes(autorange="reversed")
    for g in range(N_CYC):
        fig.update_yaxes(title_text="profondeur (µm)" if g == N_CYC // 2 else None,
                         row=g + 1, col=1)
    for j in range(len(SLUGS)):
        fig.update_xaxes(title_text="latéral (µm)", row=N_CYC, col=j + 1)
    fig = mise_en_page(fig, titre, hauteur=210 * N_CYC + 120, legende=False)
    fig.update_layout(
        margin={"t": 90, "b": 60},
        coloraxis={"colorscale": "RdBu_r", "cmin": -lim, "cmax": lim, "cmid": 0,
                   "colorbar": {"title": {"text": unite, "side": "right"},
                                "thickness": 12, "len": 0.5}})
    for a in fig.layout.annotations:
        a.font.size = 10
    enregistrer(fig, nom)
    return lim


RESUME["uy_lim_p99_um"] = fr(carte(
    "fig_uy", "u_y", True,
    "Déplacement axial u_y (µm), bin mince → bin épais, rétine + choroïde<br>"
    "une ligne par one-cycle, une colonne par réplicat · pointillés : bandes latérales",
    "u_y (µm)", ".2f"), 2)
RESUME["strain_lim_p99"] = f"{carte('fig_strain', 'strain', False, 'Strain axial e_yy = ∂u_y/∂y, bin mince → bin épais, rétine + choroïde<br>une ligne par one-cycle, une colonne par réplicat · pointillés : bandes latérales', 'e_yy', '.4f'):.4f}"


# --------------------------------------------------------------------------- #
# 3. Boxplots du strain retinien moyen par region
# --------------------------------------------------------------------------- #
def fig_box():
    fig = go.Figure()
    for s in SLUGS:
        rep = replicat(s)
        d = MARK[MARK["slug"] == s].sort_values(["k", "one_cycle"])
        fig.add_trace(go.Box(
            x=d["region"], y=d["strain_retine"] * 1e3, name=f"réplicat {rep} ({s[-3:]})",
            offsetgroup=str(rep), marker={"color": C_REPLICAT[rep], "size": 8},
            line={"color": C_REPLICAT[rep], "width": 2}, fillcolor="rgba(0,0,0,0)",
            boxpoints="all", jitter=0.35, pointpos=0,
            customdata=np.stack([d["one_cycle"], d["dCT_region_um"]], axis=-1),
            hovertemplate=(f"réplicat {rep} · %{{x}}<br>cycle %{{customdata[0]}} : "
                           "%{y:.3f} × 10⁻³<br>dCT bande %{customdata[1]:.1f} µm<extra></extra>")))
    fig.add_hline(y=0, line={"color": "rgba(150,150,150,0.7)", "width": 1})
    fig.update_xaxes(categoryorder="array", categoryarray=REGIONS, title_text="bande latérale")
    fig.update_yaxes(title_text="strain rétinien moyen (× 10⁻³)")
    fig = mise_en_page(fig, "Strain rétinien moyen par bande, bin mince → bin épais<br>"
                            "un box par réplicat · un point par one-cycle", hauteur=500)
    fig.update_layout(boxmode="group", boxgroupgap=0.25, boxgap=0.15,
                      margin={"t": 90, "b": 110}, legend={"y": -0.2})
    enregistrer(fig, "fig_box")


fig_box()


# --------------------------------------------------------------------------- #
# 4. ICC et resume
# --------------------------------------------------------------------------- #
def tab_icc():
    lignes = ["| marqueur | ICC(1,1) | ICC(A,1) | ICC(C,1) |", "|:--|--:|--:|--:|"]
    for col, d in ICC.groupby("marqueur", sort=False):
        d = d.set_index("type")
        cells = []
        for typ in ("ICC(1,1)", "ICC(A,1)", "ICC(C,1)"):
            r = d.loc[typ]
            cells.append(f"**{fr(r.icc, 2)}** [{fr(r.ci_bas, 2)} ; {fr(r.ci_haut, 2)}] "
                         f"p = {fr(r.p, 3)}")
            RESUME[f"icc_{col}_{typ}"] = f"{r.icc:.3f} [{r.ci_bas:.2f}, {r.ci_haut:.2f}] p={r.p:.3f}"
        lignes.append(f"| {d['libelle'].iloc[0]} | " + " | ".join(cells) + " |")
    manuel = ICC.dropna(subset=["icc11_manuel"])
    RESUME["icc11_ecart_max_pingouin_manuel"] = f"{(manuel.icc - manuel.icc11_manuel).abs().max():.2e}"
    ecrire_table(nl.join(lignes) + nl + nl
                 + f": ICC entre réplicats, sujets = les {int(ICC.n_regions.iloc[0])} bandes "
                   f"latérales, évaluateurs = les {int(ICC.n_replicats.iloc[0])} réplicats, valeur "
                   "= médiane des one-cycles de la paire bin mince → bin épais. Entre crochets : "
                   "IC 95 %. Avec 5 sujets seulement, les intervalles sont très larges. "
                   "{.striped}", "tab_icc")


def tab_resume():
    b = BINS[BINS["is_thick"]].sort_values(["slug", "one_cycle"])
    lignes = ["| réplicat | cycle | bins mince → épais | frames (épais) | ΔCT (µm) | |u| max (µm) "
              "| jacobien ≤ 0 (%) | RMS avant → après | strain rétine moy. (× 10⁻³) "
              "| strain choroïde méd. (× 10⁻³) |",
              "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in b.itertuples():
        lignes.append(f"| {r.slug} | {r.one_cycle} | {r.ref_bin} → {r.bin} | {r.n_frames_bin} "
                      f"| {fr(r.dCT_um)} | {fr(r.u_max_um)} | {fr(r.jac_neg_pct, 3)} "
                      f"| {fr(r.rms_avant, 2)} → {fr(r.rms_apres, 2)} "
                      f"| {fr(r.strain_retine_moy * 1e3, 3)} | {fr(r.strain_choroide_med * 1e3, 3)} |")
    RESUME["dCT_um_med"] = fr(b["dCT_um"].median())
    RESUME["u_max_um_med"] = fr(b["u_max_um"].median())
    RESUME["jac_neg_pct_max"] = fr(b["jac_neg_pct"].max(), 3)
    RESUME["rms_reduction_med_pct"] = fr(100 * (1 - b["rms_apres"] / b["rms_avant"]).median())
    RESUME["strain_retine_moy_med_e3"] = fr(b["strain_retine_moy"].median() * 1e3, 3)
    RESUME["strain_choroide_med_med_e3"] = fr(b["strain_choroide_med"].median() * 1e3, 3)
    for s in SLUGS:
        d = MARK[MARK["slug"] == s]
        RESUME[f"strain_retine_med_{s[-3:]}"] = " | ".join(
            f"{reg}: {fr(d[d.region == reg].strain_retine.median() * 1e3, 3)}" for reg in REGIONS)
    ecrire_table(nl.join(lignes) + nl + nl
                 + ": Une ligne par (réplicat, one-cycle), paire marqueur seulement. ΔCT : "
                   "épaisseur choroïdienne du bin épais moins celle du bin mince. RMS : écart "
                   "d'intensité à l'image fixe dans le tissu, avant et après démons. "
                   "{.striped}", "tab_resume")


tab_icc()
tab_resume()

(SORTIE / "plotlyjs.html").write_text(
    '<script type="text/javascript">' + nl + get_plotlyjs() + nl + "</script>" + nl,
    encoding="utf-8")
print("  plotlyjs.html")
(SORTIE / "resume.txt").write_text(nl.join(f"{k} = {v}" for k, v in RESUME.items()) + nl,
                                   encoding="utf-8")
print("  resume.txt")
