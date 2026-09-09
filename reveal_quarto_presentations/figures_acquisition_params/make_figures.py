# -*- coding: utf-8 -*-
"""
Figures et tables -- page « Acquisition parameters » (section Methods).

Ce script NE LIT QUE les tables ecrites par
``Astronauts/compute_acquisition_params.py`` ; il ne rouvre aucun XML et ne
recalcule rien. Toute correction de fond (un champ oublie, une frequence mal
deduite) se fait la-bas, pas ici.

Entrees
-------
    E:/NASA_Rigidity/AcquisitionParameters/
        constants.csv    parametres identiques sur les 107 conditions
        conditions.csv   1 ligne / condition, tous les parametres
        series.csv       1 ligne / B-scan OCT (qualite, ART, intervalle)

Sorties (fragments Quarto, inclus par ``sans-acquisition-parameters.qmd``)
-------------------------------------------------------------------------
    tab_communs.qmd       table markdown des parametres communs
    tab_conditions.qmd    LA grande table : une COLONNE par condition
    tab_resume.qmd        distribution de chaque parametre variable
    hist_frequence.qmd    frequence moyenne par video + intervalles inter-images
    hist_numeriques.qmd   histogramme de chaque parametre numerique variable
    hist_categoriels.qmd  effectifs de chaque parametre categoriel variable
    hist_angle.qmd        lignes de balayage dans le plan du fond d'oeil + angle
    hist_qualite.qmd      qualite d'image : par condition et par B-scan
    plotlyjs.html         plotly.js, insere une seule fois par page
    resume.txt            les chiffres cites dans la prose de la page

La grande table est du HTML BRUT et non une table markdown : 107 colonnes ne
tiennent dans aucune largeur de page, il lui faut un conteneur qui defile
horizontalement, une premiere colonne collante et un en-tete collant. Pandoc ne
sait produire aucun des trois a partir d'une table pipe.

Le fond est transparent et le texte gris : les memes figures passent sur le
theme clair et sur le theme sombre du site.

Lancer DEPUIS LA RACINE DU DEPOT (ce dossier est une JONCTION vers E:, un chemin
relatif calcule ici sortirait du depot) :
    python reveal_quarto_presentations/figures_acquisition_params/make_figures.py
"""

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------- #
# Entrees / sorties
# --------------------------------------------------------------------------- #
SOURCE = Path("E:/NASA_Rigidity/AcquisitionParameters")

# `.absolute()` et surtout PAS `.resolve()` : ce dossier est une JONCTION vers
# E:, la resolution ferait sortir SORTIE du depot.
SORTIE = Path(__file__).absolute().parent

COULEUR_TEXTE = "#c9c9c9"
GRILLE = "rgba(150,150,150,0.18)"
C_BARRE = "#2a78d6"
C_BARRE2 = "#27a567"
C_REF = "#c2453f"

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
                "y": -0.08, "xanchor": "left", "x": 0},
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
    """Fragment Quarto : le HTML Plotly dans un bloc brut ```{=html}.

    Sans ce bloc, `{{< include >}}` livre le fragment au lecteur markdown de
    Pandoc, qui prend le `<div>` de Plotly pour un div natif et se plaint de ne
    pas trouver sa fermeture.
    """
    html = fig.to_html(full_html=False, include_plotlyjs=False,
                       config=CONFIG, div_id=f"plot-{nom}")
    (SORTIE / f"{nom}.qmd").write_text("```{=html}\n" + html + "\n```\n",
                                       encoding="utf-8")
    FICHIERS.append(f"{nom}.qmd")
    print(f"  {nom}.qmd")


def ecrire_table(texte, nom):
    entete = ("<!-- Genere par figures_acquisition_params/make_figures.py"
              " -- ne pas editer a la main. -->" + chr(10) + chr(10))
    (SORTIE / f"{nom}.qmd").write_text(entete + texte + chr(10), encoding="utf-8")
    FICHIERS.append(f"{nom}.qmd")
    print(f"  {nom}.qmd")


def fr(x, chiffres=2):
    """Nombre a virgule decimale, comme le reste du site."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{{:.{chiffres}f}}".format(x).replace(".", ",")


def texte_valeur(v) -> str:
    """Une valeur de parametre, telle qu'elle doit s'AFFICHER.

    Trois corrections, qui reviennent partout : le booleen d'un export XML se
    lit « oui / non » et non « True » ; un entier stocke en flottant (pandas
    promeut la colonne des le premier NaN) ne doit pas s'afficher « 100.0 » ;
    et l'absence de champ se dit, plutot que de disparaitre du decompte -- dix
    conditions n'ont pas de balise ``EVI`` du tout, ce qui est une information.
    """
    if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
        return "absent"
    if isinstance(v, (bool, np.bool_)):
        return "oui" if v else "non"
    s = str(v).strip()
    if s in VALEUR_LISIBLE:
        return VALEUR_LISIBLE[s]
    try:
        f = float(s)
    except ValueError:
        return s
    if f == int(f) and abs(f) < 1e15:
        return str(int(f))
    return s.replace(".", ",")


def colonne_cat(col) -> pd.Series:
    """Colonne categorielle prete a compter/tracer, valeurs deja mises en forme.

    ``.astype(str)`` ne suffit pas : depuis pandas 2, il laisse les NaN d'une
    colonne objet en NaN, que ``value_counts`` ecarte ensuite silencieusement --
    les dix conditions sans ``EVI`` disparaissaient de la table sans que le
    total 107 ne soit jamais affiche pour le trahir.
    """
    return cond[col].map(texte_valeur)


def barres_binnees(valeurs, n_bins, couleur, unite, fmt="{:.3g}"):
    """Histogramme PRE-BINNE, rendu en ``go.Bar`` plutot qu'en ``go.Histogram``.

    ``go.Histogram`` embarque le vecteur BRUT dans le HTML et bine dans le
    navigateur : sur les 47 301 B-scans, cela faisait 500 Ko de fragment par
    figure, et une page de 6 Mo qui mettait plusieurs secondes a se peindre. Le
    binning est identique -- largeur constante, memes bords -- mais seuls les
    ``n_bins`` effectifs voyagent. Zoom, survol et bascule de serie restent
    exactement ceux de Plotly ; c'est le seul point ou l'interactivite ne perd
    rien a ce qu'on compte avant d'envoyer.
    """
    v = pd.Series(valeurs).dropna().to_numpy(dtype=float)
    comptes, bords = np.histogram(v, bins=n_bins)
    centres = 0.5 * (bords[:-1] + bords[1:])
    return go.Bar(
        x=centres, y=comptes, width=np.diff(bords), marker_color=couleur,
        customdata=np.stack([bords[:-1], bords[1:]], axis=1),
        hovertemplate=("%{customdata[0]:" + fmt[2:-1] + "} – "
                       "%{customdata[1]:" + fmt[2:-1] + "} : %{y} "
                       + unite + "<extra></extra>"))


def med_iqr(s, chiffres=2):
    s = pd.Series(s).dropna()
    if s.empty:
        return "—"
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return f"{fr(s.median(), chiffres)} [{fr(q1, chiffres)} – {fr(q3, chiffres)}]"


# --------------------------------------------------------------------------- #
# Donnees
# --------------------------------------------------------------------------- #
cst = pd.read_csv(SOURCE / "constants.csv")
cond = pd.read_csv(SOURCE / "conditions.csv")
serie = pd.read_csv(SOURCE / "series.csv")

# Etiquette de colonne : sujet, moment, oeil (+ rang du replicat). Le nom complet
# du dossier (`210713001before_rigidity_OS2`) est illisible en en-tete ; il reste
# accessible en infobulle (`title=`) sur la cellule.
cond["sujet"] = cond["astro"].str.slice(0, 2)
cond["phase"] = np.where(cond["moment"].str.contains("before"), "before", "post")
cond["oeil"] = cond["condition"].str.rsplit("_", n=1).str[-1]
cond = cond.sort_values(["astro", "moment", "condition"]).reset_index(drop=True)

N = len(cond)
RESUME["n_conditions"] = N
RESUME["n_bscans"] = len(serie)
RESUME["n_sujets"] = cond["astro"].nunique()
RESUME["n_constantes"] = len(cst)


# --------------------------------------------------------------------------- #
# 1. Les parametres communs a TOUTES les acquisitions
# --------------------------------------------------------------------------- #
# Libelle lisible + balise XML d'origine, pour qu'un lecteur puisse rouvrir un
# export et retrouver la ligne. L'ordre est celui du tableau, pas celui du CSV :
# appareil, puis B-scan OCT, puis localisateur infrarouge.
COMMUNS = [
    ("manufacturer", "Fabricant", "`GeneralEquipment/Manufacturer`"),
    ("model_name", "Appareil", "`ManufacturerModelName`"),
    ("modality", "Modality", "`Series/Modality`"),
    ("modality_procedure", "Modality-procedure", "`Series/ModalityProcedure`"),
    ("series_type", "Type de série", "`Series/Type`"),
    ("examined_structure", "Structure examinée", "`Series/ExaminedStructure`"),
    ("images_per_series", "Images par série", "`Series/NumImages`"),
    ("export_type", "Type d'export", "`BODY/ExportType`"),
    ("sw_viewing", "Logiciel d'export", "`SWVersion/Name`"),
    ("oct_width_px", "OCT — largeur (pixels)", "`Width`"),
    ("oct_height_px", "OCT — hauteur (pixels)", "`Height`"),
    ("oct_scale_y_mm", "OCT — scale Y (mm/pixel)", "`ScaleY`"),
    ("oct_field_width_deg", "OCT — champ balayé (°)", "`OCTFieldSize/Width`"),
    ("oct_resolution", "OCT — résolution", "`Resolution`"),
    ("oct_edi", "OCT — EDI (imagerie profondeur accrue)", "`EDI`"),
    ("oct_position_tolerance", "OCT — position dans la tolérance",
     "`PositionWithinTolerance`"),
    ("ir_width_px", "IR — largeur (pixels)", "`Width`"),
    ("ir_height_px", "IR — hauteur (pixels)", "`Height`"),
    ("ir_angle_deg", "IR — angle (°)", "`Angle`"),
    ("ir_resolution", "IR — résolution", "`Resolution`"),
    ("ir_light_source", "IR — source lumineuse", "`ImageType/LightSource`"),
    ("laterality_matches_folder", "Latéralité XML = latéralité du dossier",
     "`Series/Laterality`"),
    ("angle_etendue_deg", "Variation de l'angle au sein d'une vidéo (°)",
     "`Start` / `End`"),
]

VALEUR_LISIBLE = {"true": "oui", "false": "non", "True": "oui", "False": "non"}


def table_communs() -> str:
    vals = dict(zip(cst["parametre"], cst["valeur"]))
    lignes = ["| paramètre | valeur | balise XML |", "|:--|:--|:--|"]
    vus = set()
    for col, label, balise in COMMUNS:
        if col not in vals:
            continue
        vus.add(col)
        lignes.append(f"| {label} | {texte_valeur(vals[col])} | {balise} |")
    # Filet de securite : si `compute_acquisition_params.py` gagne un champ
    # constant que la liste ci-dessus ignore, il apparait quand meme, sous son
    # nom de colonne brut, plutot que de disparaitre sans bruit de la page.
    for _, r in cst.iterrows():
        if r["parametre"] not in vus:
            lignes.append(f"| `{r['parametre']}` | {texte_valeur(r['valeur'])} | — |")
    return chr(10).join(lignes)


ecrire_table(table_communs(), "tab_communs")


# --------------------------------------------------------------------------- #
# 2. LA grande table : une colonne par condition
# --------------------------------------------------------------------------- #
def _plage(r, a, b, chiffres=0):
    if not np.isfinite(r[a]) or not np.isfinite(r[b]):
        return "—"
    return f"{fr(r[a], chiffres)}–{fr(r[b], chiffres)}"


def _oui_non(v):
    return VALEUR_LISIBLE.get(str(v), "—" if pd.isna(v) else str(v))


# (libelle de ligne, unite, fonction de la ligne -> texte de cellule). L'unite
# est dans le libelle : repeter « Hz » sur 107 cellules n'apporte rien et
# elargit chaque colonne d'autant.
LIGNES = [
    ("Œil", lambda r: r["oeil"]),
    ("Date", lambda r: str(r["study_date"])),
    ("Images (B-scans)", lambda r: f"{int(r['n_frames'])}"),
    ("Durée (s)", lambda r: fr(r["duree_s"], 1)),
    ("<b>Fréquence moyenne (Hz)</b>", lambda r: fr(r["fs_moy_hz"], 2)),
    ("Intervalle moyen (ms)", lambda r: fr(r["dt_moy_ms"], 0)),
    ("Écart-type intervalle (ms)", lambda r: fr(r["dt_sd_ms"], 0)),
    ("Intervalle min–max (ms)", lambda r: _plage(r, "dt_min_ms", "dt_max_ms", 0)),
    ("Scale X (mm/pixel)", lambda r: fr(r["oct_scale_x_mm"], 4)),
    ("Longueur du B-scan (mm)", lambda r: fr(r["oct_longueur_mm"], 2)),
    ("Coord. Start X / Y (mm)", lambda r: f'{fr(r["start_x_mm"], 3)} / {fr(r["start_y_mm"], 3)}'),
    ("Coord. End X / Y (mm)", lambda r: f'{fr(r["end_x_mm"], 3)} / {fr(r["end_y_mm"], 3)}'),
    ("ΔX / ΔY (mm)", lambda r: f'{fr(r["delta_x_mm"], 3)} / {fr(r["delta_y_mm"], 3)}'),
    ("<b>Angle du B-scan (°)</b>", lambda r: fr(r["angle_deg"], 2)),
    ("ART max (NumAve)", lambda r: fr(r["oct_num_ave_max"], 0)),
    ("Image quality (moyenne)", lambda r: fr(r["quality_moy"], 1)),
    ("Image quality (min–max)", lambda r: _plage(r, "quality_min", "quality_max", 0)),
    ("Sensor gain (moyenne)", lambda r: fr(r["sensor_gain_moy"], 1)),
    ("Sensor gain (min–max)", lambda r: _plage(r, "sensor_gain_min",
                                               "sensor_gain_max", 0)),
    ("Auto sensor gain", lambda r: _oui_non(r["ir_auto_sensor_gain"])),
    ("Focus IR (D, moyenne)", lambda r: fr(r["focus_moy_d"], 2)),
    ("IR NumAve", lambda r: fr(r["ir_num_ave"], 0)),
    ("Cible de fixation", lambda r: fr(r["fixation_target"], 0)),
    ("EVI", lambda r: _oui_non(r["oct_evi"])),
    ("Version module d'acquisition", lambda r: str(r["aqm_version"])),
    ("Version module d'export", lambda r: str(r["sw_viewing_version"])),
    ("Décalage UTC (min)", lambda r: fr(r["utc_bias_min"], 0)),
]

STYLE_TABLE = """<style>
.acq-wrap { overflow-x: auto; max-height: 80vh; overflow-y: auto;
            border: 1px solid rgba(150,150,150,0.35); border-radius: 4px; }
.acq { border-collapse: separate; border-spacing: 0; font-size: 0.72rem;
       white-space: nowrap; margin: 0; }
.acq th, .acq td { padding: 2px 6px;
                   border-bottom: 1px solid rgba(150,150,150,0.18); }
.acq td { text-align: right; font-variant-numeric: tabular-nums; }
/* En-tete collant ET premiere colonne collante : leur intersection doit
   passer AU-DESSUS des deux, d'ou le z-index a trois niveaux. Le fond opaque
   est indispensable -- sans lui, les cellules qui defilent se lisent au
   travers de la colonne figee. */
.acq thead th { position: sticky; top: 0; z-index: 2;
                background: var(--bs-body-bg, #fff); text-align: center;
                font-weight: 600; line-height: 1.15;
                border-bottom: 2px solid rgba(150,150,150,0.5); }
.acq tbody th, .acq thead th.acq-coin { position: sticky; left: 0; z-index: 1;
                background: var(--bs-body-bg, #fff); text-align: left;
                font-weight: 500; white-space: normal; min-width: 13rem;
                border-right: 2px solid rgba(150,150,150,0.5); }
.acq thead th.acq-coin { z-index: 3; }
.acq tbody tr:hover td { background: rgba(42,120,214,0.10); }
.acq .acq-sujet { font-size: 0.78rem; font-weight: 700; }
.acq .acq-phase { font-weight: 400; opacity: 0.75; }
</style>"""


def table_conditions() -> str:
    entetes = []
    for _, r in cond.iterrows():
        titre = escape(f"{r['astro']} / {r['condition']}")
        entetes.append(
            f'<th title="{titre}"><span class="acq-sujet">{escape(r["sujet"])}</span>'
            f'<br><span class="acq-phase">{escape(r["phase"])}</span>'
            f'<br>{escape(r["oeil"])}</th>')

    corps = []
    for label, f in LIGNES:
        cellules = "".join(f"<td>{escape(str(f(r)))}</td>"
                           for _, r in cond.iterrows())
        corps.append(f"<tr><th>{label}</th>{cellules}</tr>")

    return (STYLE_TABLE + chr(10)
            + '<div class="acq-wrap">' + chr(10)
            + '<table class="acq">' + chr(10)
            + '<thead><tr><th class="acq-coin">paramètre</th>'
            + "".join(entetes) + "</tr></thead>" + chr(10)
            + "<tbody>" + chr(10) + chr(10).join(corps) + chr(10) + "</tbody>"
            + chr(10) + "</table>" + chr(10) + "</div>")


ecrire_table("```{=html}" + chr(10) + table_conditions() + chr(10) + "```",
             "tab_conditions")


# --------------------------------------------------------------------------- #
# 3. Distribution de chaque parametre variable (lecture de la grande table)
# --------------------------------------------------------------------------- #
# (colonne, libelle, chiffres apres la virgule) -- l'ordre est celui de la
# grande table, pour que les deux se lisent l'une a cote de l'autre.
NUMERIQUES = [
    ("n_frames", "Images par vidéo", 0),
    ("duree_s", "Durée de la vidéo (s)", 1),
    ("fs_moy_hz", "Fréquence moyenne (Hz)", 2),
    ("dt_moy_ms", "Intervalle moyen (ms)", 0),
    ("dt_sd_ms", "Écart-type de l'intervalle (ms)", 0),
    ("dt_max_ms", "Intervalle maximal (ms)", 0),
    ("oct_scale_x_mm", "Scale X (mm/pixel)", 4),
    ("oct_longueur_mm", "Longueur du B-scan (mm)", 2),
    ("start_x_mm", "Coord. Start X (mm)", 3),
    ("start_y_mm", "Coord. Start Y (mm)", 3),
    ("end_x_mm", "Coord. End X (mm)", 3),
    ("end_y_mm", "Coord. End Y (mm)", 3),
    ("angle_deg", "Angle du B-scan (°)", 2),
    ("oct_num_ave_max", "ART max (NumAve)", 0),
    ("quality_moy", "Image quality moyenne", 1),
    ("sensor_gain_moy", "Sensor gain moyen", 1),
    ("focus_moy_d", "Focus IR (dioptries)", 2),
]

CATEGORIELS = [
    ("eye", "Œil"),
    ("phase", "Moment"),
    ("ir_auto_sensor_gain", "Auto sensor gain"),
    ("fixation_target", "Cible de fixation"),
    ("oct_evi", "EVI"),
    ("ir_num_ave", "IR NumAve"),
    ("aqm_version", "Version du module d'acquisition"),
    ("sw_viewing_version", "Version du module d'export"),
    ("utc_bias_min", "Décalage UTC (min)"),
]


def table_resume() -> str:
    lignes = ["| paramètre | médiane [IQR] | min | max | valeurs distinctes |",
              "|:--|--:|--:|--:|--:|"]
    for col, label, chiffres in NUMERIQUES:
        s = pd.to_numeric(cond[col], errors="coerce").dropna()
        lignes.append(
            f"| {label} | {med_iqr(s, chiffres)} | {fr(s.min(), chiffres)} "
            f"| {fr(s.max(), chiffres)} | {s.nunique()} |")
    for col, label in CATEGORIELS:
        vc = colonne_cat(col).value_counts()
        detail = " · ".join(f"{v} ({n})" for v, n in vc.head(4).items())
        if len(vc) > 4:
            detail += f" · … ({len(vc) - 4} autres)"
        lignes.append(f"| {label} | {detail} | | | {len(vc)} |")
    return chr(10).join(lignes)


ecrire_table(table_resume(), "tab_resume")


# --------------------------------------------------------------------------- #
# 4. La frequence d'acquisition
# --------------------------------------------------------------------------- #
def fig_frequence():
    """A gauche la video (107 valeurs), a droite le B-scan (47 000 intervalles).

    Les deux repondent a des questions differentes : « a quelle cadence cette
    video a-t-elle ete acquise » et « cette cadence est-elle reguliere ». La
    seconde est en abscisse LOGARITHMIQUE : les intervalles vont de 10 ms a plus
    de 10 s (les clignements), une echelle lineaire ecraserait tout le mode.
    """
    fs = pd.to_numeric(cond["fs_moy_hz"], errors="coerce").dropna()
    dt = pd.to_numeric(serie["dt_ms"], errors="coerce").dropna()
    dt = dt[dt > 0]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Fréquence moyenne par vidéo (107)",
                        "Intervalle entre B-scans (47 301)"),
        horizontal_spacing=0.10)

    fig.add_trace(go.Histogram(
        x=fs, nbinsx=40, marker_color=C_BARRE, name="vidéos",
        hovertemplate="%{x:.2f} Hz — %{y} vidéo(s)<extra></extra>"), 1, 1)
    fig.add_vline(x=float(fs.median()), line_color=C_REF, line_dash="dash",
                  row=1, col=1,
                  annotation_text=f"médiane {fr(fs.median(), 2)} Hz",
                  annotation_font_color=C_REF, annotation_font_size=10)

    # `np.log10` puis un axe en puissances de dix : `go.Histogram` bine en
    # LINEAIRE meme quand l'axe est logarithmique, ce qui donnerait un seul bin
    # utile. On bine donc le logarithme et on reetiquette l'axe a la main.
    fig.add_trace(barres_binnees(np.log10(dt), 60, C_BARRE2, "B-scans",
                                 "{:.2f}"), 1, 2)
    fig.add_vline(x=float(np.log10(dt.median())), line_color=C_REF,
                  line_dash="dash", row=1, col=2,
                  annotation_text=f"médiane {fr(dt.median(), 0)} ms",
                  annotation_font_color=C_REF, annotation_font_size=10)

    ticks = [10, 30, 100, 300, 1000, 3000, 10000]
    fig.update_xaxes(title_text="fréquence (Hz)", row=1, col=1)
    fig.update_xaxes(title_text="intervalle (ms, échelle log)", row=1, col=2,
                     tickmode="array", tickvals=np.log10(ticks),
                     ticktext=[str(t) for t in ticks])
    fig.update_yaxes(title_text="vidéos", row=1, col=1)
    fig.update_yaxes(title_text="B-scans", row=1, col=2)
    return mise_en_page(fig, "La cadence d'acquisition, par vidéo et par image",
                        hauteur=420, legende=False)


enregistrer(fig_frequence(), "hist_frequence")


# --------------------------------------------------------------------------- #
# 5. Un histogramme par parametre numerique variable
# --------------------------------------------------------------------------- #
def fig_numeriques():
    n_cols = 3
    n_rows = int(np.ceil(len(NUMERIQUES) / n_cols))
    fig = make_subplots(rows=n_rows, cols=n_cols,
                        subplot_titles=[lab for _, lab, _ in NUMERIQUES],
                        horizontal_spacing=0.07, vertical_spacing=0.09)
    for i, (col, label, chiffres) in enumerate(NUMERIQUES):
        r, c = divmod(i, n_cols)
        s = pd.to_numeric(cond[col], errors="coerce").dropna()
        fig.add_trace(go.Histogram(
            x=s, nbinsx=25, marker_color=C_BARRE, name=label,
            hovertemplate=f"%{{x}} — %{{y}} condition(s)<extra>{label}</extra>"),
            r + 1, c + 1)
        fig.add_vline(x=float(s.median()), line_color=C_REF, line_dash="dash",
                      row=r + 1, col=c + 1)
        fig.update_yaxes(title_text="conditions" if c == 0 else None,
                         row=r + 1, col=c + 1)
    return mise_en_page(
        fig, "Chaque paramètre numérique variable, sur les 107 conditions "
             "(tiret rouge : médiane)",
        hauteur=260 * n_rows, legende=False)


enregistrer(fig_numeriques(), "hist_numeriques")


# --------------------------------------------------------------------------- #
# 6. Les parametres categoriels
# --------------------------------------------------------------------------- #
def fig_categoriels():
    n_cols = 3
    n_rows = int(np.ceil(len(CATEGORIELS) / n_cols))
    fig = make_subplots(rows=n_rows, cols=n_cols,
                        subplot_titles=[lab for _, lab in CATEGORIELS],
                        horizontal_spacing=0.07, vertical_spacing=0.14)
    for i, (col, label) in enumerate(CATEGORIELS):
        r, c = divmod(i, n_cols)
        vc = colonne_cat(col).value_counts()
        fig.add_trace(go.Bar(
            x=[str(v) for v in vc.index], y=vc.to_numpy(),
            marker_color=C_BARRE2, name=label,
            text=vc.to_numpy(), textposition="outside", cliponaxis=False,
            hovertemplate=f"%{{x}} — %{{y}} condition(s)<extra>{label}</extra>"),
            r + 1, c + 1)
        fig.update_xaxes(type="category", row=r + 1, col=c + 1)
        fig.update_yaxes(title_text="conditions" if c == 0 else None,
                         range=[0, 1.18 * float(vc.max())], row=r + 1, col=c + 1)
    return mise_en_page(
        fig, "Les paramètres catégoriels : combien de conditions par valeur",
        hauteur=250 * n_rows, legende=False)


enregistrer(fig_categoriels(), "hist_categoriels")


# --------------------------------------------------------------------------- #
# 6 bis. La geometrie du balayage : ou passe la ligne, et sous quel angle
# --------------------------------------------------------------------------- #
def fig_geometrie():
    """A gauche les 107 lignes de balayage dans le plan du fond d'oeil, a droite
    l'histogramme de leur angle.

    L'angle est ``atan(dy / dx)``, en degres, calcule dans
    ``compute_acquisition_params.py`` a partir des coordonnees ``Start`` et
    ``End`` de ``OphthalmicAcquisitionContext``. Il est constant a l'interieur
    d'une video (l'etendue intra-video vaut zero sur les 107 conditions), ce qui
    en fait bien un parametre d'acquisition et non une mesure bruitee.

    Le SIGNE n'est pas une inclinaison differente : c'est le sens de parcours.
    Une ligne parcourue de son extremite basse vers son extremite haute donne
    +20 degres, la meme parcourue en sens inverse -20 -- d'ou les deux modes
    symetriques, et d'ou le second histogramme, en VALEUR ABSOLUE, qui les
    reunit et montre le vrai reglage.
    """
    a = pd.to_numeric(cond["angle_deg"], errors="coerce")
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Les 107 lignes de balayage (plan du fond d'œil)",
                        "Angle signé — atan(ΔY/ΔX)",
                        "Angle en valeur absolue"),
        horizontal_spacing=0.10,
        column_widths=[0.40, 0.30, 0.30])

    # --- les lignes ---------------------------------------------------------
    # Un seul trace pour les 107 segments, separes par des NaN : 107 traces
    # feraient 107 entrees de legende et autant d'objets a dessiner, pour un
    # rendu identique.
    for signe, coul, nom in ((1, C_BARRE, "angle > 0"), (-1, C_REF, "angle < 0")):
        sel = cond[np.sign(a) == signe]
        if sel.empty:
            continue
        xs, ys, txt = [], [], []
        for _, r in sel.iterrows():
            xs += [r["start_x_mm"], r["end_x_mm"], None]
            ys += [r["start_y_mm"], r["end_y_mm"], None]
            etiquette = (f'{r["astro"]} · {r["condition"]}<br>'
                         f'angle {fr(r["angle_deg"], 2)}°')
            txt += [etiquette, etiquette, None]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=nom, text=txt,
            line={"color": coul, "width": 1}, opacity=0.55,
            hovertemplate="%{text}<br>(%{x:.3f} ; %{y:.3f}) mm<extra></extra>"),
            1, 1)
    fig.update_xaxes(title_text="X (mm)", row=1, col=1)
    # `scaleanchor` : sans lui, l'axe Y s'etire pour remplir le panneau et un
    # angle de 20 degres s'y lit comme 45. Une figure d'angles doit etre
    # isometrique, sinon elle ment.
    fig.update_yaxes(title_text="Y (mm)", scaleanchor="x", scaleratio=1,
                     row=1, col=1)

    # --- l'angle ------------------------------------------------------------
    for j, (v, coul, titre) in enumerate([(a, C_BARRE, "signé"),
                                          (a.abs(), C_BARRE2, "absolu")]):
        fig.add_trace(go.Histogram(
            x=v, nbinsx=45, marker_color=coul, showlegend=False,
            hovertemplate="%{x:.1f}° — %{y} condition(s)<extra></extra>"),
            1, j + 2)
        fig.update_xaxes(title_text="angle (°)", row=1, col=j + 2)
        fig.update_yaxes(title_text="conditions" if j == 0 else None,
                         row=1, col=j + 2)
    fig = mise_en_page(fig, "Où passe le B-scan, et sous quel angle",
                       hauteur=450, legende=True)
    # La legende par defaut se pose a y = -0,08, c'est-a-dire SUR le titre
    # « X (mm) » du premier panneau -- les autres figures de la page n'ont pas
    # de titre d'axe X et ne rencontrent pas le probleme. On la descend, et on
    # rend au bas de figure la place correspondante.
    fig.update_layout(legend={"y": -0.20}, margin={"b": 110})
    return fig


enregistrer(fig_geometrie(), "hist_angle")


# --------------------------------------------------------------------------- #
# 7. La qualite d'image, aux deux grains
# --------------------------------------------------------------------------- #
def fig_qualite():
    """La moyenne par condition cache la dispersion INTERNE a la video.

    Une condition a 33 de moyenne peut etre une video reguliere ou une video qui
    oscille entre 20 et 45 selon le clignement : les deux histogrammes cote a
    cote le montrent, et le troisieme panneau donne l'ecart-type intra-video.
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Moyenne par condition (107)",
                        "Chaque B-scan (47 301)",
                        "Écart-type intra-vidéo (107)"),
        horizontal_spacing=0.07)
    q_cond = pd.to_numeric(cond["quality_moy"], errors="coerce").dropna()
    q_bscan = pd.to_numeric(serie["quality"], errors="coerce").dropna()
    q_sd = pd.to_numeric(cond["quality_sd"], errors="coerce").dropna()
    for j, (s, coul, unite) in enumerate([(q_cond, C_BARRE, "condition(s)"),
                                          (q_bscan, C_BARRE2, "B-scan(s)"),
                                          (q_sd, C_BARRE, "condition(s)")]):
        fig.add_trace(barres_binnees(s, 35, coul, unite, "{:.1f}"), 1, j + 1)
        fig.add_vline(x=float(s.median()), line_color=C_REF, line_dash="dash",
                      row=1, col=j + 1)
    fig.update_xaxes(title_text="image quality (dB)", row=1, col=1)
    fig.update_xaxes(title_text="image quality (dB)", row=1, col=2)
    fig.update_xaxes(title_text="écart-type (dB)", row=1, col=3)
    fig.update_yaxes(title_text="effectif", row=1, col=1)
    return mise_en_page(fig, "Image quality : par vidéo, par image, et sa "
                             "dispersion interne", hauteur=380, legende=False)


enregistrer(fig_qualite(), "hist_qualite")


# --------------------------------------------------------------------------- #
# 8. Les chiffres cites dans la prose
# --------------------------------------------------------------------------- #
fs = pd.to_numeric(cond["fs_moy_hz"], errors="coerce")
dt = pd.to_numeric(serie["dt_ms"], errors="coerce")
dt = dt[dt > 0]
q = pd.to_numeric(cond["quality_moy"], errors="coerce")

RESUME.update({
    "fs_med": fr(fs.median(), 2),
    "fs_iqr": med_iqr(fs, 2),
    "fs_min": fr(fs.min(), 2),
    "fs_max": fr(fs.max(), 2),
    "n_frames_med": med_iqr(cond["n_frames"], 0),
    "duree_med": med_iqr(cond["duree_s"], 1),
    "dt_med_ms": fr(dt.median(), 0),
    "dt_p99_ms": fr(dt.quantile(0.99), 0),
    "dt_max_ms": fr(dt.max(), 0),
    "part_dt_sup_500ms": fr(100 * float((dt > 500).mean()), 1) + " %",
    "scale_x": med_iqr(cond["oct_scale_x_mm"], 4),
    "longueur_mm": med_iqr(cond["oct_longueur_mm"], 2),
    "quality_med": med_iqr(q, 1),
    "quality_sd_med": fr(pd.to_numeric(cond["quality_sd"],
                                       errors="coerce").median(), 1),
    "gain_med": med_iqr(cond["sensor_gain_moy"], 1),
    "focus_min": fr(cond["focus_moy_d"].min(), 2),
    "focus_max": fr(cond["focus_moy_d"].max(), 2),
    "angle_med": fr(pd.to_numeric(cond["angle_deg"], errors="coerce").median(), 2),
    "angle_abs_20": int((cond["angle_deg"].abs().round(1) == 20.0).sum()),
    "angle_positif": int((cond["angle_deg"] > 0).sum()),
    "angle_negatif": int((cond["angle_deg"] < 0).sum()),
    "angle_abs_min": fr(cond["angle_deg"].abs().min(), 2),
    "angle_abs_max": fr(cond["angle_deg"].abs().max(), 2),
    "angle_etendue_intra_max": fr(pd.to_numeric(
        cond.get("angle_etendue_deg", pd.Series(dtype=float)),
        errors="coerce").abs().max(), 3),
    "start_x": med_iqr(cond["start_x_mm"], 3),
    "start_y": med_iqr(cond["start_y_mm"], 3),
    "end_x": med_iqr(cond["end_x_mm"], 3),
    "end_y": med_iqr(cond["end_y_mm"], 3),
    "delta_x": med_iqr(cond["delta_x_mm"], 3),
    "delta_y": med_iqr(cond["delta_y_mm"], 3),
    "art_10": int((cond["oct_num_ave_max"] == 10).sum()),
    "art_5": int((cond["oct_num_ave_max"] == 5).sum()),
    "n_variables": len(NUMERIQUES) + len(CATEGORIELS),
})

(SORTIE / "plotlyjs.html").write_text(
    '<script type="text/javascript">' + chr(10) + get_plotlyjs() + chr(10)
    + "</script>" + chr(10), encoding="utf-8")
print("  plotlyjs.html")

(SORTIE / "resume.txt").write_text(
    chr(10).join(f"{k} = {v}" for k, v in RESUME.items()) + chr(10),
    encoding="utf-8")
print("  resume.txt")
print(f"{chr(10)}{len(FICHIERS)} fragment(s) écrit(s) dans {SORTIE}")
