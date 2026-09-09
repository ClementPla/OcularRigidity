# -*- coding: utf-8 -*-
"""
Figures et tables -- page « Data exploration » (section Reproducibility).

Ce script NE LIT QUE les tables ecrites par
``Reproducibility/compute_acquisition_params.py`` ; il ne rouvre aucun XML et ne
recalcule rien a partir des images. Toute correction de fond (un champ oublie,
une frequence mal deduite) se fait la-bas, pas ici.

Entrees
-------
    E:/NASA_Rigidity/Reproducibility/AcquisitionParameters/
        constants.csv    parametres identiques sur les 32 acquisitions
        conditions.csv   1 ligne / acquisition, tous les parametres
        series.csv       1 ligne / B-scan OCT (qualite, ART, intervalle)

Sorties (fragments Quarto, inclus par ``repro-data-exploration.qmd``)
--------------------------------------------------------------------
    tab_replicats.qmd     LE denombrement : participants x (OD, OS)
    txt_anomalies.qmd     les irregularites de l'effectif, redigees
    tab_communs.qmd       les parametres communs aux acquisitions
    tab_od_os.qmd         OD contre OS, chiffre par chiffre
    hist_od_os.qmd        OD contre OS, parametre par parametre
    hist_cadence.qmd      ART -> intervalle, et regularite de la cadence
    hist_angle.qmd        les lignes de balayage dans le plan du fond d'oeil
    hist_qualite.qmd      qualite d'image : par acquisition et par B-scan
    tab_conditions.qmd    une COLONNE par acquisition, OD puis OS
    tab_resume.qmd        distribution de chaque parametre variable
    plotlyjs.html         plotly.js, insere une seule fois par page
    resume.txt            les chiffres cites dans la prose de la page

Les deux grandes tables sont du HTML BRUT et non des tables markdown : il leur
faut un conteneur qui defile horizontalement, une premiere colonne collante et
un en-tete collant, et Pandoc ne sait produire aucun des trois a partir d'une
table pipe.

Tout ce qui est CHIFFRE est genere. Ce lot grandit -- il est passe de 32 a
46 acquisitions et de 6 a 7 participants entre deux versions de cette page -- et
un chiffre recopie a la main dans la prose devient faux au lot suivant sans que
rien ne le signale. Les deux endroits ou la redaction etait le plus exposee
(le tableau OD/OS et l'inventaire des irregularites) sont donc eux aussi des
fragments generes, redigees ici plutot que dans le .qmd.

Le fond est transparent et le texte gris : les memes figures passent sur le
theme clair et sur le theme sombre du site.

Lancer DEPUIS LA RACINE DU DEPOT (ce dossier est une JONCTION vers E:, un chemin
relatif calcule ici sortirait du depot) :
    python reveal_quarto_presentations/figures_repro_data/make_figures.py
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
SOURCE = Path("E:/NASA_Rigidity/Reproducibility/AcquisitionParameters")

# `.absolute()` et surtout PAS `.resolve()` : ce dossier est une JONCTION vers
# E:, la resolution ferait sortir SORTIE du depot.
SORTIE = Path(__file__).absolute().parent

COULEUR_TEXTE = "#c9c9c9"
GRILLE = "rgba(150,150,150,0.18)"
C_OD = "#2a78d6"
C_OS = "#27a567"
C_REF = "#c2453f"
COULEUR_OEIL = {"OD": C_OD, "OS": C_OS}

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
    entete = ("<!-- Genere par figures_repro_data/make_figures.py"
              " -- ne pas editer a la main. -->" + chr(10) + chr(10))
    (SORTIE / f"{nom}.qmd").write_text(entete + texte + chr(10), encoding="utf-8")
    FICHIERS.append(f"{nom}.qmd")
    print(f"  {nom}.qmd")


def fr(x, chiffres=2):
    """Nombre a virgule decimale, comme le reste du site."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{{:.{chiffres}f}}".format(x).replace(".", ",")


VALEUR_LISIBLE = {"true": "oui", "false": "non", "True": "oui", "False": "non"}


def texte_valeur(v) -> str:
    """Une valeur de parametre, telle qu'elle doit s'AFFICHER.

    Trois corrections, qui reviennent partout : le booleen d'un export XML se
    lit « oui / non » et non « True » ; un entier stocke en flottant (pandas
    promeut la colonne des le premier NaN) ne doit pas s'afficher « 768.0 » ;
    et l'absence de champ se dit, plutot que de disparaitre du decompte.
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
    une acquisition sans le champ disparaitrait du total sans rien pour le
    trahir.
    """
    return cond[col].map(texte_valeur)


def med_iqr(s, chiffres=2):
    s = pd.Series(s).dropna()
    if s.empty:
        return "—"
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return f"{fr(s.median(), chiffres)} [{fr(q1, chiffres)} – {fr(q3, chiffres)}]"


def nom_lisible(participant: str) -> str:
    """``BELANGER_CHARLES`` -> ``Belanger Charles``.

    Les accents ont ete retires a la migration pour garder les chemins en pur
    ASCII ; ils ne sont pas reintroduits ici, pour que le nom affiche sur cette
    page reste celui du dossier qui porte les donnees.
    """
    return " ".join(p.capitalize() for p in participant.split("_"))


# --------------------------------------------------------------------------- #
# Donnees
# --------------------------------------------------------------------------- #
cst = pd.read_csv(SOURCE / "constants.csv")
cond = pd.read_csv(SOURCE / "conditions.csv")
serie = pd.read_csv(SOURCE / "series.csv")

# L'intervalle MEDIAN par acquisition n'est pas dans conditions.csv, et c'est
# lui qui compte : la moyenne melange le rythme reel de l'appareil avec les
# arrets du suivi (clignements), qui durent des centaines de millisecondes. Il
# se calcule au grain du B-scan, donc ici.
cond = cond.merge(
    serie.groupby("slug")["dt_ms"].median().rename("dt_med_ms"),
    left_on="slug", right_index=True, how="left")

cond["nom"] = cond["participant"].map(nom_lisible)
cond = cond.sort_values(["participant", "eye", "replicate"]).reset_index(drop=True)

N = len(cond)
RESUME["n_acquisitions"] = N
RESUME["n_bscans"] = len(serie)
RESUME["n_participants"] = cond["participant"].nunique()
RESUME["n_constantes"] = len(cst)
RESUME["n_od"] = int((cond["eye"] == "OD").sum())
RESUME["n_os"] = int((cond["eye"] == "OS").sum())


# --------------------------------------------------------------------------- #
# 1. LE denombrement des replicats
# --------------------------------------------------------------------------- #
STYLE_REPL = """<style>
.rep { border-collapse: collapse; font-size: 0.9rem; margin: 0 0 0.4rem 0; }
.rep th, .rep td { padding: 4px 14px;
                   border-bottom: 1px solid rgba(150,150,150,0.25); }
.rep thead th { text-align: center; font-weight: 600;
                border-bottom: 2px solid rgba(150,150,150,0.5); }
.rep tbody th { text-align: left; font-weight: 500; }
.rep td { text-align: center; font-variant-numeric: tabular-nums; }
.rep tbody tr:hover td { background: rgba(42,120,214,0.10); }
.rep .rep-zero { opacity: 0.35; }
.rep .rep-flag { color: #c2453f; font-weight: 600; }
.rep tfoot th, .rep tfoot td { font-weight: 700;
                               border-top: 2px solid rgba(150,150,150,0.5);
                               border-bottom: none; }
</style>"""

# Une acquisition est dite TRONQUEE si elle compte moins de 80 % des images de
# l'acquisition mediane. Le seuil n'a pas a etre fin : la coupure observee est
# franche (95 et 252 images contre 348 partout ailleurs), et le detail chiffre
# est de toute facon en infobulle.
N_MEDIAN = float(cond["n_frames"].median())
SEUIL_TRONQUE = 0.8 * N_MEDIAN
cond["tronquee"] = cond["n_frames"] < SEUIL_TRONQUE
RESUME["n_tronquees"] = int(cond["tronquee"].sum())
RESUME["n_frames_median"] = int(N_MEDIAN)


def table_replicats() -> str:
    """Participants en lignes, OD / OS en colonnes, nombre de replicats en case.

    La case porte le COMPTE ; le detail (rang, images, duree) vit en infobulle,
    parce que c'est le compte qu'on vient lire et le detail qu'on vient
    verifier. Une case a zero est estompee plutot qu'omise : « aucun OS » est un
    fait de la cohorte, pas une ligne manquante.
    """
    lignes = []
    for nom, g_nom in cond.groupby("nom", sort=True):
        cellules = []
        for eye in ("OD", "OS"):
            g = g_nom[g_nom["eye"] == eye].sort_values("replicate")
            if g.empty:
                cellules.append('<td class="rep-zero" '
                                'title="aucune acquisition">0</td>')
                continue
            detail = " · ".join(
                f"{eye}{int(r.replicate)} : {int(r.n_frames)} images, "
                f"{fr(r.duree_s, 1)} s"
                for r in g.itertuples())
            n_tronq = int(g["tronquee"].sum())
            marque = (f'<span class="rep-flag" title="acquisition(s) tronquee(s)">'
                      f' ({n_tronq} tronquée{"s" if n_tronq > 1 else ""})</span>'
                      if n_tronq else "")
            cellules.append(f'<td title="{escape(detail)}">{len(g)}{marque}</td>')
        lignes.append(f"<tr><th>{escape(nom)}</th>" + "".join(cellules) + "</tr>")

    tot_od, tot_os = RESUME["n_od"], RESUME["n_os"]
    return (STYLE_REPL + chr(10)
            + '<table class="rep">' + chr(10)
            + "<thead><tr><th>participant</th><th>OD</th><th>OS</th></tr></thead>"
            + chr(10) + "<tbody>" + chr(10) + chr(10).join(lignes) + chr(10)
            + "</tbody>" + chr(10)
            + f"<tfoot><tr><th>total ({N})</th><td>{tot_od}</td>"
              f"<td>{tot_os}</td></tr></tfoot>" + chr(10)
            + "</table>")


ecrire_table("```{=html}" + chr(10) + table_replicats() + chr(10) + "```",
             "tab_replicats")


def texte_anomalies() -> str:
    """Les irregularites de l'effectif, en puces -- CALCULEES, pas recopiees.

    Ce paragraphe est le plus volatil de la page : au premier lot il disait
    « Belanger n'a pas d'oeil gauche », ce qui a cesse d'etre vrai au lot
    suivant. Le rediger ici garantit qu'il decrit les donnees presentes, et
    qu'il disparait tout seul le jour ou il n'y a plus rien a signaler.
    """
    puces = []

    # 1. un oeil totalement absent chez un participant
    manquants = []
    for nom, g in cond.groupby("nom", sort=True):
        for eye in ("OD", "OS"):
            if not (g["eye"] == eye).any():
                manquants.append(f"{nom} (aucun {eye})")
    if manquants:
        puces.append("- **Un œil manque** chez " + ", ".join(manquants)
                     + " : cet œil ne contribuera à aucune paire de réplicats.")

    # 2. acquisitions tronquees
    tronq = cond[cond["tronquee"]]
    if len(tronq):
        detail = ", ".join(f"`{r.slug}` ({int(r.n_frames)} images)"
                           for r in tronq.itertuples())
        puces.append(
            f"- **{len(tronq)} acquisition(s) tronquée(s)** — {detail}, contre "
            f"{int(N_MEDIAN)} images pour une acquisition médiane. Elles gardent "
            f"leur rang plutôt que d'être écartées ici : l'exclusion est une "
            f"décision d'analyse, qui se prend avec le chiffre sous les yeux.")

    # 3. plus de trois replicats sur un oeil
    gros = [(nom, eye, len(g)) for (nom, eye), g in cond.groupby(["nom", "eye"])
            if len(g) > 3]
    if gros:
        detail = ", ".join(f"{nom} {eye} ({n})" for nom, eye, n in sorted(gros))
        puces.append(f"- **Plus de trois réplicats** sur {detail}. La "
                     f"numérotation n'est pas plafonnée.")

    # 4. ART qui s'ecarte du reglage dominant de son oeil
    exceptions = []
    for eye in ("OD", "OS"):
        g = cond[cond["eye"] == eye]
        if g.empty:
            continue
        dominant = g["oct_num_ave_max"].mode()
        if dominant.empty:
            continue
        dominant = float(dominant.iloc[0])
        for r in g[g["oct_num_ave_max"] != dominant].itertuples():
            exceptions.append((r.slug, int(r.oct_num_ave_max), eye,
                               int(dominant)))
    if exceptions:
        detail = ", ".join(f"`{slug}` (ART {a} au lieu de {d})"
                           for slug, a, _, d in sorted(exceptions))
        puces.append(
            f"- **{len(exceptions)} acquisition(s) au réglage ART inhabituel** "
            f"pour leur œil — {detail}. Leur cadence suit leur ART et non leur "
            f"œil : c'est ce qui montre que le facteur est l'ART.")

    RESUME["n_anomalies"] = len(puces)
    if not puces:
        return ("Aucune irrégularité : chaque participant a trois réplicats "
                "complets de chaque œil, tous au même réglage.")
    return chr(10).join(puces)


ecrire_table(texte_anomalies(), "txt_anomalies")


# --------------------------------------------------------------------------- #
# 2. Les parametres communs aux 32 acquisitions
# --------------------------------------------------------------------------- #
# Libelle lisible + balise XML d'origine, pour qu'un lecteur puisse rouvrir un
# export et retrouver la ligne. L'ordre est celui du tableau, pas celui du CSV :
# appareil, puis B-scan OCT, puis localisateur infrarouge, puis les controles.
COMMUNS = [
    ("manufacturer", "Fabricant", "`GeneralEquipment/Manufacturer`"),
    ("model_name", "Appareil", "`ManufacturerModelName`"),
    ("aqm_version", "Version du module d'acquisition", "`AQMVersion/Version`"),
    ("sw_viewing_version", "Version du module d'export", "`SWVersion/Version`"),
    ("study_description", "Description de l'étude", "`StudyDescription`"),
    ("modality", "Modality", "`Series/Modality`"),
    ("modality_procedure", "Modality-procedure", "`Series/ModalityProcedure`"),
    ("series_type", "Type de série", "`Series/Type`"),
    ("examined_structure", "Structure examinée", "`Series/ExaminedStructure`"),
    ("export_type", "Type d'export", "`BODY/ExportType`"),
    ("oct_width_px", "OCT — largeur (pixels)", "`Width`"),
    ("oct_height_px", "OCT — hauteur (pixels)", "`Height`"),
    ("oct_scale_y_mm", "OCT — scale Y (mm/pixel)", "`ScaleY`"),
    ("oct_field_width_deg", "OCT — champ balayé (°)", "`OCTFieldSize/Width`"),
    ("oct_resolution", "OCT — résolution", "`Resolution`"),
    ("oct_edi", "OCT — EDI (imagerie profondeur accrue)", "`EDI`"),
    ("oct_evi", "OCT — EVI", "`EVI`"),
    ("oct_position_tolerance", "OCT — position dans la tolérance",
     "`PositionWithinTolerance`"),
    ("ir_width_px", "IR — largeur (pixels)", "`Width`"),
    ("ir_height_px", "IR — hauteur (pixels)", "`Height`"),
    ("ir_angle_deg", "IR — angle (°)", "`Angle`"),
    ("ir_num_ave", "IR — NumAve", "`NumAve`"),
    ("ir_resolution", "IR — résolution", "`Resolution`"),
    ("ir_light_source", "IR — source lumineuse", "`ImageType/LightSource`"),
    ("ir_auto_sensor_gain", "IR — gain automatique", "`AutoSensorGain`"),
    ("fixation_target", "Cible de fixation", "`FixationTarget`"),
    ("utc_bias_min", "Décalage UTC (min)", "`AcquisitionTime/UTCBias`"),
    ("laterality_matches_folder", "Latéralité XML = latéralité du dossier",
     "`Series/Laterality`"),
    ("angle_etendue_deg", "Variation de l'angle au sein d'une vidéo (°)",
     "`Start` / `End`"),
]


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
# 3. OD contre OS -- la figure centrale de la page
# --------------------------------------------------------------------------- #
# (colonne, libelle, chiffres) -- six parametres, choisis pour couvrir ce qui
# separe les deux yeux (ART, intervalle, frequence, duree, angle) et un temoin
# qui ne les separe PAS (la qualite d'image).
OD_OS = [
    ("oct_num_ave_max", "ART (NumAve max)", 0),
    ("dt_med_ms", "Intervalle médian (ms)", 0),
    ("fs_moy_hz", "Fréquence moyenne (Hz)", 2),
    ("duree_s", "Durée de la vidéo (s)", 1),
    ("angle_deg", "Angle du B-scan (°)", 2),
    ("quality_moy", "Image quality moyenne (dB)", 1),
]


def _essaim(v, largeur=0.34, n_bins=36):
    """Decalages horizontaux d'un essaim : les valeurs egales s'ECARTENT.

    Sans cela, la moitie de cette figure serait illisible : l'ART vaut 4 sur
    seize acquisitions et l'intervalle median 52 ms sur onze, si bien qu'un
    nuage centre affiche UN point la ou il y en a seize -- l'effectif, qui est
    justement ce qu'on vient lire, disparait.

    Le jitter aleatoire de ``go.Box`` ne suffit pas non plus : il est tire au
    hasard, donc les points se recouvrent encore par paquets et la figure change
    a chaque generation. Ici les valeurs sont rangees en ``n_bins`` classes, et
    les points d'une meme classe sont etales REGULIEREMENT, ce qui les rend
    denombrables. La largeur croit avec l'effectif jusqu'a un plafond, pour
    qu'un groupe de deux ne s'affiche pas aussi large qu'un groupe de seize.
    """
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


def fig_od_os():
    """Un essaim par oeil et par parametre, avec la mediane en barre.

    A trente-deux acquisitions, un histogramme mentirait par binning : chaque
    barre reposerait sur deux ou trois valeurs. Une boite ne vaut guere mieux
    ici, car la plupart de ces parametres sont quasi discrets -- une boite d'ART
    se reduit a un trait. L'essaim montre l'effectif reel, et le survol nomme
    l'acquisition : on voit du meme coup COMBIEN il y a d'exceptions et
    LESQUELLES.
    """
    centre = {"OD": 0.0, "OS": 1.0}
    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=[lab for _, lab, _ in OD_OS],
                        horizontal_spacing=0.08, vertical_spacing=0.16)
    for k, (col, _, chiffres) in enumerate(OD_OS):
        r, c = divmod(k, 3)
        for eye in ("OD", "OS"):
            sel = cond[cond["eye"] == eye]
            v = pd.to_numeric(sel[col], errors="coerce")
            x0 = centre[eye]
            fig.add_trace(go.Scatter(
                x=x0 + _essaim(v), y=v, mode="markers",
                name=eye, legendgroup=eye, showlegend=(k == 0),
                marker={"color": COULEUR_OEIL[eye], "size": 7, "opacity": 0.8,
                        "line": {"width": 0}},
                text=sel["slug"],
                hovertemplate="%{text}<br>%{y}<extra></extra>"),
                r + 1, c + 1)
            # La mediane en barre : sans elle, deux essaims qui se chevauchent
            # ne se comparent qu'a vue.
            med = float(v.median())
            fig.add_trace(go.Scatter(
                x=[x0 - 0.42, x0 + 0.42], y=[med, med], mode="lines",
                showlegend=False, legendgroup=eye,
                line={"color": COULEUR_OEIL[eye], "width": 2},
                hovertemplate=f"médiane {eye} : {fr(med, chiffres)}<extra></extra>"),
                r + 1, c + 1)
        fig.update_xaxes(tickvals=[0, 1], ticktext=["OD", "OS"],
                         range=[-0.6, 1.6], row=r + 1, col=c + 1)
    fig = mise_en_page(
        fig, "OD contre OS : ce que le protocole a réglé différemment",
        hauteur=620, legende=True)
    fig.update_layout(legend={"y": -0.06})
    enregistrer(fig, "hist_od_os")


fig_od_os()


# Comment chaque parametre se resume dans la table ci-dessous. « compte » pour
# un reglage discret (une mediane d'ART ne veut rien dire), « plage » pour
# l'angle (dont tout l'interet est qu'il ne se recouvre pas d'un oeil a
# l'autre), « mediane » pour le reste.
RESUME_OD_OS = {
    "oct_num_ave_max": "compte",
    "angle_deg": "plage",
}


def table_od_os() -> str:
    """Le tableau OD/OS, GENERE -- il etait ecrit a la main dans le .qmd.

    C'est le passage qui vieillissait le plus mal : six lignes de chiffres
    recopies, tous faux des qu'une acquisition s'ajoute, et rien pour le
    signaler au lecteur ni a la personne qui met la page a jour.
    """
    n = {eye: int((cond["eye"] == eye).sum()) for eye in ("OD", "OS")}
    lignes = [f'| paramètre | OD ({n["OD"]}) | OS ({n["OS"]}) |', "|:--|:--|:--|"]
    for col, label, chiffres in OD_OS:
        mode = RESUME_OD_OS.get(col, "mediane")
        cellules = []
        for eye in ("OD", "OS"):
            v = pd.to_numeric(cond.loc[cond["eye"] == eye, col],
                              errors="coerce").dropna()
            if v.empty:
                cellules.append("—")
            elif mode == "compte":
                cellules.append(" · ".join(
                    f"{texte_valeur(k)} sur {c}"
                    for k, c in v.value_counts().sort_index().items()))
            elif mode == "plage":
                cellules.append(f"{fr(v.min(), 3)} … {fr(v.max(), 3)}")
            else:
                cellules.append(med_iqr(v, chiffres))
        lignes.append(f"| {label} | {cellules[0]} | {cellules[1]} |")
    return chr(10).join(lignes)


ecrire_table(table_od_os(), "tab_od_os")


# --------------------------------------------------------------------------- #
# 4. La cadence : d'ou vient l'intervalle entre B-scans
# --------------------------------------------------------------------------- #
def fig_cadence():
    """A gauche l'ART contre l'intervalle median, a droite tous les intervalles.

    Le panneau de gauche est le mecanisme : l'appareil moyenne ``ART`` images
    brutes pour produire un B-scan, donc l'intervalle vaut ART fois la periode
    de balayage. La droite de reference n'est pas ajustee -- c'est
    ``dt = ART x 26,1 ms``, la periode deduite du rapport median, tracee pour
    qu'on voie si les points s'y posent.

    Le panneau de droite est en abscisse LOGARITHMIQUE : les intervalles vont de
    quelques dizaines de millisecondes a plus d'une seconde (les clignements),
    et une echelle lineaire ecraserait tout le mode.
    """
    art = pd.to_numeric(cond["oct_num_ave_max"], errors="coerce")
    dtm = pd.to_numeric(cond["dt_med_ms"], errors="coerce")
    periode = float((dtm / art).median())
    RESUME["periode_balayage_ms"] = fr(periode, 2)
    RESUME["cadence_brute_hz"] = fr(1000.0 / periode, 1)
    RESUME["ratio_min"] = fr(float((dtm / art).min()), 2)
    RESUME["ratio_max"] = fr(float((dtm / art).max()), 2)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("ART et intervalle médian, par acquisition",
                        f"Les {len(serie):,} intervalles entre B-scans"
                        .replace(",", " ")),
        horizontal_spacing=0.11)

    xs = np.array([art.min() - 0.4, art.max() + 0.4])
    fig.add_trace(go.Scatter(
        x=xs, y=periode * xs, mode="lines", name=f"ART × {fr(periode, 1)} ms",
        line={"color": C_REF, "width": 1.4, "dash": "dash"},
        hoverinfo="skip"), 1, 1)
    for eye in ("OD", "OS"):
        sel = cond[cond["eye"] == eye]
        fig.add_trace(go.Scatter(
            x=pd.to_numeric(sel["oct_num_ave_max"], errors="coerce"),
            y=pd.to_numeric(sel["dt_med_ms"], errors="coerce"),
            mode="markers", name=eye, legendgroup=eye,
            marker={"color": COULEUR_OEIL[eye], "size": 9, "opacity": 0.75,
                    "line": {"width": 0}},
            text=sel["slug"],
            hovertemplate="%{text}<br>ART %{x} — %{y:.0f} ms<extra></extra>"),
            1, 1)
    fig.update_xaxes(title_text="ART (NumAve max)", dtick=1, row=1, col=1)
    fig.update_yaxes(title_text="intervalle médian (ms)", row=1, col=1)

    # `np.log10` puis un axe en puissances de dix : `go.Histogram` bine en
    # LINEAIRE meme quand l'axe est logarithmique, ce qui donnerait un unique
    # bin pour tout le mode et une longue queue vide a droite.
    for eye in ("OD", "OS"):
        dt = pd.to_numeric(
            serie.loc[serie["eye"] == eye, "dt_ms"], errors="coerce").dropna()
        dt = dt[dt > 0]
        fig.add_trace(go.Histogram(
            x=np.log10(dt), nbinsx=70, marker_color=COULEUR_OEIL[eye],
            name=eye, legendgroup=eye, showlegend=False, opacity=0.65,
            hovertemplate="10^%{x:.2f} ms — %{y} intervalle(s)<extra></extra>"),
            1, 2)
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text="intervalle (ms, échelle log)",
                     tickvals=[np.log10(v) for v in (25, 50, 100, 200, 500, 1000)],
                     ticktext=["25", "50", "100", "200", "500", "1000"],
                     row=1, col=2)
    fig.update_yaxes(title_text="B-scans", row=1, col=2)

    fig = mise_en_page(fig, "D'où vient la cadence, et à quel point elle tient",
                       hauteur=460, legende=True)
    fig.update_layout(legend={"y": -0.22}, margin={"b": 110})
    enregistrer(fig, "hist_cadence")


fig_cadence()


# --------------------------------------------------------------------------- #
# 5. Ou passe le B-scan, et sous quel angle
# --------------------------------------------------------------------------- #
def fig_angle():
    """Les 32 lignes de balayage, puis l'angle signe et l'angle absolu.

    Le premier panneau est ISOMETRIQUE (``scaleanchor``) : sans lui, l'axe Y
    s'etire pour remplir le cadre et un angle de 20 degres s'y lit comme 45.
    Une figure d'angles qui n'est pas isometrique ment.
    """
    a = pd.to_numeric(cond["angle_deg"], errors="coerce")
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(f"Les {N} lignes de balayage (plan du fond d'œil)",
                        "Angle signé — atan(ΔY/ΔX)",
                        "Angle en valeur absolue"),
        horizontal_spacing=0.10, column_widths=[0.40, 0.30, 0.30])

    # Un seul trace par oeil, les segments separes par des NaN : 32 traces
    # feraient 32 entrees de legende et autant d'objets a dessiner, pour un
    # rendu identique.
    for eye in ("OD", "OS"):
        sel = cond[cond["eye"] == eye]
        xs, ys, txt = [], [], []
        for r in sel.itertuples():
            xs += [r.start_x_mm, r.end_x_mm, None]
            ys += [r.start_y_mm, r.end_y_mm, None]
            etiquette = f"{r.slug}<br>angle {fr(r.angle_deg, 2)}°"
            txt += [etiquette, etiquette, None]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=eye, legendgroup=eye, text=txt,
            line={"color": COULEUR_OEIL[eye], "width": 1.2}, opacity=0.6,
            hovertemplate="%{text}<br>(%{x:.3f} ; %{y:.3f}) mm<extra></extra>"),
            1, 1)
    fig.update_xaxes(title_text="X (mm)", row=1, col=1)
    fig.update_yaxes(title_text="Y (mm)", scaleanchor="x", scaleratio=1,
                     row=1, col=1)

    for j, (serie_v, titre) in enumerate([(a, "signé"), (a.abs(), "absolu")]):
        for eye in ("OD", "OS"):
            fig.add_trace(go.Histogram(
                x=serie_v[cond["eye"] == eye], nbinsx=40,
                marker_color=COULEUR_OEIL[eye], name=eye, legendgroup=eye,
                showlegend=False, opacity=0.7,
                hovertemplate="%{x:.2f}° — %{y} acquisition(s)<extra></extra>"),
                1, j + 2)
        fig.update_xaxes(title_text="angle (°)", row=1, col=j + 2)
        fig.update_yaxes(title_text="acquisitions" if j == 0 else None,
                         row=1, col=j + 2)
    fig.update_layout(barmode="overlay")

    fig = mise_en_page(fig, "Où passe le B-scan, et sous quel angle",
                       hauteur=450, legende=True)
    # La legende par defaut se pose a y = -0,08, c'est-a-dire SUR le titre
    # « X (mm) » du premier panneau. On la descend, et on rend au bas de figure
    # la place correspondante.
    fig.update_layout(legend={"y": -0.20}, margin={"b": 110})
    enregistrer(fig, "hist_angle")


fig_angle()


# --------------------------------------------------------------------------- #
# 6. Image quality
# --------------------------------------------------------------------------- #
def fig_qualite():
    """La qualite par acquisition, puis par B-scan, puis sa dispersion interne.

    C'est le seul parametre que l'appareil recalcule a chaque B-scan : la
    moyenne par video et la distribution des valeurs individuelles ne disent
    pas la meme chose, et l'ecart-type intra-video dit si une video mediocre
    l'est du debut a la fin ou seulement par acces.
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(f"Moyenne par acquisition ({N})",
                        f"Chaque B-scan ({len(serie):,})".replace(",", " "),
                        "Écart-type intra-acquisition"),
        horizontal_spacing=0.09)
    for eye in ("OD", "OS"):
        sel = cond[cond["eye"] == eye]
        fig.add_trace(go.Histogram(
            x=pd.to_numeric(sel["quality_moy"], errors="coerce"), nbinsx=24,
            marker_color=COULEUR_OEIL[eye], name=eye, legendgroup=eye,
            opacity=0.7,
            hovertemplate="%{x:.1f} dB — %{y} acquisition(s)<extra></extra>"),
            1, 1)
        fig.add_trace(go.Histogram(
            x=pd.to_numeric(serie.loc[serie["eye"] == eye, "quality"],
                            errors="coerce"), nbinsx=40,
            marker_color=COULEUR_OEIL[eye], name=eye, legendgroup=eye,
            showlegend=False, opacity=0.7,
            hovertemplate="%{x:.0f} dB — %{y} B-scan(s)<extra></extra>"), 1, 2)
        fig.add_trace(go.Histogram(
            x=pd.to_numeric(sel["quality_sd"], errors="coerce"), nbinsx=24,
            marker_color=COULEUR_OEIL[eye], name=eye, legendgroup=eye,
            showlegend=False, opacity=0.7,
            hovertemplate="%{x:.2f} dB — %{y} acquisition(s)<extra></extra>"),
            1, 3)
    fig.update_layout(barmode="overlay")
    for col, titre in ((1, "qualité moyenne (dB)"), (2, "qualité (dB)"),
                       (3, "écart-type (dB)")):
        fig.update_xaxes(title_text=titre, row=1, col=col)
    fig.update_yaxes(title_text="effectif", row=1, col=1)
    fig = mise_en_page(fig, "Image quality", hauteur=420, legende=True)
    fig.update_layout(legend={"y": -0.24}, margin={"b": 110})
    enregistrer(fig, "hist_qualite")


fig_qualite()


# --------------------------------------------------------------------------- #
# 7. Une colonne par acquisition
# --------------------------------------------------------------------------- #
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
.acq .acq-oeil-od { color: #2a78d6; font-weight: 600; }
.acq .acq-oeil-os { color: #27a567; font-weight: 600; }
</style>"""


def _plage(r, a, b, chiffres=0):
    if not np.isfinite(r[a]) or not np.isfinite(r[b]):
        return "—"
    return f"{fr(r[a], chiffres)}–{fr(r[b], chiffres)}"


def _oui_non(v):
    return VALEUR_LISIBLE.get(str(v), "—" if pd.isna(v) else str(v))


# (libelle de ligne, fonction de la ligne -> texte de cellule). L'unite est dans
# le libelle : la repeter sur 32 cellules n'apporte rien et elargit chaque
# colonne d'autant.
LIGNES = [
    ("Œil", lambda r: r["eye"]),
    ("Réplicat", lambda r: fr(r["replicate"], 0)),
    ("Date", lambda r: str(r["study_date"])),
    ("Début", lambda r: str(r["t_debut"])[:8]),
    ("Opérateur", lambda r: str(r["operator"])),
    ("Images (B-scans)", lambda r: f"{int(r['n_frames'])}"),
    ("Durée (s)", lambda r: fr(r["duree_s"], 1)),
    ("<b>ART (NumAve max)</b>", lambda r: fr(r["oct_num_ave_max"], 0)),
    ("<b>Intervalle médian (ms)</b>", lambda r: fr(r["dt_med_ms"], 0)),
    ("Fréquence moyenne (Hz)", lambda r: fr(r["fs_moy_hz"], 2)),
    ("Intervalle moyen (ms)", lambda r: fr(r["dt_moy_ms"], 0)),
    ("Écart-type intervalle (ms)", lambda r: fr(r["dt_sd_ms"], 0)),
    ("Intervalle min–max (ms)", lambda r: _plage(r, "dt_min_ms", "dt_max_ms", 0)),
    ("Scale X (mm/pixel)", lambda r: fr(r["oct_scale_x_mm"], 4)),
    ("Longueur du B-scan (mm)", lambda r: fr(r["oct_longueur_mm"], 2)),
    ("Coord. Start X / Y (mm)",
     lambda r: f'{fr(r["start_x_mm"], 3)} / {fr(r["start_y_mm"], 3)}'),
    ("Coord. End X / Y (mm)",
     lambda r: f'{fr(r["end_x_mm"], 3)} / {fr(r["end_y_mm"], 3)}'),
    ("ΔX / ΔY (mm)",
     lambda r: f'{fr(r["delta_x_mm"], 3)} / {fr(r["delta_y_mm"], 3)}'),
    ("<b>Angle du B-scan (°)</b>", lambda r: fr(r["angle_deg"], 2)),
    ("Image quality (moyenne)", lambda r: fr(r["quality_moy"], 1)),
    ("Image quality (écart-type)", lambda r: fr(r["quality_sd"], 2)),
    ("Image quality (min–max)", lambda r: _plage(r, "quality_min", "quality_max", 0)),
    ("Sensor gain (moyenne)", lambda r: fr(r["sensor_gain_moy"], 1)),
    ("Focus IR (D)", lambda r: fr(r["focus_moy_d"], 2)),
    ("Scale IR (mm/pixel)", lambda r: fr(r["ir_scale_mm"], 4)),
    ("Latéralité XML", lambda r: str(r["laterality_xml"])),
    ("Sexe", lambda r: str(r["sex"])),
]


def table_conditions() -> str:
    """OD d'abord, OS ensuite : c'est la comparaison que la page vient faire.

    Les colonnes sont donc groupees par oeil PUIS par participant, a l'inverse
    de l'ordre naturel du dossier -- lire les dix-huit OD cote a cote, puis les
    quatorze OS, montre le contraste que la figure OD/OS resume.
    """
    ordre = cond.sort_values(["eye", "participant", "replicate"])
    entetes = []
    for r in ordre.itertuples():
        titre = escape(r.slug)
        classe = "acq-oeil-od" if r.eye == "OD" else "acq-oeil-os"
        entetes.append(
            f'<th title="{titre}">'
            f'<span class="acq-sujet">{escape(r.nom.split()[0])}</span><br>'
            f'<span class="{classe}">{r.eye}{int(r.replicate)}</span></th>')
    corps = []
    for label, f in LIGNES:
        cellules = "".join(f"<td>{escape(str(f(r)))}</td>"
                           for _, r in ordre.iterrows())
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
# 8. Distribution de chaque parametre variable
# --------------------------------------------------------------------------- #
NUMERIQUES = [
    ("n_frames", "Images par vidéo", 0),
    ("duree_s", "Durée de la vidéo (s)", 1),
    ("oct_num_ave_max", "ART max (NumAve)", 0),
    ("dt_med_ms", "Intervalle médian (ms)", 0),
    ("fs_moy_hz", "Fréquence moyenne (Hz)", 2),
    ("dt_moy_ms", "Intervalle moyen (ms)", 0),
    ("dt_sd_ms", "Écart-type de l'intervalle (ms)", 0),
    ("dt_max_ms", "Intervalle maximal (ms)", 0),
    ("oct_scale_x_mm", "Scale X (mm/pixel)", 4),
    ("oct_longueur_mm", "Longueur du B-scan (mm)", 2),
    ("angle_deg", "Angle du B-scan (°)", 2),
    ("quality_moy", "Image quality moyenne", 1),
    ("quality_sd", "Image quality — écart-type intra", 2),
    ("sensor_gain_moy", "Sensor gain moyen", 1),
    ("focus_moy_d", "Focus IR (dioptries)", 2),
    ("ir_scale_mm", "Scale IR (mm/pixel)", 4),
]

CATEGORIELS = [
    ("eye", "Œil"),
    ("laterality_xml", "Latéralité (XML)"),
    ("sex", "Sexe"),
    ("study_date", "Date d'acquisition"),
    ("operator", "Opérateur"),
    ("images_per_series", "Images par série (`NumImages`)"),
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
        if col not in cond.columns:
            continue
        vc = colonne_cat(col).value_counts()
        detail = " · ".join(f"{v} ({n})" for v, n in vc.head(4).items())
        if len(vc) > 4:
            detail += f" · … ({len(vc) - 4} autres)"
        lignes.append(f"| {label} | {detail} | | | {len(vc)} |")
    return chr(10).join(lignes)


ecrire_table(table_resume(), "tab_resume")


# --------------------------------------------------------------------------- #
# 9. Les chiffres cites dans la prose
# --------------------------------------------------------------------------- #
def par_oeil(col, chiffres=2):
    return {eye: med_iqr(pd.to_numeric(cond.loc[cond["eye"] == eye, col],
                                       errors="coerce"), chiffres)
            for eye in ("OD", "OS")}


dt_tous = pd.to_numeric(serie["dt_ms"], errors="coerce").dropna()
dt_tous = dt_tous[dt_tous > 0]
art = pd.to_numeric(cond["oct_num_ave_max"], errors="coerce")

RESUME.update({
    "n_variables": len(NUMERIQUES) + len(CATEGORIELS),
    "art_od": " · ".join(f"ART {int(k)} : {v}"
                         for k, v in art[cond["eye"] == "OD"]
                         .value_counts().sort_index().items()),
    "art_os": " · ".join(f"ART {int(k)} : {v}"
                         for k, v in art[cond["eye"] == "OS"]
                         .value_counts().sort_index().items()),
    "art_5_slugs": ", ".join(cond.loc[art == 5, "slug"]),
    "dt_med_od": par_oeil("dt_med_ms", 0)["OD"],
    "dt_med_os": par_oeil("dt_med_ms", 0)["OS"],
    "fs_od": par_oeil("fs_moy_hz", 2)["OD"],
    "fs_os": par_oeil("fs_moy_hz", 2)["OS"],
    "duree_od": par_oeil("duree_s", 1)["OD"],
    "duree_os": par_oeil("duree_s", 1)["OS"],
    "qualite_od": par_oeil("quality_moy", 1)["OD"],
    "qualite_os": par_oeil("quality_moy", 1)["OS"],
    "qualite_sd_med": fr(pd.to_numeric(cond["quality_sd"],
                                       errors="coerce").median(), 2),
    "angle_od": f'{fr(cond.loc[cond["eye"] == "OD", "angle_deg"].min(), 3)} … '
                f'{fr(cond.loc[cond["eye"] == "OD", "angle_deg"].max(), 3)}',
    "angle_os": f'{fr(cond.loc[cond["eye"] == "OS", "angle_deg"].min(), 3)} … '
                f'{fr(cond.loc[cond["eye"] == "OS", "angle_deg"].max(), 3)}',
    "dt_pct_sup_500": fr(100.0 * float((dt_tous > 500).mean()), 2),
    "dt_median_global": fr(float(dt_tous.median()), 0),
    "dt_p99": fr(float(dt_tous.quantile(0.99)), 0),
    "dt_max": fr(float(dt_tous.max()), 0),
    "dt_sd_min": fr(pd.to_numeric(cond["dt_sd_ms"], errors="coerce").min(), 0),
    "dt_sd_max": fr(pd.to_numeric(cond["dt_sd_ms"], errors="coerce").max(), 0),
    "scale_x_n": int(cond["oct_scale_x_mm"].nunique()),
    "scale_x": med_iqr(cond["oct_scale_x_mm"], 4),
    "longueur": med_iqr(cond["oct_longueur_mm"], 2),
    "focus": med_iqr(cond["focus_moy_d"], 2),
    "focus_min": fr(cond["focus_moy_d"].min(), 2),
    "focus_max": fr(cond["focus_moy_d"].max(), 2),
    "gain": med_iqr(cond["sensor_gain_moy"], 1),
    "n_frames_uniques": " · ".join(
        f"{int(k)} images : {v}"
        for k, v in cond["n_frames"].value_counts().sort_index().items()),
    "tronquees_slugs": ", ".join(cond.loc[cond["tronquee"], "slug"]),
    "operateurs": " · ".join(f"{k} ({v})"
                             for k, v in cond["operator"].value_counts().items()),
    "sans_os": ", ".join(sorted(
        set(cond["nom"]) - set(cond.loc[cond["eye"] == "OS", "nom"]))) or "aucun",
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
