"""
compute_sans_predictors.py

Repetabilite et prediction de SANS, pour TOUS les parametres candidats de la
chaine SANSORI -- les nouveaux (strain par region, pentes, correlations
CT / pouls) comme les anciens (qualite du pouls, coefficient k de Sayah).

Ce script ne recale rien et ne replie rien : il lit les tables deja ecrites par
les autres scripts de lot, les met au meme format, et repond a deux questions.

1. REPETABILITE -- un parametre mesure deux fois sur le MEME oeil, dans la MEME
   seance, donne-t-il deux fois la meme valeur ? Certaines conditions ont
   jusqu'a cinq repliques (``..._OD1``, ``..._OD2``, ...). Trois ICC de pingouin
   sont rapportes ENSEMBLE (Liljequist et al. 2019) : ICC(1,1) suppose les
   repliques interchangeables, ICC(A,1) (accord absolu) et ICC(C,1)
   (consistance) le testent separement -- si ICC(C,1) >> ICC(A,1), un biais
   systematique entre repliques (derive de la FC, fatigue) est probable.
   ``pingouin.intraclass_corr`` exigeant un plan EQUILIBRE, ils sont calcules
   une PAIRE a la fois (replique 1 vs 2, 1 vs 3, ...), chacune restreinte aux
   sujets possedant les deux repliques.
   S'y ajoutent les COMPOSANTES DE VARIANCE, que l'ICC seul ne donne pas :
   l'ANOVA a un facteur (plan desequilibre, tous les replicats a la fois) fournit
   ``var_intra`` et ``var_inter``, donc l'ecart-type de repetabilite DANS L'UNITE
   du parametre et le coefficient de repetabilite 2,77 x sd_intra. Un ICC de 0,9
   sur une cohorte tres dispersee peut cacher une repetabilite absolue mediocre ;
   les deux lectures sont necessaires.

2. PREDICTION DE SANS -- le parametre est-il lie au syndrome post-vol ? SIX
   issues, toutes lues dans ``E:/SANSORI/sansori_db.db`` (table
   ``Measurements``), toutes definies comme APRES - AVANT, et qui couvrent DEUX
   signes cliniques distincts :
     - le GONFLEMENT RETINIEN, ``delta_TRT_x = TRT250_x(apres) - TRT250_x(avant)``
       pour les quatre quadrants (S, I, N, T) plus leur moyenne -- 18 sujets ;
     - l'APLATISSEMENT DU GLOBE, ``delta_AL_mm = AL(apres) - AL(avant)``, la
       longueur axiale (description ``Biometry``) -- 20 sujets, valeurs toutes
       negatives ou nulles.
   Les deux ne se remplacent pas : ils ne correlent entre eux (r = -0,54) que
   sur le quadrant nasal. Voir l'avertissement de CIRCULARITE au niveau de
   ``OUTCOMES`` avant de lire le k de Sayah contre ``delta_AL_mm``.
   Les predicteurs sont agreges par (sujet, moment) -- moyenne des repliques --
   et correles selon TROIS vues : sur les mesures d'AVANT le vol (peut-on
   predire ?), sur celles d'APRES (le parametre a-t-il bouge comme l'oeil ?), et
   sur leur DIFFERENCE apres - avant.
   Le panel x sigma x cas x region x 6 issues x 3 vues fait plusieurs dizaines de
   milliers de tests : une q-valeur de Benjamini-Hochberg est jointe a chaque p,
   faute de quoi la table n'est pas lisible.

Arborescence lue
----------------
    E:/NASA_Rigidity/SegmentationVariations/<variante>/
        demons_strain/{one_cycles,regions,slopes,bins}.csv
        pulse_from_data/{conditions,methods}.csv
        registered_masks/<NN_id>/<...>/mask.npz        <- pour k (Sayah)
    E:/SANSORI/<NN_id>/<...>/<condition>/Data Files/visit_data.csv  <- HR/IOP/OPA
    E:/SANSORI/sansori_db.db                                        <- AL, TRT250_*

Sorties (sous ``SEGVAR_ROOT/<variante>/sans_predictors/``)
---------------------------------------------------------
  - ``predictors.csv``       format long, une ligne par (condition, predicteur)
  - ``repeatability.csv``    ICC (3 types x paires) + composantes de variance,
    REPLIQUE = une acquisition repetee du meme oeil dans la meme seance, cible =
    le sujet
  - ``predictors_one_cycle.csv`` le meme panel sans agreger les one-cycles : une
    ligne par (condition, one-cycle, predicteur)
  - ``repeatability_one_cycles.csv`` le meme ICC avec l'AUTRE definition de la
    replique -- REPLIQUE = un one-cycle, cible = la VIDEO. Trois familles en sont
    structurellement absentes (``pente_pool``, ``pouls``, ``rigidite``) : elles
    n'existent qu'a l'echelle de la video. Meme schema que ci-dessus, plus une
    colonne ``cible``
  - ``sans_correlations.csv`` r/rho, p, q (BH), n, par predicteur x issue x vue.
    TROIS q par test : ``*_q`` sur toute la table, ``*_q_famille`` dans la
    famille, ``*_q_categorie`` dans (categorie, vue, issue) -- la categorie
    (rigidite / strain / viscosite) est une colonne de la table, vide pour les
    predicteurs qu'aucune page n'affiche ; voir ``categorie_de``
  - ``pooled_slopes.csv``    une ligne par (condition, cas, sigma, region,
    abscisse, tissu) : pente, ordonnee, r, p et n de l'ajustement POOLE sur tous
    les couples (one-cycle, bin) de la condition -- c'est ce que le volcano de la
    page Quarto trace
  - ``one_cycle_filter.csv`` une ligne par (condition, one-cycle) : la valeur de
    chaque critere du portail, le verdict par critere et le verdict global, plus
    le sort de la condition entiere. C'est la SEULE definition du portail :
    ``figures_sans_predictors/make_figures.py`` la relit pour filtrer ses
    propres lectures, afin que le site et les predicteurs portent exactement le
    meme jeu de one-cycles.
  - ``stability.csv``        une ligne par predicteur : correlation AVANT / APRES
    le vol, meme oeil et meme astronaute (Pearson et Spearman, avec q de BH).
    C'est une repetabilite a LONG terme -- des mois separent les deux mesures,
    avec un vol au milieu -- a lire a cote de l'ICC, qui mesure la repetabilite
    a court terme entre repliques d'une meme seance.
  - ``subject_values.csv``   une ligne par (predicteur, sujet) : la valeur avant,
    apres, leur difference, et les six issues -- c'est ce qui permet de RETRACER
    un nuage de points, une correlation ne se lisant pas sans lui
  - ``outcomes.csv``         une ligne par SUJET : les six issues seules, avec
    les valeurs brutes des deux visites (``TRT250_x_before/after``,
    ``AL_before_mm`` / ``AL_after_mm``) et le ``sexe`` de l'astronaute. Le meme contenu que les colonnes
    d'issue de ``subject_values.csv``, mais sans les predicteurs -- donc 4 ko au
    lieu de 245 Mo, ce qui la rend relisible par le site pour tracer les issues
    les unes contre les autres
  - ``rigidity_cache.csv``   k et delta_CT par condition (recalcul couteux, mis
    en cache : supprimer le fichier pour le refaire)
  - ``summary.json``         effectifs et meilleurs resultats, pour la prose

Lancer (kernel pyOR, depuis la racine du depot) :
    C:/Users/transformer/anaconda3/envs/pyOR/python.exe Astronauts/compute_sans_predictors.py
"""

from __future__ import annotations

import csv as csvmod
import json
import sqlite3
import sys
import time
import warnings
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import (ConstantInputWarning, NearConstantInputWarning,
                         pearsonr, spearmanr)
from scipy.stats import t as stats_t

# Un predicteur presque constant sur les 18 sujets apparies donne un r
# ininterpretable, et scipy le signale une fois par test -- soit des milliers de
# lignes. L'information n'est pas perdue : elle est portee par
# ``ecart_type_predicteur`` dans ``sans_correlations.csv``, colonne a lire avant
# tout r eleve.
warnings.filterwarnings("ignore", category=NearConstantInputWarning)
warnings.filterwarnings("ignore", category=ConstantInputWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ocularrigidity.data.io import load_mask  # noqa: E402

from compute_rigidity_time_series import (  # noqa: E402
    DB_PATH,
    AL_DESCRIPTION,
    astro_code_from_folder,
    load_clinical_db,
    lookup_clinical,
    parse_eye,
    parse_moment,
)
from compute_rigidity_compare_mask_model import (  # noqa: E402
    ICC_TYPES,
    raw_timestamps_us,
    rigidity_hilbert,
)

# --------------------------------------------------------------------------- #
# Parametres
# --------------------------------------------------------------------------- #
PATH_GENERAL = Path("E:/SANSORI")
SEGVAR_ROOT = Path("E:/NASA_Rigidity/SegmentationVariations")
MASK_VARIANT = "model1_scale_1.0_flatten_choroid_xcorr"
MASKS_SUBDIR = "registered_masks"
DEMONS_SUBDIR = "demons_strain"
PULSE_SUBDIR = "pulse_from_data"
OUTPUT_SUBDIR = "sans_predictors"

# Demi-bande du passe-bande de la methode Hilbert (k de Sayah). +/- 7 BPM est la
# valeur historique de compute_rigidity_time_series ; le script de comparaison
# des masques en balayait trois, on garde la mediane pour ne pas tripler le panel.
HR_HALF_BAND_BPM = 20.0

# DEUX MARQUEURS DE SANS, pas un seul.
#
# 1. Le GONFLEMENT RETINIEN : les quatre quadrants du TRT a 250 um, plus leur
#    moyenne. C'est l'issue historique du depot.
# 2. L'APLATISSEMENT DU GLOBE : la variation de longueur axiale entre les deux
#    visites (``Biometry`` dans la base, la meme grandeur qui sert d'ENTREE au k
#    de Sayah -- voir l'avertissement de circularite plus bas).
#
# Meme convention de signe pour les deux : APRES - AVANT. Le delta_AL est donc
# NEGATIF quand le globe s'aplatit, ce qui est le cas des 20 sujets renseignes
# (mediane -0,085 mm, etendue -0,21 a 0,00). Ne pas le retourner : un signe
# retourne pour une seule issue rendrait toutes les cartes de chaleur illisibles,
# ou la couleur d'une case doit vouloir dire la meme chose sur les six colonnes.
#
# Les deux marqueurs ne mesurent PAS la meme chose et ne se remplacent pas : sur
# les 16 sujets qui ont les deux, ils correlent a r = -0,54 (p = 0,03) contre le
# quadrant nasal seulement. Ils sont donc traites comme six issues au meme rang,
# et la q-valeur par categorie est calculee A ISSUE FIXEE -- ajouter delta_AL ne
# durcit pas ``pearson_q_categorie`` des cinq delta_TRT deja publies (elle
# durcit ``pearson_q`` et ``pearson_q_famille``, qui comptent toute la table).
TRT_QUADRANTS = ("S", "I", "N", "T")
AL_OUTCOME = "delta_AL_mm"
OUTCOMES = ([f"delta_TRT_{q}" for q in TRT_QUADRANTS] + ["delta_TRT_moyen"]
            + [AL_OUTCOME])
VIEWS = ("before", "after", "delta")

# ATTENTION, CIRCULARITE PARTIELLE. ``k_sayah`` et ``delta_V_mm3`` sont calcules
# A PARTIR de la longueur axiale du meme moment (``rigidity_hilbert`` la recoit
# en argument). Les correler a delta_AL = AL(apres) - AL(avant) fait donc
# partager AL(avant) aux deux membres de la vue « avant ». Ces marqueurs sont
# NEANMOINS gardes dans le panel : les retirer pour une seule issue casserait
# l'uniformite du decompte de Benjamini-Hochberg d'une colonne a l'autre, et en
# pratique l'artefact ne mord pas (|r| <= 0,45 pour toute la categorie rigidite,
# aucune q sous 0,05). Le piege est documente sur la page Quarto « rigidity ».

# Methodes de pulse_from_data retenues dans le panel : les deux que
# compute_one_cycle_compare_methods a departagees, plus le brut comme temoin.
PULSE_METHODS = ("0_sans_filtre", "1_fir", "3a_mssa")

MIN_N_CORR = 4  # sous 4 sujets apparies, une correlation ne veut rien dire

# Les TROIS agregats du strain, tous RETINIENS et tous au meme rang : deux
# mesures de centre (moyenne, mediane) et une mesure de queue (p95 de la
# magnitude, rendu avec son signe -- voir ``compute_demons_strain.strain_p95_signe``).
# Le centre du strain etant domine par le biais negatif du lissage, la queue
# pose une question que les deux autres ne posent pas.
#
# Le strain CHOROIDIEN est absent du panel : il ne sert que de controle de
# methode a la figure de validation, qui le lit directement dans ``bins.csv``.
# L'EPAISSEUR choroidienne (CT), elle, reste l'abscisse de toute l'analyse.
STRAIN_AGREGATS = ("strain_retine", "strain_retine_med", "strain_retine_p95")

# LES TROIS CATEGORIES DE MARQUEURS. Elles decoupent le panel en trois questions
# distinctes, une page Quarto chacune, et servent de FAMILLE a la correction de
# Benjamini-Hochberg publiee dans ``pearson_q_categorie`` :
#
#   rigidite  -- tout ce qui se mesure sur l'EPAISSEUR choroidienne et sa
#                pulsation, sans passer par le champ de deformation : le k de
#                Sayah et ses entrees (famille ``rigidite``), les metriques
#                CT / pouls (``ct_pouls``), et l'epaisseur de chaque bande
#                (``CT_region_um``, publiee sous la famille ``strain`` par
#                commodite de calcul, mais qui n'est pas un strain).
#   strain    -- les trois agregats du strain retinien par bande.
#   viscosite -- les pentes du strain contre l'epaisseur et contre la
#                dilatation, dans leurs deux estimateurs (par one-cycle et
#                poolee), et les coefficients r de ces ajustements.
#
# Deux familles n'ont PAS de categorie et ne sont donc affichees nulle part :
# ``qc`` (jacobien, |u|max, gain de RMS -- des parametres de recalage) et
# ``pouls`` (les metriques de qualite de pulse_from_data). Elles restent dans
# ``sans_correlations.csv`` avec ``categorie`` vide.
#
# Cette table est DUPLIQUEE dans ``figures_sans_predictors/make_figures.py``
# (constante ``CATEGORIES_P4``) : les deux doivent bouger ensemble, sinon la
# q-valeur publiee ne porte plus sur les cases dessinees.
def categorie_de(famille: str, predicteur: str) -> str:
    """Categorie d'un predicteur, ou une chaine vide s'il n'en a aucune."""
    if predicteur == "CT_region_um" or famille in ("rigidite", "ct_pouls"):
        return "rigidite"
    if famille == "strain":
        return "strain"
    if famille in ("pente", "pente_pool"):
        return "viscosite"
    return ""


CATEGORIES = ("rigidite", "strain", "viscosite")


# --------------------------------------------------------------------------- #
# Portail de validite des one-cycles
# --------------------------------------------------------------------------- #
# Quatre criteres, tous calculables SANS REGARDER LE STRAIN. C'est la condition
# qui rend le filtrage legitime : ecarter les one-cycles dont le strain s'ecarte
# des autres fabriquerait la coherence qu'on pretend ensuite mesurer, et ce
# degre de liberte echapperait entierement a la correction de Benjamini-Hochberg
# appliquee plus bas.
#
# 1. PULSATILITE -- |dCT| / CT du one-cycle. La pulsation choroidienne
#    cardiaque vaut 1 a 2 % ; la mediane observee (2,0 %) le confirme. Mais la
#    queue monte a 43 %, ce qui n'est pas une pulsation : c'est un echec de
#    segmentation ou de repliement. Mesure sur les seuls MASQUES.
# 2. PHASE VALIDE -- p de la correlation entre le CT (voie masque) et le pouls
#    d'intensite (voie SVD), sur le support TEMPOREL. Les deux voies ne
#    partagent aucune etape de calcul : leur accord est une preuve independante
#    que le repliement est en phase. Le support a 6 bins est ecarte -- |r| du
#    hasard y vaut deja 0,34, filtrer dessus reviendrait a filtrer au hasard.
#    C'est p (et non le r signe) qui est teste : le signe du pouls est
#    arbitraire et n'est pas stable d'un one-cycle au suivant.
# 3. OCCUPATION -- nombre de frames dans le bin le plus pauvre.
# 4. JACOBIEN -- transformation non repliee. Evalue a sigma >= 10 SEULEMENT :
#    a sigma = 5 le jacobien se replie sur 1,5 % des pixels, mais cette valeur
#    de sigma est deja ecartee par ailleurs, et faire dependre le portail de
#    sigma donnerait un jeu de one-cycles DIFFERENT par sigma -- on ne pourrait
#    plus comparer les sigmas entre eux. Le portail doit rester le meme pour
#    toutes les variantes.
GATE_PULSATILITE_MAX_PCT = 5.0
GATE_PHASE_P_MAX = 0.05
GATE_FRAMES_MIN = 5
GATE_JAC_SIGMA_MIN = 10.0
GATE_MIN_ONE_CYCLES = 3  # une condition sous ce seuil est ecartee en entier

VARIANT_ROOT = SEGVAR_ROOT / MASK_VARIANT
DEMONS_DIR = VARIANT_ROOT / DEMONS_SUBDIR
PULSE_DIR = VARIANT_ROOT / PULSE_SUBDIR
OUT_DIR = VARIANT_ROOT / OUTPUT_SUBDIR
CSV_PREDICTORS = OUT_DIR / "predictors.csv"
CSV_REPEAT = OUT_DIR / "repeatability.csv"
CSV_REPEAT_OC = OUT_DIR / "repeatability_one_cycles.csv"
CSV_PREDICTORS_OC = OUT_DIR / "predictors_one_cycle.csv"
CSV_CORR = OUT_DIR / "sans_correlations.csv"
CSV_SUBJECTS = OUT_DIR / "subject_values.csv"
CSV_OUTCOMES = OUT_DIR / "outcomes.csv"
CSV_POOLED = OUT_DIR / "pooled_slopes.csv"
CSV_STABILITY = OUT_DIR / "stability.csv"
CSV_GATE = OUT_DIR / "one_cycle_filter.csv"
CSV_RIGIDITY = OUT_DIR / "rigidity_cache.csv"
JSON_SUMMARY = OUT_DIR / "summary.json"


# --------------------------------------------------------------------------- #
# Format long commun : une ligne = (condition, predicteur)
# --------------------------------------------------------------------------- #
LONG_COLS = ["slug", "astro", "moment", "condition", "eye", "subject",
             "famille", "predicteur", "cas", "sigma", "region", "valeur"]

# Le meme format, une ligne par (condition, ONE-CYCLE, predicteur) : c'est ce que
# consomme la repetabilite « one-cycles comme repliques ».
LONG_COLS_OC = LONG_COLS[:-1] + ["one_cycle", "valeur"]


def load_sex(db_path) -> dict:
    """Sexe par NUMERO DE DOSSIER -- l'entier qui prefixe ``01_210713001``.

    ``Astronauts.Id`` est cette meme cle (``astro_code_from_folder`` s'en sert
    pour remonter au ``Code``), et c'est la seule qui marche pour les DEUX
    astronautes sans ``Code`` : ils n'ont aucune mesure clinique, donc aucune
    issue, mais ils gardent un sexe et un dossier.

    Les libelles de la base (``Male`` / ``Female``) sont traduits ici, une fois,
    pour que le site n'ait pas a le refaire a chaque figure. Une valeur inconnue
    devient une chaine vide plutot que de lever : un sexe manquant doit ecarter
    le sujet des tests, pas arreter le lot.
    """
    con = sqlite3.connect(str(db_path))
    try:
        rows = con.execute("SELECT Id, Sex FROM Astronauts").fetchall()
    finally:
        con.close()
    traduction = {"male": "homme", "female": "femme"}
    return {int(i): traduction.get(str(sx).strip().lower(), "")
            for i, sx in rows}


def _identity(slug: str) -> dict:
    """(astro, moment, condition, oeil, sujet) a partir du slug
    ``<astro>__<condition>``. Le SUJET est ``astro + oeil`` -- PAS le code
    clinique, pour ne pas perdre les sujets absents de sansori_db.db."""
    astro, condition = slug.split("__", 1)
    eye = parse_eye(condition)
    return {
        "slug": slug, "astro": astro, "condition": condition,
        "moment": parse_moment(condition), "eye": eye,
        "subject": f"{astro}_{eye}",
    }


def long_rows(df: pd.DataFrame, famille: str, value_cols: dict,
              by=("cas", "sigma", "region")) -> list:
    """Passe une table large au format long. ``value_cols`` associe le nom de
    colonne au nom de predicteur publie."""
    rows = []
    ident_cache = {}
    for _, r in df.iterrows():
        slug = r["slug"]
        if slug not in ident_cache:
            ident_cache[slug] = _identity(slug)
        ident = ident_cache[slug]
        base = dict(ident)
        base["famille"] = famille
        for key in by:
            base[key] = r[key] if key in r.index else ""
        for col, name in value_cols.items():
            val = r.get(col, np.nan)
            if pd.isna(val):
                continue
            rows.append({**base, "predicteur": name, "valeur": float(val)})
    return rows


# --------------------------------------------------------------------------- #
# 1. Panel issu de compute_demons_strain.py
# --------------------------------------------------------------------------- #
def build_gate() -> pd.DataFrame:
    """Verdict de validite de chaque one-cycle, et sort de chaque condition.

    Retourne une table (slug, one_cycle) portant la valeur de chaque critere,
    son verdict, le verdict combine ``garde_one_cycle``, le nombre de survivants
    de la condition et ``garde_condition``.

    Aucun critere ne regarde le strain : voir le commentaire des constantes
    ``GATE_*`` pour le pourquoi.
    """
    f_oc = DEMONS_DIR / "one_cycles.csv"
    f_bin = DEMONS_DIR / "bins.csv"
    if not f_oc.exists():
        raise SystemExit(f"{f_oc} absent -- lancer d'abord compute_demons_strain.py")
    oc = pd.read_csv(f_oc)

    g = oc[["slug", "astro", "moment", "condition", "one_cycle"]].copy()
    g["pulsatilite_pct"] = 100.0 * oc["deltaCT_um"].abs() / oc["CT_moyen_um"]
    # p bilateral de la correlation CT / pouls, deduit de r et n (Student).
    r = oc["r_temps"].to_numpy(dtype=float)
    n = oc["n_temps"].to_numpy(dtype=float)
    ok = np.isfinite(r) & np.isfinite(n) & (n > 2) & (np.abs(r) < 1)
    p_temps = np.full(r.shape, np.nan)
    t = r[ok] * np.sqrt((n[ok] - 2) / np.maximum(1 - r[ok] ** 2, 1e-15))
    p_temps[ok] = 2 * stats_t.sf(np.abs(t), n[ok] - 2)
    g["p_phase"] = p_temps
    g["n_frames_min_bin"] = oc["n_frames_min_bin"]

    if f_bin.exists():
        b = pd.read_csv(f_bin)
        b = b[b["sigma"] >= GATE_JAC_SIGMA_MIN]
        jac = (b.groupby(["slug", "one_cycle"])["jac_neg_pct"].max()
               .rename("jac_neg_pct_max").reset_index())
        g = g.merge(jac, on=["slug", "one_cycle"], how="left")
    else:
        g["jac_neg_pct_max"] = 0.0

    g["ok_pulsatilite"] = g["pulsatilite_pct"] < GATE_PULSATILITE_MAX_PCT
    g["ok_phase"] = g["p_phase"] < GATE_PHASE_P_MAX
    g["ok_frames"] = g["n_frames_min_bin"] >= GATE_FRAMES_MIN
    g["ok_jacobien"] = g["jac_neg_pct_max"].fillna(0.0) <= 0.0
    criteres = ["ok_pulsatilite", "ok_phase", "ok_frames", "ok_jacobien"]
    g["n_criteres_echoues"] = (~g[criteres]).sum(axis=1)
    g["garde_one_cycle"] = g[criteres].all(axis=1)

    survivants = (g.groupby("slug")["garde_one_cycle"].sum()
                  .rename("n_one_cycles_retenus"))
    total = g.groupby("slug")["one_cycle"].size().rename("n_one_cycles_total")
    g = g.merge(survivants, on="slug").merge(total, on="slug")
    g["frac_retenus"] = g["n_one_cycles_retenus"] / g["n_one_cycles_total"]
    g["garde_condition"] = g["n_one_cycles_retenus"] >= GATE_MIN_ONE_CYCLES
    # Un one-cycle d'une condition ecartee ne sert a rien, meme s'il est valide.
    g["retenu"] = g["garde_one_cycle"] & g["garde_condition"]
    return g


def apply_gate(df: pd.DataFrame, gate: pd.DataFrame) -> pd.DataFrame:
    """Ne garde que les lignes dont le (slug, one_cycle) a passe le portail."""
    keep = gate.loc[gate["retenu"], ["slug", "one_cycle"]]
    return df.merge(keep, on=["slug", "one_cycle"], how="inner")


def demons_predictors(gate: pd.DataFrame) -> list:
    """Agrege au niveau CONDITION : mediane sur les one-cycles (et sur les bins
    pour le strain).

    La mediane plutot que la moyenne : sur une condition courte un one-cycle
    peut ne compter que 4 frames par bin, et son strain est alors une valeur
    aberrante qu'il ne faut pas laisser tirer la condition entiere.
    """
    rows = []

    # --- strain par region ---------------------------------------------------
    f_reg = DEMONS_DIR / "regions.csv"
    if f_reg.exists():
        reg = apply_gate(pd.read_csv(f_reg), gate)
        agg = (reg.groupby(["slug", "cas", "sigma", "region"], sort=False)
               [list(STRAIN_AGREGATS) + ["CT_region_um", "dCT_region_um"]]
               .median().reset_index())
        rows += long_rows(agg, "strain", {c: c for c in STRAIN_AGREGATS})
        # L'epaisseur de la bande est une mesure sur les MASQUES : elle ne depend
        # pas de sigma. La laisser dans la boucle ci-dessus la republierait a
        # l'identique quatre fois, ce qui gonflerait le nombre de tests -- et
        # donc durcirait la correction de Benjamini-Hochberg -- sans apporter la
        # moindre information. Une seule fois par (condition, cas, region).
        ct_reg = (reg.groupby(["slug", "cas", "region"], sort=False)["CT_region_um"]
                  .median().reset_index())
        ct_reg["sigma"] = np.nan
        rows += long_rows(ct_reg, "strain", {"CT_region_um": "CT_region_um"})

    # --- pentes strain ~ abscisse -------------------------------------------
    f_slo = DEMONS_DIR / "slopes.csv"
    if f_slo.exists():
        slo = apply_gate(pd.read_csv(f_slo), gate)
        agg = (slo.groupby(["slug", "cas", "sigma", "region", "abscisse", "tissu"],
                           sort=False)[["pente_par_um", "r"]]
               .median().reset_index())
        # Le nom du predicteur porte l'abscisse et le tissu ; cas / sigma /
        # region restent des colonnes, pour pouvoir filtrer dessus.
        agg["predicteur_pente"] = "pente_" + agg["abscisse"] + "_" + agg["tissu"]
        agg["predicteur_r"] = "r_" + agg["abscisse"] + "_" + agg["tissu"]
        for _, r in agg.iterrows():
            ident = _identity(r["slug"])
            for col, name_col in (("pente_par_um", "predicteur_pente"), ("r", "predicteur_r")):
                if pd.isna(r[col]):
                    continue
                rows.append({**ident, "famille": "pente", "predicteur": r[name_col],
                             "cas": r["cas"], "sigma": r["sigma"], "region": r["region"],
                             "valeur": float(r[col])})

    # --- pentes POOLEES sur tous les one-cycles d'une condition -------------
    # Les pentes par one-cycle demandees plus haut sont, contre ``dCT_region`` et
    # contre ``CT_region``, MATHEMATIQUEMENT LA MEME : a l'interieur d'un
    # one-cycle, dCT = CT - CT(bin de reference) et la reference est une
    # constante, donc seule l'ordonnee change. La distinction que le carnet
    # faisait -- "la retine suit-elle le gonflement" contre "le strain
    # depend-il de l'epaisseur" -- n'a de sens qu'en POOLANT les one-cycles,
    # ou la reference varie d'un cycle a l'autre. Elle est donc refaite ici a
    # l'echelle de la condition, sur tous les couples (one-cycle, bin).
    if f_reg.exists():
        pool_rows, detail_rows = [], []
        keys = ["slug", "cas", "sigma", "region"]
        for (slug, cas, sigma, region), sub in reg.groupby(keys, sort=False):
            ident = _identity(slug)
            for xcol in ("dCT_region_um", "CT_region_um", "dCT_um", "CT_um"):
                for ycol in STRAIN_AGREGATS:
                    x = sub[xcol].to_numpy(dtype=float)
                    y = sub[ycol].to_numpy(dtype=float)
                    m = np.isfinite(x) & np.isfinite(y)
                    n = int(m.sum())
                    if n < 5 or np.std(x[m]) == 0:
                        continue
                    slope, intercept = (float(v) for v in np.polyfit(x[m], y[m], 1))
                    if np.std(y[m]) > 0:
                        r, pval = pearsonr(x[m], y[m])
                        r, pval = float(r), float(pval)
                    else:
                        r, pval = np.nan, np.nan
                    base = {**ident, "famille": "pente_pool", "cas": cas,
                            "sigma": sigma, "region": region}
                    pool_rows.append({**base, "predicteur": f"poolpente_{xcol}_{ycol}",
                                      "valeur": slope})
                    if np.isfinite(r):
                        pool_rows.append({**base, "predicteur": f"poolr_{xcol}_{ycol}",
                                          "valeur": r})
                    # Table dediee : le volcano de la page Quarto a besoin de la
                    # pente ET de sa p-valeur, ce que le format long des
                    # predicteurs (une seule colonne ``valeur``) ne peut pas
                    # porter.
                    detail_rows.append({
                        "slug": slug, "astro": ident["astro"],
                        "moment": ident["moment"], "condition": ident["condition"],
                        "cas": cas, "sigma": sigma, "region": region,
                        "abscisse": xcol, "tissu": ycol,
                        "pente_par_um": slope, "ordonnee": intercept,
                        "r": r, "p": pval, "n": n,
                    })
        rows += pool_rows
        if detail_rows:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(detail_rows).to_csv(CSV_POOLED, index=False)
            print(f"  pentes poolees : {len(detail_rows)} ajustements -> {CSV_POOLED}")

    # --- CT / pouls et CT par one-cycle -------------------------------------
    f_oc = DEMONS_DIR / "one_cycles.csv"
    if f_oc.exists():
        oc = apply_gate(pd.read_csv(f_oc), gate)
        # Le SIGNE du pouls est fixe arbitrairement par ``compute_pulse_from_data``
        # (c'est une combinaison de composantes SVD), et il n'est pas comparable
        # d'une condition a l'autre. La mediane de ``r`` sur la cohorte est donc
        # une statistique vide ; c'est |r| qui porte l'information. Les deux sont
        # publies -- ``r`` reste lisible A L'INTERIEUR d'une condition.
        oc = oc.assign(abs_r_temps=oc["r_temps"].abs(), abs_r_bins=oc["r_bins"].abs())
        agg = (oc.groupby("slug", sort=False)
               [["r_temps", "cov_temps", "r_bins", "cov_bins",
                 "abs_r_temps", "abs_r_bins",
                 "CT_moyen_um", "deltaCT_um"]].median().reset_index())
        agg["cas"] = ""
        agg["sigma"] = np.nan
        agg["region"] = "toutes"
        rows += long_rows(agg, "ct_pouls", {
            "r_temps": "r_CT_pouls_temps", "cov_temps": "cov_CT_pouls_temps",
            "r_bins": "r_CT_pouls_bins", "cov_bins": "cov_CT_pouls_bins",
            "abs_r_temps": "absr_CT_pouls_temps", "abs_r_bins": "absr_CT_pouls_bins",
            "CT_moyen_um": "CT_moyen_um", "deltaCT_um": "deltaCT_one_cycle_um",
        })

    # --- controle qualite du recalage ---------------------------------------
    f_bin = DEMONS_DIR / "bins.csv"
    if f_bin.exists():
        b = apply_gate(pd.read_csv(f_bin), gate)
        # Le gain de RMS est une colonne AVANT l'agregation : le calculer apres,
        # via un second groupby aligne par position, romprait au moindre
        # changement d'ordre des groupes.
        b = b.assign(gain_rms=b["rms_avant"] - b["rms_apres"])
        agg = (b.groupby(["slug", "cas", "sigma"], sort=False)
               [["jac_neg_pct", "u_max_um", "gain_rms"]].median().reset_index())
        agg["region"] = "toutes"
        rows += long_rows(agg, "qc", {
            "jac_neg_pct": "jac_neg_pct", "u_max_um": "u_max_um",
            "gain_rms": "gain_rms",
        })
    return rows


# --------------------------------------------------------------------------- #
# 1 bis. Le MEME panel, sans agreger les one-cycles
# --------------------------------------------------------------------------- #
def demons_predictors_one_cycle(gate: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par (condition, ONE-CYCLE, predicteur), format long.

    C'est ``demons_predictors`` prive de sa derniere agregation : la mediane sur
    les one-cycles n'est pas prise, chaque one-cycle garde sa valeur. Les
    agregations INTERNES a un one-cycle restent (mediane sur les bins pour le
    strain et le controle qualite) : un one-cycle doit rendre UN nombre.

    Trois familles ne peuvent pas y figurer, et c'est structurel :

    - ``pente_pool`` -- l'ajustement poole est defini sur tous les couples
      (one-cycle, bin) d'une condition ; il n'existe pas par one-cycle. Sa
      version par one-cycle porte deja un nom : c'est la famille ``pente``.
    - ``pouls`` -- les metriques de pulse_from_data sont produites par video.
    - ``rigidite`` -- le k de Sayah est ajuste sur la serie temporelle entiere.

    Retourne un DataFrame plutot qu'une liste de dictionnaires : a quelques
    centaines de milliers de lignes, la construction ligne a ligne de
    ``long_rows`` couterait des minutes.
    """
    parts = []
    ID_OC = ["slug", "one_cycle", "cas", "sigma", "region"]

    # --- strain par bande ----------------------------------------------------
    f_reg = DEMONS_DIR / "regions.csv"
    if f_reg.exists():
        reg = apply_gate(pd.read_csv(f_reg), gate)
        agg = (reg.groupby(["slug", "one_cycle", "cas", "sigma", "region"], sort=False)
               [list(STRAIN_AGREGATS)].median().reset_index())
        m = agg.melt(id_vars=ID_OC, value_vars=list(STRAIN_AGREGATS),
                     var_name="predicteur", value_name="valeur")
        m["famille"] = "strain"
        parts.append(m)
        # Meme raison qu'au niveau condition : l'epaisseur de la bande est une
        # mesure sur les MASQUES, elle ne depend pas de sigma. Une seule fois.
        ct = (reg.groupby(["slug", "one_cycle", "cas", "region"], sort=False)
              ["CT_region_um"].median().reset_index()
              .rename(columns={"CT_region_um": "valeur"}))
        ct["sigma"] = np.nan
        ct["predicteur"] = "CT_region_um"
        ct["famille"] = "strain"
        parts.append(ct)

    # --- pentes strain ~ abscisse, DEJA par one-cycle ------------------------
    f_slo = DEMONS_DIR / "slopes.csv"
    if f_slo.exists():
        slo = apply_gate(pd.read_csv(f_slo), gate)
        for col, prefixe in (("pente_par_um", "pente_"), ("r", "r_")):
            d = slo[ID_OC].copy()
            d["predicteur"] = prefixe + slo["abscisse"] + "_" + slo["tissu"]
            d["valeur"] = slo[col].to_numpy()
            d["famille"] = "pente"
            parts.append(d)

    # --- CT / pouls ----------------------------------------------------------
    f_oc = DEMONS_DIR / "one_cycles.csv"
    if f_oc.exists():
        oc = apply_gate(pd.read_csv(f_oc), gate)
        oc = oc.assign(abs_r_temps=oc["r_temps"].abs(), abs_r_bins=oc["r_bins"].abs())
        noms = {"r_temps": "r_CT_pouls_temps", "cov_temps": "cov_CT_pouls_temps",
                "r_bins": "r_CT_pouls_bins", "cov_bins": "cov_CT_pouls_bins",
                "abs_r_temps": "absr_CT_pouls_temps", "abs_r_bins": "absr_CT_pouls_bins",
                "CT_moyen_um": "CT_moyen_um", "deltaCT_um": "deltaCT_one_cycle_um"}
        m = oc.melt(id_vars=["slug", "one_cycle"], value_vars=list(noms),
                    var_name="predicteur", value_name="valeur")
        m["predicteur"] = m["predicteur"].map(noms)
        m["cas"] = ""
        m["sigma"] = np.nan
        m["region"] = "toutes"
        m["famille"] = "ct_pouls"
        parts.append(m)

    # --- controle qualite du recalage ---------------------------------------
    f_bin = DEMONS_DIR / "bins.csv"
    if f_bin.exists():
        b = apply_gate(pd.read_csv(f_bin), gate)
        b = b.assign(gain_rms=b["rms_avant"] - b["rms_apres"])
        cols = ["jac_neg_pct", "u_max_um", "gain_rms"]
        agg = (b.groupby(["slug", "one_cycle", "cas", "sigma"], sort=False)
               [cols].median().reset_index())
        m = agg.melt(id_vars=["slug", "one_cycle", "cas", "sigma"], value_vars=cols,
                     var_name="predicteur", value_name="valeur")
        m["region"] = "toutes"
        m["famille"] = "qc"
        parts.append(m)

    if not parts:
        return pd.DataFrame(columns=LONG_COLS_OC)

    long = pd.concat(parts, ignore_index=True).dropna(subset=["valeur"])
    ident = pd.DataFrame([_identity(sl) for sl in long["slug"].unique()])
    long = long.merge(ident, on="slug", how="left")
    long["cas"] = long["cas"].fillna("")
    long["region"] = long["region"].fillna("toutes")
    long["valeur"] = long["valeur"].astype(float)
    return long[LONG_COLS_OC]


# --------------------------------------------------------------------------- #
# 2. Panel issu de compute_pulse_from_data.py (deja calcule, cout nul)
# --------------------------------------------------------------------------- #
def pulse_predictors() -> list:
    """Qualite de l'extraction du pouls : ce que le lot mesure deja par
    condition et par methode. Ce ne sont pas des parametres physiologiques mais
    des indicateurs de QUALITE du signal -- s'ils predisent le SANS mieux que le
    strain, c'est un avertissement, pas un resultat."""
    f = PULSE_DIR / "conditions.csv"
    if not f.exists():
        return []
    cond = pd.read_csv(f)
    cond = cond[cond["status"] == "ok"]
    rows = []
    for _, r in cond.iterrows():
        ident = _identity(r["slug"])
        base = {**ident, "famille": "pouls", "cas": "", "sigma": np.nan}
        scalars = {
            "hr_BPM": "hr_BPM",
            "n_channels": "n_canaux_cardiaques",
            "var_1er_triplet": "var_1er_triplet",
            "n_comp_pic_en_bande": "n_comp_pic_en_bande",
            "mssa_pc1_var": "mssa_pc1_var",
        }
        if np.isfinite(r.get("hr_measured_BPM", np.nan)) and np.isfinite(r.get("hr_BPM", np.nan)):
            rows.append({**base, "predicteur": "ecart_HR_mesuree_BPM", "region": "toutes",
                         "valeur": float(r["hr_measured_BPM"] - r["hr_BPM"])})
        for col, name in scalars.items():
            val = r.get(col, np.nan)
            if pd.notna(val):
                rows.append({**base, "predicteur": name, "region": "toutes",
                             "valeur": float(val)})
        for m in PULSE_METHODS:
            for col, name in (("HRmed", "HR_med_BPM"), ("HRiqr", "HR_IQR_BPM"),
                              ("fneg", "f_neg_frac"), ("corrfir", "corr_fir")):
                val = r.get(f"{col}_{m}", np.nan)
                if pd.notna(val):
                    rows.append({**base, "predicteur": f"{name}[{m}]",
                                 "region": "toutes", "valeur": float(val)})

    f_meth = PULSE_DIR / "methods.csv"
    if f_meth.exists():
        meth = pd.read_csv(f_meth)
        meth = meth[meth["methode"].isin(PULSE_METHODS)]
        for _, r in meth.iterrows():
            ident = _identity(r["slug"])
            base = {**ident, "famille": "pouls", "cas": "", "sigma": np.nan,
                    "region": "toutes"}
            for col, name in (("en_bande_frac", "en_bande_frac"),
                              ("pic_LS_BPM", "pic_LS_BPM"),
                              ("hors_bande_frac", "hors_bande_frac")):
                val = r.get(col, np.nan)
                if pd.notna(val):
                    rows.append({**base, "predicteur": f"{name}[{r['methode']}]",
                                 "valeur": float(val)})
    return rows


# --------------------------------------------------------------------------- #
# 3. Panel historique : k de Sayah et delta_CT (recalcul, mis en cache)
# --------------------------------------------------------------------------- #
def rigidity_predictors(meas, id2code) -> list:
    """``k`` (Sayah et al. 2020) et ``delta_CT`` par la voie Hilbert, sur les
    masques recales de la MEME variante.

    C'est le predicteur historique du depot : il sert de reference de lecture
    pour les nouveaux. ``delta_CT`` est rapporte a cote de ``k`` parce qu'il ne
    depend NI de AL, NI de IOP, NI de OPA -- si une correlation existe pour k
    mais pas pour delta_CT, elle vient du modele de coque spherique et des
    donnees cliniques, pas du signal OCT.

    Le calcul relit tous les masques (~1 min par condition) : il est mis en
    cache dans ``rigidity_cache.csv``. Supprimer ce fichier pour le refaire.
    """
    if CSV_RIGIDITY.exists():
        cache = pd.read_csv(CSV_RIGIDITY)
        print(f"  k / delta_CT : cache relu ({len(cache)} conditions)")
    else:
        rows = []
        t0 = time.perf_counter()
        conditions = sorted((VARIANT_ROOT / MASKS_SUBDIR).glob("*/*/*/mask.npz"))
        print(f"  k / delta_CT : {len(conditions)} masques a relire "
              f"(cache absent, compter quelques minutes)")
        for i, mask_path in enumerate(conditions, start=1):
            condition = mask_path.parent.name
            moment_dir = mask_path.parent.parent.name
            astro = mask_path.parent.parent.parent.name
            slug = f"{astro}__{condition}"
            path_condi = PATH_GENERAL / astro / moment_dir / condition
            try:
                visit = path_condi / "Data Files" / "visit_data.csv"
                if not visit.exists():
                    continue
                # quoting=QUOTE_NONE est OBLIGATOIRE : la colonne Notes contient
                # un guillemet non appaire qui casse le parseur par defaut.
                df = pd.read_csv(visit, quoting=csvmod.QUOTE_NONE)
                hr = float(np.nanmean(pd.to_numeric(df["HR"], errors="coerce")))
                if not np.isfinite(hr):
                    continue
                iop = float(np.nanmean(pd.to_numeric(df["PascalIOP"], errors="coerce")))
                opa = float(np.nanmean(pd.to_numeric(df["PascalOPA"], errors="coerce")))
                code = astro_code_from_folder(astro, id2code)
                eye = parse_eye(condition)
                moment = parse_moment(moment_dir)
                al = lookup_clinical(meas, code, eye, moment, AL_DESCRIPTION)

                raw_dir = None
                for name in ("RawImages", "RawData"):
                    if (path_condi / name).is_dir():
                        raw_dir = path_condi / name
                        break
                if raw_dir is None:
                    continue
                ts_us = raw_timestamps_us(raw_dir)
                mask = np.asarray(load_mask(mask_path), dtype=bool)
                if mask.shape[0] != ts_us.shape[0]:
                    continue
                res = rigidity_hilbert(mask, ts_us, hr, HR_HALF_BAND_BPM, al, iop, opa)
                rows.append({"slug": slug, "CT_mm": res["CT_mm"],
                             "delta_CT_mm": res["delta_CT_mm"], "k": res["k"],
                             "delta_V": res["delta_V"], "AL_mm": al,
                             "IOP_mmHg": iop, "OPA_mmHg": opa})
            except Exception as exc:  # noqa: BLE001 - une condition ne doit pas tout arreter
                print(f"    [{i}/{len(conditions)}] {slug} : echec ({exc})")
                continue
        cache = pd.DataFrame(rows)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        cache.to_csv(CSV_RIGIDITY, index=False)
        print(f"  k / delta_CT : {len(cache)} conditions en "
              f"{(time.perf_counter() - t0) / 60:.1f} min -> {CSV_RIGIDITY}")

    cache = cache.copy()
    cache["cas"] = ""
    cache["sigma"] = np.nan
    cache["region"] = "toutes"
    return long_rows(cache, "rigidite", {
        "k": "k_sayah", "delta_CT_mm": "delta_CT_mm", "CT_mm": "CT_mm",
        "delta_V": "delta_V_mm3",
    })


# --------------------------------------------------------------------------- #
# Repetabilite : ICC (pingouin, paires equilibrees) + composantes de variance
# --------------------------------------------------------------------------- #
GROUP_KEYS = ["famille", "predicteur", "cas", "sigma", "region", "moment"]


def variance_components(g: pd.DataFrame, cible: str = "subject") -> dict:
    """ANOVA a un facteur (la cible), plan DESEQUILIBRE : tous les replicats a la
    fois, contrairement aux paires de pingouin.

    ``cible`` nomme la colonne qui porte l'unite a separer : ``subject`` quand
    les repliques sont des acquisitions repetees du meme oeil, ``slug`` quand ce
    sont les one-cycles d'une meme video. Les cles de sortie gardent le mot
    « sujets » dans les deux cas, pour que les deux tables aient le meme schema.

    Estimateur classique du modele a effets aleatoires :
        MSB = sum_i n_i (moy_i - moy)^2 / (k - 1)
        MSW = sum_i sum_j (x_ij - moy_i)^2 / (N - k)
        n0  = (N - sum n_i^2 / N) / (k - 1)
        var_intra = MSW ;  var_inter = (MSB - MSW) / n0
    ``var_inter`` peut sortir NEGATIVE quand la dispersion entre sujets est plus
    petite que le bruit de mesure : elle est laissee telle quelle plutot que
    ramenee a zero, parce que c'est exactement le cas qu'il faut voir -- le
    parametre ne separe alors pas les sujets.

    ``sd_intra`` est la repetabilite ABSOLUE, dans l'unite du parametre ; le
    coefficient de repetabilite ``2,77 x sd_intra`` est l'ecart maximal attendu
    entre deux mesures du meme oeil dans 95 % des cas (Bland & Altman).
    """
    g = g.dropna(subset=["valeur"])
    counts = g.groupby(cible)["valeur"].size()
    counts = counts[counts >= 2]  # un sujet a une seule mesure n'informe pas l'intra
    g = g[g[cible].isin(counts.index)]
    k = int(g[cible].nunique())
    n_tot = int(len(g))
    out = {"n_sujets_var": k, "n_mesures_var": n_tot,
           "var_intra": np.nan, "var_inter": np.nan, "sd_intra": np.nan,
           "cv_intra_pct": np.nan, "coef_repetabilite": np.nan,
           "icc1_desequilibre": np.nan}
    if k < 2 or n_tot - k < 1:
        return out

    grand = float(g["valeur"].mean())
    means = g.groupby(cible)["valeur"].mean()
    sizes = g.groupby(cible)["valeur"].size()
    ss_b = float((sizes * (means - grand) ** 2).sum())
    ss_w = float(g.groupby(cible)["valeur"].apply(lambda s: ((s - s.mean()) ** 2).sum()).sum())
    ms_b = ss_b / (k - 1)
    ms_w = ss_w / (n_tot - k)
    n0 = (n_tot - float((sizes ** 2).sum()) / n_tot) / (k - 1)
    var_inter = (ms_b - ms_w) / n0 if n0 > 0 else np.nan
    sd_intra = float(np.sqrt(ms_w)) if ms_w >= 0 else np.nan
    denom = ms_b + (n0 - 1) * ms_w
    out.update({
        "var_intra": float(ms_w), "var_inter": float(var_inter),
        "sd_intra": sd_intra,
        "cv_intra_pct": (100.0 * sd_intra / abs(grand)) if grand != 0 and np.isfinite(sd_intra) else np.nan,
        "coef_repetabilite": 2.77 * sd_intra if np.isfinite(sd_intra) else np.nan,
        "icc1_desequilibre": float((ms_b - ms_w) / denom) if denom not in (0.0,) else np.nan,
    })
    return out


def repeatability(predictors: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par (predicteur, ..., moment, paire de repliques, type d'ICC),
    plus une ligne ``pair = "toutes"`` portant les composantes de variance.

    Sujet = ``astro + oeil``, rater = rang de la replique (ordre alphabetique de
    ``condition``, donc OD1 -> 0, OD2 -> 1, ...). Le chiffre du nom n'est jamais
    lu : c'est le RANG qui fait le rater, comme dans
    ``compute_rigidity_compare_mask_model.compute_repeatability_icc``.
    """
    rows = []
    for keys, g_full in predictors.groupby(GROUP_KEYS, dropna=False, sort=False):
        info = dict(zip(GROUP_KEYS, keys))
        g_full = g_full.dropna(subset=["valeur"]).copy()
        if g_full.empty:
            continue
        g_full["rater"] = (g_full.sort_values("condition")
                           .groupby("subject").cumcount())

        vc = variance_components(g_full)
        rows.append({**info, "pair": "toutes", "icc_type": "variance",
                     "n_subjects": vc["n_sujets_var"], "icc": vc["icc1_desequilibre"],
                     "icc_ci95_low": np.nan, "icc_ci95_high": np.nan,
                     "icc_f": np.nan, "icc_pval": np.nan, **vc})

        max_rater = int(g_full["rater"].max())
        for other in range(1, max_rater + 1):
            pair = f"1v{other + 1}"
            g = g_full[g_full["rater"].isin([0, other])]
            # Ne garder que les sujets qui ont BIEN les deux repliques de cette
            # paire (1v3 exige la 3e replique, pas juste >= 2 au total).
            g = g[g.groupby("subject")["rater"].transform("nunique") == 2]
            n_subjects = int(g["subject"].nunique())
            if n_subjects < 3 or len(g) < 6:
                continue
            try:
                icc_table = pg.intraclass_corr(data=g, targets="subject", raters="rater",
                                               ratings="valeur", nan_policy="omit")
            except Exception:  # noqa: BLE001 - variance nulle, plan degenere...
                continue
            for icc_type in ICC_TYPES:
                sel = icc_table.loc[icc_table["Type"] == icc_type]
                if sel.empty:
                    continue
                r = sel.iloc[0]
                rows.append({**info, "pair": pair, "icc_type": icc_type,
                             "n_subjects": n_subjects, "icc": r["ICC"],
                             "icc_ci95_low": r["CI95"][0], "icc_ci95_high": r["CI95"][1],
                             "icc_f": r["F"], "icc_pval": r["pval"],
                             "n_sujets_var": vc["n_sujets_var"],
                             "n_mesures_var": vc["n_mesures_var"],
                             "var_intra": vc["var_intra"], "var_inter": vc["var_inter"],
                             "sd_intra": vc["sd_intra"], "cv_intra_pct": vc["cv_intra_pct"],
                             "coef_repetabilite": vc["coef_repetabilite"],
                             "icc1_desequilibre": vc["icc1_desequilibre"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Repetabilite, SECONDE definition du replicat : les one-cycles d'une video
# --------------------------------------------------------------------------- #
# La table ``repeatability.csv`` prend pour replicats les ACQUISITIONS REPETEES
# du meme oeil dans la meme seance (``..._OD1``, ``..._OD2``). C'est la bonne
# definition pour un biomarqueur, mais elle est chere : seuls les sujets ayant
# deux acquisitions comptent, ce qui laisse une douzaine de cibles.
#
# Ici la replique est le ONE-CYCLE, et la cible est la VIDEO. Un one-cycle est un
# battement replie : deux one-cycles d'une meme video sont deux mesures du meme
# oeil, dans la meme seconde, sur le meme enregistrement -- rien ne devrait les
# distinguer qu'un vrai battement a l'autre et le bruit de la chaine. L'ICC
# repond alors a une question PLUS ETROITE que celle de l'autre table :
#
#   « la valeur publiee pour une video est-elle stable d'un battement a
#     l'autre ? »
#
# et non « le parametre separe-t-il les sujets ». Un ICC eleve ici ne dit donc
# PAS que le parametre est un bon marqueur : entre deux videos il y a aussi
# l'oeil, la seance et la derive de l'acquisition, alors qu'entre deux one-cycles
# il n'y a que le battement. C'est un plancher de bruit, pas un verdict -- et
# c'est justement pour cela qu'il est informatif : un parametre qui echoue DEJA
# ici ne peut rien reussir plus loin.
#
# Le ``moment`` ne sert pas de cle de groupe : la cible etant la video, une video
# appartient a un seul moment et les 76 peuvent etre traitees ensemble. La
# colonne est conservee a la valeur ``toutes`` pour que les deux tables aient
# exactement le meme schema (plus la colonne ``cible``).
# Nombre maximal de PAIRES pingouin par predicteur (1v2, 1v3, ...). Sans borne,
# une video a 9 one-cycles en produit 8, et les dernieres ne portent que 3 a 9
# videos -- du bruit, pour 6 minutes de calcul supplementaires sur le lot. Les
# figures ne lisent de toute facon que la paire 1v2 ; la lecture qui utilise TOUS
# les one-cycles a la fois est la ligne ``pair = "toutes"`` (ICC(1) sur le plan
# desequilibre), qui elle ne coute rien.
MAX_PAIRES_OC = 4  # -> 1v2, 1v3, 1v4, 1v5

GROUP_KEYS_OC = ["famille", "predicteur", "cas", "sigma", "region"]


def repeatability_one_cycles(pred_oc: pd.DataFrame) -> pd.DataFrame:
    """Meme machinerie que ``repeatability``, cible = la video, rater = le rang
    du one-cycle dans la video (ordre croissant du numero de one-cycle).

    Comme dans l'autre table : une ligne ``pair = "toutes"`` porte les
    composantes de variance sur le plan DESEQUILIBRE (tous les one-cycles a la
    fois, de 3 a 9 selon la video), puis une ligne par (paire, type d'ICC) pour
    pingouin, qui exige un plan equilibre.
    """
    rows = []
    if pred_oc.empty:
        return pd.DataFrame(rows)
    for keys, g_full in pred_oc.groupby(GROUP_KEYS_OC, dropna=False, sort=False):
        info = dict(zip(GROUP_KEYS_OC, keys))
        info["moment"] = "toutes"
        info["cible"] = "condition"
        g_full = g_full.dropna(subset=["valeur"]).copy()
        if g_full.empty:
            continue
        g_full["rater"] = (g_full.sort_values("one_cycle")
                           .groupby("slug").cumcount())

        vc = variance_components(g_full, cible="slug")
        rows.append({**info, "pair": "toutes", "icc_type": "variance",
                     "n_subjects": vc["n_sujets_var"], "icc": vc["icc1_desequilibre"],
                     "icc_ci95_low": np.nan, "icc_ci95_high": np.nan,
                     "icc_f": np.nan, "icc_pval": np.nan, **vc})

        max_rater = min(int(g_full["rater"].max()), MAX_PAIRES_OC)
        for other in range(1, max_rater + 1):
            pair = f"1v{other + 1}"
            g = g_full[g_full["rater"].isin([0, other])]
            # Meme exigence que pour les repliques : la paire 1v4 ne compte que
            # les videos qui possedent BIEN un 4e one-cycle retenu.
            g = g[g.groupby("slug")["rater"].transform("nunique") == 2]
            n_cibles = int(g["slug"].nunique())
            if n_cibles < 3 or len(g) < 6:
                continue
            try:
                icc_table = pg.intraclass_corr(data=g, targets="slug", raters="rater",
                                               ratings="valeur", nan_policy="omit")
            except Exception:  # noqa: BLE001 - variance nulle, plan degenere...
                continue
            for icc_type in ICC_TYPES:
                sel = icc_table.loc[icc_table["Type"] == icc_type]
                if sel.empty:
                    continue
                r = sel.iloc[0]
                rows.append({**info, "pair": pair, "icc_type": icc_type,
                             "n_subjects": n_cibles, "icc": r["ICC"],
                             "icc_ci95_low": r["CI95"][0], "icc_ci95_high": r["CI95"][1],
                             "icc_f": r["F"], "icc_pval": r["pval"],
                             "n_sujets_var": vc["n_sujets_var"],
                             "n_mesures_var": vc["n_mesures_var"],
                             "var_intra": vc["var_intra"], "var_inter": vc["var_inter"],
                             "sd_intra": vc["sd_intra"], "cv_intra_pct": vc["cv_intra_pct"],
                             "coef_repetabilite": vc["coef_repetabilite"],
                             "icc1_desequilibre": vc["icc1_desequilibre"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Prediction de SANS : delta_TRT par quadrant, trois vues
# --------------------------------------------------------------------------- #
def build_outcomes(meas, id2code, id2sex, subjects: list) -> pd.DataFrame:
    """Les SIX issues par sujet -- cinq de gonflement retinien, une
    d'aplatissement du globe :

        delta_TRT_x = TRT250_x(apres vol) - TRT250_x(avant vol)   x = S, I, N, T
        delta_TRT_moyen                                           leur moyenne
        delta_AL_mm = AL(apres vol) - AL(avant vol)

    ``_date_to_moment`` (compute_rigidity_time_series) traduit les etiquettes de
    la base : ``L-*`` -> before, ``L+*`` / ``R-*`` -> during, ``R+*`` -> after.
    NaN si l'une des deux mesures manque : un delta a moitie renseigne n'est pas
    un delta. Les valeurs brutes des deux visites sont publiees a cote du delta,
    pour les deux issues : sans elles, un delta aberrant ne se diagnostique pas.

    La longueur axiale est lue par la MEME fonction que le reste du depot,
    ``lookup_clinical(..., AL_DESCRIPTION)`` (description ``Biometry``), celle
    qui alimente deja ``rigidity_predictors`` -- d'ou la circularite partielle
    signalee au niveau de ``OUTCOMES``. Elle n'existe qu'aux moments ``before``
    et ``after`` : 20 sujets sur 28 ont le couple complet, contre 18 pour le TRT.

    La colonne ``sexe`` est jointe ici et pas plus loin : c'est la seule table du
    lot ou une ligne vaut un SUJET, donc la seule ou une covariable par personne
    a sa place. Le site s'en sert pour colorer les points et pour ajuster les
    comparaisons de moyennes.
    """
    rows = []
    for subject in sorted(set(subjects)):
        astro, eye = subject.rsplit("_", 1)
        code = astro_code_from_folder(astro, id2code)
        entry = {"subject": subject, "astro": astro, "eye": eye, "code": code,
                 "sexe": id2sex.get(_folder_id(astro), "")}
        vals = []
        for q in TRT_QUADRANTS:
            before = lookup_clinical(meas, code, eye, "before", f"TRT250_{q}")
            after = lookup_clinical(meas, code, eye, "after", f"TRT250_{q}")
            d = float(after - before) if np.isfinite(before) and np.isfinite(after) else np.nan
            entry[f"delta_TRT_{q}"] = d
            entry[f"TRT250_{q}_before"] = before
            entry[f"TRT250_{q}_after"] = after
            vals.append(d)
        entry["delta_TRT_moyen"] = float(np.mean(vals)) if np.all(np.isfinite(vals)) else np.nan

        al_before = lookup_clinical(meas, code, eye, "before", AL_DESCRIPTION)
        al_after = lookup_clinical(meas, code, eye, "after", AL_DESCRIPTION)
        entry["AL_before_mm"] = al_before
        entry["AL_after_mm"] = al_after
        entry[AL_OUTCOME] = (float(al_after - al_before)
                             if np.isfinite(al_before) and np.isfinite(al_after)
                             else np.nan)
        rows.append(entry)
    return pd.DataFrame(rows)


def _folder_id(astro: str):
    """``'01_210713001'`` -> ``1``, ou ``None``. Meme decoupage que
    ``astro_code_from_folder``, mais on garde l'Id au lieu du Code."""
    try:
        return int(astro.split("_")[0])
    except (ValueError, IndexError):
        return None


def subject_views(predictors: pd.DataFrame) -> pd.DataFrame:
    """Agrege les predicteurs par (sujet, predicteur, moment) -- MOYENNE des
    repliques -- puis ajoute la vue ``delta`` = apres - avant.

    Moyenne et non mediane ici : a ce niveau les repliques sont 1 a 5 mesures du
    meme oeil dans la meme seance, deja debruitees par la mediane sur les
    one-cycles ; la moyenne est l'estimateur naturel de leur valeur commune, et
    c'est le geste de ``compute_rigidity_compare_mask_model``.
    """
    keys = ["famille", "predicteur", "cas", "sigma", "region", "subject"]
    agg = (predictors.dropna(subset=["valeur"])
           .groupby(keys + ["moment"], dropna=False, sort=False)["valeur"]
           .agg(["mean", "size"]).reset_index()
           .rename(columns={"mean": "valeur", "size": "n_repliques"}))
    # dropna=False : ``sigma`` vaut NaN pour les familles qui n'ont pas de
    # balayage (pouls, rigidite) ; le defaut de pandas retirerait ces lignes de
    # l'index et ferait disparaitre SILENCIEUSEMENT tout le panel historique.
    wide = agg.pivot_table(index=keys, columns="moment", values="valeur",
                           dropna=False).reset_index()
    for col in ("before", "after"):
        if col not in wide.columns:
            wide[col] = np.nan
    wide["delta"] = wide["after"] - wide["before"]
    return wide


def stability_before_after(views: pd.DataFrame) -> pd.DataFrame:
    """Correlation AVANT / APRES le vol de chaque predicteur, sur les sujets.

    Un sujet est un couple (astronaute, oeil) : on compare donc la valeur
    mesuree sur le MEME oeil du MEME astronaute, avant puis apres le vol. Des
    mois separent les deux mesures, avec un vol au milieu.

    Ce que cela mesure n'est PAS ce que mesure l'ICC. L'ICC porte sur des
    repliques d'une meme seance : c'est la reproductibilite instrumentale. Ici
    l'intervalle est de plusieurs mois : un parametre de TRAIT (une propriete
    stable de l'oeil, comme son epaisseur choroidienne) doit rester correle,
    tandis qu'un parametre qui repond reellement au vol doit se decorreler. Une
    correlation avant/apres nulle ne tranche donc pas entre « le parametre est du
    bruit » et « le parametre a change avec le vol » -- c'est l'ICC, mesure a
    court terme, qui separe les deux.

    Pearson ET Spearman sont rapportes : sur 26 sujets, un seul point extreme
    suffit a fabriquer un Pearson eleve, et l'ecart entre les deux le revele.
    """
    rows = []
    keys = ["famille", "predicteur", "cas", "sigma", "region"]
    for group_keys, g in views.groupby(keys, dropna=False, sort=False):
        sub = g[["before", "after"]].dropna()
        n = len(sub)
        if n < MIN_N_CORR or sub["before"].std() == 0 or sub["after"].std() == 0:
            continue
        r_p, p_p = pearsonr(sub["before"], sub["after"])
        r_s, p_s = spearmanr(sub["before"], sub["after"])
        rows.append({**dict(zip(keys, group_keys)), "n_sujets": n,
                     "pearson_r": float(r_p), "pearson_p": float(p_p),
                     "spearman_rho": float(r_s), "spearman_p": float(p_s),
                     "moyenne_avant": float(sub["before"].mean()),
                     "moyenne_apres": float(sub["after"].mean())})
    stab = pd.DataFrame(rows)
    if stab.empty:
        return stab
    for src, dst in (("pearson_p", "pearson_q"), ("spearman_p", "spearman_q")):
        stab[dst] = _bh_fdr(stab[src].to_numpy())
    for src, dst in (("pearson_p", "pearson_q_famille"),
                     ("spearman_p", "spearman_q_famille")):
        stab[dst] = stab.groupby("famille")[src].transform(
            lambda s: _bh_fdr(s.to_numpy()))
    return stab.sort_values("pearson_q")


def sans_correlations(views: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Pearson et Spearman de chaque (predicteur, vue) contre chaque issue."""
    merged = views.merge(outcomes, on="subject", how="left")
    keys = ["famille", "predicteur", "cas", "sigma", "region"]
    rows = []
    for group_keys, g in merged.groupby(keys, dropna=False, sort=False):
        info = dict(zip(keys, group_keys))
        for view in VIEWS:
            if view not in g.columns:
                continue
            x_all = g[view]
            for outcome in OUTCOMES:
                sub = g[[view, outcome]].dropna()
                n = len(sub)
                if n < MIN_N_CORR or sub[view].std() == 0 or sub[outcome].std() == 0:
                    continue
                r_p, p_p = pearsonr(sub[view], sub[outcome])
                r_s, p_s = spearmanr(sub[view], sub[outcome])
                rows.append({**info, "vue": view, "issue": outcome, "n_sujets": n,
                             "pearson_r": float(r_p), "pearson_p": float(p_p),
                             "spearman_rho": float(r_s), "spearman_p": float(p_s),
                             "moyenne_predicteur": float(sub[view].mean()),
                             "ecart_type_predicteur": float(sub[view].std())})
    corr = pd.DataFrame(rows)
    if corr.empty:
        return corr
    # DEUX corrections, et il faut les deux.
    #
    # ``*_q`` -- Benjamini-Hochberg sur TOUTE la table. C'est la lecture
    # conservatrice, et la seule honnete si l'on considere que l'etude a
    # reellement essaye tous ces predicteurs sans hypothese prealable.
    #
    # ``*_q_famille`` -- BH a l'interieur de chaque famille seulement. Le
    # decompte global est domine par les 640 variantes de pente (16 pentes x 2
    # cas x 4 sigmas x 5 regions), qui sont fortement redondantes entre elles :
    # les y noyer penaliserait injustement les quatre predicteurs de la famille
    # ``rigidite``, qui eux constituent une hypothese ancienne et etroite. Une
    # correlation qui ne passe QUE le seuil par famille est une piste, pas un
    # resultat -- et la difference entre les deux colonnes dit exactement combien
    # l'ampleur du balayage a coute.
    for src, dst in (("pearson_p", "pearson_q"), ("spearman_p", "spearman_q")):
        corr[dst] = _bh_fdr(corr[src].to_numpy())
    for src, dst in (("pearson_p", "pearson_q_famille"),
                     ("spearman_p", "spearman_q_famille")):
        corr[dst] = corr.groupby("famille")[src].transform(
            lambda s: _bh_fdr(s.to_numpy()))
    # ``*_q_categorie`` -- la TROISIEME correction, a l'echelle de la question
    # que pose chaque page : parmi les tentatives d'UNE categorie (rigidite,
    # strain, viscosite) contre UNE issue, laquelle survit ? La famille est donc
    # (categorie, vue, issue). Elle ne corrige PAS le fait d'avoir essaye les
    # cinq issues ni les trois categories : c'est le prix a payer pour qu'un
    # graphique par issue et par page soit lisible, et c'est dit sur les pages.
    # C'est la lecture la plus INDULGENTE des trois.
    # NaN pour les predicteurs sans categorie : la colonne n'y veut rien dire.
    corr["categorie"] = [categorie_de(f, p)
                         for f, p in zip(corr["famille"], corr["predicteur"])]
    avec_cat = corr["categorie"] != ""
    for src, dst in (("pearson_p", "pearson_q_categorie"),
                     ("spearman_p", "spearman_q_categorie")):
        corr[dst] = np.nan
        if avec_cat.any():
            corr.loc[avec_cat, dst] = (corr.loc[avec_cat]
                                       .groupby(["categorie", "vue", "issue"])[src]
                                       .transform(lambda s: _bh_fdr(s.to_numpy())))
    return corr.sort_values("pearson_q")


def _bh_fdr(p: np.ndarray) -> np.ndarray:
    """q-valeurs de Benjamini-Hochberg (implementees ici pour ne pas dependre de
    statsmodels, absent de l'environnement)."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.size, np.nan)
    # Les p non finis sont ecartes du DECOMPTE, pas seulement du tri : les
    # compter gonflerait n et durcirait la correction pour des tests qui n'ont
    # pas eu lieu. (``np.argsort`` range les NaN en queue et les propagerait.)
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
# Boucle principale
# --------------------------------------------------------------------------- #
def _top(corr: pd.DataFrame, n: int = 15) -> list:
    if corr.empty:
        return []
    cols = ["famille", "predicteur", "cas", "sigma", "region", "vue", "issue",
            "n_sujets", "pearson_r", "pearson_p", "pearson_q", "pearson_q_famille"]
    return corr.nsmallest(n, "pearson_q")[cols].to_dict("records")


def main() -> None:
    t_start = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"sortie   : {OUT_DIR}")
    print(f"variante : {MASK_VARIANT}")

    meas, id2code = load_clinical_db(DB_PATH)
    print(f"base clinique : {len(meas)} mesures, {len(id2code)} astronautes")

    print("\nportail de validite des one-cycles")
    gate = build_gate()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gate.to_csv(CSV_GATE, index=False)
    n_oc = len(gate)
    n_oc_ok = int(gate["garde_one_cycle"].sum())
    par_cond = gate.groupby("slug").first()
    n_cond_ok = int(par_cond["garde_condition"].sum())
    print(f"  criteres : pulsatilite < {GATE_PULSATILITE_MAX_PCT:g} %, "
          f"p(phase) < {GATE_PHASE_P_MAX:g}, >= {GATE_FRAMES_MIN} frames/bin, "
          f"jacobien non replie (sigma >= {GATE_JAC_SIGMA_MIN:g})")
    for _c, _lab in (("ok_pulsatilite", "pulsatilite"), ("ok_phase", "phase"),
                     ("ok_frames", "frames/bin"), ("ok_jacobien", "jacobien")):
        print(f"    {_lab:12s} garde {100 * gate[_c].mean():5.1f} %")
    print(f"  one-cycles retenus : {n_oc_ok}/{n_oc} ({100 * n_oc_ok / n_oc:.1f} %)")
    print(f"  conditions retenues (>= {GATE_MIN_ONE_CYCLES} one-cycles) : "
          f"{n_cond_ok}/{len(par_cond)} ({100 * n_cond_ok / len(par_cond):.1f} %)")
    print(f"  -> {CSV_GATE}")

    print(f"\nconstitution du panel de predicteurs")
    rows = []
    rows += demons_predictors(gate)
    print(f"  demons     : {len(rows)} lignes")
    n0 = len(rows)
    rows += pulse_predictors()
    print(f"  pouls      : {len(rows) - n0} lignes")
    n0 = len(rows)
    rows += rigidity_predictors(meas, id2code)
    print(f"  rigidite   : {len(rows) - n0} lignes")

    if not rows:
        raise SystemExit("aucun predicteur : lancer d'abord compute_demons_strain.py")

    predictors = pd.DataFrame(rows)
    for col in LONG_COLS:
        if col not in predictors.columns:
            predictors[col] = np.nan
    predictors = predictors[LONG_COLS]
    predictors["cas"] = predictors["cas"].fillna("")
    predictors["region"] = predictors["region"].fillna("toutes")
    predictors.to_csv(CSV_PREDICTORS, index=False)
    n_pred = predictors.groupby(["famille", "predicteur", "cas", "sigma", "region"],
                                dropna=False).ngroups
    print(f"\n{len(predictors)} lignes, {n_pred} predicteurs distincts, "
          f"{predictors['slug'].nunique()} conditions, "
          f"{predictors['subject'].nunique()} sujets (astronaute x oeil)")

    print("\nrepetabilite (ICC + composantes de variance)")
    repeat = repeatability(predictors)
    repeat.to_csv(CSV_REPEAT, index=False)
    n_icc = int((repeat["icc_type"] == "ICC(A,1)").sum()) if len(repeat) else 0
    print(f"  {len(repeat)} lignes ({n_icc} ICC(A,1))")
    if n_icc:
        best = (repeat[(repeat["icc_type"] == "ICC(A,1)") & (repeat["pair"] == "1v2")]
                .nlargest(8, "icc"))
        print("  meilleurs ICC(A,1), paire 1v2 :")
        for _, r in best.iterrows():
            print(f"    {r['predicteur']:<28s} {str(r['region']):<20s} "
                  f"cas={r['cas'] or '-':<7s} sigma={r['sigma']!s:<6s} "
                  f"ICC {r['icc']:+.3f}  (n={int(r['n_subjects'])})")

    print("\nrepetabilite, ONE-CYCLES comme repliques (cible = la video)")
    pred_oc = demons_predictors_one_cycle(gate)
    pred_oc.to_csv(CSV_PREDICTORS_OC, index=False)
    print(f"  {len(pred_oc)} lignes, "
          f"{pred_oc.groupby(GROUP_KEYS_OC, dropna=False).ngroups} predicteurs, "
          f"{pred_oc['slug'].nunique()} videos, "
          f"{pred_oc.groupby(['slug', 'one_cycle']).ngroups} one-cycles")
    repeat_oc = repeatability_one_cycles(pred_oc)
    repeat_oc.to_csv(CSV_REPEAT_OC, index=False)
    n_icc_oc = 0
    if len(repeat_oc):
        _a1 = repeat_oc[(repeat_oc["icc_type"] == "ICC(A,1)")
                        & (repeat_oc["pair"] == "1v2")]
        _vc = repeat_oc[repeat_oc["pair"] == "toutes"]
        n_icc_oc = int(len(_a1))
        print(f"  {len(repeat_oc)} lignes ({n_icc_oc} ICC(A,1) paire 1v2, "
              f"{int(_a1['n_subjects'].median())} videos par ICC)")
        print(f"  ICC(A,1) median {_a1['icc'].median():+.3f}, "
              f"CV intra median {_vc['cv_intra_pct'].median():.0f} %, "
              f"variance inter negative sur {100 * (_vc['var_inter'] < 0).mean():.1f} %")
        for fam, gf in _a1.groupby("famille"):
            print(f"    {fam:<10s} {len(gf):>5d} predicteurs, "
                  f"ICC median {gf['icc'].median():+.3f}")

    print("\nprediction de SANS (delta_TRT)")
    id2sex = load_sex(DB_PATH)
    outcomes = build_outcomes(meas, id2code, id2sex,
                              predictors["subject"].unique().tolist())
    n_out = {o: int(outcomes[o].notna().sum()) for o in OUTCOMES}
    print(f"  {len(outcomes)} sujets, issues renseignees : "
          + ", ".join(f"{k}={v}" for k, v in n_out.items()))
    # Table minuscule (une ligne par sujet) mais indispensable au site : c'est la
    # SEULE source ou les issues se lisent seules, sans les predicteurs.
    # ``subject_values.csv`` les porte aussi, mais pese 245 Mo -- le script de
    # figures ne peut pas le relire pour tracer une issue contre une autre.
    outcomes.to_csv(CSV_OUTCOMES, index=False)
    views = subject_views(predictors)
    # Le detail par sujet est publie : avec une douzaine de points, un seul sujet
    # extreme suffit a fabriquer une correlation, et il faut pouvoir le voir.
    views.merge(outcomes, on="subject", how="left").to_csv(CSV_SUBJECTS, index=False)
    stab = stability_before_after(views)
    stab.to_csv(CSV_STABILITY, index=False)
    if len(stab):
        n_sig = int((stab["pearson_q"] < 0.05).sum())
        print(f"  stabilite avant/apres : {len(stab)} predicteurs, "
              f"{int(stab['n_sujets'].median())} sujets, r median "
              f"{stab['pearson_r'].median():+.3f}, rho median "
              f"{stab['spearman_rho'].median():+.3f}, "
              f"{100 * (stab['pearson_p'] < 0.05).mean():.1f} % a p < 0,05, "
              f"{n_sig} a q < 0,05")
    corr = sans_correlations(views, outcomes)
    corr.to_csv(CSV_CORR, index=False)
    print(f"  {len(corr)} correlations")
    if len(corr):
        n_sig = int((corr["pearson_q"] < 0.05).sum())
        n_sig_f = int((corr["pearson_q_famille"] < 0.05).sum())
        n_sig_s = int((corr["pearson_q_categorie"] < 0.05).sum())
        _panel = corr[corr["categorie"] != ""]
        n_panel = int(len(_panel))
        n_par_cat = {c: int(g.groupby(["vue", "issue"]).size().median())
                     for c, g in _panel.groupby("categorie")}
        q_min_issue = {c: {i: round(float(g2["pearson_q_categorie"].min()), 4)
                           for i, g2 in g[g["vue"] == "before"].groupby("issue")}
                       for c, g in _panel.groupby("categorie")}
        print(f"  {n_sig} avec q < 0,05 (BH sur les {len(corr)} tests)")
        print(f"  {n_sig_f} avec q < 0,05 par famille")
        print(f"  {n_sig_s} avec q < 0,05 dans leur categorie "
              f"({n_panel} tests categorises)")
        for _c, _n in sorted(n_par_cat.items()):
            _g = _panel[_panel["categorie"] == _c]
            print(f"    {_c:<10s} {len(_g):>5d} tests, {_n} par vue x issue, "
                  f"q min {_g['pearson_q_categorie'].min():.3f}, "
                  f"{int((_g['pearson_q_categorie'] < 0.05).sum())} a q < 0,05")
        for fam, g in corr.groupby("famille"):
            print(f"    {fam:<10s} {len(g):>5d} tests, "
                  f"{int((g['pearson_q_famille'] < 0.05).sum()):>3d} a q_famille < 0,05, "
                  f"|r| max {g['pearson_r'].abs().max():.2f}")
        print("  les plus fortes :")
        for r in _top(corr, 10):
            print(f"    {r['predicteur']:<28s} {str(r['region']):<20s} "
                  f"vue={r['vue']:<7s} {r['issue']:<18s} "
                  f"r={r['pearson_r']:+.3f} p={r['pearson_p']:.2g} "
                  f"q={r['pearson_q']:.2g} (n={int(r['n_sujets'])})")

    summary = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "mask_variant": MASK_VARIANT,
        "portail": {
            "pulsatilite_max_pct": GATE_PULSATILITE_MAX_PCT,
            "phase_p_max": GATE_PHASE_P_MAX,
            "frames_min": GATE_FRAMES_MIN,
            "jac_sigma_min": GATE_JAC_SIGMA_MIN,
            "min_one_cycles": GATE_MIN_ONE_CYCLES,
            "n_one_cycles": int(len(gate)),
            "n_one_cycles_retenus": int(gate["garde_one_cycle"].sum()),
            "n_conditions": int(gate["slug"].nunique()),
            "n_conditions_retenues": int(gate.groupby("slug").first()["garde_condition"].sum()),
        },
        "n_lignes_predicteurs": int(len(predictors)),
        "n_predicteurs": int(n_pred),
        "n_conditions": int(predictors["slug"].nunique()),
        "n_sujets": int(predictors["subject"].nunique()),
        "n_predicteurs_one_cycle": int(pred_oc.groupby(GROUP_KEYS_OC, dropna=False).ngroups) if len(pred_oc) else 0,
        "n_one_cycles_repetabilite": int(pred_oc.groupby(["slug", "one_cycle"]).ngroups) if len(pred_oc) else 0,
        "n_icc_one_cycles": n_icc_oc,
        "icc_one_cycles_median": float(_a1["icc"].median()) if n_icc_oc else None,
        "n_correlations": int(len(corr)),
        "n_correlations_q05": int((corr["pearson_q"] < 0.05).sum()) if len(corr) else 0,
        "n_correlations_q05_famille": int((corr["pearson_q_famille"] < 0.05).sum()) if len(corr) else 0,
        "n_correlations_q05_categorie": n_sig_s if len(corr) else 0,
        "n_tests_categorises": n_panel if len(corr) else 0,
        "n_tests_par_categorie_vue_issue": n_par_cat if len(corr) else {},
        "q_categorie_min_par_issue": q_min_issue if len(corr) else {},
        "issues_renseignees": n_out,
        "hr_half_band_bpm": HR_HALF_BAND_BPM,
        "min_n_corr": MIN_N_CORR,
        "top_correlations": _top(corr, 20),
    }
    JSON_SUMMARY.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"\ntermine {datetime.now():%Y-%m-%d %H:%M} "
          f"({(time.perf_counter() - t_start) / 60:.1f} min)")
    for path in (CSV_PREDICTORS, CSV_REPEAT, CSV_PREDICTORS_OC, CSV_REPEAT_OC,
                 CSV_CORR, CSV_SUBJECTS, CSV_OUTCOMES, CSV_POOLED, CSV_STABILITY,
                 CSV_GATE, JSON_SUMMARY):
        print(f"  {path}")


if __name__ == "__main__":
    main()
