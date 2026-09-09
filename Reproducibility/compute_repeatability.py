# -*- coding: utf-8 -*-
"""
compute_repeatability.py

De combien deux mesures du MEME oeil, prises a quelques minutes d'intervalle sur
le meme appareil, different-elles ? C'est la seule question de ce script, et
c'est la borne sans laquelle aucune difference observee dans l'experience SANS
n'est interpretable.

Il ne recalcule aucun strain : il LIT les tables de ``demons_strain`` produites
par le meme lot que la cohorte SANS (``Reproducibility/run_batch.py strain``) et
n'en fait que trois choses -- reduire, accorder, decomposer.

1. REDUIRE : d'un one-cycle a UN marqueur
-----------------------------------------
``regions.csv`` porte une ligne par (slug, one-cycle, cas, sigma, bin, region).
Le marqueur retenu est le strain mesure entre **le bin de choroide la plus MINCE
et le bin de choroide la plus EPAISSE** du one-cycle :

  - le bin MINCE est deja l'image fixe du recalage (``ref_bin``, choisi par
    ``compute_demons_strain`` comme le bin de CT minimum) ;
  - le bin EPAIS est celui de CT maximum, selectionne ici.

C'est la deformation sur l'amplitude COMPLETE de la pulsation choroidienne,
donc le plus grand rapport signal/bruit que le cycle puisse offrir. Les autres
bins restent dans la table et ne sont pas utilises.

Le repliement se fait ensuite en deux temps, comme demande :

  - par REGION : les trois agregats sont deja calcules par le lot (moyenne,
    mediane, p95 signe de la magnitude) sur les pixels de la bande ;
  - entre CYCLES : la MEDIANE des one-cycles d'une acquisition. La mediane et
    non la moyenne, parce qu'un one-cycle rate (clignement, perte de suivi)
    produit un strain aberrant et non un strain bruite -- la moyenne le
    suivrait.

Il reste alors UN nombre par (acquisition, cas, sigma, region, agregat).

2. ACCORDER : ICC entre replicats
---------------------------------
Le "sujet" est l'OEIL (participant x lateralite), le "replicat" son rang
chronologique. Trois ICC, calcules par ``pingouin``, qui ne repondent pas a la
meme question :

  - ``ICC(1,1)`` modele a un facteur aleatoire -- les replicats sont
    INTERCHANGEABLES. C'est le modele juste ici : rien ne distingue le 1er du
    3e enregistrement, ils ne different que par l'ordre.
  - ``ICC(A,1)`` accord ABSOLU, replicats traites comme un facteur ;
  - ``ICC(C,1)`` COHERENCE, qui ignore un decalage systematique entre rangs.

Leur ecart est informatif : ``ICC(C,1)`` nettement superieur a ``ICC(A,1)``
signalerait une DERIVE au fil de la seance (le 3e enregistrement
systematiquement plus haut que le 1er, par exemple parce que l'oeil s'assseche
ou que le sujet se fatigue). Les trois sont donc rapportes ensemble.

3. DECOMPOSER : ou passe la variance
------------------------------------
L'ICC est un rapport ; il ne dit pas dans quelle unite on se trompe. La
decomposition le dit, et separe DEUX niveaux que l'ICC melange :

    participant  ->  oeil(participant)  ->  replicat(oeil)

C'est une ANOVA hierarchique a effets aleatoires. Le plan est EXACTEMENT
equilibre -- chaque participant a ses deux yeux, chaque oeil ses trois premiers
replicats -- ce qui rend les composantes de variance exactes plutot
qu'approchees, et permet de les calculer directement par sommes de carres sans
recourir a un ajustement iteratif (REML) qui, sur un plan equilibre, rendrait
les memes valeurs.

    sigma2_residuel  = MS_residuel                              (intra-oeil)
    sigma2_oeil      = (MS_oeil - MS_residuel) / n_rep          (entre yeux d'un meme sujet)
    sigma2_sujet     = (MS_sujet - MS_oeil) / (n_oeil * n_rep)  (entre sujets)

Une composante negative n'a pas de sens physique : elle signale que le facteur
n'explique rien de plus que le bruit, et elle est ramenee a zero (convention
usuelle), le fait etant signale par ``composante_negative``.

Sont rapportes en plus, dans l'unite du marqueur :

  - ``sd_intra``  ecart-type intra-oeil = la borne cherchee ;
  - ``rc``        coefficient de repetabilite = 2,77 x sd_intra : deux mesures
    du meme oeil different de moins que cela dans 95 % des cas ;
  - ``cv_intra``  le meme, rapporte a la moyenne -- sans valeur quand le
    marqueur change de signe (le strain le fait), d'ou ``cv_utilisable``.

Arborescence lue
----------------
    E:/NASA_Rigidity/Reproducibility/SegmentationVariations/<variante>/
        demons_strain/regions.csv      strain par bande laterale
        demons_strain/bins.csv         strain sur toute la retine + CT du bin
        demons_strain/one_cycles.csv   CT et accord CT/pouls par one-cycle

Sorties (sous ``.../repeatability/``)
-------------------------------------
    markers.csv     1 ligne / (slug, cas, sigma, region, agregat) -- LE marqueur
    icc.csv         1 ligne / (cas, sigma, region, agregat, type d'ICC)
    variance.csv    1 ligne / (cas, sigma, region, agregat) -- les composantes
    pairs.csv       1 ligne / paire de replicats du meme oeil -- pour les nuages
    summary.txt     les chiffres cites dans la page Quarto

Lancer (kernel pyOR, depuis la racine du depot) :
    C:/Users/transformer/anaconda3/envs/pyOR/python.exe \
        Reproducibility/compute_repeatability.py
"""

from __future__ import annotations

import itertools
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Chemins et parametres
# --------------------------------------------------------------------------- #
SEGVAR_ROOT = Path("E:/NASA_Rigidity/Reproducibility/SegmentationVariations")
VARIANT = "model1_scale_1.0_flatten_choroid_xcorr"
VARIANT_ROOT = SEGVAR_ROOT / VARIANT
STRAIN_DIR = VARIANT_ROOT / "demons_strain"
OUT_DIR = VARIANT_ROOT / "repeatability"

# Rangs de replicat retenus. Trois : c'est le plan nominal, et c'est le nombre
# que TOUS les yeux atteignent -- au-dela, seuls quatre yeux suivraient et le
# plan cesserait d'etre equilibre (l'ANOVA hierarchique ci-dessous en depend).
N_REP = 3

# Les trois agregats du strain, tels que le lot les nomme -- et il ne les nomme
# PAS pareil dans les deux tables : la moyenne est ``strain_retine`` dans
# ``regions.csv`` mais ``strain_retine_moy`` dans ``bins.csv``. Les deux
# dictionnaires existent pour cette seule raison ; les confondre leve un
# ``KeyError`` au premier agregat.
AGREGATS_REGIONS = {
    "moyenne": "strain_retine",
    "mediane": "strain_retine_med",
    "p95_signe": "strain_retine_p95",
}
AGREGATS_BINS = {
    "moyenne": "strain_retine_moy",
    "mediane": "strain_retine_med",
    "p95_signe": "strain_retine_p95",
}

# Les bandes laterales sont nommees par le lot (« colonne -2 » ... « colonne +2 »,
# ``k`` etant leur indice). Cette etiquette-la designe la retine ENTIERE, lue
# dans ``bins.csv`` : c'est le marqueur de reference auquel les cinq bandes se
# comparent, et ``k = -1`` le place avant elles au tri.
REGION_TOUTE = "retine entiere"
K_TOUTE = -1

RE_SLUG = re.compile(r"^(?P<participant>.+)_(?P<eye>OD|OS)(?P<replicate>\d+)$")


# --------------------------------------------------------------------------- #
# Lecture
# --------------------------------------------------------------------------- #
def decompose_slug(df: pd.DataFrame) -> pd.DataFrame:
    """``BELANGER_CHARLES_OD1`` -> participant / oeil / rang / identifiant d'oeil."""
    parts = df["slug"].str.extract(RE_SLUG)
    df = df.copy()
    df["participant"] = parts["participant"]
    df["eye"] = parts["eye"]
    df["replicate"] = pd.to_numeric(parts["replicate"], errors="coerce")
    df["eye_id"] = df["participant"] + "_" + df["eye"]
    return df


def charger() -> tuple[pd.DataFrame, pd.DataFrame]:
    for f in ("regions.csv", "bins.csv"):
        if not (STRAIN_DIR / f).exists():
            raise SystemExit(
                f"{STRAIN_DIR / f} absent -- lancer d'abord :\n"
                f"    python Reproducibility/run_batch.py strain")
    regions = decompose_slug(pd.read_csv(STRAIN_DIR / "regions.csv"))
    bins = decompose_slug(pd.read_csv(STRAIN_DIR / "bins.csv"))
    return regions, bins


# --------------------------------------------------------------------------- #
# 0. Quelles acquisitions sont exploitables
# --------------------------------------------------------------------------- #
DATA_ROOT = Path("E:/SANSORI/Reproducibility")
# Une image OCT 768 x 496 exportee par le Spectralis pese 400 a 530 ko. Un
# export tronque en pese ~24 et se decode en image NOIRE, sans erreur : ni le
# XML (qui la liste normalement) ni le decodeur ne le signalent. La taille du
# fichier est donc le seul controle qui les attrape, et il ne coute qu'un stat.
SEUIL_IMAGE_VIDE_OCTETS = 60_000


def qualite_images() -> pd.DataFrame:
    """Une ligne par acquisition : combien de ses .tif sont vides.

    Ce controle ne regarde AUCUN pixel -- il lit la taille des fichiers. Il est
    ici, et non dans le lot d'acquisition, parce que c'est une question de
    repetabilite : une acquisition majoritairement noire n'est pas un replicat,
    et il faut le dire avant de compter les yeux.
    """
    lignes = []
    if not DATA_ROOT.is_dir():
        return pd.DataFrame()
    for d in sorted(DATA_ROOT.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        tifs = list((d / "RawImages").glob("*.tif"))
        if not tifs:
            continue
        tailles = np.array([t.stat().st_size for t in tifs])
        vides = int((tailles < SEUIL_IMAGE_VIDE_OCTETS).sum())
        lignes.append({
            "slug": d.name, "n_images": len(tifs), "n_images_vides": vides,
            "pct_vides": 100.0 * vides / len(tifs),
            "taille_mediane_ko": float(np.median(tailles) / 1024),
        })
    return decompose_slug(pd.DataFrame(lignes)) if lignes else pd.DataFrame()


# --------------------------------------------------------------------------- #
# 1. Le marqueur : bin le plus MINCE -> bin le plus EPAIS, puis mediane des cycles
# --------------------------------------------------------------------------- #
def bin_le_plus_epais(df: pd.DataFrame) -> pd.DataFrame:
    """Ne garde, par (slug, one-cycle, cas, sigma), que les lignes du bin de CT MAXIMUM.

    Le bin de CT minimum est deja l'image fixe du recalage : les lignes retenues
    portent donc la deformation entre les deux extremes du cycle.

    Le choix se fait sur le NUMERO DE BIN, puis toutes les lignes de ce bin sont
    gardees -- et non par ``idxmax`` sur ``CT_um``. La difference n'est pas
    cosmetique : dans ``regions.csv``, ``CT_um`` est l'epaisseur du BIN, repetee
    a l'identique sur ses cinq bandes laterales. Un ``idxmax`` y trouve donc
    cinq ex aequo et n'en rend qu'UN, c'est-a-dire une bande arbitraire -- trois
    des cinq colonnes disparaissaient de l'analyse, et les yeux auxquels il
    manquait une combinaison sortaient du plan equilibre.
    """
    cle = ["slug", "one_cycle", "cas", "sigma"]
    # Le bin de CT max, un par groupe : on passe par les couples (groupe, bin)
    # distincts pour que la selection porte sur le bin et non sur une ligne.
    par_bin = df[cle + ["bin", "CT_um"]].drop_duplicates(cle + ["bin"])
    choisi = par_bin.loc[par_bin.groupby(cle, dropna=False)["CT_um"].idxmax(),
                         cle + ["bin"]]
    return df.merge(choisi, on=cle + ["bin"], how="inner")


def marqueurs(regions: pd.DataFrame, bins: pd.DataFrame) -> pd.DataFrame:
    """Un nombre par (slug, cas, sigma, region, agregat)."""
    lignes = []

    # --- les cinq bandes laterales ---------------------------------------- #
    r = bin_le_plus_epais(regions)
    for nom, col in AGREGATS_REGIONS.items():
        g = (r.groupby(["slug", "participant", "eye", "eye_id", "replicate",
                        "cas", "sigma", "region", "k"], dropna=False)
               .agg(valeur=(col, "median"),
                    n_one_cycles=(col, "size"),
                    CT_um=("CT_um", "median"),
                    dCT_um=("dCT_um", "median"))
               .reset_index())
        g["agregat"] = nom
        lignes.append(g)

    # --- la retine entiere (region = -1) ----------------------------------- #
    b = bin_le_plus_epais(bins)
    for nom, col in AGREGATS_BINS.items():
        g = (b.groupby(["slug", "participant", "eye", "eye_id", "replicate",
                        "cas", "sigma"], dropna=False)
               .agg(valeur=(col, "median"),
                    n_one_cycles=(col, "size"),
                    CT_um=("CT_um", "median"),
                    dCT_um=("dCT_um", "median"))
               .reset_index())
        g["region"] = REGION_TOUTE
        g["k"] = K_TOUTE
        g["agregat"] = nom
        lignes.append(g)

    out = pd.concat(lignes, ignore_index=True)
    return out.sort_values(["cas", "sigma", "k", "agregat", "slug"])


# --------------------------------------------------------------------------- #
# 2. ICC
# --------------------------------------------------------------------------- #
def plan_equilibre(g: pd.DataFrame) -> pd.DataFrame | None:
    """Les yeux ayant les ``N_REP`` premiers rangs, et ces rangs seulement.

    Un plan desequilibre rendrait les composantes de variance approchees et
    ferait echouer ``pingouin`` ; mieux vaut dire quels yeux sont retenus que
    produire un chiffre dont on ne saurait pas de quoi il est la moyenne.
    """
    g = g[g["replicate"] <= N_REP]
    complets = (g.groupby("eye_id")["replicate"].nunique() == N_REP)
    complets = complets[complets].index
    g = g[g["eye_id"].isin(complets)]
    return g if not g.empty else None


def icc_table(marq: pd.DataFrame) -> pd.DataFrame:
    import pingouin as pg

    # Les libelles sont ceux que ``pingouin`` ecrit lui-meme dans sa colonne
    # ``Type`` -- « ICC(1,1) », pas « ICC1 » : les versions recentes rendent
    # deja la forme lisible, et traduire depuis l'ancienne ferait silencieusement
    # rater TOUTES les lignes (le tableau sortait vide, sans erreur).
    # Meme liste que ``Astronauts/compute_rigidity_compare_mask_model.py``.
    types = ["ICC(1,1)", "ICC(A,1)", "ICC(C,1)"]
    out = []
    cles = ["cas", "sigma", "region", "agregat"]
    for cle, g in marq.groupby(cles, dropna=False):
        base = dict(zip(cles, cle))
        g = plan_equilibre(g)
        if g is None or g["eye_id"].nunique() < 3:
            continue
        base["n_yeux"] = int(g["eye_id"].nunique())
        base["n_obs"] = int(len(g))
        # Le nombre de one-cycles sur lequel repose le marqueur le plus mince
        # de cette configuration. Il varie d'un facteur cinq dans la cohorte
        # (2 a 11), et le repliement inter-cycles est une MEDIANE : sur deux
        # valeurs elle ne resiste a rien. Un ICC lu sans ce chiffre traiterait
        # a egalite des marqueurs qui n'ont pas la meme robustesse.
        base["n_one_cycles_min"] = int(g["n_one_cycles"].min())
        base["n_one_cycles_med"] = float(g["n_one_cycles"].median())
        try:
            tab = pg.intraclass_corr(data=g, targets="eye_id", raters="replicate",
                                     ratings="valeur", nan_policy="omit")
        except Exception as exc:  # un marqueur constant, par exemple
            out.append({**base, "icc_type": "—", "icc": np.nan,
                        "erreur": f"{type(exc).__name__}: {exc}"})
            continue
        tab = tab.set_index("Type")
        manquants = [t for t in types if t not in tab.index]
        if manquants:
            raise SystemExit(
                f"pingouin n'a pas rendu {manquants} mais {list(tab.index)} -- "
                "la liste TYPES doit suivre ses libelles.")
        # Le nom de la colonne d'intervalle a change entre versions de
        # pingouin (« CI95% » puis « CI95 ») : on prend celle qui est la.
        col_ci = next((c for c in ("CI95", "CI95%") if c in tab.columns), None)
        for t in types:
            r = tab.loc[t]
            ci = r[col_ci] if col_ci else (np.nan, np.nan)
            out.append({
                **base, "icc_type": t, "icc": float(r["ICC"]),
                "icc_ci95_low": float(ci[0]), "icc_ci95_high": float(ci[1]),
                "f": float(r["F"]), "pval": float(r["pval"]), "erreur": "",
            })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# 3. Decomposition de la variance (ANOVA hierarchique, plan equilibre)
# --------------------------------------------------------------------------- #
def composantes(g: pd.DataFrame) -> dict:
    """sujet / oeil(sujet) / replicat(oeil), sur un plan strictement equilibre."""
    n_sujets = g["participant"].nunique()
    n_oeil = int(g.groupby("participant")["eye_id"].nunique().iloc[0])
    n_rep = N_REP
    x = g["valeur"].to_numpy(dtype=float)
    if not np.isfinite(x).all() or len(x) != n_sujets * n_oeil * n_rep:
        return {}

    moy = float(x.mean())
    moy_sujet = g.groupby("participant")["valeur"].mean()
    moy_oeil = g.groupby(["participant", "eye_id"])["valeur"].mean()

    # Les moyennes par oeil, remises a plat avec la moyenne de LEUR sujet en
    # regard. La mise a plat n'est pas cosmetique : additionner des Series
    # indexees par des yeux DIFFERENTS les aligne sur l'union de leurs index et
    # rend des NaN, que ``.sum()`` ecarte ensuite -- la somme des carres valait
    # alors zero sur tous les jeux de donnees, et la variance inter-yeux
    # disparaissait au profit de la variance inter-sujets.
    par_oeil = moy_oeil.rename("m").reset_index()
    par_oeil["m_sujet"] = par_oeil["participant"].map(moy_sujet)

    ss_sujet = n_oeil * n_rep * float(((moy_sujet - moy) ** 2).sum())
    ss_oeil = n_rep * float(((par_oeil["m"] - par_oeil["m_sujet"]) ** 2).sum())
    ss_res = float(sum(
        ((sub["valeur"] - moy_oeil.loc[(p, e)]) ** 2).sum()
        for (p, e), sub in g.groupby(["participant", "eye_id"])))
    # Garde-fou : la decomposition doit etre exacte sur un plan equilibre.
    ss_total = float(((x - moy) ** 2).sum())
    if not np.isclose(ss_sujet + ss_oeil + ss_res, ss_total, rtol=1e-6, atol=1e-9):
        raise AssertionError(
            f"decomposition incoherente : {ss_sujet:.6g} + {ss_oeil:.6g} + "
            f"{ss_res:.6g} != {ss_total:.6g}")

    ddl_sujet = n_sujets - 1
    ddl_oeil = n_sujets * (n_oeil - 1)
    ddl_res = n_sujets * n_oeil * (n_rep - 1)
    ms_sujet, ms_oeil, ms_res = (ss_sujet / ddl_sujet, ss_oeil / ddl_oeil,
                                 ss_res / ddl_res)

    v_res = ms_res
    v_oeil = (ms_oeil - ms_res) / n_rep
    v_sujet = (ms_sujet - ms_oeil) / (n_oeil * n_rep)
    negatives = [n for n, v in (("oeil", v_oeil), ("sujet", v_sujet)) if v < 0]
    v_oeil, v_sujet = max(v_oeil, 0.0), max(v_sujet, 0.0)
    total = v_sujet + v_oeil + v_res

    sd_intra = float(np.sqrt(v_res))
    moyenne_abs = float(np.abs(x).mean())
    # Le strain change de signe d'une bande a l'autre : un CV rapporte a une
    # moyenne proche de zero explose sans rien vouloir dire. On ne le publie que
    # lorsque la moyenne domine franchement sa propre dispersion.
    cv_utilisable = bool(moyenne_abs > 3.0 * sd_intra) and moyenne_abs > 0
    return {
        "n_sujets": n_sujets, "n_yeux": n_sujets * n_oeil, "n_obs": len(x),
        "moyenne": moy, "moyenne_abs": moyenne_abs,
        "var_sujet": v_sujet, "var_oeil": v_oeil, "var_residuelle": v_res,
        "var_totale": total,
        "pct_sujet": 100 * v_sujet / total if total > 0 else np.nan,
        "pct_oeil": 100 * v_oeil / total if total > 0 else np.nan,
        "pct_residuelle": 100 * v_res / total if total > 0 else np.nan,
        "sd_intra": sd_intra,
        "sd_inter_sujet": float(np.sqrt(v_sujet)),
        "rc": 2.77 * sd_intra,
        "cv_intra_pct": (100 * sd_intra / moyenne_abs) if cv_utilisable else np.nan,
        "cv_utilisable": cv_utilisable,
        "composante_negative": ",".join(negatives),
        # ICC implicite de la decomposition : ce qui n'est PAS du bruit intra.
        "icc_variance": (v_sujet + v_oeil) / total if total > 0 else np.nan,
    }


def variance_table(marq: pd.DataFrame) -> pd.DataFrame:
    out = []
    cles = ["cas", "sigma", "region", "agregat"]
    for cle, g in marq.groupby(cles, dropna=False):
        g = plan_equilibre(g)
        if g is None:
            continue
        # Ne garder que les participants ayant leurs DEUX yeux complets : c'est
        # ce qui rend le plan equilibre et la decomposition exacte.
        n_par_sujet = g.groupby("participant")["eye_id"].nunique()
        g = g[g["participant"].isin(n_par_sujet[n_par_sujet == 2].index)]
        if g["participant"].nunique() < 3:
            continue
        comp = composantes(g)
        if comp:
            comp["n_one_cycles_min"] = int(g["n_one_cycles"].min())
            comp["n_one_cycles_med"] = float(g["n_one_cycles"].median())
            out.append({**dict(zip(cles, cle)), **comp})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# 4. Paires de replicats (pour les nuages de la page)
# --------------------------------------------------------------------------- #
def paires(marq: pd.DataFrame) -> pd.DataFrame:
    out = []
    cles = ["cas", "sigma", "region", "agregat"]
    for cle, g in marq.groupby(cles, dropna=False):
        base = dict(zip(cles, cle))
        for eye_id, ge in g.groupby("eye_id"):
            ge = ge.sort_values("replicate")
            for a, b in itertools.combinations(ge.itertuples(), 2):
                out.append({
                    **base, "eye_id": eye_id, "participant": a.participant,
                    "eye": a.eye, "rep_a": int(a.replicate),
                    "rep_b": int(b.replicate),
                    "valeur_a": a.valeur, "valeur_b": b.valeur,
                    "ecart": b.valeur - a.valeur,
                    "moyenne": 0.5 * (a.valeur + b.valeur),
                })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Lot
# --------------------------------------------------------------------------- #
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    qual = qualite_images()
    if not qual.empty:
        qual.to_csv(OUT_DIR / "quality.csv", index=False, encoding="utf-8")
        abimees = qual[qual["n_images_vides"] > 0]
        print(f"{OUT_DIR / 'quality.csv'}  ({len(qual)} acquisitions)")
        if len(abimees):
            print(f"  ! {len(abimees)} acquisition(s) avec des images vides "
                  f"a la source :")
            for r in abimees.itertuples():
                print(f"      {r.slug:26} {r.n_images_vides:4d} / {r.n_images} "
                      f"({r.pct_vides:.0f} %)")

    regions, bins = charger()
    print(f"regions.csv : {len(regions)} lignes, {regions['slug'].nunique()} acquisitions")
    print(f"bins.csv    : {len(bins)} lignes")

    marq = marqueurs(regions, bins)
    marq.to_csv(OUT_DIR / "markers.csv", index=False, encoding="utf-8")
    print(f"\n{OUT_DIR / 'markers.csv'}  ({len(marq)} marqueurs)")

    complets = plan_equilibre(marq)
    n_yeux = complets["eye_id"].nunique() if complets is not None else 0
    tous = set(marq["eye_id"])
    retenus = set(complets["eye_id"]) if complets is not None else set()
    ecartes = sorted(tous - retenus)
    print(f"  {n_yeux} yeux avec {N_REP} replicats complets sur {len(tous)}")
    if ecartes:
        # Nommer les yeux ecartes, et pourquoi : un effectif qui baisse sans
        # qu'on sache lequel est parti se lit comme un choix arbitraire.
        n_dispo = marq.groupby("eye_id")["replicate"].nunique()
        print("  yeux ecartes (moins de "
              f"{N_REP} replicats exploitables) : "
              + ", ".join(f"{e} ({int(n_dispo.get(e, 0))})" for e in ecartes))
    # La decomposition hierarchique exige en plus les DEUX yeux du participant.
    if complets is not None:
        n_par_sujet = complets.groupby("participant")["eye_id"].nunique()
        sujets_partiels = sorted(n_par_sujet[n_par_sujet < 2].index)
        if sujets_partiels:
            print("  participants hors decomposition (un seul oeil complet) : "
                  + ", ".join(sujets_partiels))
    else:
        sujets_partiels = []

    icc = icc_table(marq)
    icc.to_csv(OUT_DIR / "icc.csv", index=False, encoding="utf-8")
    print(f"{OUT_DIR / 'icc.csv'}  ({len(icc)} lignes)")

    var = variance_table(marq)
    var.to_csv(OUT_DIR / "variance.csv", index=False, encoding="utf-8")
    print(f"{OUT_DIR / 'variance.csv'}  ({len(var)} lignes)")

    pr = paires(marq)
    pr.to_csv(OUT_DIR / "pairs.csv", index=False, encoding="utf-8")
    print(f"{OUT_DIR / 'pairs.csv'}  ({len(pr)} paires)")

    # --- resume lisible ----------------------------------------------------- #
    lignes = [f"n_acquisitions = {marq['slug'].nunique()}",
              f"n_yeux_complets = {n_yeux}",
              f"n_yeux_total = {len(tous)}",
              f"yeux_ecartes = {', '.join(ecartes) if ecartes else 'aucun'}",
              f"sujets_hors_decomposition = "
              f"{', '.join(sujets_partiels) if sujets_partiels else 'aucun'}",
              f"n_participants = {marq['participant'].nunique()}",
              f"n_sujets_decomposition = "
              f"{int(var['n_sujets'].iloc[0]) if not var.empty else 0}",
              f"n_rep = {N_REP}"]
    if not qual.empty:
        abimees = qual[qual["n_images_vides"] > 0]
        lignes += [
            f"n_acq_avec_images_vides = {len(abimees)}",
            "acq_images_vides = " + (", ".join(
                f"{r.slug} ({r.n_images_vides}/{r.n_images})"
                for r in abimees.itertuples()) if len(abimees) else "aucune"),
        ]
    if not icc.empty:
        a1 = icc[icc["icc_type"] == "ICC(1,1)"].dropna(subset=["icc"])
        if not a1.empty:
            best = a1.loc[a1["icc"].idxmax()]
            lignes += [
                f"icc11_median = {a1['icc'].median():.3f}",
                f"icc11_max = {best['icc']:.3f}",
                f"icc11_max_config = cas={best['cas']} sigma={best['sigma']:g} "
                f"region={best['region']} agregat={best['agregat']}",
                f"icc11_n_sup_0.75 = {int((a1['icc'] >= 0.75).sum())} / {len(a1)}",
                f"icc11_n_sup_0.50 = {int((a1['icc'] >= 0.50).sum())} / {len(a1)}",
            ]
    if not var.empty:
        lignes += [
            f"pct_residuelle_median = {var['pct_residuelle'].median():.1f}",
            f"pct_sujet_median = {var['pct_sujet'].median():.1f}",
            f"pct_oeil_median = {var['pct_oeil'].median():.1f}",
        ]
    (OUT_DIR / "summary.txt").write_text("\n".join(lignes) + "\n", encoding="utf-8")
    print(f"{OUT_DIR / 'summary.txt'}")
    print("\n" + "\n".join(lignes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
