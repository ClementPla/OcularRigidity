"""
compute_rigidity_compare_mask_model.py

Teste la SENSIBILITE de la conclusion "rigidite oculaire k <-> gonflement
retinien post-vol" au choix de la segmentation de la choroide.

Pour chaque condition AVANT LE VOL (``parse_moment(...) == "before"``) du jeu
SANSORI, calcule k (Sayah et al. 2020) via UNIQUEMENT la Methode 1 (passe-bande
Butterworth + enveloppe de Hilbert, ``method1_bandpass_hilbert`` de
``compute_rigidity_time_series.py``), a partir des masques DEJA recales et
DEJA disponibles sous
``E:/NASA_Rigidity/SegmentationVariations/<variante>/registered_masks/...``
pour CHACUNE des variantes de segmentation listees dans ``MASK_VARIANTS``, et
pour CHACUNE des demi-largeurs de bande listees dans ``HR_HALF_BANDS_BPM``
(+/-10, +/-20, +/-50 BPM autour de la HR mesuree).

Aucun recalage ni segmentation n'est refait ici : seul ``mask.npz`` (par
variante, deja recale) est necessaire. Les horodatages ne sont PAS relus
depuis ``RawImages/registered/timestamp.txt`` (celui-ci est deja rogne du
skip/drop de la ``RegistrationConfig`` de production -- ex. 371 frames sur
401 -- alors que les masques de ``SegmentationVariations`` couvrent TOUTES
les frames brutes, skip=drop=0, cf. ``transform.npz``) : ils sont recalcules
directement depuis l'export XML Spectralis via
``scripts.registration.astronauts.load_ordered_oct_series`` (memes
horodatages pour toutes les variantes d'une condition -- calcules une seule
fois, partages entre les 4 variantes de masque).

Pour chaque (variante, demi-bande), agrege k par (astronaute, oeil) -- moyenne
des repliques "before" (OS1/OS2/... ou OD1/OD2/...) -- et correle avec
``delta_TRT_N = TRT250_N(apres vol) - TRT250_N(avant vol)`` (base
``sansori_db.db``, table ``Measurements``) : c'est le gonflement retinien
(quadrant nasal, indicateur SANS) que k est cense predire.

⚠️ CONSTAT DE DONNEES (a la date d'ecriture) : sur les 107 conditions
presentes, ``model1_scale_1.0`` == ``model1_scale_2.0`` et
``model2_scale_1.0`` == ``model2_scale_2.0`` -- masques ET transforms de
recalage identiques BIT A BIT. L'axe "scale" ne fait donc encore varier AUCUNE
donnee sur disque ; les deux variantes d'un meme modele produiront
necessairement le meme k et la meme correlation. Inclus quand meme ici (choix
de l'utilisateur) -- ``flag_duplicate_variants`` le detecte automatiquement et
l'affiche/l'ecrit en sortie, pour que ce ne soit pas lu par erreur comme "4
segmentations independantes qui concordent".

Sorties (sous ``E:/NASA_Rigidity/SegmentationVariations/``) :
  - ``rigidity_compare_mask_model_detail.csv`` : k par condition x variante x
    demi-bande (granularite replique, pour tracabilite/figures).
  - ``rigidity_compare_mask_model_correlations.csv`` : correlation
    k ~ delta_TRT_N (Pearson + Spearman) par variante x demi-bande, agregee
    par (astronaute, oeil).
  - ``rigidity_compare_mask_model_correlations_deltaCT.csv`` : meme chose
    pour delta_CT (amplitude de pulsatilite Hilbert, en mm, AVANT le modele
    de Sayah -- ne depend pas de AL/IOP/OPA) ~ delta_TRT_N -- isole si la
    correlation k ~ delta_TRT_N vient du signal OCT brut ou de sa
    transformation en k (coque spherique + AL/IOP/OPA).
  - ``rigidity_compare_mask_model_icc.csv`` : repetabilite (ICC(1,1),
    ICC(A,1) et ICC(C,1) de pingouin) de k entre repliques "meme oeil, meme
    moment", par variante x demi-bande x PAIRE de repliques (1v2, 1v3, 1v4,
    ... selon le nombre de repliques disponibles) x type d'ICC.
  - ``rigidity_compare_mask_model_icc.png`` : k de chaque replique, par sujet
    (astronaute + oeil), une grille variante x demi-bande -- illustre
    visuellement la dispersion intra-sujet que resument les trois ICC (paire
    1v2, la mieux dotee en sujets ; les autres paires sont dans le CSV).

Repetabilite (ICC) : certaines conditions "before" ont plusieurs repliques
(memes oeil et moment, ex. OS1/OS2/OD1..OD5 -- plusieurs acquisitions du meme
oeil dans la meme seance, jusqu'a 5 pour un sujet). ``compute_repeatability_icc``
calcule, pour chaque (variante, demi-bande), ICC(1,1), ICC(A,1) et ICC(C,1)
(Shrout & Fleiss / McGraw & Wong, cf. docstring de ``pingouin.intraclass_corr``
-- rapportes ENSEMBLE plutot qu'un seul choisi a priori, pour detecter un
biais systematique entre repliques si ICC(C,1) et ICC(A,1) divergent).
``pingouin.intraclass_corr`` exige un plan EQUILIBRE (memes "raters" pour
tous les sujets) : plutot qu'utiliser seulement les 2 premieres repliques de
chaque sujet (perdant l'info des sujets a 3+ repliques au-dela de la 2e), on
construit une PAIRE equilibree a la fois -- replique 1 vs 2, 1 vs 3, 1 vs 4,
... -- chacune n'incluant que les sujets qui possedent les deux repliques en
question. Sujet = (dossier astronaute, oeil) -- PAS le code de la base
clinique, pour ne pas perdre les sujets non apparies dans ``sansori_db.db``.
"""

from __future__ import annotations

import csv
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # backend non interactif : sauvegarde sur disque
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import pearsonr, spearmanr

from ocularrigidity.data.io import load_mask
from ocularrigidity.thickness.features import compute_deltaY_masks
from ocularrigidity.scripts.registration.astronauts import load_ordered_oct_series

from compute_rigidity_time_series import (
    PATH_GENERAL,
    PIX_Y_SIZE,
    DB_PATH,
    AL_DESCRIPTION,
    sayah_k,
    load_clinical_db,
    lookup_clinical,
    astro_code_from_folder,
    parse_eye,
    parse_moment,
    method1_bandpass_hilbert,
)

# --------------------------------------------------------------------------- #
# Parametres
# --------------------------------------------------------------------------- #
SEGVAR_ROOT = Path("E:/NASA_Rigidity/SegmentationVariations")
MASK_VARIANTS = ["model1_scale_1.0", "model1_scale_2.0", "model2_scale_1.0", "model2_scale_2.0"]
REGISTERED_MASKS_SUBDIR = "registered_masks"

HR_HALF_BANDS_BPM = [10.0, 20.0, 50.0]

TRT_DESCRIPTION = "TRT250_N"  # gonflement retinien post-vol (quadrant nasal)

DETAIL_CSV = SEGVAR_ROOT / "rigidity_compare_mask_model_detail.csv"
CORR_CSV = SEGVAR_ROOT / "rigidity_compare_mask_model_correlations.csv"
CORR_DELTACT_CSV = SEGVAR_ROOT / "rigidity_compare_mask_model_correlations_deltaCT.csv"
ICC_CSV = SEGVAR_ROOT / "rigidity_compare_mask_model_icc.csv"
ICC_FIG = SEGVAR_ROOT / "rigidity_compare_mask_model_icc.png"


# --------------------------------------------------------------------------- #
# Resolution des chemins (arborescence SANSORI, cf. compute_rigidity_time_series)
# --------------------------------------------------------------------------- #
def find_raw_dir(condition_dir: Path) -> Path | None:
    """Sous-dossier contenant les .tif bruts + l'export XML Spectralis."""
    for name in ("RawImages", "RawData"):
        d = condition_dir / name
        if d.is_dir():
            return d
    return None


def find_before_conditions(path_general: Path) -> list[tuple[Path, Path, Path]]:
    """(path_astro, path_moment, path_condi) pour toutes les conditions AVANT LE VOL."""
    out = []
    for path_astro in path_general.iterdir():
        if not path_astro.is_dir():
            continue
        for path_moment in path_astro.iterdir():
            if not path_moment.match("*rigidity"):
                continue
            if parse_moment(path_moment.name) != "before":
                continue
            for path_condi in path_moment.iterdir():
                if path_condi.is_dir():
                    out.append((path_astro, path_moment, path_condi))
    return out


# --------------------------------------------------------------------------- #
# Horodatages bruts (independants de la segmentation -> calcules une seule
# fois par condition, partages entre les variantes de masque)
# --------------------------------------------------------------------------- #
def raw_timestamps_us(raw_dir: Path) -> np.ndarray:
    """Horodatages bruts (us), un par frame, MEME ORDRE que les masques de
    ``SegmentationVariations`` (skip=drop=0 -> toutes les frames brutes,
    triees par horodatage croissant comme ``build_cube_and_timestamps``)."""
    series = load_ordered_oct_series(raw_dir)
    return np.array(
        [int(round(s.acquisition_time.seconds_of_day * 1e6)) for s in series],
        dtype=np.int64,
    )


# --------------------------------------------------------------------------- #
# Rigidite k (Sayah), Methode 1 (passe-bande + Hilbert) SEULEMENT
# --------------------------------------------------------------------------- #
def rigidity_hilbert(
    mask: np.ndarray, ts_us: np.ndarray, hr: float, half_band_bpm: float,
    AL: float, IOP: float, OPA: float,
) -> dict:
    """k (Sayah) a partir d'UN masque recale (une variante de segmentation),
    passe-bande centre sur ``hr`` +/- ``half_band_bpm``."""
    deltaY = compute_deltaY_masks(mask)  # (T, W) px

    ts = (ts_us - ts_us[0]) / 1e6  # secondes, origine au premier frame
    order = np.argsort(ts)
    ts_sorted = ts[order]
    dY = deltaY[order] - deltaY[order].mean(0)

    band = ((hr - half_band_bpm) / 60.0, (hr + half_band_bpm) / 60.0)
    fc0 = 0.5 * (band[0] + band[1])
    _t1, A1, f_hil, *_ = method1_bandpass_hilbert(ts_sorted, dY, band, fc0)

    amp1 = 2.0 * np.median(A1, axis=0)  # amplitude crete-a-crete par A-scan (px)
    deltaCT_mm = np.median(amp1) * PIX_Y_SIZE
    CT_mm = np.median(deltaY) * PIX_Y_SIZE
    k, deltaV = sayah_k(deltaCT_mm, CT_mm, AL, IOP, OPA)

    return {
        "CT_mm": CT_mm,
        "delta_CT_mm": deltaCT_mm,
        "delta_V": deltaV,
        "k": k,
        "hr_hilbert_bpm": float(np.median(f_hil) * 60.0) if f_hil.size else np.nan,
    }


# --------------------------------------------------------------------------- #
# delta_TRT_N : gonflement retinien post-vol (base clinique)
# --------------------------------------------------------------------------- #
def lookup_delta_trt(meas, code, eye) -> float:
    """TRT250_N(apres) - TRT250_N(avant), par (astronaute, oeil). NaN si
    l'une des deux valeurs est absente de la base."""
    before = lookup_clinical(meas, code, eye, "before", TRT_DESCRIPTION)
    after = lookup_clinical(meas, code, eye, "after", TRT_DESCRIPTION)
    if not (np.isfinite(before) and np.isfinite(after)):
        return np.nan
    return float(after - before)


# --------------------------------------------------------------------------- #
# Detection automatique de variantes dupliquees (cf. avertissement en tete)
# --------------------------------------------------------------------------- #
def flag_duplicate_variants(detail: pd.DataFrame) -> dict[str, list[str]]:
    """Regroupe les variantes dont TOUS les k (toutes conditions x bandes)
    sont identiques -- signale un doublon de segmentation (masques identiques
    en amont), plutot que deux methodes reellement independantes."""
    key_cols = ["patient", "moment", "condition", "half_band_bpm"]
    groups: dict[tuple, list[str]] = {}
    for variant, g in detail.groupby("mask_variant"):
        g_sorted = g.sort_values(key_cols)
        # chaine plutot que float brut : NaN != NaN casserait l'egalite de tuple
        # (deux groupes avec exactement les memes k manquants ne matcheraient jamais).
        key = tuple(
            "nan" if not np.isfinite(v) else f"{v:.9g}"
            for v in g_sorted["k"].to_numpy(dtype=float)
        )
        groups.setdefault(key, []).append(variant)
    return {v: vs for vs in groups.values() if len(vs) > 1 for v in vs}


# --------------------------------------------------------------------------- #
# Correlation d'une grandeur agregee par sujet (k ou delta_CT_mm) avec
# delta_TRT_N, par variante x demi-bande
# --------------------------------------------------------------------------- #
def compute_outcome_correlation(agg: pd.DataFrame, value_col: str, dup: dict[str, list[str]]) -> pd.DataFrame:
    """Correlation (Pearson + Spearman) entre ``value_col`` (deja agrege par
    sujet -- moyenne des repliques "before") et ``delta_TRT_N``, par
    (variante, demi-bande). Meme fonction pour k et pour delta_CT_mm : seul
    change le nombre de sujets valides (delta_CT_mm ne depend pas de
    AL/IOP/OPA, donc disponible pour des sujets ou k est NaN faute de
    donnees cliniques -- ``dropna`` est refait ici, par colonne)."""
    rows = []
    for (variant, half_band), g in agg.groupby(["mask_variant", "half_band_bpm"]):
        valid = g.dropna(subset=[value_col, "delta_TRT_N"])
        n = len(valid)
        if n >= 3:
            r_p, p_p = pearsonr(valid[value_col], valid["delta_TRT_N"])
            r_s, p_s = spearmanr(valid[value_col], valid["delta_TRT_N"])
        else:
            r_p = p_p = r_s = p_s = np.nan
        rows.append({
            "mask_variant": variant,
            "half_band_bpm": half_band,
            "n_patients_eyes": n,
            "duplicate_of": ", ".join(sorted(v for v in dup.get(variant, []) if v != variant)) or "",
            "pearson_r": r_p,
            "pearson_p": p_p,
            "spearman_r": r_s,
            "spearman_p": p_s,
        })
    return pd.DataFrame(rows).sort_values(["mask_variant", "half_band_bpm"])


# --------------------------------------------------------------------------- #
# Repetabilite : ICC(1,1)/ICC(A,1)/ICC(C,1) (pingouin) entre repliques
# "meme oeil, meme moment"
# --------------------------------------------------------------------------- #
ICC_TYPES = ["ICC(1,1)", "ICC(A,1)", "ICC(C,1)"]


def compute_repeatability_icc(detail: pd.DataFrame) -> pd.DataFrame:
    """ICC(1,1), ICC(A,1) et ICC(C,1) de k entre repliques, pour chaque
    (variante, demi-bande, paire de repliques) -- une ligne par combinaison x
    type d'ICC.

    Les trois types sont rapportes ENSEMBLE (recommandation de Liljequist et
    al. 2019, cf. docstring de ``pingouin.intraclass_corr``) plutot qu'un
    seul choisi a priori : ICC(1,1) suppose les "raters" interchangeables
    (pas de biais systematique) ; ICC(A,1) (accord absolu) et ICC(C,1)
    (consistance, rang) le testent separement. S'ils divergent notablement
    (ICC(C,1) >> ICC(A,1)), un biais systematique entre repliques (ex. derive
    de la FC, fatigue) est probable et ICC(1,1) seul serait trompeur.

    ``pingouin.intraclass_corr`` implemente le modele ANOVA classique de
    Shrout & Fleiss, qui exige un plan EQUILIBRE (memes "raters" pour tous les
    "targets") : ``nan_policy="omit"`` ne fait qu'un pivot target x rater
    suivi d'une suppression des LIGNES incompletes -- un sujet avec moins de
    repliques que le maximum observe serait entierement rejete, pas juste ses
    repliques manquantes. Comme le nombre de repliques varie ici de 1 a 5
    selon le sujet, on ne peut pas passer TOUTES les repliques d'un coup :
    on construit plutot une paire equilibree a la fois, replique 1 vs
    replique 2, replique 1 vs replique 3, etc. (jusqu'au plus grand nombre de
    repliques observe pour cette variante/bande) -- chaque paire n'incluant
    que les sujets qui possedent les DEUX repliques en question, pour ne rien
    perdre des sujets a 3+ repliques au-dela de la 2e.

    ``targets`` = sujet (``patient`` + ``eye`` -- PAS le code clinique, pour
    ne pas perdre les sujets non apparies dans ``sansori_db.db``) ; ``raters``
    = index de replique (ordre alphabetique de ``condition``, ex. OD1 -> 0,
    OD2 -> 1, ...).
    """
    rows = []
    for (variant, half_band), g_full in detail.groupby(["mask_variant", "half_band_bpm"]):
        g_full = g_full.dropna(subset=["k"]).copy()
        g_full["subject"] = g_full["patient"] + "_" + g_full["eye"].astype(str)
        g_full["rater"] = g_full.sort_values("condition").groupby("subject").cumcount()

        max_rater = int(g_full["rater"].max()) if len(g_full) else 0
        for other_rater in range(1, max_rater + 1):
            pair = f"1v{other_rater + 1}"
            g = g_full[g_full["rater"].isin([0, other_rater])]
            # ne garder que les sujets qui ont BIEN les deux repliques de cette paire
            # (ex. paire 1v3 exige la 3e replique, pas juste >=2 repliques au total)
            g = g[g.groupby("subject")["rater"].transform("nunique") == 2]

            n_subjects = g["subject"].nunique()
            if n_subjects < 2 or len(g) < 5:
                for icc_type in ICC_TYPES:
                    rows.append({
                        "mask_variant": variant, "half_band_bpm": half_band, "pair": pair,
                        "icc_type": icc_type, "n_subjects": n_subjects,
                        "icc": np.nan, "icc_ci95_low": np.nan, "icc_ci95_high": np.nan,
                        "icc_f": np.nan, "icc_pval": np.nan,
                    })
                continue

            icc_table = pg.intraclass_corr(
                data=g, targets="subject", raters="rater", ratings="k", nan_policy="omit"
            )
            for icc_type in ICC_TYPES:
                row = icc_table.loc[icc_table["Type"] == icc_type].iloc[0]
                rows.append({
                    "mask_variant": variant, "half_band_bpm": half_band, "pair": pair,
                    "icc_type": icc_type, "n_subjects": n_subjects,
                    "icc": row["ICC"], "icc_ci95_low": row["CI95"][0], "icc_ci95_high": row["CI95"][1],
                    "icc_f": row["F"], "icc_pval": row["pval"],
                })
    return pd.DataFrame(rows).sort_values(["mask_variant", "half_band_bpm", "pair", "icc_type"])


def plot_repeatability_by_subject(detail: pd.DataFrame, icc: pd.DataFrame, out_path: Path) -> None:
    """k de chaque replique, par sujet (astronaute + oeil) : une grille
    variante (ligne) x demi-bande (colonne). Chaque sujet est un ``x`` ;
    plusieurs points a la meme abscisse = plusieurs repliques -- leur
    dispersion (verticale) est justement ce que resume l'ICC annonce en
    titre de chaque case."""
    d = detail.dropna(subset=["k"]).copy()
    d["subject"] = d["patient"] + "_" + d["eye"].astype(str)
    subjects = sorted(d["subject"].unique())
    x_pos = {s: i for i, s in enumerate(subjects)}

    fig, axes = plt.subplots(
        len(MASK_VARIANTS), len(HR_HALF_BANDS_BPM),
        figsize=(4.5 * len(HR_HALF_BANDS_BPM), 3.8 * len(MASK_VARIANTS)),
        sharex=True, squeeze=False,
    )
    for i, variant in enumerate(MASK_VARIANTS):
        for j, half_band in enumerate(HR_HALF_BANDS_BPM):
            ax = axes[i, j]
            g = d[(d["mask_variant"] == variant) & (d["half_band_bpm"] == half_band)]
            x = g["subject"].map(x_pos).to_numpy()
            jitter = np.random.default_rng(0).uniform(-0.12, 0.12, size=x.shape)
            ax.scatter(x + jitter, g["k"], s=14, alpha=0.7, color="C0")

            # titre : paire "1v2" (replique 1 vs 2) -- la mieux dotee en sujets ;
            # les paires 1v3/1v4/... (sujets a 3+ repliques) sont dans le CSV.
            g_icc = icc[
                (icc["mask_variant"] == variant) & (icc["half_band_bpm"] == half_band)
                & (icc["pair"] == "1v2")
            ]
            lines = [f"{variant}, +/-{half_band:.0f} BPM (paire 1v2)"]
            for icc_type in ICC_TYPES:
                row = g_icc[g_icc["icc_type"] == icc_type]
                val = row["icc"].iloc[0] if len(row) else np.nan
                p = row["icc_pval"].iloc[0] if len(row) else np.nan
                lines.append(
                    f"{icc_type} = {val:.2f} (p={p:.3g})" if np.isfinite(val) else f"{icc_type} : n/a"
                )
            ax.set_title("\n".join(lines), fontsize=7)
            ax.set_ylabel("k (mm$^{-3}$)", fontsize=8)
            ax.tick_params(labelsize=7)
            if i == len(MASK_VARIANTS) - 1:
                ax.set_xticks(range(len(subjects)))
                ax.set_xticklabels(subjects, rotation=90, fontsize=6)

    fig.suptitle("Repetabilite de k par sujet (astronaute + oeil) -- repliques 'avant le vol'")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Boucle principale
# --------------------------------------------------------------------------- #
def _fmt(x):
    return f"{x:.6g}" if isinstance(x, (int, float)) and np.isfinite(x) else "<NA>"


def main():
    meas, id2code = load_clinical_db(DB_PATH)  # base clinique chargee une fois

    conditions = find_before_conditions(PATH_GENERAL)
    print(f"{len(conditions)} conditions 'avant le vol' trouvees sous {PATH_GENERAL}")

    rows = []
    for path_astro, path_moment, path_condi in conditions:
        print(path_condi)

        raw_dir = find_raw_dir(path_condi)
        if raw_dir is None:
            print(f"  [skip] aucun dossier RawImages/RawData : {path_condi}")
            continue

        path_heartbeat = path_condi / "Data Files" / "visit_data.csv"
        if not path_heartbeat.exists():
            print(f"  [skip] visit_data.csv manquant : {path_condi}")
            continue
        df = pd.read_csv(path_heartbeat, quoting=csv.QUOTE_NONE)
        hr = np.nanmean(pd.to_numeric(df["HR"], errors="coerce"))
        if not np.isfinite(hr):
            print(f"  [skip] HR non exploitable : {path_condi}")
            continue
        IOP = np.nanmean(pd.to_numeric(df["PascalIOP"], errors="coerce"))
        OPA = np.nanmean(pd.to_numeric(df["PascalOPA"], errors="coerce"))

        code = astro_code_from_folder(path_astro.name, id2code)
        eye = parse_eye(path_condi.name)
        AL = lookup_clinical(meas, code, eye, "before", AL_DESCRIPTION)
        delta_trt_n = lookup_delta_trt(meas, code, eye)

        try:
            ts_us = raw_timestamps_us(raw_dir)
        except Exception as e:  # noqa: BLE001
            print(f"  [erreur] horodatages bruts : {e} : {path_condi}")
            continue
        if ts_us.size < 2:
            print(f"  [skip] pas assez de frames brutes ({ts_us.size}) : {path_condi}")
            continue

        relpath = Path(path_astro.name) / path_moment.name / path_condi.name

        for variant in MASK_VARIANTS:
            mask_path = SEGVAR_ROOT / variant / REGISTERED_MASKS_SUBDIR / relpath / "mask.npz"
            if not mask_path.exists():
                print(f"  [skip] {variant} : mask.npz absent ({mask_path})")
                continue

            mask = np.asarray(load_mask(mask_path), dtype=bool)
            if mask.shape[0] != ts_us.shape[0]:
                print(
                    f"  [skip] {variant} : {mask.shape[0]} frames masque != "
                    f"{ts_us.shape[0]} horodatages : {path_condi}"
                )
                continue

            for half_band in HR_HALF_BANDS_BPM:
                try:
                    res = rigidity_hilbert(mask, ts_us, hr, half_band, AL, IOP, OPA)
                except Exception as e:  # noqa: BLE001
                    print(f"  [erreur] {variant} +/-{half_band:.0f} BPM : {e} : {path_condi}")
                    traceback.print_exc()
                    continue

                rows.append({
                    "patient": path_astro.name,
                    "moment": path_moment.name,
                    "condition": path_condi.name,
                    "code": code,
                    "eye": eye,
                    "hr": hr,
                    "AL": AL,
                    "IOP": IOP,
                    "OPA": OPA,
                    "delta_TRT_N": delta_trt_n,
                    "mask_variant": variant,
                    "half_band_bpm": half_band,
                    **res,
                })
                print(
                    f"  -> {variant} +/-{half_band:.0f} BPM : "
                    f"k = {_fmt(res['k'])} mm^-3  (delta_TRT_N = {_fmt(delta_trt_n)} um)"
                )

    if not rows:
        print("Aucune condition traitee : rien a ecrire.")
        return

    detail = pd.DataFrame(rows)
    detail.to_csv(DETAIL_CSV, index=False, na_rep="<NA>")
    print(f"\n{len(detail)} lignes (condition x variante x bande) ecrites dans {DETAIL_CSV}")

    dup = flag_duplicate_variants(detail)
    if dup:
        print(
            "\n[AVERTISSEMENT] variantes de segmentation produisant EXACTEMENT le meme k "
            "(masques identiques en amont) :"
        )
        seen = set()
        for variant, group_variants in dup.items():
            key = tuple(sorted(group_variants))
            if key in seen:
                continue
            seen.add(key)
            print(f"    {', '.join(key)}")

    # --- agregation par (astronaute, oeil) : moyenne des repliques "before" ---
    agg = (
        detail.groupby(["code", "eye", "mask_variant", "half_band_bpm"], dropna=False)
        .agg(
            k=("k", "mean"),
            delta_CT_mm=("delta_CT_mm", "mean"),
            delta_TRT_N=("delta_TRT_N", "mean"),
            n_replicates=("k", "size"),
        )
        .reset_index()
    )

    # --- correlation k ~ delta_TRT_N, par variante x demi-bande ---
    corr = compute_outcome_correlation(agg, "k", dup)
    corr.to_csv(CORR_CSV, index=False, na_rep="<NA>")
    print(f"\nCorrelations k ~ delta_TRT_N ecrites dans {CORR_CSV} :\n")
    print(corr.to_string(index=False))

    # --- correlation delta_CT (amplitude Hilbert, avant le modele de Sayah) ~
    # delta_TRT_N -- isole si la correlation k~delta_TRT_N vient du signal OCT
    # brut ou de sa transformation en k (AL/IOP/OPA, modele coque spherique) ---
    corr_deltact = compute_outcome_correlation(agg, "delta_CT_mm", dup)
    corr_deltact.to_csv(CORR_DELTACT_CSV, index=False, na_rep="<NA>")
    print(f"\nCorrelations delta_CT ~ delta_TRT_N ecrites dans {CORR_DELTACT_CSV} :\n")
    print(corr_deltact.to_string(index=False))

    # --- repetabilite (ICC(1,1)/ICC(A,1)/ICC(C,1), pingouin) entre repliques "before" ---
    icc = compute_repeatability_icc(detail)
    icc.to_csv(ICC_CSV, index=False, na_rep="<NA>")
    print(f"\nRepetabilite (ICC) ecrite dans {ICC_CSV} :\n")
    print(icc.to_string(index=False))

    plot_repeatability_by_subject(detail, icc, ICC_FIG)
    print(f"\nGraphique de repetabilite par sujet ecrit dans {ICC_FIG}")


if __name__ == "__main__":
    main()
