"""
compute_registration.py

Recalage par lot de TOUTES les videos SANSORI, avec une strategie en QUATRE TEMPS
demandee explicitement (et differente de celle de ``register_videos``, qui
estime le dx lateral par correlation d'image entiere sur les images brutes) :

  1. RECALAGE LATERAL GROSSIER SUR LA FOVEE. La fovee est localisee par le creux
     de l'ILM (``segmentation.fovea.from_ilm.estimate_fovea``) et chaque frame
     est translatee en x pour l'amener a la position qu'elle occupe dans la
     frame de reference. C'est un repere ANATOMIQUE, insensible au speckle : il
     enleve les grands deplacements avant que la correlation, qui ne cherche que
     dans une fenetre de +-MAX_LATERAL_SHIFT px, n'ait a s'en occuper.
     (Seul le dx de la fovee est applique : son dy serait exactement annule par
     l'aplatissement de l'etape 2, cf. ``fovea_dx``.) La trace est ecartee en
     bloc quand elle n'est pas credible -- mesure sur la cohorte : 12 conditions
     sur 107 ou le creux de l'ILM se fixe sur un artefact, avec des amplitudes de
     100 a 9000 px la ou le mouvement reel mediane 4 px. Ces conditions sont
     recalees par la seule correlation, et ``fovea_used`` le dit dans le CSV.
  2. APLATISSEMENT DE LA RPE (axial, y). Chaque colonne est translatee pour que
     la membrane de Bruch tombe sur UNE MEME ligne constante (``flatten_rpe``),
     pas sur la courbe de BM d'une frame de reference. La choroide devient donc
     une bande horizontale de profondeur quasi constante, ce qui rend le
     probleme restant purement lateral.
  3. CROSS-CORRELATION EN X sur les PIXELS DE LA CHOROIDE, restreints au TIERS
     CENTRAL des A-scans -- raffinement fin du recalage grossier de l'etape 1.
     Pour chaque decalage entier ``s`` de -MAX_LATERAL_SHIFT a +MAX_LATERAL_SHIFT,
     on correle le pave d'image aplati de la frame avec celui de la frame de
     reference, en ne sommant QUE sur les pixels ou le masque de choroide est
     vrai (le reste est mis a zero apres soustraction de la moyenne
     intra-choroide, donc ne contribue pas). Score normalise (NCC), pic +
     interpolation parabolique sous-pixel.
  4. ALIGNEMENT DES A-SCANS sur la mediane temporelle
     (``axial.median_registration.register_ascans_to_median``) : correlation de
     phase 1D par colonne, le long de l'axe axial, contre le template median du
     volume -- un raffinement axial FIN, colonne par colonne, la ou l'etape 2 ne
     corrige que la position de la BM segmentee.

L'etape 4 vient APRES l'etape 3, et non a la fin de ``register_videos`` comme le
ferait ``axial_refinement=True`` : sa reference est la MEDIANE TEMPORELLE du
volume, qui n'est nette que si les frames sont deja alignees lateralement. La
calculer avant le recalage en x reviendrait a aligner chaque A-scan sur un
template flou de son propre bouge.

Le dx applique est ``dx_fovee + dx_xcorr``. Le rejet temporel des valeurs
aberrantes (``robust_temporal_dx``) intervient DEUX fois, pour deux raisons
distinctes -- ce n'est pas une redondance :

  - sur ``dx_fovee`` AVANT son application, parce que l'etape 1 translate
    reellement les pixels. Sur certaines conditions le creux de l'ILM saute de
    plusieurs dizaines de px sur quelques frames ; les translater ainsi les fait
    sortir du cadre et y laisse une bande noire que RIEN ne rattrape ensuite,
    la fenetre de l'etape 3 ne faisant que +-MAX_LATERAL_SHIFT px.
  - sur le TOTAL ``dx_fovee + dx_xcorr``, et non sur la correction de
    correlation seule : la ou la fovee derape encore un peu, la correlation
    produit legitimement une correction opposee, et filtrer les deux separement
    rejetterait cette correction en la prenant pour une aberration -- laissant
    la frame decalee.

Pourquoi cette ROI : le tiers central evite les bords lateraux, ou le recalage
est le moins fiable et ou la BM sort souvent du champ ; se limiter aux pixels
de la segmentation fait porter la correlation sur la texture vasculaire de la
choroide elle-meme plutot que sur la retine ou le fond de l'image, qui ne
bougent pas de la meme facon. C'est une correlation 2D sommee en y mais decalee
en x SEULEMENT (le y est deja regle par l'etape 1).

Le dx est ensuite applique aux frames et masques deja aplatis (2e ``grid_sample``,
x uniquement -- meme decoupage en deux warps successifs que ``register_videos``),
puis les colonnes dont la BM reste inexploitable sont noircies
(``filter_bad_ascans_per_bms``, re-evalue APRES le decalage lateral pour que la
sortie soit coherente dans son propre repere).

La correction de fovea est faite ICI plutot que par ``register_videos``
(``fovea_correction_enabled=False`` dans la config) uniquement pour que son dx
soit EXPOSE : le moteur l'applique en interne sans le retourner dans ``params``,
ce qui rendrait le dx total intracable dans ``transform.npz``. Meme raison de
forme pour l'etape 4 (``axial_refinement=False`` dans la config, appel explicite
plus loin) -- mais la raison principale y est l'ORDRE, cf. ci-dessus.

Arborescence lue
----------------
    E:/SANSORI/<NN_id>/<...>_rigidity/<..._OD|OS...>/
        RawImages/ (ou RawData/)   <- images .tif + export XML Spectralis

Sorties (nouvelle variante, MEME arborescence que les variantes existantes --
les scripts aval n'ont qu'a changer leur ``MASK_VARIANT``)
-----------------------------------------------------------------------------
    E:/NASA_Rigidity/SegmentationVariations/<VARIANT>/
        registered_frames/<NN_id>/<...>_rigidity/<...>/cube.mp4
        registered_masks/<NN_id>/<...>_rigidity/<...>/mask.npz
        registered_masks/<NN_id>/<...>_rigidity/<...>/transform.npz  (dx, dy, ...)
        registration_summary.csv    1 ligne / condition
        registration_params.json    parametres de la variante (tracabilite)

Comme les variantes existantes, la sortie couvre TOUTES les frames brutes
(skip = drop = 0) : aucun ``timestamp.txt`` n'est ecrit, les horodatages se
recalculent depuis l'export XML (``load_ordered_oct_series``), exactement comme
le font ``compute_pulse_from_data.py`` et ``compute_rigidity_compare_mask_model.py``.

Le CSV est reecrit apres CHAQUE condition et relu au demarrage : le script est
interruptible et reprend ou il s'est arrete (``OVERWRITE`` pour tout refaire,
``LIMIT`` pour un essai sur les N premieres conditions).

Lancer (kernel pyOR, depuis la racine du depot) :
    C:/Users/transformer/anaconda3/envs/pyOR/python.exe \
        Astronauts/compute_registration.py
"""

from __future__ import annotations

import os
import csv
import dataclasses
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ocularrigidity.scripts import batch_layout as layout
from ocularrigidity.data.io import save_mask
from ocularrigidity.registration.axial.median_registration import (
    register_ascans_to_median,
)
from ocularrigidity.registration.config import RegistrationConfig
from ocularrigidity.registration.lateral.utils import (
    robust_temporal_dx,
    smooth_translations,
)
from ocularrigidity.registration.postprocess import filter_bad_ascans_per_bms
from ocularrigidity.registration.rigid import register_videos
from ocularrigidity.segmentation.fovea.from_ilm import estimate_fovea
from ocularrigidity.segmentation.inference import infer
from ocularrigidity.segmentation.utils import get_choroid_segmentation_model

# Importe aussi pour l'effet de bord : ce module retire IMAGEIO_FFMPEG_EXE de
# l'environnement (chemin Linux code en dur par data/compression.py).
from ocularrigidity.scripts.registration.astronauts import (
    build_cube_and_timestamps,
    estimate_fps,
    fill_empty_columns,
    load_ordered_oct_series,
    write_gray_mp4,
)

# --------------------------------------------------------------------------- #
# Parametres
# --------------------------------------------------------------------------- #
# Surchargeables par l'environnement (cf. ``scripts/batch_layout``) : le meme lot
# doit pouvoir tourner sur la cohorte SANS et sur l'etude de repetabilite, dont
# l'arborescence est plate. Non definies, ces variables laissent exactement le
# comportement d'origine.
PATH_GENERAL = layout.env_path("OR_PATH_GENERAL", "E:/SANSORI")
SEGVAR_ROOT = layout.env_path("OR_SEGVAR_ROOT", "E:/NASA_Rigidity/SegmentationVariations")
# Variante : meme segmentation que ``model1_scale_1.0`` (modele HuggingFace par
# defaut, scale 1.0), mais recalage "RPE aplatie + xcorr choroide".
VARIANT = layout.env_str("OR_VARIANT", "model1_scale_1.0_flatten_choroid_xcorr")
FRAMES_SUBDIR = "registered_frames"
MASKS_SUBDIR = "registered_masks"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OVERWRITE = False  # True = re-recaler les conditions deja ecrites
LIMIT = (int(os.environ["OR_LIMIT"]) if os.environ.get("OR_LIMIT")
         else None)  # int = ne traiter que les N premieres conditions (essai)

# --- Segmentation (identique a la variante model1_scale_1.0) -----------------
SEG_SCALE_FACTOR = 1.0
SEG_BATCH_SIZE = 8

# --- Etape 1 : recalage lateral grossier sur la fovee ------------------------
FOVEA_CORRECTION = True  # False = aller directement a la correlation
# Fraction de frames sans fovee exploitable au-dela de laquelle la trace est
# jugee trop interpolee pour servir de repere.
MAX_FOVEA_FAILED_FRAC = 0.10

# --- Etape 3 : cross-correlation en x sur la choroide ------------------------
# TIERS CENTRAL des A-scans : les bords lateraux sont ceux ou le recalage est le
# moins fiable (BM souvent hors champ, ombres vasculaires retiniennes).
COL_FRAC = (1.0 / 3.0, 2.0 / 3.0)
MAX_LATERAL_SHIFT = 16  # px, amplitude maximale cherchee (comme RegistrationConfig)
# Garde-fou de l'ETAPE 1, exprime dans l'unite de l'etape 3 : la fovee n'est la
# que pour retirer un mouvement que la fenetre +-MAX_LATERAL_SHIFT ne pourrait pas
# rattraper. En deca, elle n'apporte rien que la correlation ne fasse deja ;
# au-dela de 2 x cette fenetre, une trace de fovee n'est plus credible (le
# mouvement lateral reel mesure ici a une mediane de 4 px crete-a-crete par
# condition), c'est un artefact que l'ILM a pris pour un creux maculaire.
MAX_FOVEA_SPREAD = 2.0 * MAX_LATERAL_SHIFT
SUBPIXEL = True  # interpolation parabolique du pic de correlation
SMOOTH_TRANSVERSAL = False  # lissage gaussien du dx temporel
SMOOTH_TRANSVERSAL_SIGMA = 2.0
BATCH_SIZE = 128  # frames par lot (correlation et warps)

# --- Etape 4 : alignement des A-scans sur la mediane temporelle --------------
AXIAL_REFINEMENT = True
MAX_AXIAL_SHIFT = 7  # px, deplacement axial maximal teste (defaut RegistrationConfig)
AXIAL_BANDPASS = (0.02, 0.5)  # passe-bande spectral de la correlation de phase

# --- Sorties -----------------------------------------------------------------
VARIANT_ROOT = SEGVAR_ROOT / VARIANT
CSV_SUMMARY = VARIANT_ROOT / "registration_summary.csv"
JSON_PARAMS = VARIANT_ROOT / "registration_params.json"

SUMMARY_FIELDS = [
    "patient", "moment", "condition", "status", "reason",
    "n_frames", "fps", "height", "width",
    "roi_x0", "roi_x1", "roi_y0", "roi_y1",
    "dx_min", "dx_max", "dx_std", "dx_ptp", "dx_conf_median",
    "dx_fovea_ptp", "dx_xcorr_ptp", "n_fovea_failed", "fovea_used",
    "dy_std", "dy_median_std", "n_bad_columns", "seconds", "created",
]

# Etape 2 : aplatissement de la RPE SEUL (le dx lateral est estime aux etapes 1
# et 3, l'alignement A-scan a l'etape 4 -- rien de tout cela par le moteur).
FLATTEN_CONFIG = RegistrationConfig(
    skip_first_n_frames=0,
    drop_last_n_frames=0,
    correct_transversal=False,
    correct_axial=True,
    flatten_rpe=True,
    axial_refinement=False,
    fovea_correction_enabled=False,
    subpixel=SUBPIXEL,
    batch_size=BATCH_SIZE,
)


# --------------------------------------------------------------------------- #
# Resolution des chemins (arborescence SANSORI)
# --------------------------------------------------------------------------- #
def find_raw_dir(condition_dir: Path) -> Path | None:
    """Sous-dossier contenant les .tif bruts + l'export XML Spectralis."""
    for name in ("RawImages", "RawData"):
        d = condition_dir / name
        if d.is_dir():
            return d
    return None


def iter_conditions():
    """``(astro, moment, path_condi)`` de toutes les conditions.

    Les deux premiers elements ne sont utilises que par leur ``.name`` (cle de
    reprise et chemin de sortie) : ce sont des ETIQUETTES. Sur l'arborescence
    plate elles ne designent aucun dossier reel, et c'est voulu -- les sorties
    se rangent quand meme a trois niveaux.
    """
    for path_condi in layout.iter_condition_dirs(PATH_GENERAL):
        astro, moment, _ = layout.labels_of(path_condi, PATH_GENERAL)
        yield Path(astro), Path(moment), path_condi


# --------------------------------------------------------------------------- #
# Etape 1 : recalage lateral grossier sur la fovee
# --------------------------------------------------------------------------- #
def fovea_dx(frames: np.ndarray, masks: np.ndarray, ref_idx: int):
    """dx (T,) amenant la fovee de chaque frame sur celle de la reference.

    Meme grandeur que celle calculee par ``registration.lateral.dx.fovea_correction``
    (``ref_fovea_x - fovea_x``, meme convention de signe), mais RETOURNEE au lieu
    d'etre appliquee en interne -- c'est ce qui permet de tracer le dx total.

    Le dy de la fovee n'est deliberement pas produit : l'aplatissement de la RPE
    (etape 2) pose la BM sur une ligne constante, donc il ABSORBE integralement
    toute translation verticale anterieure -- appliquer dy_fovee reviendrait a
    payer une interpolation supplementaire pour un resultat identique.

    La trace est ECARTEE EN BLOC (dx nul, ``used=False``) si trop de frames n'ont
    pas de fovee, ou si son amplitude depasse ``MAX_FOVEA_SPREAD`` : le rejet
    temporel ne sait retirer que des valeurs ISOLEES, alors qu'une trace fausse
    sur une bonne part des frames entraine avec elle la mediane glissante qui
    sert a la juger. La correlation de l'etape 3 fait alors seule le travail
    lateral -- ce qu'elle sait faire, la fovee n'etant qu'un degrossissage.

    Retourne ``(dx, n_failed, used)``, ``n_failed`` etant le nombre de frames ou
    l'ILM n'a pas donne de fovee exploitable (leur dx est interpole temporellement).
    """
    loc = np.asarray(estimate_fovea(frames, masks), dtype=np.float64)  # (T, 2) = (x, y)
    x = loc[:, 0]
    valid = np.isfinite(x)
    n_failed = int((~valid).sum())
    unusable = np.zeros(x.shape[0], dtype=np.float32)
    if not valid.any() or n_failed > MAX_FOVEA_FAILED_FRAC * x.shape[0]:
        # Aucune fovee trouvee (ex. B-scan hors macula), ou trop peu : etape 1
        # sans effet, l'etape 3 fait alors tout le travail lateral.
        return unusable, n_failed, False

    # Cible = la fovee de la frame de reference ; si c'est justement elle qui a
    # echoue, la mediane des frames valides evite de perdre toute la condition.
    target = x[ref_idx] if valid[ref_idx] else float(np.median(x[valid]))
    dx = target - x
    if n_failed:
        idx = np.arange(x.shape[0], dtype=np.float64)
        dx = np.interp(idx, idx[valid], dx[valid])

    # Rejet temporel AVANT application, et non plus tard sur le dx total. Sur
    # certaines conditions le creux de l'ILM saute d'un coup de plusieurs dizaines
    # de px (mesure : 3 a 5 frames sur ~470, jusqu'a 120 px d'ecart a la mediane
    # locale) -- la fenetre de recherche de l'etape 3 ne fait que +-MAX_LATERAL_SHIFT
    # px, elle ne peut donc pas rattraper une telle erreur PAR LA CORRELATION. Le
    # filtre du dx total la rattrape bien numeriquement, mais trop tard : l'etape 1
    # a deja translate la frame de 120 px avec remplissage a zero, et le contenu
    # sorti du cadre ne revient pas -- la frame garde une bande noire.
    dx = robust_temporal_dx(torch.from_numpy(dx.astype(np.float32))).numpy()
    if float(np.ptp(dx)) > MAX_FOVEA_SPREAD:
        return unusable, n_failed, False
    return dx, n_failed, True


# --------------------------------------------------------------------------- #
# Etape 3 : cross-correlation en x sur les pixels de choroide du tiers central
# --------------------------------------------------------------------------- #
def choroid_roi(masks: np.ndarray, col_frac=COL_FRAC, max_shift=MAX_LATERAL_SHIFT):
    """Boite (y0, y1, x0, x1) de la ROI de correlation.

    En x : le tiers central des A-scans (elargi si besoin pour rester plus large
    que la plage de decalages testee, sinon la correlation n'aurait plus rien a
    superposer). En y : les lignes effectivement occupees par la choroide dans
    cette bande de colonnes -- apres aplatissement de la RPE c'est une bande
    etroite, donc autant ne pas correler des centaines de lignes vides.
    """
    T, H, W = masks.shape
    x0 = int(round(W * col_frac[0]))
    x1 = int(round(W * col_frac[1]))
    min_width = 4 * max_shift + 1
    if x1 - x0 < min_width:  # ROI degeneree (image tres etroite)
        center = (x0 + x1) // 2
        x0 = max(0, center - min_width // 2)
        x1 = min(W, x0 + min_width)
    rows = np.flatnonzero(masks[:, :, x0:x1].any(axis=(0, 2)))
    if rows.size == 0:
        raise ValueError("aucun pixel de choroide dans le tiers central des A-scans")
    return int(rows[0]), int(rows[-1]) + 1, x0, x1


def _roi_patches(frames, masks, box, device):
    """Paves de correlation ``(t, h, w)`` : intensite moins la moyenne INTRA-choroide,
    puis mise a zero hors du masque -- seuls les pixels de choroide comptent dans
    la somme de correlation, et un fond a zero n'y ajoute aucun terme."""
    y0, y1, x0, x1 = box
    f = torch.as_tensor(np.ascontiguousarray(frames[:, y0:y1, x0:x1])).to(
        device, torch.float32
    )
    m = torch.as_tensor(np.ascontiguousarray(masks[:, y0:y1, x0:x1])).to(
        device, torch.float32
    )
    n = m.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
    mean = (f * m).sum(dim=(1, 2), keepdim=True) / n
    return (f - mean) * m


def choroid_xcorr_dx(
    frames: np.ndarray,
    masks: np.ndarray,
    ref_idx: int,
    box: tuple[int, int, int, int],
    *,
    max_shift: int = MAX_LATERAL_SHIFT,
    subpixel: bool = SUBPIXEL,
    batch_size: int = BATCH_SIZE,
    device: str = DEVICE,
):
    """dx (T,) alignant chaque frame sur ``ref_idx``, par cross-correlation en x.

    Convention de signe identique a ``profile_correlation_dx`` : le score du
    decalage ``s`` compare ``frame[x]`` a ``ref[x + s]``, et le dx retourne
    s'applique tel quel dans ``norm_x = (grid_x - dx)``.

    Estimation BRUTE : ni rejet temporel ni lissage ici, ils sont appliques par
    l'appelant au dx TOTAL (fovee + correlation), cf. docstring du module.

    Retourne ``(dx, conf)``, ``conf`` etant la nettete du pic (z-score du maximum
    par rapport a l'ensemble des decalages testes) -- utilisee pour ecarter les
    frames dont la correlation n'a rien trouve de franc.
    """
    T = frames.shape[0]
    w = box[3] - box[2]
    ref = _roi_patches(frames[ref_idx : ref_idx + 1], masks[ref_idx : ref_idx + 1], box, device)[0]
    ref_energy_full = (ref * ref).sum()
    if not torch.isfinite(ref_energy_full) or ref_energy_full <= 0:
        raise ValueError("frame de reference sans texture de choroide exploitable")

    shifts = list(range(-max_shift, max_shift + 1))
    dx = torch.empty(T, device=device, dtype=torch.float32)
    conf = torch.empty(T, device=device, dtype=torch.float32)

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        patch = _roi_patches(frames[start:end], masks[start:end], box, device)
        scores = torch.empty(end - start, len(shifts), device=device)
        for i, s in enumerate(shifts):
            if s >= 0:
                a, b = patch[:, :, : w - s], ref[:, s:]
            else:
                a, b = patch[:, :, -s:], ref[:, : w + s]
            # NCC : sans normalisation, le recouvrement qui retrecit avec |s|
            # biaiserait mecaniquement le score vers s = 0.
            num = (a * b).sum(dim=(1, 2))
            den = torch.sqrt((a * a).sum(dim=(1, 2)) * (b * b).sum()) + 1e-8
            scores[:, i] = num / den

        best = scores.argmax(dim=1)
        peak = torch.tensor(shifts, device=device, dtype=torch.float32)[best]
        if subpixel:
            # Parabole sur les 3 scores autour du pic ; pas de fit si le maximum
            # touche un bord de la fenetre de recherche (voisin manquant).
            inner = (best > 0) & (best < len(shifts) - 1)
            if inner.any():
                i0 = best[inner]
                s_m1 = scores[inner, i0 - 1]
                s_0 = scores[inner, i0]
                s_p1 = scores[inner, i0 + 1]
                den = s_m1 - 2 * s_0 + s_p1
                offset = torch.where(
                    den != 0, 0.5 * (s_m1 - s_p1) / den, torch.zeros_like(s_0)
                ).clamp(-1.0, 1.0)
                peak[inner] += offset
        dx[start:end] = peak
        conf[start:end] = (
            scores.gather(1, best[:, None]).squeeze(1) - scores.mean(dim=1)
        ) / (scores.std(dim=1) + 1e-8)

    return dx, conf


def apply_lateral_shift(
    frames: np.ndarray,
    masks: np.ndarray,
    dx: torch.Tensor,
    *,
    batch_size: int = BATCH_SIZE,
    device: str = DEVICE,
):
    """Translate frames et masques de ``dx`` en x (bilineaire, bords a zero).

    Le masque est re-seuille a 0.5 et non "tout pixel non nul" (ce que ferait un
    cast direct en booleen) : avec un dx sous-pixel, garder les pixels partiels
    dilaterait le masque d'un pixel de chaque cote, et cette dilatation
    varierait AVEC dx -- soit une epaisseur choroidienne qui oscillerait au
    rythme du recalage, exactement le signal qu'on cherche a mesurer.
    """
    T, H, W = frames.shape
    xs = torch.arange(W, device=device, dtype=torch.float32)
    ys = torch.arange(H, device=device, dtype=torch.float32)
    out_frames = np.empty_like(frames)
    out_masks = np.empty_like(masks)

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        t = end - start
        f = torch.as_tensor(np.ascontiguousarray(frames[start:end])).to(
            device, torch.float32
        )
        m = torch.as_tensor(np.ascontiguousarray(masks[start:end])).to(
            device, torch.float32
        )
        data = torch.stack([m, f], dim=1)  # t x 2 x H x W

        grid_x = xs.view(1, 1, W).expand(t, H, W)
        grid_y = ys.view(1, H, 1).expand(t, H, W)
        d = dx[start:end].to(device).view(t, 1, 1)
        norm_x = (grid_x - d) / (W - 1) * 2 - 1
        norm_y = grid_y / (H - 1) * 2 - 1
        reg = F.grid_sample(
            data,
            torch.stack([norm_x, norm_y], dim=-1),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        out_masks[start:end] = (reg[:, 0] >= 0.5).cpu().numpy()
        out_frames[start:end] = (
            reg[:, 1].round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        )
    return out_frames, out_masks


# --------------------------------------------------------------------------- #
# Traitement d'une condition
# --------------------------------------------------------------------------- #
def process_condition(raw_dir: Path, model, frames_dir: Path, masks_dir: Path) -> dict:
    """Segmente, aplatit la RPE, recale en x sur la choroide, et ecrit la sortie."""
    series = load_ordered_oct_series(raw_dir)
    if len(series) < 2:
        return {"status": "skipped", "reason": "no_series"}

    cube, ts_us = build_cube_and_timestamps(raw_dir, series)

    # --- Segmentation de la choroide (variante model1, scale 1.0) ------------
    masks = np.asarray(
        infer(
            model,
            cube,
            scale_factor=SEG_SCALE_FACTOR,
            batch_size=SEG_BATCH_SIZE,
            device=DEVICE,
        ),
        dtype=bool,
    )
    # Indispensable avant l'aplatissement : une colonne sans choroide donne une
    # BM NaN, et ``nanmean(ref_bm)`` propagerait le NaN -> deplacement nul
    # partout, donc AUCUN recalage vertical (cf. fill_empty_columns).
    masks = fill_empty_columns(masks)

    # Frame de reference, unique pour toute la chaine : celle dont l'aire de
    # masque est la plus proche de la mediane temporelle (regle de ``register_videos``).
    areas = masks.sum(axis=(1, 2)).astype(np.float64)
    ref_idx = int(np.abs(areas - np.median(areas)).argmin())

    # --- Etape 1 : recalage lateral grossier sur la fovee --------------------
    if FOVEA_CORRECTION:
        dx_fovea, n_fovea_failed, fovea_used = fovea_dx(cube, masks, ref_idx)
        if fovea_used:
            cube, masks = apply_lateral_shift(
                cube, masks, torch.as_tensor(dx_fovea, dtype=torch.float32)
            )
            # La translation vide les colonnes de bord (padding a zero) : on les
            # recomble, sinon leur BM redevient NaN et l'aplatissement les laisse
            # sur place au lieu de les recaler.
            masks = fill_empty_columns(masks)
    else:
        dx_fovea = np.zeros(cube.shape[0], dtype=np.float32)
        n_fovea_failed, fovea_used = 0, False

    # --- Etape 2 : aplatissement de la RPE -----------------------------------
    flat_masks, flat_frames, params = register_videos(
        masks, cube, FLATTEN_CONFIG, device=DEVICE, verbose=True, return_params=True
    )
    flat_masks = flat_masks.cpu().numpy() > 0
    flat_frames = flat_frames.cpu().numpy().astype(np.uint8)

    # --- Etape 3 : cross-correlation en x sur la choroide (tiers central) ----
    box = choroid_roi(flat_masks)
    dx_xcorr, conf = choroid_xcorr_dx(flat_frames, flat_masks, ref_idx, box)

    # Rejet temporel sur le dx TOTAL (cf. docstring) : une frame ou la fovee a
    # derape et ou la correlation la rattrape est coherente, pas aberrante.
    dx_fovea_t = torch.as_tensor(dx_fovea, dtype=torch.float32, device=dx_xcorr.device)
    dx = robust_temporal_dx(dx_fovea_t + dx_xcorr, conf=conf)
    if SMOOTH_TRANSVERSAL:
        dx = smooth_translations(dx, sigma=SMOOTH_TRANSVERSAL_SIGMA)
    if not SUBPIXEL:
        dx = dx.round()

    # Le dx de la fovee est DEJA applique : il ne reste que le complement.
    reg_frames, reg_masks = apply_lateral_shift(
        flat_frames, flat_masks, dx - dx_fovea_t
    )

    # --- Etape 4 : alignement des A-scans sur la mediane temporelle ----------
    # Fait ICI, une fois le recalage lateral acquis : la mediane qui sert de
    # template ne vaut que si le volume est deja aligne en x (cf. docstring).
    if AXIAL_REFINEMENT:
        reg_frames_t, reg_masks_t, dy_median = register_ascans_to_median(
            reg_frames,
            reg_masks,
            max_vshift=MAX_AXIAL_SHIFT,
            subpixel=SUBPIXEL,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            bandpass=AXIAL_BANDPASS,
            verbose=True,
        )
        reg_frames = reg_frames_t.numpy()
        reg_masks = reg_masks_t.numpy()
        dy_median_np = dy_median.numpy().astype(np.float32)
    else:
        dy_median_np = None

    # --- Colonnes inexploitables, re-evaluees APRES tous les deplacements ----
    bad_cols = filter_bad_ascans_per_bms(reg_masks)
    if bool(bad_cols.any()):
        bad = bad_cols.cpu().numpy()
        reg_frames[:, :, bad] = 0
        reg_masks[:, :, bad] = 0

    # --- Ecriture ------------------------------------------------------------
    frames_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    fps = estimate_fps(ts_us)
    write_gray_mp4(reg_frames, frames_dir / "cube.mp4", fps)
    save_mask(reg_masks, masks_dir / "mask.npz")

    dx_np = dx.detach().cpu().numpy().astype(np.float32)
    dy_np = params["dy"].detach().cpu().numpy().astype(np.float32)
    extra = {} if dy_median_np is None else {"dy_median": dy_median_np}
    np.savez(
        masks_dir / "transform.npz",
        **extra,
        dx=dx_np,  # total applique = fovee + correlation, apres rejet temporel
        dx_fovea=np.asarray(dx_fovea, dtype=np.float32),
        dx_xcorr=dx_xcorr.detach().cpu().numpy().astype(np.float32),
        dy=dy_np,
        dx_confidence=conf.detach().cpu().numpy().astype(np.float32),
        bad_columns=bad_cols.cpu().numpy(),
        ref_idx=np.int64(ref_idx),
        roi_box=np.asarray(box, dtype=np.int64),  # (y0, y1, x0, x1)
    )

    T, H, W = reg_frames.shape
    return {
        "status": "ok",
        "reason": "",
        "n_frames": T,
        "fps": round(float(fps), 3),
        "height": H,
        "width": W,
        "roi_y0": box[0], "roi_y1": box[1], "roi_x0": box[2], "roi_x1": box[3],
        "dx_min": float(dx_np.min()),
        "dx_max": float(dx_np.max()),
        "dx_std": float(dx_np.std()),
        "dx_ptp": float(dx_np.max() - dx_np.min()),
        "dx_conf_median": float(np.median(conf.detach().cpu().numpy())),
        "dx_fovea_ptp": float(np.ptp(dx_fovea)),
        "dx_xcorr_ptp": float(np.ptp(dx_xcorr.detach().cpu().numpy())),
        "n_fovea_failed": int(n_fovea_failed),
        "fovea_used": bool(fovea_used),
        "dy_std": float(np.nanstd(dy_np)),
        "dy_median_std": (
            float(np.nanstd(dy_median_np)) if dy_median_np is not None else ""
        ),
        "n_bad_columns": int(bad_cols.sum()),
    }


# --------------------------------------------------------------------------- #
# Table de suivi (reecrite apres chaque condition -> reprise possible)
# --------------------------------------------------------------------------- #
def load_summary() -> dict[tuple[str, str, str], dict]:
    if not CSV_SUMMARY.exists():
        return {}
    with CSV_SUMMARY.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {(r["patient"], r["moment"], r["condition"]): r for r in rows}


def write_summary(rows: dict[tuple[str, str, str], dict]) -> None:
    CSV_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with CSV_SUMMARY.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow(rows[key])


def write_params_json() -> None:
    JSON_PARAMS.parent.mkdir(parents=True, exist_ok=True)
    JSON_PARAMS.write_text(
        json.dumps(
            {
                "variant": VARIANT,
                "strategy": "recalage sur la fovee, puis flatten_rpe, puis xcorr "
                "en x sur les pixels de choroide du tiers central des A-scans, "
                "puis alignement des A-scans sur la mediane temporelle",
                "fovea_correction": FOVEA_CORRECTION,
                "max_fovea_spread": MAX_FOVEA_SPREAD,
                "max_fovea_failed_frac": MAX_FOVEA_FAILED_FRAC,
                "axial_refinement": AXIAL_REFINEMENT,
                "max_axial_shift": MAX_AXIAL_SHIFT,
                "axial_bandpass": list(AXIAL_BANDPASS),
                "flatten_config": dataclasses.asdict(FLATTEN_CONFIG),
                "seg_scale_factor": SEG_SCALE_FACTOR,
                "seg_batch_size": SEG_BATCH_SIZE,
                "col_frac": list(COL_FRAC),
                "max_lateral_shift": MAX_LATERAL_SHIFT,
                "subpixel": SUBPIXEL,
                "smooth_transversal": SMOOTH_TRANSVERSAL,
                "smooth_transversal_sigma": SMOOTH_TRANSVERSAL_SIGMA,
                "skip_first_n_frames": 0,
                "drop_last_n_frames": 0,
                "created": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Boucle principale
# --------------------------------------------------------------------------- #
def main():
    if not PATH_GENERAL.exists():
        print(f"Dossier introuvable : {PATH_GENERAL}")
        return

    conditions = list(iter_conditions())
    print(f"{len(conditions)} condition(s) trouvee(s) sous {PATH_GENERAL}")
    print(f"Sortie : {VARIANT_ROOT}\n")

    write_params_json()
    summary = load_summary()
    model = get_choroid_segmentation_model()  # telecharge au 1er appel

    n_done = n_skipped = n_error = 0
    for path_astro, path_moment, path_condi in conditions:
        key = (path_astro.name, path_moment.name, path_condi.name)
        relpath = Path(path_astro.name) / path_moment.name / path_condi.name
        frames_dir = VARIANT_ROOT / FRAMES_SUBDIR / relpath
        masks_dir = VARIANT_ROOT / MASKS_SUBDIR / relpath

        if LIMIT is not None and n_done >= LIMIT:
            break

        already = (frames_dir / "cube.mp4").exists() and (masks_dir / "mask.npz").exists()
        if already and not OVERWRITE:
            print(f"[deja fait] {relpath}")
            continue

        print(path_condi)
        t0 = time.time()
        raw_dir = find_raw_dir(path_condi)
        if raw_dir is None:
            print("  [skip] aucun dossier RawImages/RawData")
            row = {"status": "skipped", "reason": "no_raw_dir"}
        else:
            try:
                row = process_condition(raw_dir, model, frames_dir, masks_dir)
            except Exception as e:  # noqa: BLE001
                print(f"  [erreur] {e}")
                traceback.print_exc()
                row = {"status": "error", "reason": str(e)[:300]}

        if row["status"] == "ok":
            n_done += 1
            dy_med = row["dy_median_std"]
            dy_med_txt = f"{dy_med:.2f} px" if isinstance(dy_med, float) else "n/a"
            print(
                f"  -> {row['n_frames']} frames @ {row['fps']:.1f} fps | "
                f"dx in [{row['dx_min']:+.2f}, {row['dx_max']:+.2f}] px "
                f"(fovea {row['dx_fovea_ptp']:.1f}{'' if row['fovea_used'] else ' ECARTEE'}"
                f" + xcorr {row['dx_xcorr_ptp']:.1f} pp, "
                f"conf med {row['dx_conf_median']:.1f}) | "
                f"ROI x[{row['roi_x0']}:{row['roi_x1']}] y[{row['roi_y0']}:{row['roi_y1']}] | "
                f"A-scan dy std {dy_med_txt} | "
                f"{row['n_bad_columns']} colonnes noircies"
            )
        elif row["status"] == "error":
            n_error += 1
        else:
            n_skipped += 1
            print(f"  [skip] {row['reason']}")

        row.update(
            patient=path_astro.name,
            moment=path_moment.name,
            condition=path_condi.name,
            seconds=round(time.time() - t0, 1),
            created=datetime.now().isoformat(timespec="seconds"),
        )
        summary[key] = row
        write_summary(summary)

    print(
        f"\n{n_done} condition(s) recalee(s), {n_skipped} ignoree(s), "
        f"{n_error} en erreur.\nSuivi : {CSV_SUMMARY}"
    )


if __name__ == "__main__":
    main()
