"""
compute_demons_strain.py

Strain retinien et choroidien par recalage demons (SimpleITK), sur TOUTES les
conditions deja traitees par ``compute_pulse_from_data.py``. Portage du carnet
``notebook/demons_sitk_variants.ipynb`` a l'echelle de la cohorte.

Le carnet mesure, sur UNE condition, que le strain retinien croit avec la
dilatation choroidienne dans les cinq regions laterales (pentes 2,1 a 2,9 e-4
par um, r = 0,27 a 0,39), mais que l'integrale du strain choroidien ne predit
que 0,1 a 0,36 fois le dCT mesure. Une condition ne permet pas de trancher entre
"la methode ne marche pas" et "le bruit domine a cette echelle" : d'ou ce lot.

Ce qui change par rapport au carnet
-----------------------------------
  - PYRAMIDE A DEUX NIVEAUX (shrink 2 puis 1, 75 + 100 iterations) au lieu de
    trois : le niveau shrink=4 coute 20 % du temps pour une plage de capture
    dont on n'a pas besoin ici (les deplacements mesures font 2 a 7 px).
  - RECALAGE SUR UN CROP retine + choroide (~284 lignes sur 496) au lieu de
    l'image entiere : le vitre et la sclere n'apportent aucune structure a
    recaler, et le crop fait passer un recalage de 4,58 s a 2,82 s.
  - BALAYAGE de ``displacement_sigma`` sur {5, 10, 12, 15} : la regularisation
    est le seul reglage qui bouge vraiment le champ (|u|max passe de 22,5 px a
    9,7 px entre sigma 5 et 15 sur la paire temoin).
  - DEUX CHOIX DE REFERENCE, calcules cote a cote :
      * ``local``  -- l'image fixe est le bin de CT MINIMUM de CE one-cycle.
        Chaque one-cycle est mesure contre lui-meme : insensible a la derive
        lente de l'oeil, mais la reference est bruitee (un bin = 4 a 16 frames).
      * ``global`` -- l'image fixe est la MOYENNE des bins de CT minimum de tous
        les one-cycles. Reference commune donc beaucoup moins bruitee, et les
        strains de deux one-cycles deviennent comparables entre eux ; en
        contrepartie elle absorbe la derive. Les SIX bins sont alors mobiles (le
        bin de reference local y porte justement cette derive : colonne
        ``is_local_ref``).

Chaine par condition
--------------------
  1. Frames et masques DEJA recales (``_prepared_registrator``), horodatages de
     l'export XML, base de temps verifiee contre le ``.npz`` du lot.
  2. Taille du pixel lue dans le XML Spectralis -- PAS ``AXIAL_PIXEL_SIZE_MM``,
     qui vaut la moitie de la resolution reelle de ces acquisitions.
  3. Pouls et phase ``1_fir`` repris de ``pulse_from_data/traces/<slug>.npz``,
     phase recentree pour que le bin 0 couvre le maximum du pouls.
  4. Repliement par groupes de N_CYCLE battements REELS (phase deroulee), puis
     egalisation d'histogramme par A-scan et recalage des bins entre eux.
  5. CT par bin (masques replies), bin de reference = CT minimum.
  6. Retine par Otsu, crop, cinq bandes laterales (+2 a gauche, -2 a droite).
  7. Demons pour chaque (one-cycle, bin mobile, sigma, cas) -> e_yy = d(u_y)/dy,
     resume par TROIS agregats du strain RETINIEN : moyenne, mediane et p95 de
     la magnitude rendu avec son signe (cf. ``strain_p95_signe``). Le strain
     CHOROIDIEN ne subsiste que dans ``bins.csv``, ou il sert de controle de
     methode (seul tissu a posseder une dilatation connue, donc une valeur
     attendue) -- il n'est ni un marqueur, ni une abscisse.
  8. Pentes strain ~ dCT et strain ~ CT par region, par agregat et par
     one-cycle.
  9. Correlation ET covariance CT filtre FIR / pouls filtre FIR, sur DEUX
     supports : la fenetre temporelle du one-cycle, et les bins replies.

Arborescence lue
----------------
    E:/NASA_Rigidity/SegmentationVariations/<variante>/
        pulse_from_data/conditions.csv          <- conditions status == ok
        pulse_from_data/traces/<slug>.npz       <- POULS + PHASES
        registered_frames/<NN_id>/<...>_rigidity/<..._OD|OS...>/cube.mp4
        registered_masks/<NN_id>/<...>_rigidity/<..._OD|OS...>/mask.npz
    E:/SANSORI/<NN_id>/<...>_rigidity/<..._OD|OS...>/
        RawImages/ (ou RawData/)                <- horodatages + echelles (XML)

Sorties (sous ``SEGVAR_ROOT/<variante>/demons_strain/``)
-------------------------------------------------------
  - ``conditions.csv``  une ligne par condition (statut, one-cycles, um/px, crop,
    temps par etape)
  - ``one_cycles.csv``  une ligne par (slug, one_cycle) : bin de reference, CT,
    et les quatre mesures CT/pouls (r et covariance, temps et bins)
  - ``bins.csv``        une ligne par (slug, one_cycle, cas, sigma, bin) :
    dCT, CT, diagnostics du recalage (jacobien negatif, |u|max, RMS), les trois
    agregats du strain retinien et le strain choroidien median (validation)
  - ``regions.csv``     une ligne par (slug, one_cycle, cas, sigma, bin, region) :
    CT et dCT de la bande, et les trois agregats du strain RETINIEN
  - ``slopes.csv``      une ligne par (slug, one_cycle, cas, sigma, region,
    abscisse, tissu) : pente, ordonnee, r -- ``tissu`` est l'agregat du strain
  - ``<astro>/<moment>/<condition>/demons_strain.npz`` + ``_params.json``

AUCUN champ de deplacement n'est conserve (ce serait ~250 Go) : pour rouvrir une
condition et regenerer les cartes de strain, voir ``replay_demons_strain.py``.

Les tables sont reecrites apres CHAQUE condition terminee : le script est
interruptible et reprend ou il s'est arrete (``OVERWRITE`` pour tout refaire,
``LIMIT`` pour un essai sur les N premieres conditions).

Lancer (kernel pyOR, depuis la racine du depot) :
    C:/Users/transformer/anaconda3/envs/pyOR/python.exe Astronauts/compute_demons_strain.py
"""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Le lot tourne sur CPU : dix processus qui initialisent CUDA sur une seule carte
# se la disputeraient pour rien (les demons de SimpleITK sont CPU de toute
# facon, et le reste du travail par condition est marginal). La variable est
# posee AVANT torch, sinon elle n'a plus d'effet.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import SimpleITK as sitk  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy import ndimage  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Les helpers de repliement sont REPRIS des scripts de lot plutot que recopies :
# le one-cycle replie ici est alors bit a bit celui de la cohorte.
import compute_one_cycle_compare_methods as occ  # noqa: E402
from compute_rigidity_time_series import parse_eye  # noqa: E402
import compute_registration as creg  # noqa: E402

from ocularrigidity.scripts import batch_layout as layout
from ocularrigidity.consts import AXIAL_PIXEL_SIZE_MM  # noqa: E402
from ocularrigidity.motion.filters._1d import spatio_temporal_filter  # noqa: E402
from ocularrigidity.motion.one_cycle import fold_video_numba_mean  # noqa: E402
from ocularrigidity.motion.pulsation import (  # noqa: E402
    CardiacBand,
    NCycleConfig,
    NCycleReconstructor,
)
from ocularrigidity.motion.video_timeline_aligner import VideoTimelineAligner  # noqa: E402
from ocularrigidity.registration.axial.median_registration import (  # noqa: E402
    register_ascans_to_median,
)
from ocularrigidity.scripts.one_cycle.astronauts import _prepared_registrator  # noqa: E402
from ocularrigidity.scripts.registration.astronauts import (  # noqa: E402
    load_ordered_oct_series,
)

# --------------------------------------------------------------------------- #
# Parametres
# --------------------------------------------------------------------------- #
# Surchargeables par l'environnement (cf. ``scripts/batch_layout``). Ce lot
# repartit les conditions sur dix PROCESSUS : le reglage doit passer par
# l'environnement, herite par les fils, et non par une variable de module que
# le parent modifierait en memoire.
PATH_GENERAL = layout.env_path("OR_PATH_GENERAL", "E:/SANSORI")
SEGVAR_ROOT = layout.env_path("OR_SEGVAR_ROOT", "E:/NASA_Rigidity/SegmentationVariations")
MASK_VARIANT = layout.env_str("OR_VARIANT", "model1_scale_1.0_flatten_choroid_xcorr")
FRAMES_SUBDIR = "registered_frames"
MASKS_SUBDIR = "registered_masks"
PULSE_SUBDIR = "pulse_from_data"
OUTPUT_SUBDIR = "demons_strain"  # PAS "demons_sitk", occupe par les sorties du carnet

DEVICE = "cpu"  # cf. CUDA_VISIBLE_DEVICES ci-dessus
OVERWRITE = False  # True = retraiter les conditions deja "ok" dans les CSV
LIMIT = (int(os.environ["OR_LIMIT"]) if os.environ.get("OR_LIMIT")
         else None)  # int = ne traiter que les N premieres conditions (essai)

# --- Parallelisme -----------------------------------------------------------
# Le scaling threads de SimpleITK est plat au-dela de 8 (mesure : 11,24 s a 20
# threads, 12,85 s a 8, 16,44 s a 4, 23,83 s a 2 pour les 4 sigmas d'une paire) :
# c'est le parallelisme PROCESSUS qui paie. 10 x 2 donne ~4,7x le debit d'un seul
# processus a 20 threads.
# Surchargeable : le bon nombre depend de la memoire LIBRE au moment du lot, pas
# de la machine. Chaque worker importe torch, qui reserve ~2 Go d'engagement
# rien qu'en chargeant ses DLL CUDA -- meme avec CUDA_VISIBLE_DEVICES vide. Dix
# workers demandent donc ~20 Go d'engagement AU DEMARRAGE, et echouent en
# WinError 1455 (fichier de pagination insuffisant) des que la machine sert
# aussi a autre chose.
N_WORKERS = int(os.environ.get("OR_WORKERS", "10"))
ITK_THREADS = int(os.environ.get("OR_ITK_THREADS", "2"))
# La PREPARATION (cube brut + masques en memoire) culmine a ~2 Go par processus,
# la ou la phase des demons n'en garde que ~150 Mo. Sur 34 Go de RAM dont 17
# libres, dix preparations simultanees passeraient en swap -- ce qui arriverait
# precisement au demarrage, quand les dix workers partent ensemble. Un semaphore
# n'en laisse que quelques-unes de front ; le cout est nul, la preparation ne
# pesant que ~1 % du temps d'une condition (24 s contre ~35 min de demons).
MAX_CONCURRENT_PREP = 3

# --- Repliement (identique au carnet) ---------------------------------------
PULSE_METHOD = "1_fir"
N_BINS = 6
N_CYCLE = 6  # battements cardiaques MOYENNES pour faire UN one-cycle
FOLD_METHOD = "median"
PHASE_ALIGN = True
HIST_MATCH_ASCAN = True
PHASE_PROBE_BINS = 60
EDGE_CYCLES = 1.0
COL_FRAC = (0.125, 0.875)
BAND_FRAC = 0.2

# --- Recalage des bins avant repliement -------------------------------------
BIN_REGISTRATION = True
BIN_XCORR_ITERS = 2
BIN_MAX_SHIFT = 16
BIN_AXIAL = True
BIN_MAX_AXIAL_SHIFT = 7
BIN_AXIAL_BANDPASS = (0.02, 0.5)
BETWEEN_BINS = True

# --- Segmentation retine / regions ------------------------------------------
OTSU_BINS = 256
RETINA_OPENING = 3
ILM_MEDIAN = 15
N_COLUMNS = 5
CROP_MARGIN = 20  # px de marge au-dessus de l'ILM et sous le bas de la choroide

# --- Demons -----------------------------------------------------------------
METHOD = "fast_symmetric_forces"
SIGMAS = (5.0, 10.0, 12.0, 15.0)
REF_CASES = ("local", "global")
STRAIN_SIGMA = 0.0  # la regularisation est deja dans displacement_sigma

# --- Sorties ----------------------------------------------------------------
VARIANT_ROOT = SEGVAR_ROOT / MASK_VARIANT
OUT_DIR = VARIANT_ROOT / OUTPUT_SUBDIR
PULSE_DIR = VARIANT_ROOT / PULSE_SUBDIR
TRACES_DIR = PULSE_DIR / "traces"
CSV_PULSE = PULSE_DIR / "conditions.csv"
CSV_CONDITIONS = OUT_DIR / "conditions.csv"
CSV_ONE_CYCLES = OUT_DIR / "one_cycles.csv"
CSV_BINS = OUT_DIR / "bins.csv"
CSV_REGIONS = OUT_DIR / "regions.csv"
CSV_SLOPES = OUT_DIR / "slopes.csv"


# --------------------------------------------------------------------------- #
# Demons : parametres, filtre, pyramide
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DemonsParams:
    """Reglages du carnet, pyramide raccourcie a deux niveaux.

    Le niveau shrink=4 du carnet servait a rattraper de grands deplacements ;
    ici les champs mesures font 2 a 7 px, largement dans la plage de capture du
    niveau shrink=2. Le supprimer coute ~20 % du temps et ne change pas le champ
    (verifie sur la paire temoin).
    """

    shrink_factors: tuple = (2, 1)
    smoothing_sigmas: tuple = (1.0, 0.0)
    iterations: tuple = (75, 100)
    # --- regularisation -------------------------------------------------------
    displacement_sigma: float = 15.0  # elastique -- c'est LUI qu'on balaie
    smooth_displacement_field: bool = True
    update_sigma: float = 0.0  # fluide
    smooth_update_field: bool = False
    # --- diffeomorphic / fast_symmetric_forces uniquement ---------------------
    max_step_length: float = 2.0
    gradient_type: int = 0
    # --- criteres -------------------------------------------------------------
    intensity_difference_threshold: float = 0.001
    max_rms_error: float = 0.0  # 0 = pas d'arret anticipe
    # --- pre-traitement -------------------------------------------------------
    histogram_matching: bool = True
    hist_levels: int = 1024
    hist_match_points: int = 7


FILTER_CLASSES = {
    "demons": sitk.DemonsRegistrationFilter,
    "diffeomorphic": sitk.DiffeomorphicDemonsRegistrationFilter,
    "symmetric_forces": sitk.SymmetricForcesDemonsRegistrationFilter,
    "fast_symmetric_forces": sitk.FastSymmetricForcesDemonsRegistrationFilter,
}


def to_sitk(arr):
    return sitk.GetImageFromArray(np.ascontiguousarray(arr, dtype=np.float32))


def smooth_and_resample(image, shrink, sigma):
    # Un niveau de la pyramide : lissage puis reechantillonnage, l'ETENDUE
    # PHYSIQUE etant conservee (c'est elle qui rend les champs de deux niveaux
    # comparables, donc reutilisables comme champ initial).
    img = sitk.SmoothingRecursiveGaussian(image, sigma) if sigma > 0 else image
    size, spacing = image.GetSize(), image.GetSpacing()
    new_size = [max(4, int(s / float(shrink) + 0.5)) for s in size]
    new_spacing = [((s - 1) * sp) / (ns - 1) for sp, s, ns in zip(spacing, size, new_size)]
    return sitk.Resample(img, new_size, sitk.Transform(), sitk.sitkLinear,
                         image.GetOrigin(), new_spacing, image.GetDirection(),
                         0.0, image.GetPixelID())


def make_filter(name, p):
    filt = FILTER_CLASSES[name]()
    filt.SetStandardDeviations(p.displacement_sigma)
    filt.SetSmoothDisplacementField(p.smooth_displacement_field)
    filt.SetSmoothUpdateField(p.smooth_update_field)
    filt.SetUpdateFieldStandardDeviations(p.update_sigma)
    filt.SetMaximumRMSError(p.max_rms_error)
    filt.SetIntensityDifferenceThreshold(p.intensity_difference_threshold)
    # Ces deux reglages n'existent que sur diffeomorphic et fast_symmetric_forces.
    if hasattr(filt, "SetMaximumUpdateStepLength"):
        filt.SetMaximumUpdateStepLength(p.max_step_length)
    if hasattr(filt, "SetUseGradientType"):
        filt.SetUseGradientType(p.gradient_type)
    return filt


def run_demons(name, fixed_arr, moving_arr, p):
    """Un recalage. ``field`` est (h, w, 2) = (u_x, u_y), convention ITK :
    ``warped(x) = moving(x + u(x))``, donc ``e_yy > 0`` = plus epais dans le
    mobile."""
    fixed_img, moving_img = to_sitk(fixed_arr), to_sitk(moving_arr)
    if p.histogram_matching:
        matcher = sitk.HistogramMatchingImageFilter()
        matcher.SetNumberOfHistogramLevels(p.hist_levels)
        matcher.SetNumberOfMatchPoints(p.hist_match_points)
        matcher.ThresholdAtMeanIntensityOn()
        moving_img = matcher.Execute(moving_img, fixed_img)

    filt = make_filter(name, p)
    field = None
    t0 = time.perf_counter()
    for shrink, sigma, iters in zip(p.shrink_factors, p.smoothing_sigmas, p.iterations):
        f_l = smooth_and_resample(fixed_img, shrink, sigma)
        m_l = smooth_and_resample(moving_img, shrink, sigma)
        filt.SetNumberOfIterations(int(iters))
        if field is None:
            field = filt.Execute(f_l, m_l)
        else:
            field = filt.Execute(f_l, m_l, sitk.Resample(field, f_l))
    elapsed = time.perf_counter() - t0

    field = sitk.Resample(field, fixed_img)
    warped = sitk.Resample(moving_img, fixed_img,
                           sitk.DisplacementFieldTransform(sitk.Image(field)),
                           sitk.sitkLinear, 0.0)
    jac = sitk.GetArrayFromImage(sitk.DisplacementFieldJacobianDeterminant(field))
    return {
        "field": sitk.GetArrayFromImage(field),
        "warped": sitk.GetArrayFromImage(warped),
        "jacobian": jac,
        "elapsed": elapsed,
        "iterations": int(filt.GetElapsedIterations()),
        "metric": float(filt.GetMetric()),
    }


def strain_yy(field_arr, sigma=STRAIN_SIGMA):
    # d(u_y)/dy : l'axe 0 du tableau est y (les lignes), pas de 1 px. Sans
    # dimension -- u_y et y partagent l'echelle axiale.
    uy = np.asarray(field_arr[..., 1], dtype=np.float64)
    if sigma > 0:
        uy = gaussian_filter(uy, sigma)
    return np.gradient(uy, axis=0)


def strain_p95_signe(valeurs):
    """95e percentile de la MAGNITUDE du strain, rendu AVEC SON SIGNE.

    La moyenne et la mediane sont deux mesures de CENTRE, et le centre de la
    distribution du strain est domine par le biais negatif du lissage. Ce
    troisieme agregat regarde la QUEUE : les plus fortes deformations locales
    suivent-elles la dilatation, meme quand le centre ne la suit pas ?

    Le rang est pris sur ``|e_yy|`` mais la valeur rendue est SIGNEE : si la plus
    grande magnitude est negative, le p95 est negatif. Un
    ``percentile(|e_yy|, 95)`` repondrait « quelle amplitude » en perdant le sens
    de la deformation, ce qui est precisement ce qu'on veut lire ici.

    Rang le PLUS PROCHE, sans interpolation : interpoler entre deux pixels de
    signes opposes ne voudrait rien dire. C'est donc la valeur d'un pixel reel.
    """
    v = np.asarray(valeurs)
    if v.size == 0:
        return float("nan")
    a = np.abs(v)
    k = int(round(0.95 * (a.size - 1)))
    return float(v[np.argpartition(a, k)[k]])


def rms(a, b, where=None):
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    d = d if where is None else d[where]
    return float(np.sqrt((d ** 2).mean())) if d.size else float("nan")


# --------------------------------------------------------------------------- #
# Recalage des bins avant repliement (carnet, cellule 21)
# --------------------------------------------------------------------------- #
def warp_dx_dy(frames, occupancy, dx, dy, device=DEVICE, batch_size=64):
    """UN SEUL reechantillonnage pour tout le recalage du groupe : le pixel
    (y, x) de la sortie est lu dans l'image D'ORIGINE en (x - dx, y + dy[x]).
    Chaque interpolation bilineaire etant un lissage, en enchainer trois (deux
    passes en x, puis les A-scans) coute de la nettete pour rien -- mesure sur
    la condition temoin : -3,4 % de |dI/dy| en les enchainant, contre -1,3 %
    ici. La composition est exacte : dx est une translation pure, et dy est
    indexe par la colonne de SORTIE, celle-la meme ou il a ete estime."""
    n, H, W = frames.shape
    out_f = np.empty((n, H, W), dtype=np.float32)
    out_o = np.empty((n, H, W), dtype=np.float32)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    ys = torch.arange(H, device=device, dtype=torch.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        t = end - start
        f = torch.as_tensor(np.ascontiguousarray(frames[start:end])).to(device, torch.float32)
        o = torch.as_tensor(np.ascontiguousarray(occupancy[start:end])).to(device, torch.float32)
        d = torch.as_tensor(np.ascontiguousarray(dx[start:end])).to(device, torch.float32)
        sample_x = xs.view(1, 1, W).expand(t, H, W) - d.view(t, 1, 1)
        sample_y = ys.view(1, H, 1).expand(t, H, W)
        if dy is not None:
            dv = torch.as_tensor(np.ascontiguousarray(dy[start:end])).to(device, torch.float32)
            sample_y = sample_y + dv.view(t, 1, W)
        grid = torch.stack([sample_x / (W - 1) * 2 - 1,
                            sample_y / (H - 1) * 2 - 1], dim=-1)
        reg = F.grid_sample(torch.stack([o, f], dim=1), grid, mode="bilinear",
                            padding_mode="zeros", align_corners=True)
        out_o[start:end] = reg[:, 0].cpu().numpy()
        out_f[start:end] = reg[:, 1].cpu().numpy()
    return out_f, out_o


def xcorr_dx_to_reference(frames, occupancy, ref_patch, box, max_shift,
                          device=DEVICE, subpixel=True):
    """NCC en x contre un pave de REFERENCE (et non contre une frame indexee,
    comme ``creg.choroid_xcorr_dx``). Meme convention de signe : le score du
    decalage s compare frame[x] a ref[x + s], et le dx s'applique tel quel."""
    w = box[3] - box[2]
    patch = creg._roi_patches(frames, occupancy >= 0.5, box, device)
    shifts = list(range(-max_shift, max_shift + 1))
    scores = torch.empty(patch.shape[0], len(shifts), device=device)
    for i, s in enumerate(shifts):
        if s >= 0:
            a, b = patch[:, :, : w - s], ref_patch[:, s:]
        else:
            a, b = patch[:, :, -s:], ref_patch[:, : w + s]
        num = (a * b).sum(dim=(1, 2))
        den = torch.sqrt((a * a).sum(dim=(1, 2)) * (b * b).sum()) + 1e-8
        scores[:, i] = num / den
    best = scores.argmax(dim=1)
    peak = torch.tensor(shifts, device=device, dtype=torch.float32)[best]
    if subpixel:
        inner = (best > 0) & (best < len(shifts) - 1)
        if inner.any():
            i0 = best[inner]
            s_m1, s_0, s_p1 = scores[inner, i0 - 1], scores[inner, i0], scores[inner, i0 + 1]
            den = s_m1 - 2 * s_0 + s_p1
            offset = torch.where(den != 0, 0.5 * (s_m1 - s_p1) / den,
                                 torch.zeros_like(s_0)).clamp(-1.0, 1.0)
            peak[inner] += offset
    return peak


def register_group(frames, occupancy, device=DEVICE, iters=BIN_XCORR_ITERS,
                   max_shift=BIN_MAX_SHIFT, axial=BIN_AXIAL):
    """Recale entre elles les images d'un groupe (les frames d'un bin, ou les
    bins replies). Retourne (images, occupation, dx applique, ecart-type du dy)."""
    f = np.ascontiguousarray(frames, dtype=np.float32)
    o = np.ascontiguousarray(occupancy, dtype=np.float32)
    dx_total = np.zeros(f.shape[0], dtype=np.float32)
    if f.shape[0] < 2:
        return f, o, dx_total, 0.0

    box = creg.choroid_roi(o >= 0.5, max_shift=max_shift)
    # Les passes qui suivent ne servent qu'a ESTIMER : elles travaillent sur une
    # copie de travail, et le deplacement total est applique une seule fois a la
    # fin, aux images d'origine.
    f_work, o_work = f, o
    for _ in range(max(1, iters)):
        # Reference = la MOYENNE du groupe, refaite a chaque passe : c'est elle
        # qui s'affine, et c'est tout l'interet d'iterer.
        ref_patch = creg._roi_patches(f_work.mean(axis=0)[None],
                                      (o_work.mean(axis=0) >= 0.5)[None], box, device)[0]
        dx = xcorr_dx_to_reference(f_work, o_work, ref_patch, box, max_shift, device)
        dx = dx - dx.mean()  # recalage RELATIF : le groupe ne derive pas en bloc
        dx_np = dx.cpu().numpy()
        f_work, o_work = warp_dx_dy(f_work, o_work, dx_np, None, device)
        dx_total += dx_np

    dy_np, dy_std = None, 0.0
    if axial:
        # Seul le dy est repris : il est estime sur la copie DEJA recalee en x
        # (une mediane floue donnerait un dy faux), puis compose avec dx dans
        # l'unique warp ci-dessous. masks=None evite que l'occupation passe par
        # son seuil a 0,5.
        _, _, dy = register_ascans_to_median(
            f_work, None, max_vshift=BIN_MAX_AXIAL_SHIFT, subpixel=True, batch_size=64,
            device=device, bandpass=BIN_AXIAL_BANDPASS, verbose=False)
        dy_np = dy.numpy()
        dy_std = float(np.std(dy_np))

    f, o = warp_dx_dy(f, o, dx_total, dy_np, device)
    return f, o, dx_total, dy_std


# --------------------------------------------------------------------------- #
# Egalisation par A-scan et seuillage de la retine (carnet, cellules 24 et 37)
# --------------------------------------------------------------------------- #
def match_ascans(stack, reference):
    """Egalisation exacte par colonne : la valeur de rang i prend la valeur de
    rang i de la reference. Change les NIVEAUX, pas la geometrie -- l'ordre des
    intensites le long de l'A-scan est preserve."""
    ref_sorted = np.sort(np.asarray(reference, dtype=np.float32), axis=0)  # (H, W)
    out = np.empty_like(stack, dtype=np.float32)
    for i, frame in enumerate(np.asarray(stack, dtype=np.float32)):
        if not np.isfinite(frame).all():  # bin vide : rien a egaliser
            out[i] = frame
            continue
        ranks = np.argsort(np.argsort(frame, axis=0), axis=0)
        out[i] = np.take_along_axis(ref_sorted, ranks, axis=0)
    return out


def otsu_threshold(values, bins=OTSU_BINS):
    """Seuil d'Otsu : celui qui maximise la variance INTER-classe."""
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    hist, edges = np.histogram(v, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w_lo = np.cumsum(hist).astype(np.float64)
    w_hi = w_lo[-1] - w_lo
    cum = np.cumsum(hist * centers)
    ok = (w_lo > 0) & (w_hi > 0)
    mean_lo = np.divide(cum, w_lo, out=np.zeros_like(cum), where=w_lo > 0)
    mean_hi = np.divide(cum[-1] - cum, w_hi, out=np.zeros_like(cum), where=w_hi > 0)
    between = np.where(ok, w_lo * w_hi * (mean_lo - mean_hi) ** 2, -1.0)
    return float(centers[int(np.argmax(between))])


def segment_tissue(fixed_ref, roi):
    """Retine (au-dessus de la choroide, plus claire que le seuil d'Otsu) et
    repere de profondeur centre sur la RPE. Retourne
    (retina, tissue, ilm, y_rpe, depth_map)."""
    h_px, w_px = fixed_ref.shape
    rows_grid = np.arange(h_px)[:, None]

    otsu = otsu_threshold(fixed_ref)
    bright = fixed_ref > otsu
    if RETINA_OPENING:
        bright = ndimage.binary_opening(bright, np.ones((RETINA_OPENING, RETINA_OPENING)))
    choroid_top = np.where(roi.any(axis=0), roi.argmax(axis=0), h_px)
    above = bright & (rows_grid < choroid_top[None, :])

    has_tissue = above.any(axis=0)
    if not has_tissue.any():
        raise ValueError("aucun tissu clair au-dessus de la choroide : seuil d'Otsu inadapte")
    ilm = np.where(has_tissue, above.argmax(axis=0), np.nan)
    ilm = np.interp(np.arange(w_px), np.flatnonzero(has_tissue), ilm[has_tissue])
    ilm = ndimage.median_filter(ilm, size=ILM_MEDIAN, mode="nearest")
    retina = (rows_grid >= ilm[None, :]) & (rows_grid < choroid_top[None, :]) & ~roi
    tissue = retina | roi

    retina_bottom = np.where(retina.any(axis=0),
                             h_px - 1 - retina[::-1].argmax(axis=0), np.nan)
    choroid_top_f = np.where(roi.any(axis=0), roi.argmax(axis=0), np.nan).astype(float)
    y_rpe = 0.5 * (retina_bottom + choroid_top_f)
    ok_cols = np.isfinite(y_rpe)
    if not ok_cols.any():
        raise ValueError("aucune colonne ou retine et choroide se touchent")
    y_rpe = np.interp(np.arange(w_px), np.flatnonzero(ok_cols), y_rpe[ok_cols])

    # floor(x + 0.5) et NON np.rint : les deux masques etant jointifs, y_rpe
    # tombe sur un demi-entier, et l'arrondi au pair le plus proche n'atteindrait
    # qu'une profondeur sur deux.
    depth_map = np.floor(y_rpe[None, :] - rows_grid + 0.5).astype(np.int64)
    return retina, tissue, ilm, y_rpe, depth_map


def column_label(k, eye=None):
    """Bandes laterales : indice 0 au centre, et le SIGNE designe un cote de la
    RETINE, pas un cote de l'image.

    ``k`` est l'indice geometrique de la bande dans la B-scan, 0 a gauche de
    l'image. Pour un oeil DROIT, l'indice croit vers la gauche de l'image
    (+2 a gauche, -2 a droite) -- convention du carnet, cellule 40.

    Pour un oeil GAUCHE, la B-scan est le MIROIR de celle de l'oeil droit :
    la meme colonne de l'image y regarde le cote oppose de la retine. La
    numerotation est donc inversee (+2 a droite de l'image), faute de quoi
    agreger la « bande +2 » sur la cohorte melangerait deux cotes de la retine
    et une eventuelle asymetrie naso-temporale s'annulerait d'elle-meme.

    ``eye`` vaut ``"OD"``, ``"OS"`` ou ``None`` (traite comme un oeil droit,
    pour que la fonction reste utilisable hors contexte).
    """
    idx = N_COLUMNS // 2 - k
    if eye == "OS":
        idx = -idx
    return f"colonne {idx:+d}" if idx else "colonne 0 (centre)"


# --------------------------------------------------------------------------- #
# Preparation d'une condition : chargement, repliement, CT, regions, crop
# --------------------------------------------------------------------------- #
@dataclass
class Prep:
    """Tout ce qui precede les demons. Partage tel quel avec
    ``replay_demons_strain.py``, pour que le rejeu d'une condition parte
    exactement du meme one-cycle que le lot."""

    slug: str
    astro: str
    moment: str
    condition: str
    hr: float
    fs: float
    n_frames: int
    um_y: float
    um_x: float
    n_groups: int
    cycles: np.ndarray  # (n_groups * N_BINS, h, w), croppees
    mask_cycles: np.ndarray
    counts_mask: np.ndarray
    thickness: np.ndarray  # px par bin, sur toute la largeur exploitable
    th_grid: np.ndarray  # (n_groups, N_BINS), um
    ref_bins: np.ndarray  # (n_groups,) int, -1 si le one-cycle est ecarte
    fixed_global: np.ndarray
    roi: np.ndarray
    retina: np.ndarray
    tissue: np.ndarray
    column_masks: list
    band_cols: list
    cols: np.ndarray
    depth_map: np.ndarray
    crop: tuple
    ct_pulse: dict  # g -> {"r_temps", "cov_temps", "r_bins", "cov_bins", ...}
    phase_max: float
    phase_min: float
    n_beats: float
    timings: dict
    notes: list

    @property
    def eye(self):
        """``"OD"`` / ``"OS"``, lu dans le nom de condition. Sert au libelle des
        bandes : la B-scan d'un oeil gauche est le miroir de celle d'un oeil
        droit (cf. ``column_label``). Ce n'est pas un champ du constructeur pour
        que ``replay_demons_strain.py``, qui reconstruit un ``Prep``, n'ait rien
        a changer."""
        return parse_eye(self.condition)


def prepare_condition(row) -> Prep:
    astro, moment, condition = row["astro"], row["moment"], row["condition"]
    slug = row["slug"]
    notes = []
    timings = {}

    frames_path = VARIANT_ROOT / FRAMES_SUBDIR / astro / moment / condition / "cube.mp4"
    mask_path = VARIANT_ROOT / MASKS_SUBDIR / astro / moment / condition / "mask.npz"
    npz_path = TRACES_DIR / f"{slug}.npz"
    for p in (frames_path, mask_path, npz_path):
        if not p.exists():
            raise FileNotFoundError(p)

    # ``pulse_from_data`` ecrit le chemin reel dans sa table : sur une
    # arborescence plate il ne se reconstruit pas a partir du triplet, qui n'y
    # est qu'une etiquette. Le repli couvre les tables ecrites avant l'ajout de
    # la colonne.
    path_str = row.get("path")
    path_condi = (Path(path_str) if isinstance(path_str, str) and path_str
                  else layout.condition_dir(astro, moment, condition, PATH_GENERAL))
    raw_dir = occ.find_raw_dir(path_condi)
    if raw_dir is None:
        raise FileNotFoundError(f"RawImages/RawData absent : {path_condi}")

    # --- 1. chargement (rien n'est re-recale) --------------------------------
    t0 = time.perf_counter()
    registrator = _prepared_registrator(frames_path, mask_path, DEVICE, verbose=False)
    frames = np.asarray(registrator.registered_frames)
    masks = np.asarray(registrator.registered_masks, dtype=bool)
    T, H, W = frames.shape

    ts_us = occ.raw_timestamps_us(raw_dir)
    if ts_us.size != T:
        raise ValueError(f"frames ({T}) != horodatages ({ts_us.size})")
    aligner = VideoTimelineAligner(registrator, ts_us)
    t = aligner.timestamps_seconds
    u_time = aligner.uniform_time
    fs = float(aligner.fs)

    data = np.load(npz_path)
    hr = float(data["hr"])
    # La phase reprise vit sur CETTE base de temps : si elle a bouge, tout le
    # repliement serait faux sans que rien ne plante.
    if not np.allclose(t, data["t"], atol=1e-6) or not np.allclose(u_time, data["u_time"], atol=1e-6):
        raise ValueError("les horodatages du .npz ne sont pas ceux de la video chargee")

    # Colonnes exploitables : la regle de compute_one_cycle_compare_methods,
    # reprise telle quelle pour que l'epaisseur calculee ici soit LA MEME
    # grandeur que le deltaY de la cohorte.
    roi_all = masks.all(axis=0)
    if roi_all.sum() < 100:  # intersection vide : repli sur la choroide majoritaire
        roi_all = masks.mean(axis=0) > 0.5
    c0, c1 = int(COL_FRAC[0] * W), int(COL_FRAC[1] * W)
    cols = np.flatnonzero(roi_all[:, c0:c1].any(axis=0)) + c0
    if cols.size == 0:
        raise ValueError("aucune colonne exploitable")

    # --- 2. taille du pixel, lue dans l'export XML ---------------------------
    # ScaleY / ScaleX sont ecrits par le Spectralis pour CHAQUE B-scan : c'est la
    # source faisant foi. La constante du depot vise un autre mode d'acquisition
    # (1024 px de haut au lieu de 496) et vaut la MOITIE de la valeur reelle.
    series = load_ordered_oct_series(raw_dir)
    ax = [s.axial_resolution for s in series if s.axial_resolution]
    lat = [s.lateral_resolution for s in series if s.lateral_resolution]
    um_y = 1000.0 * float(np.median(ax)) if ax else 1000.0 * AXIAL_PIXEL_SIZE_MM
    um_x = 1000.0 * float(np.median(lat)) if lat else float("nan")
    if not ax:
        notes.append("echelle axiale absente du XML : REPLI sur la constante du depot")
    elif abs(um_y - 1000 * AXIAL_PIXEL_SIZE_MM) > 0.05 * um_y:
        notes.append(f"um/px axial {um_y:.3f} contre {1000 * AXIAL_PIXEL_SIZE_MM:.3f} "
                     f"pour la constante du depot (facteur "
                     f"{um_y / (1000 * AXIAL_PIXEL_SIZE_MM):.2f}) -- le XML fait foi")
    timings["t_load_s"] = round(time.perf_counter() - t0, 1)
    return _prepare_fold(row, slug, astro, moment, condition, registrator, aligner,
                         frames, masks, data, t, u_time, fs, hr, cols, um_y, um_x,
                         notes, timings)


def _prepare_fold(row, slug, astro, moment, condition, registrator, aligner,
                  frames, masks, data, t, u_time, fs, hr, cols, um_y, um_x,
                  notes, timings) -> Prep:
    """Suite de ``prepare_condition`` : phase, repliement, CT, regions, crop.

    Separee uniquement pour que chaque fonction tienne sous les yeux -- elle
    n'est jamais appelee ailleurs."""
    T, H, W = frames.shape

    # --- 3. phase reprise du lot, recentree sur le maximum du pouls ----------
    good_u = occ.edge_mask(u_time, hr) & ~aligner.gap_mask(np.zeros(T, dtype=bool))
    extractor = occ.build_extractor(
        registrator, aligner,
        data[f"pulse_{PULSE_METHOD}"], data[f"phase_{PULSE_METHOD}"], hr, good_u,
    )
    phase_raw = extractor.phase_per_frame
    good_f = extractor.good_per_frame

    # Ou tombe le maximum du pouls dans le cycle ? Mesure plutot que suppose :
    # la convention de Hilbert place theoriquement le pic en 0, mais le
    # resampling de build_track et les conventions de signe du lot peuvent
    # l'avoir deplace.
    pulse_on_frames = np.interp(extractor.timestamps_seconds, u_time,
                                np.asarray(data[f"pulse_{PULSE_METHOD}"], dtype=float))
    probe_bin = (phase_raw * PHASE_PROBE_BINS).astype(int) % PHASE_PROBE_BINS
    probe_sum = np.bincount(probe_bin[good_f], weights=pulse_on_frames[good_f],
                            minlength=PHASE_PROBE_BINS)
    probe_cnt = np.bincount(probe_bin[good_f], minlength=PHASE_PROBE_BINS).astype(float)
    probe = np.where(probe_cnt > 0, probe_sum / np.maximum(probe_cnt, 1), np.nan)
    phase_max = float((np.nanargmax(probe) + 0.5) / PHASE_PROBE_BINS)
    phase_min = float((np.nanargmin(probe) + 0.5) / PHASE_PROBE_BINS)

    # Le repliement calcule ses bins par floor(phase * n_bins), donc un bin
    # COMMENCE a une frontiere : on avance d'un demi-bin pour que la frontiere
    # devienne un centre, et que le bin 0 couvre le maximum.
    phase_f = (np.mod(phase_raw - phase_max + 0.5 / N_BINS, 1.0).astype(np.float32)
               if PHASE_ALIGN else phase_raw)

    # --- 4. groupes de N_CYCLE battements REELS ------------------------------
    # La phase deroulee compte les battements ; diviser la duree par un rythme
    # nominal supposerait une frequence cardiaque constante, qu'elle n'est pas.
    cycle_pos = np.unwrap(2 * np.pi * phase_raw) / (2 * np.pi)
    cycle_pos = cycle_pos - cycle_pos[good_f].min()
    n_beats = float(cycle_pos[good_f].max())
    group_of = np.floor(cycle_pos / N_CYCLE).astype(int)
    group_of = np.clip(group_of, 0, max(int(np.floor(n_beats / N_CYCLE)) - 1, 0))
    n_groups = int(group_of[good_f].max()) + 1

    t0 = time.perf_counter()
    reconstructor = NCycleReconstructor(
        extractor,
        NCycleConfig(n_bins=N_BINS, n_cycle=1, fold_method=FOLD_METHOD, verbose=False),
    )
    cycle_chunks, mask_chunks, count_chunks = [], [], []
    for g in range(n_groups):
        keep_g = good_f & (group_of == g)
        if int(keep_g.sum()) < N_BINS:
            cycle_chunks.append(np.full((N_BINS, H, W), np.nan, np.float32))
            mask_chunks.append(np.zeros((N_BINS, H, W), np.float32))
            count_chunks.append(np.zeros(N_BINS, np.int32))
            continue
        cyc_g, _ = reconstructor.compute(phase_per_frame=phase_f, good_per_frame=keep_g,
                                         n_cycle=1, n_bins=N_BINS)
        mc, cc = fold_video_numba_mean(masks.astype(np.float32), phase_f, keep_g,
                                       n_bins=N_BINS, verbose=False)
        cycle_chunks.append(cyc_g)
        mask_chunks.append(mc)
        count_chunks.append(cc)

    cycles = np.concatenate(cycle_chunks, axis=0)
    mask_cycles = np.concatenate(mask_chunks, axis=0)
    counts_mask = np.concatenate(count_chunks, axis=0)
    timings["t_fold_s"] = round(time.perf_counter() - t0, 1)

    # --- 5. egalisation par A-scan puis recalage des bins --------------------
    if HIST_MATCH_ASCAN:
        t0 = time.perf_counter()
        cycles = match_ascans(cycles, frames.mean(axis=0).astype(np.float32))
        timings["t_hist_s"] = round(time.perf_counter() - t0, 1)

    bin_of = (phase_f * N_BINS).astype(np.int32) % N_BINS
    slot_of = group_of * N_BINS + bin_of
    good_slot = good_f.copy()
    for g in range(n_groups):  # groupe trop pauvre : ecarte en bloc
        sel_g = group_of == g
        if int((good_f & sel_g).sum()) < N_BINS:
            good_slot &= ~sel_g

    if BIN_REGISTRATION:
        t0 = time.perf_counter()
        cycles_reg = np.full_like(cycles, np.nan, dtype=np.float32)
        mask_reg = np.zeros_like(mask_cycles, dtype=np.float32)
        for s in range(n_groups * N_BINS):
            sel = good_slot & (slot_of == s)
            if not sel.any():
                continue
            f_s, o_s, _, _ = register_group(frames[sel], masks[sel].astype(np.float32))
            cycles_reg[s] = (np.median(f_s, axis=0) if FOLD_METHOD == "median"
                             else f_s.mean(axis=0))
            mask_reg[s] = o_s.mean(axis=0)  # occupation du bin, dans [0, 1]
        if BETWEEN_BINS:
            filled = counts_mask > 0
            if filled.sum() >= 2:
                f_b, o_b, _, _ = register_group(cycles_reg[filled], mask_reg[filled])
                cycles_reg[filled] = f_b
                mask_reg[filled] = o_b
        cycles, mask_cycles = cycles_reg, mask_reg
        timings["t_binreg_s"] = round(time.perf_counter() - t0, 1)

    return _prepare_regions(row, slug, astro, moment, condition, frames, masks, data,
                            t, u_time, fs, hr, cols, um_y, um_x, cycles, mask_cycles,
                            counts_mask, n_groups, group_of, good_f, good_slot, slot_of,
                            pulse_on_frames, phase_max, phase_min, n_beats,
                            notes, timings)


def _prepare_regions(row, slug, astro, moment, condition, frames, masks, data,
                     t, u_time, fs, hr, cols, um_y, um_x, cycles, mask_cycles,
                     counts_mask, n_groups, group_of, good_f, good_slot, slot_of,
                     pulse_on_frames, phase_max, phase_min, n_beats,
                     notes, timings) -> Prep:
    """Fin de la preparation : CT par bin, bins de reference, retine, crop,
    bandes laterales, et les quatre mesures CT / pouls."""
    T, H, W = frames.shape
    t0 = time.perf_counter()

    # --- 6. CT par bin -------------------------------------------------------
    # Un bin VIDE vaut NaN, jamais 0 : un seul 0 transformerait un crete-a-crete
    # de 0,5 px en 30 px.
    thickness = np.asarray(mask_cycles, np.float64)[:, :, cols].sum(axis=1).mean(axis=1)
    thickness[counts_mask == 0] = np.nan
    th_grid = thickness.reshape(n_groups, N_BINS) * um_y

    # Bin de reference PROPRE A CHAQUE one-cycle : sa choroide la plus fine.
    ref_bins = np.full(n_groups, -1, dtype=int)
    for g in range(n_groups):
        prof = th_grid[g]
        if np.isfinite(prof).sum() >= 2:
            ref_bins[g] = int(np.nanargmin(prof))
    valid_groups = np.flatnonzero(ref_bins >= 0)
    if valid_groups.size == 0:
        raise ValueError("aucun one-cycle exploitable (tous les bins vides)")

    # Reference GLOBALE : moyenne des bins de reference de tous les one-cycles.
    # Elle sert a la fois de repere anatomique (masques, regions, crop) et
    # d'image fixe du cas "global".
    ref_slots = [int(g * N_BINS + ref_bins[g]) for g in valid_groups]
    ref_slots = [s for s in ref_slots if counts_mask[s] > 0]
    if not ref_slots:
        raise ValueError("tous les bins de reference sont vides")
    fixed_global = np.nanmean(cycles[ref_slots], axis=0).astype(np.float32)
    roi = mask_cycles[ref_slots].mean(axis=0) > 0.5
    if not roi.any():
        raise ValueError("choroide vide sur la reference globale")

    # --- 7. retine, crop, bandes laterales -----------------------------------
    retina, tissue, ilm, y_rpe, depth_map = segment_tissue(fixed_global, roi)

    # Crop : de l'ILM la plus haute au bas de la choroide, avec une marge. Le
    # vitre et la sclere n'apportent aucune structure a recaler et coutent 40 %
    # du temps des demons.
    top = int(np.floor(np.nanmin(ilm))) - CROP_MARGIN
    bottom = int(np.flatnonzero(roi.any(axis=1))[-1]) + 1 + CROP_MARGIN
    y0 = max(0, top)
    y1 = min(H, bottom)
    if y1 - y0 < 32:
        raise ValueError(f"crop degenere : {y0}-{y1}")
    cycles = np.ascontiguousarray(cycles[:, y0:y1])
    mask_cycles = np.ascontiguousarray(mask_cycles[:, y0:y1])
    fixed_global = np.ascontiguousarray(fixed_global[y0:y1])
    roi = np.ascontiguousarray(roi[y0:y1])
    retina = np.ascontiguousarray(retina[y0:y1])
    tissue = np.ascontiguousarray(tissue[y0:y1])
    depth_map = np.ascontiguousarray(depth_map[y0:y1])

    retina_cols = np.flatnonzero(retina.any(axis=0))
    if retina_cols.size == 0:
        raise ValueError("retine vide apres le crop")
    xr_lo, xr_hi = int(retina_cols[0]), int(retina_cols[-1]) + 1
    edges_x = np.linspace(xr_lo, xr_hi, N_COLUMNS + 1).round().astype(int)
    column_masks, band_cols = [], []
    for k in range(N_COLUMNS):
        sl = slice(int(edges_x[k]), int(edges_x[k + 1]))
        m = np.zeros_like(tissue, dtype=bool)
        m[:, sl] = True
        column_masks.append(m)
        # CT de la region : sur les SEULES colonnes exploitables de la bande --
        # une region fine et une region epaisse ne partagent pas la meme
        # epaisseur de depart, et c'est justement ce qu'on veut pouvoir lire.
        band = cols[(cols >= sl.start) & (cols < sl.stop)]
        band_cols.append(band if band.size else np.arange(sl.start, sl.stop))
    timings["t_regions_s"] = round(time.perf_counter() - t0, 1)

    ct_pulse = _ct_pulse_correlations(masks, data, t, u_time, fs, hr, cols,
                                      thickness, um_y, counts_mask, n_groups,
                                      group_of, good_f, good_slot, slot_of,
                                      pulse_on_frames)

    return Prep(
        slug=slug, astro=astro, moment=moment, condition=condition,
        hr=hr, fs=fs, n_frames=T, um_y=um_y, um_x=um_x, n_groups=n_groups,
        cycles=cycles, mask_cycles=mask_cycles, counts_mask=counts_mask,
        thickness=thickness, th_grid=th_grid, ref_bins=ref_bins,
        fixed_global=fixed_global, roi=roi, retina=retina, tissue=tissue,
        column_masks=column_masks, band_cols=band_cols, cols=cols,
        depth_map=depth_map, crop=(y0, y1), ct_pulse=ct_pulse,
        phase_max=phase_max, phase_min=phase_min, n_beats=n_beats,
        timings=timings, notes=notes,
    )


# --------------------------------------------------------------------------- #
# CT filtre FIR contre pouls filtre FIR, par one-cycle, sur deux supports
# --------------------------------------------------------------------------- #
def _r_cov(a, b):
    """(r de Pearson, covariance) sur les points finis communs. NaN si moins de
    trois points ou si l'une des deux series est constante."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    n = int(ok.sum())
    if n < 3:
        return np.nan, np.nan, n
    x, y = a[ok], b[ok]
    cov = float(np.cov(x, y, ddof=1)[0, 1])
    if x.std() == 0 or y.std() == 0:
        return np.nan, cov, n
    return float(np.corrcoef(x, y)[0, 1]), cov, n


def _ct_pulse_correlations(masks, data, t, u_time, fs, hr, cols, thickness, um_y,
                           counts_mask, n_groups, group_of, good_f, good_slot,
                           slot_of, pulse_on_frames):
    """Deux lectures de la meme question, par one-cycle.

    TEMPOREL -- l'epaisseur PAR FRAME (somme du masque par colonne, moyennee sur
    les colonnes exploitables), interpolee sur la grille uniforme et passee au
    MEME FIR que la methode ``1_fir`` (bande HR +/- 20 %, phase lineaire), contre
    le pouls ``pulse_1_fir``. Les deux signaux subissent donc exactement le meme
    traitement -- c'est la condition pour que la comparaison veuille dire quelque
    chose. Restreint aux echantillons de la fenetre du one-cycle ET a ``core``
    (les 20 % de bord que le lot ecarte deja, ou le FIR a son transitoire).
    Quelques dizaines a centaines de points par one-cycle.

    BINS -- l'epaisseur PAR BIN du one-cycle contre le pouls replie sur les memes
    bins. Six points seulement, donc tres bruite, mais c'est le support EXACT sur
    lequel le strain est mesure : si le strain doit suivre le pouls, c'est la
    qu'il faut regarder.

    Le carnet a montre que l'epaisseur par frame ne porte AUCUNE pulsation
    cardiaque mesurable sur au moins une condition (puissance en bande de 12,3 %
    pour une bande occupant 11,4 % de la grille) : la version temporelle est donc
    autant un temoin qu'une mesure, et un desaccord entre les deux supports est
    un resultat en soi.
    """
    thickness_t = masks[:, :, cols].sum(axis=1).mean(axis=1).astype(np.float64)
    band = CardiacBand(expected_bpm=hr, expected_bpm_band_frac=BAND_FRAC)
    lo_bpm, hi_bpm = band.effective_bpm_range
    nyq = 0.5 * fs
    low_cut = (lo_bpm / 60.0) / nyq
    high_cut = min((hi_bpm / 60.0) / nyq, 0.99)
    thickness_u = np.interp(u_time, t, thickness_t)
    # En um : la covariance a alors une unite lisible (um x unite de pouls). Le r
    # de Pearson, lui, ne change pas -- c'est un rapport.
    ct_fir = spatio_temporal_filter(thickness_u[:, None], 0.0, low_cut, high_cut,
                                    fs, None)[:, 0] * um_y
    pulse_u = np.asarray(data[f"pulse_{PULSE_METHOD}"], dtype=float)
    core = occ.edge_mask(u_time, hr)

    out = {}
    for g in range(n_groups):
        sel = good_f & (group_of == g)
        entry = {"n_frames": int(sel.sum())}
        if sel.any():
            t_lo, t_hi = float(t[sel].min()), float(t[sel].max())
            win = core & (u_time >= t_lo) & (u_time <= t_hi)
            entry["t_lo_s"] = round(t_lo, 3)
            entry["t_hi_s"] = round(t_hi, 3)
            r, cov, n = _r_cov(ct_fir[win], pulse_u[win])
            entry.update(r_temps=r, cov_temps=cov, n_temps=n)
        else:
            entry.update(t_lo_s=np.nan, t_hi_s=np.nan,
                         r_temps=np.nan, cov_temps=np.nan, n_temps=0)

        # Support "bins" : le pouls replie avec la MEME affectation (groupe, bin)
        # que les images, donc la moyenne du pouls sur les frames de ce bin.
        ct_bins = thickness[g * N_BINS:(g + 1) * N_BINS] * um_y
        pulse_bins = np.full(N_BINS, np.nan)
        for b in range(N_BINS):
            sel_b = good_slot & (slot_of == g * N_BINS + b)
            if sel_b.any():
                pulse_bins[b] = float(pulse_on_frames[sel_b].mean())
        r, cov, n = _r_cov(ct_bins, pulse_bins)
        entry.update(r_bins=r, cov_bins=cov, n_bins_used=n)
        out[g] = entry
    return out


# --------------------------------------------------------------------------- #
# Les demons : (one-cycle, bin mobile) x sigma x cas de reference
# --------------------------------------------------------------------------- #
def moving_bins_for(prep: Prep, case: str, g: int) -> list:
    """Cas ``local`` : tous les bins SAUF celui de reference, qui est l'image
    fixe. Cas ``global`` : les six, y compris le bin de reference local -- contre
    la reference commune il porte la derive d'un one-cycle a l'autre, ce qui est
    precisement l'information que ce cas apporte."""
    ref = int(prep.ref_bins[g])
    if case == "local":
        return [b for b in range(N_BINS) if b != ref]
    return list(range(N_BINS))


def fixed_for(prep: Prep, case: str, g: int) -> np.ndarray:
    if case == "local":
        return prep.cycles[g * N_BINS + int(prep.ref_bins[g])]
    return prep.fixed_global


def run_condition_demons(prep: Prep, keep_fields: bool = False):
    """Retourne (bin_rows, region_rows, fields). ``fields`` reste vide sauf en
    rejeu : garder les champs de deplacement de toute la cohorte ferait ~250 Go
    pour une information qu'un seul recalage suffit a reconstruire."""
    bin_rows, region_rows = [], []
    fields = {}
    retina, roi, tissue = prep.retina, prep.roi, prep.tissue
    um_y, um_x = prep.um_y, prep.um_x

    # La reference du cas "global" est la meme pour tous les one-cycles : son CT
    # et ses CT par region sont donc calcules UNE fois.
    gref = _global_ref_slots(prep)
    ct_ref_global = float(np.nanmean([prep.thickness[s] for s in gref])) * um_y
    ct_reg_ref_global = [
        float(np.mean([prep.mask_cycles[s][:, prep.band_cols[k]].sum(axis=0).mean()
                       for s in gref])) * um_y
        for k in range(N_COLUMNS)
    ]

    for case in REF_CASES:
        # Avec un seul one-cycle, la reference globale EST le bin de reference
        # de ce one-cycle : le cas "global" refarait a l'identique le cas
        # "local", plus un recalage degenere du bin de reference sur lui-meme
        # (champ nul, strain nul) qui viendrait fausser les agregats.
        if case == "global" and int((prep.ref_bins >= 0).sum()) < 2:
            continue
        for sigma in SIGMAS:
            params = DemonsParams(displacement_sigma=float(sigma))
            for g in range(prep.n_groups):
                if prep.ref_bins[g] < 0:
                    continue
                ref_slot = g * N_BINS + int(prep.ref_bins[g])
                if case == "local" and prep.counts_mask[ref_slot] == 0:
                    continue
                fixed_g = fixed_for(prep, case, g)
                ct_ref = (float(prep.th_grid[g, prep.ref_bins[g]])
                          if case == "local" else ct_ref_global)
                for b in moving_bins_for(prep, case, g):
                    slot = g * N_BINS + b
                    if prep.counts_mask[slot] == 0:
                        continue
                    moving = prep.cycles[slot]
                    if not np.isfinite(moving).all() or not np.isfinite(fixed_g).all():
                        continue
                    res = run_demons(METHOD, fixed_g, moving, params)
                    u = res["field"]
                    strain = strain_yy(u).astype(np.float32)
                    ct_um = float(prep.th_grid[g, b])
                    d_ct = ct_um - ct_ref

                    bin_rows.append({
                        "slug": prep.slug, "one_cycle": g + 1, "cas": case,
                        "sigma": float(sigma), "bin": int(b),
                        "is_local_ref": bool(b == int(prep.ref_bins[g])),
                        "ref_bin": int(prep.ref_bins[g]),
                        "n_frames_bin": int(prep.counts_mask[slot]),
                        "CT_um": round(ct_um, 3), "dCT_um": round(d_ct, 3),
                        "rms_avant": round(rms(fixed_g, moving, tissue), 3),
                        "rms_apres": round(rms(fixed_g, res["warped"], tissue), 3),
                        "u_max_um": round(float(np.hypot(u[..., 0] * um_x,
                                                         u[..., 1] * um_y).max()), 2),
                        "jac_neg_pct": round(100 * float((res["jacobian"] <= 0).mean()), 4),
                        "strain_retine_moy": round(float(np.mean(strain[retina])), 6),
                        "strain_retine_med": round(float(np.median(strain[retina])), 6),
                        "strain_retine_p95": round(strain_p95_signe(strain[retina]), 6),
                        # Le strain CHOROIDIEN ne subsiste QUE sur cette ligne, et
                        # pour une seule raison : la reference etant le bin de
                        # choroide la plus fine, la choroide est le seul tissu
                        # dont la dilatation est connue, donc le seul a posseder
                        # une valeur de strain ATTENDUE. C'est le controle de
                        # methode de la page « Stress-strain », jamais un
                        # marqueur -- il est absent des regions, des pentes et
                        # des predicteurs.
                        "strain_choroide_med": round(float(np.median(strain[roi])), 6),
                        "iterations": res["iterations"],
                        "metric": round(res["metric"], 6),
                        "temps_s": round(res["elapsed"], 2),
                    })

                    for k in range(N_COLUMNS):
                        bc = prep.band_cols[k]
                        # Rétine seule : le strain choroidien est un controle de
                        # methode (cf. bin_rows), pas un marqueur, et ne descend
                        # donc pas au niveau des bandes.
                        m_ret = retina & prep.column_masks[k]
                        ct_reg = float(prep.mask_cycles[slot][:, bc].sum(axis=0).mean()) * um_y
                        ct_reg_ref = (
                            float(prep.mask_cycles[ref_slot][:, bc].sum(axis=0).mean()) * um_y
                            if case == "local" else ct_reg_ref_global[k])
                        region_rows.append({
                            "slug": prep.slug, "one_cycle": g + 1, "cas": case,
                            "sigma": float(sigma), "bin": int(b),
                            # ``k`` reste l'indice GEOMETRIQUE de la bande
                            # (0 a gauche de l'image) ; ``region`` porte le
                            # libelle anatomique, mire pour l'oeil gauche.
                            "region": column_label(k, prep.eye), "k": k,
                            "CT_um": round(ct_um, 3), "dCT_um": round(d_ct, 3),
                            "CT_region_um": round(ct_reg, 3),
                            "dCT_region_um": round(ct_reg - ct_reg_ref, 3),
                            # Les TROIS agregats du strain retinien, au meme rang :
                            # deux mesures de centre et une mesure de queue.
                            "strain_retine": round(float(np.mean(strain[m_ret])), 7)
                            if m_ret.any() else np.nan,
                            "strain_retine_med": round(float(np.median(strain[m_ret])), 7)
                            if m_ret.any() else np.nan,
                            "strain_retine_p95": round(strain_p95_signe(strain[m_ret]), 7)
                            if m_ret.any() else np.nan,
                        })

                    if keep_fields:
                        fields[(case, float(sigma), g, int(b))] = {
                            "u_x": u[..., 0].astype(np.float32),
                            "u_y": u[..., 1].astype(np.float32),
                            "strain": strain,
                            "jacobian": res["jacobian"].astype(np.float32),
                            "warped": res["warped"].astype(np.float32),
                            "fixed": np.asarray(fixed_g, np.float32),
                        }
    return bin_rows, region_rows, fields


def _global_ref_slots(prep: Prep) -> list:
    return [int(g * N_BINS + prep.ref_bins[g]) for g in range(prep.n_groups)
            if prep.ref_bins[g] >= 0 and prep.counts_mask[g * N_BINS + prep.ref_bins[g]] > 0]


# --------------------------------------------------------------------------- #
# Pentes strain ~ abscisse, par region ET par one-cycle
# --------------------------------------------------------------------------- #
SLOPE_ABSCISSAS = ("dCT_um", "CT_um", "dCT_region_um", "CT_region_um")
# Les trois agregats du strain RETINIEN. Le choroidien n'y figure pas : il ne
# sert que de controle de methode a la validation (cf. bin_rows).
SLOPE_TISSUES = ("strain_retine", "strain_retine_med", "strain_retine_p95")


def compute_slopes(region_rows: list, slug: str) -> list:
    """Pente et correlation de ``strain ~ abscisse``, ajustees SEPAREMENT pour
    chaque (one-cycle, cas, sigma, region, abscisse, tissu).

    Le carnet ajustait sur tous les one-cycles a la fois ; ici l'ajustement est
    PAR one-cycle, parce que c'est le one-cycle qui est l'unite de repetition de
    l'analyse aval (repetabilite, prediction de SANS). Cela ne laisse que 5 ou 6
    points par ajustement : la pente d'un seul one-cycle n'est pas un resultat,
    c'est leur distribution qui en est un.

    Quatre abscisses, deux questions differentes. ``dCT`` demande si la retine
    suit le GONFLEMENT (difference, centree sur zero, commune aux regions) ;
    ``CT`` si le strain depend de l'EPAISSEUR sous-jacente (absolue, qui separe
    les regions). Les variantes ``_region`` posent la meme question avec
    l'epaisseur PROPRE a la bande, la seule que cette bande voit reellement.
    """
    if not region_rows:
        return []
    df = pd.DataFrame(region_rows)
    rows = []
    keys = ["one_cycle", "cas", "sigma", "region", "k"]
    for (one_cycle, cas, sigma, region, k), sub in df.groupby(keys, sort=True):
        for xcol in SLOPE_ABSCISSAS:
            x = sub[xcol].to_numpy(dtype=float)
            for ycol in SLOPE_TISSUES:
                y = sub[ycol].to_numpy(dtype=float)
                ok = np.isfinite(x) & np.isfinite(y)
                n = int(ok.sum())
                slope = intercept = r = np.nan
                if n >= 3 and np.std(x[ok]) > 0:
                    slope, intercept = np.polyfit(x[ok], y[ok], 1)
                    if np.std(y[ok]) > 0:
                        r = float(np.corrcoef(x[ok], y[ok])[0, 1])
                rows.append({
                    "slug": slug, "one_cycle": int(one_cycle), "cas": cas,
                    "sigma": float(sigma), "region": region, "k": int(k),
                    "abscisse": xcol, "tissu": ycol,
                    "pente_par_um": float(slope), "ordonnee": float(intercept),
                    "r": float(r), "n": n,
                })
    return rows


def one_cycle_rows(prep: Prep) -> list:
    """Une ligne par one-cycle : sa reference, son CT, et les quatre mesures
    CT / pouls (r et covariance sur les deux supports)."""
    rows = []
    for g in range(prep.n_groups):
        prof = prep.th_grid[g]
        cp = prep.ct_pulse.get(g, {})
        rows.append({
            "slug": prep.slug, "astro": prep.astro, "moment": prep.moment,
            "condition": prep.condition, "one_cycle": g + 1,
            "ref_bin": int(prep.ref_bins[g]),
            "n_bins_remplis": int((prep.counts_mask[g * N_BINS:(g + 1) * N_BINS] > 0).sum()),
            "n_frames_min_bin": int(prep.counts_mask[g * N_BINS:(g + 1) * N_BINS].min()),
            "n_frames_max_bin": int(prep.counts_mask[g * N_BINS:(g + 1) * N_BINS].max()),
            "CT_moyen_um": round(float(np.nanmean(prof)), 3),
            "CT_min_um": round(float(np.nanmin(prof)), 3) if np.isfinite(prof).any() else np.nan,
            "CT_max_um": round(float(np.nanmax(prof)), 3) if np.isfinite(prof).any() else np.nan,
            "deltaCT_um": round(float(np.nanmax(prof) - np.nanmin(prof)), 3)
            if np.isfinite(prof).any() else np.nan,
            "r_temps": cp.get("r_temps", np.nan),
            "cov_temps": cp.get("cov_temps", np.nan),
            "n_temps": cp.get("n_temps", 0),
            "r_bins": cp.get("r_bins", np.nan),
            "cov_bins": cp.get("cov_bins", np.nan),
            "n_bins_used": cp.get("n_bins_used", 0),
            "t_lo_s": cp.get("t_lo_s", np.nan),
            "t_hi_s": cp.get("t_hi_s", np.nan),
            "n_frames": cp.get("n_frames", 0),
        })
    return rows


# --------------------------------------------------------------------------- #
# Traitement d'une condition (ce que chaque worker execute)
# --------------------------------------------------------------------------- #
def _params_dict() -> dict:
    return {
        "mask_variant": MASK_VARIANT, "pulse_method": PULSE_METHOD,
        "n_bins": N_BINS, "n_cycle": N_CYCLE, "fold_method": FOLD_METHOD,
        "phase_align": PHASE_ALIGN, "hist_match_ascan": HIST_MATCH_ASCAN,
        "edge_cycles": EDGE_CYCLES, "col_frac": list(COL_FRAC), "band_frac": BAND_FRAC,
        "bin_registration": BIN_REGISTRATION, "bin_xcorr_iters": BIN_XCORR_ITERS,
        "bin_max_shift": BIN_MAX_SHIFT, "bin_axial": BIN_AXIAL,
        "bin_max_axial_shift": BIN_MAX_AXIAL_SHIFT,
        "bin_axial_bandpass": list(BIN_AXIAL_BANDPASS), "between_bins": BETWEEN_BINS,
        "otsu_bins": OTSU_BINS, "retina_opening": RETINA_OPENING,
        "ilm_median": ILM_MEDIAN, "n_columns": N_COLUMNS, "crop_margin": CROP_MARGIN,
        "demons_method": METHOD, "sigmas": list(SIGMAS), "ref_cases": list(REF_CASES),
        "strain_sigma": STRAIN_SIGMA, "demons_params": asdict(DemonsParams()),
    }


def save_payload(prep: Prep) -> str:
    """Le .npz par condition : tout ce qui permet de relire l'analyse SANS les
    champs de deplacement (profils de profondeur du strain, geometrie, CT).
    Les champs eux-memes se regenerent avec ``replay_demons_strain.py``."""
    out_rel = f"{prep.astro}/{prep.moment}/{prep.condition}"
    out_dir = OUT_DIR / prep.astro / prep.moment / prep.condition
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "slug": prep.slug, "astro": prep.astro, "moment": prep.moment,
        "condition": prep.condition, "hr": prep.hr, "fs": prep.fs,
        "um_per_px_y": prep.um_y, "um_per_px_x": prep.um_x,
        "n_groups": prep.n_groups, "n_bins": N_BINS, "n_cycle": N_CYCLE,
        "th_grid_um": prep.th_grid.astype(np.float32),
        "thickness_px": prep.thickness.astype(np.float32),
        "counts_mask": prep.counts_mask.astype(np.int32),
        "ref_bins": prep.ref_bins.astype(np.int16),
        "crop": np.asarray(prep.crop, dtype=np.int32),
        "cols": prep.cols.astype(np.int32),
        "roi": prep.roi, "retina": prep.retina,
        "depth_map": prep.depth_map.astype(np.int16),
        "fixed_global": prep.fixed_global.astype(np.float32),
        "phase_max": prep.phase_max, "phase_min": prep.phase_min,
        "n_beats": prep.n_beats,
        "column_edges": np.asarray([int(np.flatnonzero(m.any(axis=0))[0])
                                    for m in prep.column_masks] +
                                   [int(np.flatnonzero(prep.column_masks[-1].any(axis=0))[-1]) + 1],
                                  dtype=np.int32),
    }
    np.savez_compressed(out_dir / "demons_strain.npz", **payload)
    (out_dir / "demons_strain_params.json").write_text(
        json.dumps(_params_dict(), indent=2, default=str), encoding="utf-8")
    return out_rel


def process_condition(row: dict, sem=None) -> dict:
    """Une condition, de bout en bout. Retourne les quatre listes de lignes plus
    la ligne de la table des conditions.

    ``sem`` limite le nombre de PREPARATIONS simultanees (cf.
    ``MAX_CONCURRENT_PREP``) : c'est la seule etape gourmande en memoire."""
    t_start = time.perf_counter()
    if sem is not None:
        with sem:
            prep = prepare_condition(row)
    else:
        prep = prepare_condition(row)

    t0 = time.perf_counter()
    bin_rows, region_rows, _ = run_condition_demons(prep)
    prep.timings["t_demons_s"] = round(time.perf_counter() - t0, 1)

    slope_rows = compute_slopes(region_rows, prep.slug)
    oc_rows = one_cycle_rows(prep)
    out_rel = save_payload(prep)

    binf = pd.DataFrame(bin_rows) if bin_rows else pd.DataFrame()
    ocf = pd.DataFrame(oc_rows) if oc_rows else pd.DataFrame()
    condition_row = {
        "slug": prep.slug, "astro": prep.astro, "moment": prep.moment,
        "condition": prep.condition, "hr_BPM": prep.hr, "fs_Hz": round(prep.fs, 4),
        "n_frames": prep.n_frames, "n_beats": round(prep.n_beats, 1),
        "n_one_cycles": prep.n_groups,
        "n_one_cycles_ok": int((prep.ref_bins >= 0).sum()),
        "n_recalages": len(bin_rows),
        "um_per_px_y": round(prep.um_y, 4), "um_per_px_x": round(prep.um_x, 4),
        "crop_y0": prep.crop[0], "crop_y1": prep.crop[1],
        "crop_rows": prep.crop[1] - prep.crop[0],
        "n_cols": int(prep.cols.size),
        "roi_pixels": int(prep.roi.sum()), "retina_pixels": int(prep.retina.sum()),
        "phase_max": round(prep.phase_max, 4), "phase_min": round(prep.phase_min, 4),
        "jac_neg_pct_max": round(float(binf["jac_neg_pct"].max()), 4) if len(binf) else np.nan,
        "u_max_um_median": round(float(binf["u_max_um"].median()), 2) if len(binf) else np.nan,
        "rms_gain_median": round(float((binf["rms_avant"] - binf["rms_apres"]).median()), 3)
        if len(binf) else np.nan,
        "r_temps_median": round(float(ocf["r_temps"].median()), 4) if len(ocf) else np.nan,
        "r_bins_median": round(float(ocf["r_bins"].median()), 4) if len(ocf) else np.nan,
        "out_rel": out_rel,
        "notes": " | ".join(prep.notes),
        "status": "ok",
    }
    condition_row.update(prep.timings)
    condition_row["t_total_s"] = round(time.perf_counter() - t_start, 1)
    return {"condition": [condition_row], "one_cycle": oc_rows, "bin": bin_rows,
            "region": region_rows, "slope": slope_rows}


# --------------------------------------------------------------------------- #
# Persistance
# --------------------------------------------------------------------------- #
TABLES = {
    "condition": CSV_CONDITIONS,
    "one_cycle": CSV_ONE_CYCLES,
    "bin": CSV_BINS,
    "region": CSV_REGIONS,
    "slope": CSV_SLOPES,
}


def load_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def save_table(rows: list, path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
_PREP_SEM = None


def _init_worker(itk_threads: int, prep_sem) -> None:
    """Chaque processus se limite a ``itk_threads``.

    Mesure sur une paire de bins replies (238 x 768, pyramide 2 niveaux, les 4
    sigmas) : 21,4 s a 20 threads, 15,9 s a 8, 16,9 s a 4, 21,4 s a 2. Le debit
    d'un LOT se lit en revanche par PROCESSUS : 1 x 20 threads donne 0,047
    recalage/s, 2 x 8 en donne 0,126, 5 x 4 en donne 0,296 et 10 x 2 en donne
    0,468. Un fil de plus ne sert presque a rien sur une image de cette taille,
    un processus de plus sert pleinement -- d'ou 10 x 2."""
    global _PREP_SEM
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(int(itk_threads))
    torch.set_num_threads(int(itk_threads))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(itk_threads)
    _PREP_SEM = prep_sem



def _worker(row: dict) -> tuple:
    """Retourne (slug, resultat, message d'erreur). Une condition qui echoue ne
    doit jamais arreter le lot."""
    try:
        return row["slug"], process_condition(row, sem=_PREP_SEM), ""
    except Exception as exc:  # noqa: BLE001 - une condition ne doit pas tout arreter
        return row["slug"], None, f"{exc}\n{traceback.format_exc()}"


# --------------------------------------------------------------------------- #
# Boucle principale
# --------------------------------------------------------------------------- #
def main() -> None:
    t_start = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"sortie   : {OUT_DIR}")
    print(f"variante : {MASK_VARIANT}   device : {DEVICE}   SimpleITK {sitk.Version.VersionString()}")
    print(f"demons   : {METHOD}, pyramide {DemonsParams().shrink_factors} "
          f"({DemonsParams().iterations} iterations), sigma {list(SIGMAS)}, "
          f"cas {list(REF_CASES)}")
    print(f"repliement : {N_CYCLE} battements par one-cycle, {N_BINS} bins, {FOLD_METHOD}")
    print(f"workers  : {N_WORKERS} x {ITK_THREADS} threads ITK")

    if not CSV_PULSE.exists():
        raise FileNotFoundError(f"{CSV_PULSE} absent : lancer compute_pulse_from_data.py")
    conditions = pd.read_csv(CSV_PULSE)
    conditions = conditions[conditions["status"] == "ok"].reset_index(drop=True)

    rows = {}
    for key, path in TABLES.items():
        df = load_table(path)
        rows[key] = df.to_dict("records") if not df.empty else []

    if OVERWRITE:
        done = set()
        rows = {k: [] for k in rows}
    else:
        done = {r["slug"] for r in rows["condition"] if r.get("status") == "ok"}
        # Une ligne en ECHEC est du travail a refaire, pas un resultat : elle est
        # retiree de la table et la condition sera retentee, sinon un correctif
        # n'atteindrait jamais les conditions qu'il etait cense debloquer.
        n_retry = len(rows["condition"]) - len(done)
        rows["condition"] = [r for r in rows["condition"] if r.get("status") == "ok"]
        if n_retry:
            print(f"{n_retry} condition(s) en echec seront retentees")

    todo = [r for _, r in conditions.iterrows() if r["slug"] not in done]
    if LIMIT is not None:
        todo = todo[:LIMIT]
    print(f"\n{len(conditions)} condition(s) exploitable(s), {len(done)} deja faite(s), "
          f"{len(todo)} a traiter\n")
    if not todo:
        print("rien a faire")
        return

    payloads = [dict(r) for r in todo]
    n_ok = n_fail = 0
    manager = multiprocessing.Manager()
    prep_sem = manager.Semaphore(MAX_CONCURRENT_PREP)
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_worker,
                             initargs=(ITK_THREADS, prep_sem)) as pool:
        futures = {pool.submit(_worker, p): p["slug"] for p in payloads}
        for i, fut in enumerate(as_completed(futures), start=1):
            slug, result, err = fut.result()
            if result is None:
                n_fail += 1
                print(f"[{i:>3d}/{len(payloads)}] {slug}  ECHEC")
                print(f"        {err.splitlines()[0] if err else '?'}")
                src = conditions[conditions["slug"] == slug].iloc[0]
                rows["condition"].append({
                    "slug": slug, "astro": src["astro"], "moment": src["moment"],
                    "condition": src["condition"],
                    "status": f"echec : {err.splitlines()[0] if err else '?'}",
                })
            else:
                n_ok += 1
                cr = result["condition"][0]
                print(f"[{i:>3d}/{len(payloads)}] {slug}  "
                      f"{cr['n_one_cycles_ok']} one-cycles, {cr['n_recalages']} recalages, "
                      f"crop {cr['crop_rows']} lignes, "
                      f"jac_neg max {cr['jac_neg_pct_max']:.3f} %  "
                      f"[{cr['t_total_s']:.0f} s]")
                if cr["notes"]:
                    print(f"        note : {cr['notes']}")
                for key in TABLES:
                    rows[key].extend(result[key])
            for key, path in TABLES.items():
                save_table(rows[key], path)

    dt = time.perf_counter() - t_start
    print(f"\ntermine {datetime.now():%Y-%m-%d %H:%M} : {n_ok} ok, {n_fail} en echec  "
          f"({dt / 3600:.2f} h)")
    for key, path in TABLES.items():
        print(f"  {path}  ({len(rows[key])} lignes)")


if __name__ == "__main__":
    main()
