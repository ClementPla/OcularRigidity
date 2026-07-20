"""
compute_rigidity_full_pipeline.py

Pipeline complet, par condition SANSORI, PARTANT DES IMAGES BRUTES (.tif +
export XML Spectralis) : segmentation -> recalage -> RESEGMENTATION de la
video DEJA recalee (``resegment_registered_video`` : remplace le masque
transporte/interpole par le recalage par un masque frais, directement sur les
frames alignees) -> rigidite oculaire k (Sayah et al. 2020), calculee de TROIS
facons a partir du MEME recalage ET du MEME masque resegmente :

  1. "toute la video, Hilbert" : deltaY brut (masque) par frame -> passe-bande
     Butterworth + enveloppe de Hilbert, centre sur la HR (Methode 1 de
     compute_rigidity_time_series.py) -> amplitude par A-scan -> k.
  2. "toute la video, package"  : MEME video recalee, mais via
     ``MaskPulseExtractor`` (motion.pulsation, contenu du package plutot
     qu'un filtrage ad hoc) -> ``filtered_signal`` (passe-bande FIR
     zero-phase, spatial + temporel — pas de Hilbert) restreint aux
     echantillons a phase IQ fiable (``phase_per_frame`` / ``good_uniform`` —
     pas la phase peak-locked, mieux adaptee au repliement one-cycle qu'a une
     lecture continue) -> amplitude crete-a-crete -> k. Deuxieme avis,
     independant de la Methode 1, sur le meme signal.
  3. "one-cycle"                : la video recalee est repliee sur un
     battement cardiaque moyen (motion.pulsation, SNR ameliore) ; segmentation
     fraiche de cette video repliee (le repliement ne conserve pas les
     masques) -> amplitude crete-a-crete DIRECTE par A-scan (le repliement a
     deja isole la composante cardiaque) -> k.

N'affiche ni n'enregistre AUCUNE figure : chaque condition ecrit un
``rigidity_full_pipeline.npz`` (a cote de la video recalee) contenant tout ce
qu'il faut pour tracer les figures depuis un notebook separe. Une table CSV
recapitulative (les trois ``k`` par condition) est ecrite a la fin.

Ne reutilise QUE des briques deja presentes dans le projet :
  - segmentation + recalage : ``scripts.registration.astronauts.export_registered_video``
    (``VideoRegistrator``, meme moteur que ``testing_app/``), avec la
    ``RegistrationConfig`` GLOBALE (``pipeline_config.REGISTRATION`` — reglee et
    persistee via ``testing_app/first_cc_registration.py``) ;
  - Methode 2 : ``motion.pulsation.MaskPulseExtractor``/``PulseExtractionConfig``
    + ``scripts.one_cycle.astronauts._prepared_registrator`` (charge la video
    DEJA recalee dans un ``VideoRegistrator`` sans re-recaler, meme pattern que
    ``export_one_cycle_video``) ;
  - repliement one-cycle : ``scripts.one_cycle.astronauts.export_one_cycle_video``
    (``motion.pulsation`` : ``MaskPulseExtractor`` + ``NCycleReconstructor``) ;
  - epaisseur choroidienne : ``thickness.features.compute_deltaY_masks`` ;
  - donnees cliniques (HR/IOP/OPA/AL), Methode 1 (passe-bande + Hilbert) et la
    formule de Sayah (``sayah_k``) : ``compute_rigidity_time_series.py``, meme
    dossier.

Arborescence attendue (identique aux scripts freres) :
    E:/SANSORI/<NN_id>/<...>_rigidity/<...>_rigidity_<OD|OS><rep?>/
        RawImages/ (ou RawData/)        <- .tif + export .xml (entree)
        RawImages/registered/           <- sorties (video, masques, npz)
        Data Files/visit_data.csv       <- HR, IOP, OPA
"""

from __future__ import annotations

import csv
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d

from ocularrigidity.pipeline_config import REGISTRATION
from ocularrigidity.segmentation.utils import get_choroid_segmentation_model
from ocularrigidity.segmentation.inference import infer
from ocularrigidity.data.io import load_mask, save_mask
from ocularrigidity.data.compression import read_gray
from ocularrigidity.thickness.features import compute_deltaY_masks
from ocularrigidity.motion.pulsation import MaskPulseExtractor, PulseExtractionConfig
from ocularrigidity.motion.video_timeline_aligner import VideoTimelineAligner
from ocularrigidity.scripts.registration.astronauts import (
    export_registered_video,
    DEFAULT_OUTPUT_SUBDIR,
)
from ocularrigidity.scripts.one_cycle.astronauts import (
    export_one_cycle_video,
    DEFAULT_ONE_CYCLE_NAME,
    _prepared_registrator,
)

from compute_rigidity_time_series import (
    PATH_GENERAL,
    PIX_Y_SIZE,
    HR_HALF_BAND_BPM,
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Segmentation de la choroide : meme echelle que les autres scripts Astronauts.
SEG_SCALE_FACTOR = 2.0
SEG_BATCH_SIZE = 8

# Re-traiter une condition deja recalee / repliee (couteux en GPU -> False par
# defaut). Le calcul de rigidite lui-meme est toujours refait (rapide).
OVERWRITE_REGISTRATION = False
OVERWRITE_ONE_CYCLE = False
OVERWRITE_RESEGMENTATION = False

OUTPUT_CSV = PATH_GENERAL / "rigidity_full_pipeline.csv"


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


# --------------------------------------------------------------------------- #
# Resegmentation de la video DEJA recalee (remplace le masque transporte)
# --------------------------------------------------------------------------- #
def resegment_registered_video(
    reg_dir: Path, model, device: str, overwrite: bool = False
) -> np.ndarray:
    """Resegmente ``registered_video.mp4`` (deja recalee) plutot que de reutiliser
    ``mask.npz`` (le masque BRUT transporte geometriquement par le recalage —
    interpolation bilineaire cumulee sur plusieurs passes : laterale, verticale,
    A-scan, fovea). Segmenter APRES coup donne des frontieres nettes,
    directement sur les frames deja alignees, sans ces artefacts d'interpolation.

    Sauvegarde le resultat dans ``mask_resegmented.npz`` (a cote de
    ``mask.npz``, laisse intact) et le retourne pour usage immediat — le reste
    de la pipeline (Methodes 1 et 2) utilise CE masque, pas l'original. Si ce
    fichier existe deja et ``overwrite`` est False, le recharge tel quel plutot
    que de refaire l'inference GPU (meme convention que ``overwrite`` sur
    ``export_registered_video``/``export_one_cycle_video``).
    """
    mask_path = reg_dir / "mask_resegmented.npz"
    if mask_path.exists() and not overwrite:
        return np.asarray(load_mask(mask_path), dtype=bool)

    frames = read_gray(str(reg_dir / "registered_video.mp4"))  # (T, H, W) uint8
    mask = np.asarray(
        infer(model, frames, scale_factor=SEG_SCALE_FACTOR, batch_size=SEG_BATCH_SIZE, device=device),
        dtype=bool,
    )
    save_mask(mask, mask_path)
    return mask


# --------------------------------------------------------------------------- #
# Rigidite "toute la video", Methode 1 : deltaY brut -> passe-bande + Hilbert
# --------------------------------------------------------------------------- #
def rigidity_from_full_video_hilbert(
    reg_dir: Path, mask: np.ndarray, hr: float, AL: float, IOP: float, OPA: float
) -> dict:
    """Rigidite k (Sayah) a partir de TOUTE la video recalee — Methode 1.

    ``registered_video.mp4`` + ``timestamp.txt`` (produits par
    ``export_registered_video``) + ``mask`` (resegmente, cf.
    ``resegment_registered_video``) -> deltaY par frame -> passe-bande
    Butterworth + enveloppe de Hilbert (Methode 1, bande centree sur ``hr``)
    -> amplitude par A-scan -> k.
    """
    deltaY = compute_deltaY_masks(mask)  # (T, W) px

    ts_us = np.loadtxt(reg_dir / "timestamp.txt", dtype=np.int64)
    ts = (ts_us - ts_us[0]) / 1e6  # secondes, origine au premier frame

    order = np.argsort(ts)
    ts_sorted = ts[order]
    dY = deltaY[order] - deltaY[order].mean(0)

    band = ((hr - HR_HALF_BAND_BPM) / 60.0, (hr + HR_HALF_BAND_BPM) / 60.0)
    fc0 = 0.5 * (band[0] + band[1])
    t1, A1, f_hil, _S1 = method1_bandpass_hilbert(ts_sorted, dY, band, fc0)

    As_sorted = np.array([np.interp(ts_sorted, t1, A1[:, i]) for i in range(A1.shape[1])]).T
    amp1 = 2.0 * np.median(As_sorted, axis=0)  # amplitude crete-a-crete par A-scan (px)
    deltaCT_mm = np.median(amp1) * PIX_Y_SIZE
    CT_mm = np.median(deltaY) * PIX_Y_SIZE
    k, deltaV = sayah_k(deltaCT_mm, CT_mm, AL, IOP, OPA)

    return {
        "ts": t1,
        "amp_per_ascan": amp1,
        "f_hil_bpm": f_hil * 60.0,
        "deltaCT_mm": deltaCT_mm,
        "CT_mm": CT_mm,
        "k": k,
        "deltaV": deltaV,
    }


# --------------------------------------------------------------------------- #
# Rigidite "toute la video", Methode 2 : MaskPulseExtractor (contenu du package)
# --------------------------------------------------------------------------- #
def rigidity_from_full_video_package(
    reg_dir: Path, mask_path: Path, hr: float, AL: float, IOP: float, OPA: float, device: str
) -> dict:
    """Rigidite k (Sayah) a partir de TOUTE la video recalee — Methode 2.

    Meme video que la Methode 1 (et meme ``mask_path`` resegmente, cf.
    ``resegment_registered_video``), mais via ``MaskPulseExtractor`` (deja
    dans le package) plutot qu'un filtrage ad hoc :
      - ``_prepared_registrator`` charge la video DEJA recalee (pas de
        re-recalage), meme pattern que ``export_one_cycle_video`` ;
      - ``filtered_signal`` : passe-bande FIR zero-phase (spatial + temporel,
        ``firwin``/``filtfilt``) — PAS Butterworth+Hilbert ;
      - phase IQ (``phase_per_frame``, PAS peak-locked) : sert uniquement a
        obtenir ``good_uniform``, un masque de fiabilite continu (densite de
        phase), plus adapte a une lecture globale du signal filtre qu'une
        phase peak-locked (pensee pour decouper des cycles individuels).

    Amplitude par A-scan = crete-a-crete de ``filtered_signal``, restreinte
    aux echantillons ou ``good_uniform`` est vrai.
    """
    registrator = _prepared_registrator(
        reg_dir / "registered_video.mp4", mask_path, device, verbose=False
    )
    aligner = VideoTimelineAligner(registrator, str(reg_dir / "timestamp.txt"))
    config = PulseExtractionConfig(
        expected_bpm=float(hr), col_slice=slice(100, 924), verbose=False
    )
    extractor = MaskPulseExtractor(registrator, aligner, config)

    _ = extractor.phase_per_frame  # IQ (pas peak-locked) -> calcule good_uniform au passage

    filtered = extractor.filtered_signal  # (T_uniform, W') passe-bande FIR
    good = extractor.good_uniform  # (T_uniform,) fiabilite de phase (IQ)
    if not good.any():
        raise RuntimeError("aucun echantillon a phase IQ fiable (good_uniform)")

    amp = extractor.amplitude_per_frame  # px
    deltaCT_mm = 2*np.nanmedian(amp) * PIX_Y_SIZE
    CT_mm = np.nanmedian(extractor.signal) * PIX_Y_SIZE
    k, deltaV = sayah_k(deltaCT_mm, CT_mm, AL, IOP, OPA)

    return {
        "amp_per_ascan": amp,
        "deltaCT_mm": deltaCT_mm,
        "CT_mm": CT_mm,
        "k": k,
        "deltaV": deltaV,
        "cardiac_bpm": extractor.cardiac_bpm,
        "confidence": extractor.confidence,
        "gap_fraction": extractor.gap_fraction,
    }


# --------------------------------------------------------------------------- #
# Rigidite "one-cycle" (video repliee sur un battement cardiaque moyen)
# --------------------------------------------------------------------------- #
def rigidity_from_one_cycle(one_cycle_path: Path, model, AL: float, IOP: float, OPA: float) -> dict:
    """Rigidite k (Sayah) a partir de la video ONE-CYCLE.

    Le repliement (``NCycleReconstructor``) ne porte que sur les frames en
    niveaux de gris, pas sur les masques : on segmente ``one_cycle.mp4`` a
    nouveau, puis on mesure l'amplitude crete-a-crete DIRECTEMENT (le
    repliement a deja isole la composante cardiaque -> pas besoin de
    passe-bande + Hilbert, contrairement a ``rigidity_from_full_video_hilbert``).
    """
    frames = read_gray(str(one_cycle_path))  # (n_bins*n_cycle, H, W) uint8
    mask = infer(
        model, frames, scale_factor=SEG_SCALE_FACTOR, batch_size=SEG_BATCH_SIZE, device=DEVICE
    )
    mask = np.asarray(mask, dtype=bool)

    deltaY = compute_deltaY_masks(mask)  # (n_bins*n_cycle, W) px
    amp = deltaY.max(axis=0) - deltaY.min(axis=0)  # crete-a-crete par A-scan (px)

    deltaCT_mm = np.median(amp) * PIX_Y_SIZE
    CT_mm = np.median(deltaY) * PIX_Y_SIZE
    k, deltaV = sayah_k(deltaCT_mm, CT_mm, AL, IOP, OPA)

    return {
        "deltaY": deltaY,
        "amp_per_ascan": amp,
        "deltaCT_mm": deltaCT_mm,
        "CT_mm": CT_mm,
        "k": k,
        "deltaV": deltaV,
    }


# --------------------------------------------------------------------------- #
# Traitement d'une condition
# --------------------------------------------------------------------------- #
def process_condition(path_condi: Path, meas, id2code, model) -> dict | None:
    """Calcule les trois rigidites (video complete x2, one-cycle) d'une condition.

    Part des .tif/.xml bruts (segmentation + recalage via
    ``export_registered_video``), puis derive les trois ``k`` du meme recalage.
    Sauvegarde les tableaux necessaires aux figures (aucune figure ici) dans
    ``rigidity_full_pipeline.npz``, a cote de la video recalee.

    Returns
    -------
    dict | None : une ligne de resultats, ou None si la condition est
    incomplete (pas de HR exploitable, pas de recalage possible, etc.).
    """
    raw_dir = find_raw_dir(path_condi)
    if raw_dir is None:
        print(f"  [skip] aucun dossier RawImages/RawData : {path_condi}")
        return None

    # --- donnees cliniques : HR/IOP/OPA (visit_data.csv), AL (sansori_db.db) ---
    path_heartbeat = path_condi / "Data Files" / "visit_data.csv"
    if not path_heartbeat.exists():
        print(f"  [skip] visit_data.csv manquant : {path_condi}")
        return None
    df = pd.read_csv(path_heartbeat, quoting=csv.QUOTE_NONE)
    hr = np.nanmean(pd.to_numeric(df["HR"], errors="coerce"))  # BPM
    if not np.isfinite(hr):
        print(f"  [skip] HR non exploitable : {path_condi}")
        return None
    IOP = np.nanmean(pd.to_numeric(df["PascalIOP"], errors="coerce"))  # mmHg
    OPA = np.nanmean(pd.to_numeric(df["PascalOPA"], errors="coerce"))  # mmHg

    code = astro_code_from_folder(path_condi.parent.parent.name, id2code)
    eye = parse_eye(path_condi.name)
    moment = parse_moment(path_condi.parent.name)
    AL = lookup_clinical(meas, code, eye, moment, AL_DESCRIPTION)  # mm

    # --- 1) segmentation + recalage, a partir des .tif/.xml bruts ---
    # RegistrationConfig GLOBALE (pipeline_config.REGISTRATION) : ce que
    # l'utilisateur a regle/enregistre depuis testing_app/first_cc_registration.py.
    reg_result = export_registered_video(
        raw_dir,
        REGISTRATION,
        model,
        device=DEVICE,
        out_subdir=DEFAULT_OUTPUT_SUBDIR,
        overwrite=OVERWRITE_REGISTRATION,
        scale_factor=SEG_SCALE_FACTOR,
        seg_batch_size=SEG_BATCH_SIZE,
        verbose=False,
    )
    if reg_result["status"] != "ok" and reg_result.get("reason") != "exists":
        print(f"  [skip] recalage : {reg_result.get('reason')} : {raw_dir}")
        return None

    reg_dir = raw_dir / DEFAULT_OUTPUT_SUBDIR
    if not (reg_dir / "mask.npz").exists():
        print(f"  [skip] mask.npz absent apres recalage : {reg_dir}")
        return None

    # --- 1bis) resegmentation de la video DEJA recalee (remplace le masque
    # transporte pour TOUT le reste de la pipeline : Methodes 1 et 2). ---
    try:
        mask = resegment_registered_video(
            reg_dir, model, DEVICE, overwrite=OVERWRITE_RESEGMENTATION
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [skip] resegmentation post-recalage : {e} : {path_condi}")
        return None
    mask_path = reg_dir / "mask_resegmented.npz"

    # --- 2) rigidite "toute la video", Methode 1 (Hilbert) ---
    try:
        res_full_hilbert = rigidity_from_full_video_hilbert(reg_dir, mask, hr, AL, IOP, OPA)
    except Exception as e:  # noqa: BLE001
        print(f"  [erreur] rigidite (video complete, Hilbert) : {e} : {path_condi}")
        res_full_hilbert = None

    # --- 2bis) rigidite "toute la video", Methode 2 (MaskPulseExtractor) ---
    try:
        res_full_package = rigidity_from_full_video_package(
            reg_dir, mask_path, hr, AL, IOP, OPA, DEVICE
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [erreur] rigidite (video complete, package) : {e} : {path_condi}")
        res_full_package = None

    # --- 3) rigidite "one-cycle" (repliement sur un battement moyen) ---
    oc_result = export_one_cycle_video(
        reg_dir,
        expected_bpm=float(hr),
        device=DEVICE,
        verbose=False,
        overwrite=OVERWRITE_ONE_CYCLE,
    )
    res_cycle = None
    one_cycle_ready = oc_result["status"] == "ok" or (
        oc_result["status"] == "skipped" and oc_result.get("reason") == "exists"
    )
    if one_cycle_ready:
        try:
            res_cycle = rigidity_from_one_cycle(
                reg_dir / DEFAULT_ONE_CYCLE_NAME, model, AL, IOP, OPA
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [erreur] rigidite (one-cycle) : {e} : {path_condi}")
    else:
        print(f"  [avert] one-cycle non genere ({oc_result.get('reason')}) : {path_condi}")

    if res_full_hilbert is None and res_full_package is None and res_cycle is None:
        return None

    # --- 4) sauvegarde des donnees -- AUCUNE figure ici (cf. notebook) ---
    np.savez_compressed(
        reg_dir / "rigidity_full_pipeline.npz",
        hr=hr, AL=AL, IOP=IOP, OPA=OPA,
        # branche "toute la video", Methode 1 (Hilbert)
        full_hilbert_ts=res_full_hilbert["ts"] if res_full_hilbert else np.array([]),
        full_hilbert_amp_per_ascan=(
            res_full_hilbert["amp_per_ascan"] if res_full_hilbert else np.array([])
        ),
        full_hilbert_f_hil_bpm=(
            res_full_hilbert["f_hil_bpm"] if res_full_hilbert else np.array([])
        ),
        full_hilbert_deltaCT_mm=res_full_hilbert["deltaCT_mm"] if res_full_hilbert else np.nan,
        full_hilbert_CT_mm=res_full_hilbert["CT_mm"] if res_full_hilbert else np.nan,
        full_hilbert_k=res_full_hilbert["k"] if res_full_hilbert else np.nan,
        full_hilbert_deltaV=res_full_hilbert["deltaV"] if res_full_hilbert else np.nan,
        # branche "toute la video", Methode 2 (MaskPulseExtractor, IQ)
        full_package_amp_per_ascan=(
            res_full_package["amp_per_ascan"] if res_full_package else np.array([])
        ),
        full_package_deltaCT_mm=res_full_package["deltaCT_mm"] if res_full_package else np.nan,
        full_package_CT_mm=res_full_package["CT_mm"] if res_full_package else np.nan,
        full_package_k=res_full_package["k"] if res_full_package else np.nan,
        full_package_deltaV=res_full_package["deltaV"] if res_full_package else np.nan,
        full_package_cardiac_bpm=(
            res_full_package["cardiac_bpm"] if res_full_package else np.nan
        ),
        full_package_confidence=res_full_package["confidence"] if res_full_package else "",
        full_package_gap_fraction=(
            res_full_package["gap_fraction"] if res_full_package else np.nan
        ),
        # branche "one-cycle"
        cycle_deltaY=res_cycle["deltaY"] if res_cycle else np.array([]),
        cycle_amp_per_ascan=res_cycle["amp_per_ascan"] if res_cycle else np.array([]),
        cycle_deltaCT_mm=res_cycle["deltaCT_mm"] if res_cycle else np.nan,
        cycle_CT_mm=res_cycle["CT_mm"] if res_cycle else np.nan,
        cycle_k=res_cycle["k"] if res_cycle else np.nan,
        cycle_deltaV=res_cycle["deltaV"] if res_cycle else np.nan,
        cycle_cardiac_bpm=oc_result.get("cardiac_bpm", np.nan),
        cycle_confidence=oc_result.get("confidence", ""),
    )

    return {
        "patient": path_condi.parent.parent.name,
        "moment": path_condi.parent.name,
        "condition": path_condi.name,
        "path": str(path_condi),
        "hr": hr,
        "AL": AL,
        "IOP": IOP,
        "OPA": OPA,
        "CT_full_hilbert": res_full_hilbert["CT_mm"] if res_full_hilbert else np.nan,
        "delta_CT_full_hilbert": res_full_hilbert["deltaCT_mm"] if res_full_hilbert else np.nan,
        "delta_V_full_hilbert": res_full_hilbert["deltaV"] if res_full_hilbert else np.nan,
        "k_full_hilbert": res_full_hilbert["k"] if res_full_hilbert else np.nan,
        "CT_full_package": res_full_package["CT_mm"] if res_full_package else np.nan,
        "delta_CT_full_package": res_full_package["deltaCT_mm"] if res_full_package else np.nan,
        "delta_V_full_package": res_full_package["deltaV"] if res_full_package else np.nan,
        "k_full_package": res_full_package["k"] if res_full_package else np.nan,
        "cardiac_bpm_full_package": (
            res_full_package["cardiac_bpm"] if res_full_package else np.nan
        ),
        "confidence_full_package": res_full_package["confidence"] if res_full_package else "",
        "CT_one_cycle": res_cycle["CT_mm"] if res_cycle else np.nan,
        "delta_CT_one_cycle": res_cycle["deltaCT_mm"] if res_cycle else np.nan,
        "delta_V_one_cycle": res_cycle["deltaV"] if res_cycle else np.nan,
        "k_one_cycle": res_cycle["k"] if res_cycle else np.nan,
        "cardiac_bpm_one_cycle": oc_result.get("cardiac_bpm", np.nan),
        "confidence_one_cycle": oc_result.get("confidence", ""),
    }


# --------------------------------------------------------------------------- #
# Boucle principale
# --------------------------------------------------------------------------- #
def _fmt(x):
    """Formatage console : valeur ou '<NA>' si non finie."""
    return f"{x:.6g}" if isinstance(x, (int, float)) and np.isfinite(x) else "<NA>"


def main():
    meas, id2code = load_clinical_db(DB_PATH)   # base clinique chargee une fois
    model = get_choroid_segmentation_model()    # telecharge au 1er appel

    rows = []
    for path_astro in PATH_GENERAL.iterdir():
        if not path_astro.is_dir():
            continue
        for path_moment in path_astro.iterdir():
            if not path_moment.match("*rigidity"):
                continue
            for path_condi in path_moment.iterdir():
                if not path_condi.is_dir():
                    continue
                print(path_condi)
                try:
                    result = process_condition(path_condi, meas, id2code, model)
                except Exception as e:  # noqa: BLE001
                    print(f"  [erreur] {path_condi} : {e}")
                    traceback.print_exc()
                    continue
                if result is None:
                    continue
                rows.append(result)
                print(
                    f"  -> k (Hilbert) = {_fmt(result['k_full_hilbert'])} mm^-3  ·  "
                    f"k (package) = {_fmt(result['k_full_package'])} mm^-3 "
                    f"(FC = {_fmt(result['cardiac_bpm_full_package'])} BPM, "
                    f"confiance = {result['confidence_full_package']})  ·  "
                    f"k (one-cycle) = {_fmt(result['k_one_cycle'])} mm^-3  "
                    f"(FC = {_fmt(result['cardiac_bpm_one_cycle'])} BPM, "
                    f"confiance = {result['confidence_one_cycle']})"
                )

    if not rows:
        print("Aucune condition traitee : table CSV non ecrite.")
        return

    columns = [
        "patient", "moment", "condition", "path", "hr", "AL", "IOP", "OPA",
        "CT_full_hilbert", "delta_CT_full_hilbert", "delta_V_full_hilbert", "k_full_hilbert",
        "CT_full_package", "delta_CT_full_package", "delta_V_full_package", "k_full_package",
        "cardiac_bpm_full_package", "confidence_full_package",
        "CT_one_cycle", "delta_CT_one_cycle", "delta_V_one_cycle", "k_one_cycle",
        "cardiac_bpm_one_cycle", "confidence_one_cycle",
    ]
    out = pd.DataFrame(rows)[columns]
    out.to_csv(OUTPUT_CSV, index=False, na_rep="<NA>")
    print(f"\n{len(out)} conditions ecrites dans {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
