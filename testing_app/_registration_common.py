"""Logique partagee par les pages de recalage (first_cc_registration + pages/).

Regroupe :
  - la selection de condition (patient / moment / oeil / replicat / point de temps)
    via ``select_condition`` — widgets a cles ``w_*`` partagees par toutes les
    pages, donc la selection et les parametres de recalage se propagent d'une page
    a l'autre (``st.session_state`` est global a la session Streamlit) ;
  - le chargement automatique de la derniere experience (``experiment_*.json``) ;
  - les outils image (chargement, overlays, decalages) ;
  - le pretraitement RPE decouple (compensation d'ombres et/ou LoG) et l'apercu
    du recalage par A-scan ;
  - l'estimation du recalage horizontal (X) et vertical (Y).

Les trois pages :
  1. ``first_cc_registration.py``   -> recalage initial (X/Y) + apercu par paire ;
  2. ``pages/1_Correction_A-scan``  -> correction par A-scan (params + apercu) ;
  3. ``pages/2_Video_recalee``      -> generation de la video recalee + apercu.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json

import streamlit as st
import numpy as np
import scipy.ndimage
import imageio.v3 as iio
import torch

from ocularrigidity.data.spectralis import SpectralisStudy
from ocularrigidity.segmentation.utils import get_choroid_segmentation_model
from ocularrigidity.segmentation.inference import infer
from ocularrigidity.registration.horizontal.phase_correlation import (
    estimate_lateral_shift_fullframe,
    estimate_lateral_shift_xcorr_subpixel,
)
from ocularrigidity.registration.rigid import register_masks_by_displacement
from ocularrigidity.registration.axial import (
    correct_shadow,
    laplacian_of_gaussian,
    estimate_ascan_vshift_to_median,
)
from ocularrigidity.pipeline_config import RegistrationConfig

from sansori_nav import (
    PATH_GENERAL,
    find_patient_dir,
    find_moment_dir,
    list_replicate_dirs,
    find_raw_dir,
    format_acq_time,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# Lecture des series OCT + derniere experience
# --------------------------------------------------------------------------- #
def load_oct_series(raw_dir: Path) -> list:
    """Series OCT exploitables de l'export XML (B-scan + ImageQuality + .tif)."""
    xml_files = sorted(raw_dir.glob("*.xml"))
    if not xml_files:
        return []
    study = SpectralisStudy.from_file(xml_files[0])
    return [
        s for s in study.series
        if s.oct is not None and s.oct.quality is not None and s.oct_file_name
    ]


def latest_experiment_params(condition_dir: Path):
    """(params, chemin) du dernier ``experiment_*.json`` de la condition."""
    exp_dir = condition_dir / "experiments"
    if not exp_dir.is_dir():
        return None, None
    files = sorted(exp_dir.glob("experiment_*.json"))
    if not files:
        return None, None
    try:
        return json.loads(files[-1].read_text(encoding="utf-8")), files[-1]
    except Exception:
        return None, files[-1]


def apply_experiment_to_state(loaded: dict, timed: list) -> None:
    """Initialise les widgets (cles ``w_*``) depuis une experience chargee.

    Appele a CHAQUE changement de condition (cf. ``select_condition``). Passer par
    ``session_state`` est le seul moyen fiable, sous Streamlit, de reinitialiser
    des widgets par programme tout en preservant les reglages manuels ensuite. Les
    memes cles etant utilisees sur toutes les pages, les parametres se partagent.
    """
    method = loaded.get("x_method", "fullframe")
    if method not in ("fullframe", "xcorr", "aucun"):
        method = "fullframe"
    xp = loaded.get("x_params") or {}

    moving = loaded.get("moving_image")
    st.session_state["w_time_idx"] = next(
        (i for i, s in enumerate(timed) if s.oct_file_name == moving), 0
    )

    st.session_state["w_x_method"] = method

    ff = xp if method == "fullframe" else {}
    st.session_state["w_ff_max_shift"] = int(ff.get("max_shift") or 64)
    st.session_state["w_ff_max_vshift"] = int(ff.get("max_vshift") or 512)
    st.session_state["w_ff_downsample"] = int(ff.get("downsample") or 512)
    st.session_state["w_ff_bp_lo"] = float(ff.get("bp_lo", 0.02))
    st.session_state["w_ff_bp_hi"] = float(ff.get("bp_hi", 0.5))

    xc = xp if method == "xcorr" else {}
    st.session_state["w_xc_max_shift"] = xc.get("max_shift")  # None -> W // 4
    st.session_state["w_xc_drop_edges"] = int(xc.get("drop_edges") or 75)

    st.session_state["w_y_enabled"] = bool(loaded.get("y_enabled", True))
    st.session_state["w_flatten"] = bool(loaded.get("flatten", True))
    st.session_state["w_subpixel"] = bool(loaded.get("subpixel", True))

    # Correction par A-scan sur la mediane (RPE) — 2e passe.
    mp = loaded.get("median_params") or {}
    st.session_state["w_median_enabled"] = bool(loaded.get("median_enabled", False))
    st.session_state["w_median_use_shadow"] = bool(mp.get("use_shadow", True))
    st.session_state["w_median_use_log"] = bool(mp.get("use_log", True))
    st.session_state["w_median_max_vshift"] = int(mp.get("max_vshift") or 30)
    st.session_state["w_median_shadow_n"] = float(mp.get("shadow_n", 4.0))
    st.session_state["w_median_shadow_a"] = float(mp.get("shadow_a", 0.8))
    st.session_state["w_median_log_k"] = int(mp.get("log_kernel_size") or 9)
    st.session_state["w_median_log_sigma"] = float(mp.get("log_sigma", 3.0))


@st.cache_data(show_spinner=False, max_entries=3)
def read_video_bytes(path: str, mtime: float) -> bytes:
    """Octets d'un .mp4 (mis en cache ; ``mtime`` invalide le cache si le fichier change)."""
    return Path(path).read_bytes()


# --------------------------------------------------------------------------- #
# Selection de condition (partagee par toutes les pages)
# --------------------------------------------------------------------------- #
@dataclass
class ConditionContext:
    """Resultat de ``select_condition`` : contexte resolu d'une condition."""

    x: int
    moment: str
    eye: str
    r: int
    patient_dir: Path
    data_dir: Path
    raw_dir: Path
    series_list: list
    best: object
    best_image_path: Path
    timed: list
    chosen: object | None
    chosen_image_path: Path | None


def select_condition(with_time_point: bool = True) -> ConditionContext:
    """Rend la selection patient/moment/oeil/replicat(/temps) et resout les chemins.

    Les widgets patient/moment/oeil et le point de temps portent des cles ``w_*``
    partagees : la condition choisie se propage donc d'une page a l'autre. Charge
    aussi les parametres de la derniere experience a chaque changement de condition.
    ``st.stop()`` si la selection est incomplete/introuvable.
    """
    st.header("Données")
    st.session_state.setdefault("w_patient", 2)
    st.session_state.setdefault("w_moment", ["before"])
    st.session_state.setdefault("w_eye", ["OD"])
    x = st.slider("Patient ID", 1, 14, step=1, key="w_patient")
    y = st.multiselect("Moment", ["before", "after"], max_selections=1, key="w_moment")
    z = st.multiselect("Oeil", ["OS", "OD"], max_selections=1, key="w_eye")

    moment = y[0] if y else None
    eye = z[0] if z else None

    patient_dir = find_patient_dir(x)
    if patient_dir is None:
        st.error(f"Aucun dossier patient pour l'ID {x} dans {PATH_GENERAL}.")
        st.stop()
    moment_dir = find_moment_dir(patient_dir, moment) if moment else None
    if moment_dir is None:
        st.warning("Selectionnez un moment disponible pour ce patient.")
        st.stop()
    replicate_dirs = list_replicate_dirs(moment_dir, eye) if eye else []
    n_rep = len(replicate_dirs)
    if n_rep == 0:
        st.warning(f"Aucun replicat '{eye}' pour {patient_dir.name} / {moment}.")
        st.stop()
    if n_rep == 1:
        r = 1
        st.info("Un seul replicat disponible pour cette selection.")
    else:
        r = st.selectbox("Replicat", options=list(range(1, n_rep + 1)))

    data_dir = replicate_dirs[r - 1]
    st.caption("Chemin vers les donnees :")
    st.code(str(data_dir))

    raw_dir = find_raw_dir(data_dir)
    if raw_dir is None:
        st.warning(f"Aucun sous-dossier 'RawImages'/'RawData' dans {data_dir}.")
        st.stop()

    series_list = load_oct_series(raw_dir)
    if not series_list:
        st.warning("Aucune image avec un champ ImageQuality dans l'export XML.")
        st.stop()

    best = max(series_list, key=lambda s: s.oct.quality)
    best_image_path = raw_dir / best.oct_file_name

    timed = sorted(
        (s for s in series_list if s.acquisition_time is not None),
        key=lambda s: s.acquisition_time.seconds_of_day,
    )
    if not timed:
        st.warning("Aucun point de temps horodate dans l'export XML.")
        st.stop()

    # Chargement automatique des parametres de la derniere experience (une fois
    # par condition ; partage entre pages via le meme ``_cond_sig``).
    loaded_exp, loaded_exp_path = latest_experiment_params(data_dir)
    cond_sig = (int(x), moment, eye, int(r))
    if st.session_state.get("_cond_sig") != cond_sig:
        st.session_state["_cond_sig"] = cond_sig
        apply_experiment_to_state(loaded_exp or {}, timed)
        # Les parametres A-scan enregistres (fichier dedie) priment sur la section
        # median de l'experience, pour que widgets et video partagent la meme source.
        saved_ascan = load_ascan_params(data_dir)
        if saved_ascan:
            apply_ascan_params_to_state(saved_ascan)
        st.session_state["_loaded_from"] = (
            loaded_exp_path.name if (loaded_exp and loaded_exp_path) else None
        )
        st.session_state["_ascan_loaded"] = bool(saved_ascan)
    if st.session_state.get("_loaded_from"):
        st.info(f"Parametres charges depuis {st.session_state['_loaded_from']}")
    if st.session_state.get("_ascan_loaded"):
        st.caption(f"Paramètres A-scan chargés depuis {ASCAN_PARAMS_NAME}.")

    chosen = None
    chosen_image_path = None
    if with_time_point:
        idx = st.select_slider(
            "Point de temps (image a recaler)",
            options=list(range(len(timed))),
            key="w_time_idx",
            format_func=lambda i: format_acq_time(timed[i].acquisition_time),
        )
        chosen = timed[idx]
        chosen_image_path = raw_dir / chosen.oct_file_name
        c1, c2 = st.columns(2)
        c1.metric("Qualite (reference)", f"{best.oct.quality:g}")
        c2.metric("Qualite (point choisi)", f"{chosen.oct.quality:g}")

    return ConditionContext(
        x=int(x), moment=moment, eye=eye, r=int(r),
        patient_dir=patient_dir, data_dir=data_dir, raw_dir=raw_dir,
        series_list=series_list, best=best, best_image_path=best_image_path,
        timed=timed, chosen=chosen, chosen_image_path=chosen_image_path,
    )


# --------------------------------------------------------------------------- #
# Lecture des parametres depuis session_state (partages entre pages)
# --------------------------------------------------------------------------- #
def read_x_params(x_method: str) -> dict:
    """Reconstruit le dict de parametres X depuis ``session_state`` (cles ``w_*``)."""
    if x_method == "fullframe":
        return {
            "max_shift": int(st.session_state.get("w_ff_max_shift", 64)),
            "max_vshift": int(st.session_state.get("w_ff_max_vshift", 512)),
            "downsample": int(st.session_state.get("w_ff_downsample", 512)),
            "bp_lo": float(st.session_state.get("w_ff_bp_lo", 0.02)),
            "bp_hi": float(st.session_state.get("w_ff_bp_hi", 0.5)),
        }
    if x_method == "xcorr":
        return {
            "max_shift": st.session_state.get("w_xc_max_shift"),
            "drop_edges": int(st.session_state.get("w_xc_drop_edges", 75)),
        }
    return {}


def read_median_params() -> dict:
    """Reconstruit le dict de parametres de correction A-scan depuis ``session_state``."""
    return {
        "use_shadow": bool(st.session_state.get("w_median_use_shadow", True)),
        "use_log": bool(st.session_state.get("w_median_use_log", True)),
        "max_vshift": int(st.session_state.get("w_median_max_vshift", 30)),
        "shadow_n": float(st.session_state.get("w_median_shadow_n", 4.0)),
        "shadow_a": float(st.session_state.get("w_median_shadow_a", 0.8)),
        "log_kernel_size": int(st.session_state.get("w_median_log_k", 9)),
        "log_sigma": float(st.session_state.get("w_median_log_sigma", 3.0)),
    }


# --------------------------------------------------------------------------- #
# Persistance dediee des parametres de correction A-scan
#
# La page « Correction A-scan » enregistre ses parametres dans un fichier dedie
# ``ascan_params.json`` (semantique de commit : on regle/previsualise, puis on
# enregistre) ; la page « Video recalee » LES CHARGE depuis ce fichier pour
# generer la video A-scan — decorrele des reglages non enregistres en session.
# --------------------------------------------------------------------------- #
ASCAN_PARAMS_NAME = "ascan_params.json"


def ascan_params_path(data_dir: Path) -> Path:
    """Chemin du fichier de parametres A-scan enregistres pour la condition."""
    return data_dir / "experiments" / ASCAN_PARAMS_NAME


def save_ascan_params(data_dir: Path, params: dict) -> Path:
    """Enregistre les parametres de correction A-scan (ecrase le fichier)."""
    path = ascan_params_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**params, "saved": datetime.now().isoformat(timespec="seconds")}
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def load_ascan_params(data_dir: Path) -> dict | None:
    """Charge ``ascan_params.json`` de la condition, ou ``None`` s'il n'existe pas."""
    path = ascan_params_path(data_dir)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def apply_ascan_params_to_state(saved: dict) -> None:
    """Injecte des parametres A-scan enregistres dans les widgets (cles ``w_median_*``)."""
    st.session_state["w_median_use_shadow"] = bool(saved.get("use_shadow", True))
    st.session_state["w_median_use_log"] = bool(saved.get("use_log", True))
    st.session_state["w_median_max_vshift"] = int(saved.get("max_vshift") or 30)
    st.session_state["w_median_shadow_n"] = float(saved.get("shadow_n", 4.0))
    st.session_state["w_median_shadow_a"] = float(saved.get("shadow_a", 0.8))
    st.session_state["w_median_log_k"] = int(saved.get("log_kernel_size") or 9)
    st.session_state["w_median_log_sigma"] = float(saved.get("log_sigma", 3.0))
    if "enabled" in saved:
        st.session_state["w_median_enabled"] = bool(saved.get("enabled"))


def build_reg_cfg(x_method: str, y_enabled: bool, flatten: bool, subpixel: bool,
                  median_enabled: bool, median_params: dict) -> RegistrationConfig:
    """Assemble une ``RegistrationConfig`` a partir des reglages de l'UI."""
    return RegistrationConfig(
        flatten=bool(flatten and y_enabled),
        horizontal_alignment=(x_method != "aucun"),
        lateral_method="xcorr" if x_method == "xcorr" else "fullframe",
        subpixel=subpixel,
        median_registration=bool(median_enabled),
        median_max_vshift=int(median_params.get("max_vshift", 30)),
        median_use_shadow=bool(median_params.get("use_shadow", True)),
        median_use_log=bool(median_params.get("use_log", True)),
        median_shadow_n=float(median_params.get("shadow_n", 4.0)),
        median_shadow_a=float(median_params.get("shadow_a", 0.8)),
        median_log_kernel_size=int(median_params.get("log_kernel_size", 9)),
        median_log_sigma=float(median_params.get("log_sigma", 3.0)),
    )


# --------------------------------------------------------------------------- #
# Outils image
# --------------------------------------------------------------------------- #
def load_gray(path) -> np.ndarray:
    """Charge une image et la ramene en niveaux de gris (H x W, float32)."""
    img = iio.imread(path)
    if img.ndim == 3:
        img = img[..., 0]  # les .tif sont du gris replique sur 3 canaux
    return img.astype(np.float32)


def vertical_mean_profile(gray: np.ndarray) -> np.ndarray:
    """Profil lateral = moyenne verticale apres un leger flou 5x5 (cf. rigid.py)."""
    return scipy.ndimage.uniform_filter(gray, size=5, mode="nearest").mean(axis=0)


def make_overlay(ref_gray: np.ndarray, mov_gray: np.ndarray) -> np.ndarray:
    """Superposition facon imshowpair : reference en magenta, ``mov`` en vert."""
    H, W = ref_gray.shape
    overlay = np.zeros((H, W, 3), dtype=np.uint8)
    overlay[..., 0] = np.clip(ref_gray, 0, 255)  # R -> magenta (reference)
    overlay[..., 1] = np.clip(mov_gray, 0, 255)  # G -> vert (image recalee)
    overlay[..., 2] = np.clip(ref_gray, 0, 255)  # B -> magenta (reference)
    return overlay


def apply_dx(gray: np.ndarray, dx: float) -> np.ndarray:
    """Applique un decalage lateral dx : output[x] = input[x - dx]."""
    return scipy.ndimage.shift(gray, (0.0, dx), order=1, mode="constant", cval=0.0)


def overlay_mask(gray: np.ndarray, mask: np.ndarray,
                 color=(255, 0, 0), alpha: float = 0.35, border: int = 2) -> np.ndarray:
    """Image grise + masque : remplissage colore transparent + bordure pleine."""
    base = np.clip(gray, 0, 255).astype(np.uint8)
    rgb = np.stack([base, base, base], axis=-1)
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return rgb
    col = np.array(color, dtype=np.float32)
    rgb[m] = (rgb[m] * (1.0 - alpha) + col * alpha).astype(np.uint8)
    edge = m & ~scipy.ndimage.binary_erosion(m, iterations=int(border))
    rgb[edge] = col.astype(np.uint8)
    return rgb


def to_display(img: np.ndarray) -> np.ndarray:
    """Normalise une carte (valeurs +/-) en uint8 pour ``st.image`` (percentiles 1-99)."""
    x = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((x - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Pretraitement RPE decouple + apercu du recalage par A-scan
# --------------------------------------------------------------------------- #
def rpe_enhance(gray: np.ndarray, use_shadow: bool, n: float, a: float,
                use_log: bool, k: int, sigma: float) -> np.ndarray:
    """Pretraitement RPE decouple : compensation d'ombres et/ou LoG, ou aucun."""
    x = gray.astype(np.float32)
    if use_shadow:
        x = correct_shadow(x, float(n), float(a))
    if use_log:
        x = laplacian_of_gaussian(np.asarray(x, dtype=np.float32), int(k), float(sigma))
    return np.asarray(x, dtype=np.float32)


def prep_label(use_shadow: bool, use_log: bool) -> str:
    """Libelle du pretraitement actif pour les legendes de l'apercu."""
    steps = []
    if use_shadow:
        steps.append("compensation")
    if use_log:
        steps.append("LoG")
    return " + ".join(steps) if steps else "aucun prétraitement"


def apply_ascan_vshift(gray: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Applique un deplacement vertical PAR COLONNE ``dy`` (W,) : out[y,x]=in[y+dy[x],x].

    Meme convention (``sample_y = grid_y + dy``) et meme moteur (``grid_sample``)
    que ``register_ascans_to_median``, pour que l'apercu reflete la video exportee.
    """
    H, W = gray.shape
    ys = torch.arange(H, dtype=torch.float32)
    xs = torch.arange(W, dtype=torch.float32)
    grid_y = ys.view(H, 1).expand(H, W) + torch.as_tensor(dy, dtype=torch.float32).view(1, W)
    grid_x = xs.view(1, W).expand(H, W)
    norm_y = grid_y / (H - 1) * 2 - 1
    norm_x = grid_x / (W - 1) * 2 - 1
    grid = torch.stack([norm_x, norm_y], dim=-1)[None]
    out = torch.nn.functional.grid_sample(
        torch.from_numpy(gray.astype(np.float32))[None, None],
        grid, mode="bilinear", padding_mode="zeros", align_corners=True,
    )
    return out[0, 0].numpy()


def estimate_chosen_ascan_dy(ref: np.ndarray, mov: np.ndarray, use_shadow: bool,
                             n: float, a: float, use_log: bool, k: int, sigma: float,
                             max_vshift: int, subpixel: bool) -> np.ndarray:
    """dy par A-scan (W,) alignant ``mov`` sur ``ref`` apres pretraitement (decouple)."""
    ref_pre = rpe_enhance(ref, use_shadow, n, a, use_log, k, sigma)
    mov_pre = rpe_enhance(mov, use_shadow, n, a, use_log, k, sigma)
    dy = estimate_ascan_vshift_to_median(
        mov_pre[None], ref_pre, max_vshift=int(max_vshift),
        subpixel=bool(subpixel), batch_size=1, device=DEVICE,
    )
    return dy[0].cpu().numpy()


# --------------------------------------------------------------------------- #
# Recalage horizontal (X) et vertical (Y)
# --------------------------------------------------------------------------- #
def estimate_dx(method: str, ref: np.ndarray, mov: np.ndarray, params: dict,
                subpixel: bool) -> float:
    """dx lateral (px) alignant ``mov`` sur ``ref`` selon la methode choisie."""
    if method == "aucun":
        return 0.0
    if method == "fullframe":
        frames = torch.from_numpy(np.stack([ref, mov]).astype(np.float32))
        dx = estimate_lateral_shift_fullframe(
            frames,
            ref=frames[0],
            downsample_to=(int(params["downsample"]), int(params["downsample"])),
            max_shift=int(params["max_shift"]),
            max_vshift=int(params["max_vshift"]),
            bandpass=(float(params["bp_lo"]), float(params["bp_hi"])),
            device=DEVICE,
            subpixel=bool(subpixel),
        )
        return float(dx[1])
    if method == "xcorr":
        curve = torch.from_numpy(vertical_mean_profile(mov)).float()[None]
        ref_curve = torch.from_numpy(vertical_mean_profile(ref)).float()
        ms = params.get("max_shift")
        dx = estimate_lateral_shift_xcorr_subpixel(
            curve, ref_curve,
            max_shift=int(ms) if ms is not None else None,
            drop_edges=int(params["drop_edges"]),
            subpixel=bool(subpixel),
        )
        return float(dx[0])
    raise ValueError(f"Methode X inconnue : {method!r}")


@st.cache_resource(show_spinner="Chargement du modele de segmentation...")
def get_seg_model():
    """Modele de segmentation de la choroide (telecharge depuis Hugging Face)."""
    return get_choroid_segmentation_model()


def fill_empty_columns(masks: np.ndarray) -> np.ndarray:
    """Remplit les colonnes sans masque par la colonne valide la plus proche."""
    out = masks.copy()
    cols = np.arange(out.shape[2])
    for i in range(out.shape[0]):
        has = out[i].any(0)
        if has.all():
            continue
        idx = np.where(has)[0]
        if idx.size == 0:
            continue
        nearest = idx[np.abs(cols[:, None] - idx[None, :]).argmin(1)]
        out[i] = out[i][:, nearest]
    return out


@st.cache_data(show_spinner="Segmentation des images (masque choroide)...")
def segment_images(ref_path, mov_path):
    """Masques de choroide (2, H, W) bool des images brutes [reference, point choisi]."""
    frames = np.stack([load_gray(ref_path), load_gray(mov_path)], axis=0)
    model = get_seg_model()
    masks = infer(model, frames, scale_factor=2.0, batch_size=2, device=DEVICE)
    return np.asarray(masks, dtype=bool)


@st.cache_data(show_spinner="Recalage Y (segmentation + register_masks_by_displacement)...")
def register_y(ref_path, mov_path, dx, flatten, subpixel):
    """Recale ``mov`` (deja recalee en x de ``dx``) sur ``ref`` en Y.

    Returns reg_frames (2, H, W) : [reference aplatie/alignee, image recalee X+Y].
    """
    ref = load_gray(ref_path)
    mov = load_gray(mov_path)
    mov_x = apply_dx(mov, dx)
    frames = np.stack([ref, mov_x], axis=0)

    model = get_seg_model()
    masks = infer(model, frames, scale_factor=2.0, batch_size=2, device=DEVICE)
    masks = fill_empty_columns(masks)

    _, reg_frames, _ = register_masks_by_displacement(
        masks, frames,
        correct_dx=False, flatten_rpe=bool(flatten), batch_size=2,
        device=DEVICE, verbose=False, return_params=True, subpixel=bool(subpixel),
    )
    if isinstance(reg_frames, torch.Tensor):
        reg_frames = reg_frames.cpu().numpy()
    return np.asarray(reg_frames)


def base_registration(ctx: ConditionContext, x_method: str, x_params: dict,
                      y_enabled: bool, flatten: bool, subpixel: bool):
    """(base_ref, base_mov, dx) apres le recalage initial X(+Y) de la paire choisie.

    Sert de point de depart au recalage par A-scan (« posterieur au recalage
    initial »). Reutilise les caches ``segment_images`` / ``register_y``.
    """
    ref_gray = load_gray(ctx.best_image_path)
    mov_gray = load_gray(ctx.chosen_image_path)
    dx = estimate_dx(x_method, ref_gray, mov_gray, x_params, subpixel)
    if y_enabled:
        reg = register_y(
            str(ctx.best_image_path), str(ctx.chosen_image_path), dx, flatten, subpixel
        )
        return reg[0], reg[1], dx
    return ref_gray, apply_dx(mov_gray, dx), dx
