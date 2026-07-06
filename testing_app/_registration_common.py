"""Logique partagee par les pages de recalage (first_cc_registration + pages/).

Regroupe :
  - la selection de condition (patient / moment / oeil / replicat / point de temps)
    via ``select_condition`` — widgets a cles ``w_*`` partagees par toutes les
    pages, donc la selection se propage d'une page a l'autre
    (``st.session_state`` est global a la session Streamlit) ;
  - les PARAMETRES DE RECALAGE : ``RegistrationConfig`` (pipeline_config.py) est la
    SEULE source de verite. Les widgets (cles ``w_reg_*``) sont rendus sur la page
    d'accueil (``registration_config_widgets``) et lus par toutes les pages
    (``registration_config_from_state``) ; il n'y a plus de fichier d'experience
    par condition. « Enregistrer » (``save_registration_config``) reecrit les
    valeurs par defaut de ``RegistrationConfig`` directement dans
    ``pipeline_config.py`` — aucun autre fichier n'est ecrit ;
  - un constructeur de ``VideoRegistrator`` sur des frames/masques DEJA en memoire
    (``build_registrator``), pour que l'apercu interactif utilise exactement le
    meme moteur de recalage que l'export de production
    (``scripts/registration/astronauts.py::export_registered_video``) ;
  - les outils image (chargement, overlays) et la segmentation.

Les trois pages :
  1. ``first_cc_registration.py``   -> reglage de TOUS les parametres de recalage
                                        + apercu du recalage initial (X/Y) sur une paire ;
  2. ``pages/1_Correction_A-scan``  -> apercu (avant/apres) de la 2e passe A-scan ;
  3. ``pages/2_Video_recalee``      -> generation de la video recalee complete.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
import numpy as np
import scipy.ndimage
import imageio.v3 as iio
import torch

from ocularrigidity.data.spectralis import SpectralisStudy
from ocularrigidity.segmentation.utils import get_choroid_segmentation_model
from ocularrigidity.segmentation.inference import infer
from ocularrigidity.registration.registration_engine import VideoRegistrator
from ocularrigidity import pipeline_config

from sansori_nav import (
    PATH_GENERAL,
    find_patient_dir,
    find_moment_dir,
    list_replicate_dirs,
    find_raw_dir,
    format_acq_time,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Echelle de la segmentation (choroide) pour l'apercu — distincte de
# RegistrationConfig.scale_factor (qui ne s'applique qu'au recalage lateral).
SEG_SCALE_FACTOR = 2.0


# --------------------------------------------------------------------------- #
# RegistrationConfig : source unique de verite (pipeline_config.py)
#
# Plus de fichier par condition : les widgets (cles ``w_reg_*``) sont partages
# par toutes les pages via ``st.session_state`` et initialises UNE FOIS depuis
# ``pipeline_config.REGISTRATION``. « Enregistrer » reecrit cette meme classe
# dans pipeline_config.py, en ne touchant QUE les champs qui y sont declares.
# --------------------------------------------------------------------------- #
_REGISTRATION_STATE_KEY = "_registration_defaults_loaded"


def _registration_field_names() -> set[str]:
    """Noms des champs de ``RegistrationConfig`` — recalcule a chaque appel pour
    rester correct apres un ``importlib.reload`` de ``pipeline_config``."""
    return {f.name for f in dataclasses.fields(pipeline_config.RegistrationConfig)}


def init_registration_state() -> None:
    """Seme les cles ``w_reg_<champ>`` depuis ``pipeline_config.REGISTRATION``.

    Une seule fois par session Streamlit (drapeau ``_REGISTRATION_STATE_KEY``) :
    les reglages suivants de l'utilisateur restent dans ``session_state`` tant
    qu'on ne clique pas sur « Enregistrer » (qui, lui, persiste ces memes
    valeurs dans pipeline_config.py — cf. ``save_registration_config``).
    """
    if st.session_state.get(_REGISTRATION_STATE_KEY):
        return
    cfg = pipeline_config.REGISTRATION
    for name in _registration_field_names():
        st.session_state[f"w_reg_{name}"] = getattr(cfg, name)
    st.session_state[_REGISTRATION_STATE_KEY] = True


def registration_config_from_state() -> pipeline_config.RegistrationConfig:
    """Assemble la ``RegistrationConfig`` courante depuis ``session_state``.

    Fonctionne que les widgets aient ete rendus ou non sur la page courante
    (repli sur ``pipeline_config.REGISTRATION`` champ par champ) : les widgets
    de reglage vivent uniquement sur la page d'accueil, les autres pages lisent
    juste cet etat partage.
    """
    init_registration_state()
    cfg = pipeline_config.REGISTRATION
    values = {
        name: st.session_state.get(f"w_reg_{name}", getattr(cfg, name))
        for name in _registration_field_names()
    }
    return pipeline_config.RegistrationConfig(**values)


def _literal_repr(value) -> str:
    """Source Python litterale de ``value``, au style de pipeline_config.py."""
    if isinstance(value, bool):
        return repr(value)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, tuple):
        return "(" + ", ".join(_literal_repr(v) for v in value) + ")"
    raise TypeError(f"Type non supporte pour pipeline_config.py : {type(value)!r}")


def save_registration_config(cfg: pipeline_config.RegistrationConfig) -> Path:
    """Persiste ``cfg`` en reecrivant les valeurs par defaut de
    ``RegistrationConfig`` DIRECTEMENT dans pipeline_config.py.

    N'ecrit QUE les champs reellement declares sur la dataclasse (obtenus via
    ``dataclasses.fields``) : rien d'autre dans le fichier n'est touche — ni les
    commentaires, ni les autres classes, ni la mise en forme. Il n'existe pas
    d'autre fichier de parametres ; le prochain processus (rerun de l'app,
    ``Astronauts/register_files.py``, ...) recharge les memes valeurs simplement
    en (re)important ``pipeline_config.REGISTRATION``.
    """
    path = Path(pipeline_config.__file__)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "RegistrationConfig"
    )
    field_names = {f.name for f in dataclasses.fields(cfg)}

    lines = source.splitlines(keepends=True)
    line_start = [0]
    for line in lines:
        line_start.append(line_start[-1] + len(line))

    def offset(lineno: int, col: int) -> int:
        return line_start[lineno - 1] + col

    spans = []
    for node in cls.body:
        if not (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)):
            continue
        name = node.target.id
        if name not in field_names or node.value is None:
            continue
        start = offset(node.value.lineno, node.value.col_offset)
        end = offset(node.value.end_lineno, node.value.end_col_offset)
        spans.append((start, end, _literal_repr(getattr(cfg, name))))

    # Du bas vers le haut : remplacer un span ne doit pas invalider les offsets
    # (deja calcules) des champs qui le precedent dans le fichier.
    for start, end, literal in sorted(spans, reverse=True):
        source = source[:start] + literal + source[end:]

    path.write_text(source, encoding="utf-8")
    # pipeline_config.REGISTRATION doit refleter le fichier immediatement (l'app
    # reste le meme process Streamlit d'un rerun a l'autre) : on recharge le
    # module plutot que de rouvrir un nouveau processus.
    importlib.reload(pipeline_config)
    return path


def build_registrator(
    frames: np.ndarray,
    masks: np.ndarray,
    cfg: pipeline_config.RegistrationConfig,
    *,
    batch_size: int | None = None,
) -> VideoRegistrator:
    """``VideoRegistrator`` sur des frames/masques DEJA en memoire (pas de disque).

    Meme pattern d'injection que ``scripts/registration/astronauts.py::
    export_registered_video`` pour une condition complete : on affecte
    directement ``_raw_frames``/``_raw_masks`` pour court-circuiter le
    chargement fichier, de sorte que l'apercu interactif et l'export de
    production partagent EXACTEMENT le meme moteur de recalage
    (``registration.rigid.register_videos``).
    """
    registrator = VideoRegistrator(
        video=Path("."),
        root_data=Path("."),
        root_masks=Path("."),
        skip_first_n_frames=0,
        drop_last_n_frames=0,
        correct_transversal=cfg.correct_transversal,
        correct_axial=cfg.correct_axial,
        flatten_rpe=cfg.flatten_rpe,
        axial_refinement=cfg.axial_refinement,
        fovea_correction_enabled=cfg.fovea_correction_enabled,
        lateral_method=cfg.lateral_method,
        max_lateral_shift=cfg.max_lateral_shift,
        smooth_transversal=cfg.smooth_transversal,
        smooth_transversal_sigma=cfg.smooth_transversal_sigma,
        crop_factor=cfg.crop_factor,
        scale_factor=cfg.scale_factor,
        transversal_bandpass=cfg.transversal_bandpass,
        axial_bandpass=cfg.axial_bandpass,
        max_axial_shift=cfg.max_axial_shift,
        subpixel=cfg.subpixel,
        device=DEVICE,
        batch_size=batch_size or cfg.batch_size,
        cache_dir=None,
        verbose=False,
    )
    registrator._raw_frames = frames
    registrator._raw_masks = masks
    registrator.compute_registration()
    return registrator


# --------------------------------------------------------------------------- #
# Widgets des parametres de recalage (page d'accueil UNIQUEMENT)
# --------------------------------------------------------------------------- #
def registration_config_widgets() -> pipeline_config.RegistrationConfig:
    """Rend un widget par champ de ``RegistrationConfig`` (cles ``w_reg_*``).

    A appeler UNE SEULE FOIS, sur la page d'accueil — les autres pages lisent
    l'etat partage via ``registration_config_from_state``.
    """
    init_registration_state()
    k = "w_reg_"

    st.header("Recalage horizontal (X)")
    correct_transversal = st.checkbox(
        "Corriger le decalage lateral (X)", key=k + "correct_transversal",
    )
    lateral_method = st.selectbox(
        "Methode", options=["xcorr", "fullframe", "both"], key=k + "lateral_method",
        disabled=not correct_transversal,
        help=(
            "xcorr = correlation 1D des profils lateraux ; "
            "fullframe = correlation de phase 2D plein champ ; "
            "both = moyenne des deux."
        ),
    )
    max_lateral_shift = st.number_input(
        "max_lateral_shift (px)", min_value=1, step=1, key=k + "max_lateral_shift",
        disabled=not correct_transversal,
    )
    smooth_transversal = st.checkbox(
        "Lisser dx dans le temps", key=k + "smooth_transversal",
        disabled=not correct_transversal,
    )
    smooth_transversal_sigma = st.number_input(
        "smooth_transversal_sigma", min_value=0.1, step=0.5, format="%.2f",
        key=k + "smooth_transversal_sigma",
        disabled=not (correct_transversal and smooth_transversal),
    )
    cc1, cc2 = st.columns(2)
    crop_factor = cc1.number_input(
        "crop_factor", min_value=0.05, max_value=1.0, step=0.05, format="%.2f",
        key=k + "crop_factor", disabled=not correct_transversal,
        help="Fraction centrale de la largeur conservee avant correlation.",
    )
    scale_factor = cc2.number_input(
        "scale_factor", min_value=0.1, max_value=2.0, step=0.1, format="%.2f",
        key=k + "scale_factor", disabled=not correct_transversal,
        help="Facteur d'echelle (downscale) applique avant correlation.",
    )
    bc1, bc2 = st.columns(2)
    bp_lo = bc1.number_input(
        "transversal_bandpass bas", min_value=0.0, max_value=0.5, step=0.01,
        format="%.3f", key=k + "transversal_bandpass_lo",
        value=st.session_state.get(k + "transversal_bandpass", (0.02, 0.5))[0],
        disabled=not correct_transversal,
    )
    bp_hi = bc2.number_input(
        "transversal_bandpass haut", min_value=0.01, max_value=1.0, step=0.01,
        format="%.3f", key=k + "transversal_bandpass_hi",
        value=st.session_state.get(k + "transversal_bandpass", (0.02, 0.5))[1],
        disabled=not correct_transversal,
    )
    st.session_state[k + "transversal_bandpass"] = (float(bp_lo), float(bp_hi))

    st.header("Correction de la fovea")
    fovea_correction_enabled = st.checkbox(
        "Recentrer sur la fovea avant le recalage lateral", key=k + "fovea_correction_enabled",
        help="segmentation.fovea.from_ilm.estimate_fovea (detection sans modele, via l'ILM).",
    )

    st.header("Recalage vertical (Y)")
    correct_axial = st.checkbox(
        "Corriger l'alignement vertical (Y)", key=k + "correct_axial",
        help="Alignement par colonne de la membrane de Bruch sur la frame de reference.",
    )
    flatten_rpe = st.checkbox(
        "flatten_rpe (aplatir sur une ligne constante)", key=k + "flatten_rpe",
        disabled=not correct_axial,
    )

    st.header("Recalage A-scan / RPE (2e passe)")
    axial_refinement = st.checkbox(
        "Activer le recalage par A-scan sur la mediane", key=k + "axial_refinement",
        help="register_ascans_to_median : correlation de phase par colonne sur la mediane temporelle.",
    )
    max_axial_shift = st.number_input(
        "max_axial_shift (px)", min_value=1, step=1, key=k + "max_axial_shift",
        disabled=not axial_refinement,
    )
    ac1, ac2 = st.columns(2)
    abp_lo = ac1.number_input(
        "axial_bandpass bas", min_value=0.0, max_value=0.5, step=0.01, format="%.3f",
        key=k + "axial_bandpass_lo",
        value=st.session_state.get(k + "axial_bandpass", (0.02, 0.5))[0],
        disabled=not axial_refinement,
    )
    abp_hi = ac2.number_input(
        "axial_bandpass haut", min_value=0.01, max_value=1.0, step=0.01, format="%.3f",
        key=k + "axial_bandpass_hi",
        value=st.session_state.get(k + "axial_bandpass", (0.02, 0.5))[1],
        disabled=not axial_refinement,
    )
    st.session_state[k + "axial_bandpass"] = (float(abp_lo), float(abp_hi))

    st.header("Options generales")
    subpixel = st.checkbox("subpixel", key=k + "subpixel")
    with st.expander("Options avancees (video complete)"):
        skip_first_n_frames = st.number_input(
            "skip_first_n_frames", min_value=0, step=1, key=k + "skip_first_n_frames",
        )
        drop_last_n_frames = st.number_input(
            "drop_last_n_frames", min_value=0, step=1, key=k + "drop_last_n_frames",
        )
        use_encoded_video = st.checkbox(
            "use_encoded_video", key=k + "use_encoded_video",
        )
        batch_size = st.number_input(
            "batch_size", min_value=1, step=1, key=k + "batch_size",
        )

    return pipeline_config.RegistrationConfig(
        skip_first_n_frames=int(skip_first_n_frames),
        drop_last_n_frames=int(drop_last_n_frames),
        use_encoded_video=bool(use_encoded_video),
        correct_transversal=bool(correct_transversal),
        correct_axial=bool(correct_axial),
        flatten_rpe=bool(flatten_rpe),
        axial_refinement=bool(axial_refinement),
        fovea_correction_enabled=bool(fovea_correction_enabled),
        lateral_method=lateral_method,
        max_lateral_shift=int(max_lateral_shift),
        smooth_transversal=bool(smooth_transversal),
        smooth_transversal_sigma=float(smooth_transversal_sigma),
        crop_factor=float(crop_factor),
        scale_factor=float(scale_factor),
        transversal_bandpass=(float(bp_lo), float(bp_hi)),
        axial_bandpass=(float(abp_lo), float(abp_hi)),
        max_axial_shift=int(max_axial_shift),
        subpixel=bool(subpixel),
        batch_size=int(batch_size),
    )


# --------------------------------------------------------------------------- #
# Lecture des series OCT
# --------------------------------------------------------------------------- #
def load_oct_series(raw_dir: Path) -> list:
    """Series OCT exploitables de l'export XML (B-scan + ImageQuality + .tif)."""
    xml_files = sorted(raw_dir.glob("*.xml"))
    if not xml_files:
        return []
    study = SpectralisStudy.from_file(xml_files[0])
    return [
        s
        for s in study.series
        if s.oct is not None and s.oct.quality is not None and s.oct_file_name
    ]


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
    partagees : la condition choisie se propage donc d'une page a l'autre.
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

    chosen = None
    chosen_image_path = None
    if with_time_point:
        st.session_state.setdefault("w_time_idx", 0)
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
        x=int(x),
        moment=moment,
        eye=eye,
        r=int(r),
        patient_dir=patient_dir,
        data_dir=data_dir,
        raw_dir=raw_dir,
        series_list=series_list,
        best=best,
        best_image_path=best_image_path,
        timed=timed,
        chosen=chosen,
        chosen_image_path=chosen_image_path,
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


def make_overlay(ref_gray: np.ndarray, mov_gray: np.ndarray) -> np.ndarray:
    """Superposition facon imshowpair : reference en magenta, ``mov`` en vert."""
    H, W = ref_gray.shape
    overlay = np.zeros((H, W, 3), dtype=np.uint8)
    overlay[..., 0] = np.clip(ref_gray, 0, 255)  # R -> magenta (reference)
    overlay[..., 1] = np.clip(mov_gray, 0, 255)  # G -> vert (image recalee)
    overlay[..., 2] = np.clip(ref_gray, 0, 255)  # B -> magenta (reference)
    return overlay


def overlay_mask(
    gray: np.ndarray,
    mask: np.ndarray,
    color=(255, 0, 0),
    alpha: float = 0.35,
    border: int = 2,
) -> np.ndarray:
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


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
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
def segment_images(*paths: str) -> np.ndarray:
    """Masques de choroide (N, H, W) bool des images brutes fournies."""
    frames = np.stack([load_gray(p) for p in paths], axis=0)
    model = get_seg_model()
    masks = infer(model, frames, scale_factor=SEG_SCALE_FACTOR, batch_size=len(paths), device=DEVICE)
    return fill_empty_columns(np.asarray(masks, dtype=bool))


# --------------------------------------------------------------------------- #
# Recalage d'une paire (reference, point choisi) — apercu interactif
# --------------------------------------------------------------------------- #
# ``register_videos``/``VideoRegistrator`` choisit AUTOMATIQUEMENT sa frame de
# reference (celle dont l'aire du masque est la plus proche de la mediane du
# volume) — pertinent sur une video complete, mais pas ici : l'utilisateur a
# deja choisi explicitement "reference" (best) vs "point a recaler" (chosen), et
# rien ne garantit que ``ref`` ait la mediane des aires sur seulement 2 frames.
# On force ce choix en dupliquant ``ref`` 3 fois : sur 4 aires (r, r, r, m), la
# mediane vaut r quel que soit m, donc l'auto-selection retombe forcement sur
# une des copies de ref (index 0). Meme dupplication rend la MEDIANE TEMPORELLE
# utilisee par le recalage A-scan (2e passe) egale a ref (3 valeurs sur 4), donc
# la 2e passe aligne bien "mov sur ref" plutot que sur un artefact du melange.
_PAIR_REF_COPIES = 3


def register_pair(
    ref_path, mov_path, cfg: pipeline_config.RegistrationConfig
) -> VideoRegistrator:
    """Segmente puis recale la paire (reference, point choisi) via ``build_registrator``.

    Meme moteur (``VideoRegistrator`` / ``registration.rigid.register_videos``) que
    l'export video complet : l'apercu reflete exactement ce que produirait
    ``pages/2_Video_recalee.py`` avec la meme ``cfg``. La frame de reference est
    ``registrator.registered_frames[0]`` et le point recale ``[-1]`` (cf.
    ``_PAIR_REF_COPIES`` ci-dessus).
    """
    ref_path, mov_path = str(ref_path), str(mov_path)
    ref_gray, mov_gray = load_gray(ref_path), load_gray(mov_path)
    ref_mask, mov_mask = segment_images(ref_path, mov_path)

    frames = np.stack([ref_gray] * _PAIR_REF_COPIES + [mov_gray], axis=0)
    masks = np.stack([ref_mask] * _PAIR_REF_COPIES + [mov_mask], axis=0)
    return build_registrator(frames, masks, cfg, batch_size=_PAIR_REF_COPIES + 1)
