"""Page Streamlit : creation de la video one-cycle a partir de la video recalee.

Une video one-cycle est la reconstruction d'un battement cardiaque moyen : les
frames recalees sont rangees par phase cardiaque puis moyennees (cf.
ocularrigidity.motion.one_cycle_export). Elle prend en entree la video recalee
produite par la page de recalage / register_files.py (RawImages/registered/), la
replie avec les parametres reglables ci-dessous, et affiche le resultat.
"""

import json
from pathlib import Path

import streamlit as st
import torch

from ocularrigidity.pipeline_config import PULSATION
from ocularrigidity.scripts.registration.astronauts import DEFAULT_OUTPUT_SUBDIR
from ocularrigidity.scripts.one_cycle.astronauts import (
    export_one_cycle_video,
    DEFAULT_ONE_CYCLE_NAME,
)

from sansori_nav import (
    PATH_GENERAL,
    find_patient_dir,
    find_moment_dir,
    list_replicate_dirs,
    find_raw_dir,
)

st.set_page_config(page_title="Video one-cycle", layout="wide")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_data(show_spinner=False, max_entries=3)
def read_video_bytes(path: str, mtime: float) -> bytes:
    """Octets d'un .mp4 (cache invalide via mtime quand le fichier change)."""
    return Path(path).read_bytes()


st.title("Video one-cycle")

col_cfg, col_out = st.columns([1, 1], gap="large")

with col_cfg:
    # ------------------------------------------------------------------- #
    # 1) Identification des donnees (meme arborescence que la page de recalage)
    # ------------------------------------------------------------------- #
    st.header("Donnees")
    x = st.slider("Patient ID", 1, 14, 2, step=1)
    y = st.multiselect(
        "Moment", ["before", "after"], default=["before"], max_selections=1
    )
    z = st.multiselect("Oeil", ["OS", "OD"], default=["OD"], max_selections=1)
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

    raw_dir = find_raw_dir(data_dir)
    if raw_dir is None:
        st.warning(f"Aucun sous-dossier RawImages/RawData dans {data_dir}.")
        st.stop()
    registered_dir = raw_dir / DEFAULT_OUTPUT_SUBDIR
    reg_video = registered_dir / "registered_video.mp4"
    st.caption("Video recalee (entree) :")
    st.code(str(reg_video))
    if not reg_video.exists():
        st.warning(
            "Video recalee introuvable. Genere-la d'abord (page de recalage : "
            "bouton « Enregistrer la video recalee », ou script register_files.py)."
        )
        st.stop()

    # ------------------------------------------------------------------- #
    # 2) Parametres du one-cycle (defauts : pipeline_config.PULSATION)
    # ------------------------------------------------------------------- #
    st.header("Parametres one-cycle")
    expected_bpm = st.number_input(
        "FC attendue (BPM) — vide = detection auto",
        value=None,
        min_value=1.0,
        step=1.0,
        placeholder="auto",
        help="Ancre la recherche de frequence autour de la FC connue (si dispo).",
    )
    method = st.selectbox(
        "Decomposition",
        ["ICA", "PCA", "SVD"],
        index=0,
        help="Separation des composantes temporelles avant scoring Lomb-Scargle.",
    )
    phase_method = st.selectbox(
        "Methode de phase (fold)",
        ["peak_locked", "iq"],
        index=0,
        help="peak_locked = phase 0 a chaque pic systolique ; iq = demodulation IQ.",
    )
    c1, c2 = st.columns(2)
    n_bins = c1.number_input(
        "n_bins (frames / cycle)",
        value=int(PULSATION.fold.n_bins),
        min_value=2,
        step=1,
        help="Nombre de casiers de phase = nombre de frames du cycle reconstruit.",
    )
    n_cycle = c2.number_input(
        "n_cycle (cycles moyennes)",
        value=int(PULSATION.fold.n_cycle),
        min_value=1,
        step=1,
        help="Nombre de cycles (tranches temporelles) moyennes et concatenes.",
    )
    c3, c4 = st.columns(2)
    fold_method = c3.selectbox(
        "Repliement (fold)",
        ["median", "mean"],
        index=0 if PULSATION.fold.fold_method == "median" else 1,
    )
    output_fps = c4.number_input(
        "fps de sortie",
        value=int(PULSATION.output_fps),
        min_value=1,
        step=1,
        help="Cadence d'affichage du .mp4 one-cycle.",
    )

    with st.expander("Parametres avances"):
        sigma_col = st.number_input(
            "sigma_col (lissage spatial)",
            value=float(PULSATION.extraction.sigma_col),
            min_value=0.0,
            step=0.5,
        )
        cs1, cs2 = st.columns(2)
        col_lo = cs1.number_input(
            "col_slice debut",
            value=int(PULSATION.extraction.col_slice.start),
            min_value=0,
            step=1,
        )
        col_hi = cs2.number_input(
            "col_slice fin",
            value=int(PULSATION.extraction.col_slice.stop),
            min_value=1,
            step=1,
        )
        target_fpb = st.number_input(
            "target_frames_per_bin",
            value=25,
            min_value=1,
            step=1,
        )
        band_frac = st.number_input(
            "expected_bpm_band_frac",
            value=float(PULSATION.extraction.expected_bpm_band_frac),
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            format="%.2f",
        )
        n_comp = st.number_input(
            "n_separable_components",
            value=16,
            min_value=2,
            step=1,
        )
        phase_cycles = st.number_input(
            "phase_smoother_cycles",
            value=2.0,
            min_value=0.5,
            step=0.5,
        )
        harmonic = st.checkbox("harmonic_correction", value=True)
        b1, b2 = st.columns(2)
        bpm_lo = b1.number_input("bpm_range min", value=30.0, min_value=1.0, step=1.0)
        bpm_hi = b2.number_input("bpm_range max", value=180.0, min_value=1.0, step=1.0)

    st.caption(
        "La video recalee est deja rognee (skip/drop du recalage) : le one-cycle "
        "est calcule sans re-rogner ni re-recaler."
    )

    if st.button("Generer la video one-cycle"):
        with st.spinner("Extraction du cycle cardiaque + repliement..."):
            try:
                result = export_one_cycle_video(
                    registered_dir,
                    overwrite=True,
                    device=DEVICE,
                    verbose=False,
                    expected_bpm=(float(expected_bpm) if expected_bpm else None),
                    bpm_range=(float(bpm_lo), float(bpm_hi)),
                    expected_bpm_band_frac=float(band_frac),
                    ICA_or_PCA=method,
                    sigma_col=float(sigma_col),
                    col_slice=(int(col_lo), int(col_hi)),
                    n_separable_components=int(n_comp),
                    phase_smoother_cycles=float(phase_cycles),
                    harmonic_correction=bool(harmonic),
                    phase_method_for_fold=phase_method,
                    n_bins=int(n_bins),
                    n_cycle=int(n_cycle),
                    target_frames_per_bin=int(target_fpb),
                    one_cycle_fold_method=fold_method,
                    output_fps=int(output_fps),
                    extra_meta={
                        "patient": patient_dir.name,
                        "moment": moment,
                        "eye": eye,
                        "replicate": int(r),
                    },
                )
            except Exception as e:  # noqa: BLE001
                st.error(f"Echec de la generation one-cycle : {e}")
            else:
                if result["status"] == "ok":
                    st.success(
                        f"One-cycle genere : {result['video']}  ·  "
                        f"{result['n_frames']} frames  ·  FC {result['cardiac_bpm']:.1f} BPM "
                        f"(confiance {result['confidence']})"
                    )
                else:
                    st.warning(f"Non genere (raison : {result.get('reason')}).")

# --------------------------------------------------------------------------- #
# Resultat : affiche la video one-cycle enregistree si elle existe.
# --------------------------------------------------------------------------- #
with col_out:
    st.header("Resultat")
    one_cycle_path = registered_dir / DEFAULT_ONE_CYCLE_NAME
    if one_cycle_path.exists():
        st.video(read_video_bytes(str(one_cycle_path), one_cycle_path.stat().st_mtime))
        st.caption(str(one_cycle_path))
        params_path = registered_dir / "one_cycle_params.json"
        if params_path.exists():
            with st.expander("Parametres du dernier one-cycle enregistre"):
                st.json(json.loads(params_path.read_text(encoding="utf-8")))
    else:
        st.info("Aucune video one-cycle enregistree pour cette condition.")
