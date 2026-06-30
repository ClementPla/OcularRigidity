import streamlit as st
import pandas as pd
import numpy as np
import re
from ocularrigidity.rigidity.features import compute_deltaY_masks
from pathlib import Path
from ocularrigidity.consts import ROOT_DATA_SMB
import numpy as np
from ocularrigidity.data.io import save_mask, load_mask
from ocularrigidity.segmentation.utils import get_choroid_segmentation_model
from ocularrigidity.segmentation.inference import infer
from ocularrigidity.data.compression import mp4_to_cube, read_gray
from ocularrigidity.data.io import load_cube
from ocularrigidity.data.spectralis import SpectralisStudy
import imageio.v3 as iio
# load the computed thicknesses
from matplotlib.pyplot import plot
import scipy
import numpy as np
from scipy.io import loadmat
from scipy.signal import lombscargle
import pandas as pd, csv

# --------------------------------------------------------------------------- #
# Resolution du chemin des donnees a partir des selections de l'utilisateur
#
# Arborescence SANSORI (cf. Astronauts/*.py) :
#   E:/SANSORI/<NN_id>/<id><before|post>_rigidity/<id><..>_rigidity_<OD|OS><rep?>/
#     - patient  : dossier prefixe par le numero zero-padde (01_..14_)
#     - moment   : "before" sur disque = "before" ; "post" sur disque = "after"
#     - oeil+rep : suffixe du dossier condition ; le numero de replicat est
#                  ABSENT s'il n'y a qu'un replicat (..._OD), sinon numerote
#                  (..._OS1, ..._OS2, ...). Le nombre de replicats varie (1 a 5).
# --------------------------------------------------------------------------- #
PATH_GENERAL = Path("E:/SANSORI")


def _moment_matches(folder_name: str, moment: str) -> bool:
    """Le moment 'after' choisi par l'utilisateur correspond au dossier 'post'."""
    low = folder_name.lower()
    if moment == "before":
        return "before" in low
    if moment == "after":
        return "post" in low or "after" in low
    return False


def find_patient_dir(patient_id: int) -> Path | None:
    """ID 1..14 -> dossier patient prefixe '01_' .. '14_'."""
    prefix = f"{patient_id:02d}_"
    if not PATH_GENERAL.exists():
        return None
    for p in PATH_GENERAL.iterdir():
        if p.is_dir() and p.name.startswith(prefix):
            return p
    return None


def find_moment_dir(patient_dir: Path, moment: str) -> Path | None:
    """Dossier '*rigidity' du patient correspondant au moment choisi."""
    for m in patient_dir.iterdir():
        if m.is_dir() and m.match("*rigidity") and _moment_matches(m.name, moment):
            return m
    return None


def list_replicate_dirs(moment_dir: Path, eye: str) -> list[Path]:
    """Dossiers condition pour un oeil donne, tries par numero de replicat.

    Le suffixe peut etre nu (..._OD = un seul replicat) ou numerote (..._OD2).
    """
    pat = re.compile(rf"{eye}(\d*)$")
    found = []
    for c in moment_dir.iterdir():
        if not c.is_dir():
            continue
        m = pat.search(c.name)
        if m:
            num = int(m.group(1)) if m.group(1) else 0
            found.append((num, c))
    found.sort(key=lambda t: t[0])
    return [c for _, c in found]


def find_raw_dir(condition_dir: Path) -> Path | None:
    """Sous-dossier contenant les images brutes (.tif) + l'export XML Spectralis.

    Le dossier s'appelle 'RawImages' dans le jeu SANSORI (anciennement 'RawData').
    """
    for name in ("RawData", "RawImages"):
        d = condition_dir / name
        if d.is_dir():
            return d
    return None


def load_oct_series(raw_dir: Path) -> list:
    """Series OCT exploitables de l'export XML (B-scan + ImageQuality + .tif).

    Chaque serie expose son B-scan OCT (.oct), son ImageQuality (.oct.quality),
    le nom du fichier .tif associe (.oct_file_name) et son horodatage
    (.acquisition_time). Renvoie une liste vide si rien d'exploitable.
    """
    xml_files = sorted(raw_dir.glob("*.xml"))
    if not xml_files:
        return []
    study = SpectralisStudy.from_file(xml_files[0])
    return [
        s for s in study.series
        if s.oct is not None and s.oct.quality is not None and s.oct_file_name
    ]


def format_acq_time(t) -> str:
    """AcquisitionTime -> 'HH:MM:SS.mmm' (millisecondes)."""
    return f"{t.hour:02d}:{t.minute:02d}:{t.second:06.3f}"


# --------------------------------------------------------------------------- #
# Mise en page : image a GAUCHE, parametres a DROITE
# --------------------------------------------------------------------------- #
st.set_page_config(layout="wide")
col_image, col_params = st.columns([2, 1], gap="large")

with col_params:
    x = st.slider("Select a patient ID", 1, 14, 2, step=1)
    y = st.multiselect("Select a moment", ["before", "after"], default=["before"], max_selections=1)
    z = st.multiselect("Select an eye", ["OS", "OD"], default=["OD"], max_selections=1)

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

    # Le choix du replicat depend du nombre reel de replicats : s'il n'y en a
    # qu'un, aucun choix n'est propose ; sinon, seuls les replicats existants.
    if n_rep == 1:
        r = 1
        st.info("Un seul replicat disponible pour cette selection.")
    else:
        r = st.selectbox("Select a replicate", options=list(range(1, n_rep + 1)))

    data_dir = replicate_dirs[r - 1]

    # --- Chemin vers les donnees obtenu ---
    st.success("Chemin vers les donnees :")
    st.code(str(data_dir))

    # --- Resolution de l'image de meilleure qualite ---
    raw_dir = find_raw_dir(data_dir)
    if raw_dir is None:
        st.warning(f"Aucun sous-dossier 'RawData'/'RawImages' dans {data_dir}.")
        st.stop()

    series_list = load_oct_series(raw_dir)
    if not series_list:
        st.warning("Aucune image avec un champ ImageQuality dans l'export XML.")
        st.stop()

    best = max(series_list, key=lambda s: s.oct.quality)
    best_image_path = raw_dir / best.oct_file_name
    st.metric("Meilleur ImageQuality", f"{best.oct.quality:g}")
    st.caption(best.oct_file_name)

    # --- Second point de temps de la meme serie ---
    # Le slider parcourt les points de temps horodates de l'export XML, tries
    # chronologiquement ; l'utilisateur en choisit un pour la seconde image.
    timed = sorted(
        (s for s in series_list if s.acquisition_time is not None),
        key=lambda s: s.acquisition_time.seconds_of_day,
    )
    if not timed:
        st.warning("Aucun point de temps horodate dans l'export XML.")
        st.stop()

    idx = st.select_slider(
        "Point de temps (seconde image)",
        options=list(range(len(timed))),
        format_func=lambda i: format_acq_time(timed[i].acquisition_time),
    )
    chosen = timed[idx]
    chosen_image_path = raw_dir / chosen.oct_file_name
    st.metric("ImageQuality (point choisi)", f"{chosen.oct.quality:g}")
    st.caption(chosen.oct_file_name)

# Panneau de gauche : les deux images cote a cote (rendu apres calcul).
with col_image:
    sub_best, sub_time = st.columns(2)

    with sub_best:
        st.subheader(f"Meilleure qualite — ImageQuality = {best.oct.quality:g}")
        if best_image_path.exists():
            st.image(
                iio.imread(best_image_path),
                caption=f"{best.oct_file_name} (ImageQuality = {best.oct.quality:g})",
                use_container_width=True,
            )
        else:
            st.error(f"Fichier image introuvable : {best_image_path}")

    with sub_time:
        st.subheader(f"Point de temps {format_acq_time(chosen.acquisition_time)}")
        if chosen_image_path.exists():
            st.image(
                iio.imread(chosen_image_path),
                caption=f"{chosen.oct_file_name} (ImageQuality = {chosen.oct.quality:g})",
                use_container_width=True,
            )
        else:
            st.error(f"Fichier image introuvable : {chosen_image_path}")