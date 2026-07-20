"""
register_files.py

Recalage par lot des videos OCT brutes du jeu SANSORI.

Pour chaque condition (patient / moment / oeil / replicat) :
  1. lit les parametres d'experience enregistres par l'app interactive
     ``testing_app/first_cc_registration.py`` (dernier ``experiment_*.json`` du
     dossier ``<condition>/experiments/``) -> ``RegistrationConfig`` ;
  2. delegue a ``ocularrigidity.registration.export.export_registered_video``,
     qui empile toutes les images .tif dans l'ordre des horodatages du XML,
     segmente la choroide, recale la video via ``RegisteredVideo`` et enregistre
     sous ``<condition>/RawImages/registered/`` :
       registered_video.mp4, mask.npz, timestamp.txt, transform.npz,
       registration_params.json.

La video, les masques et les horodatages sont deja rognes du skip/drop de la
config ; lancer ``pulsation.py`` avec ``skip_first_n_frames=0`` /
``drop_last_n_frames=0`` sur ces sorties.

Arborescence SANSORI (cf. les scripts freres ``segment_files.py`` et
``compute_rigidity_time_series.py``) :
    E:/SANSORI/<NN_id>/<...>_rigidity/<...>_rigidity_<OD|OS><rep?>/
        experiments/experiment_*.json   <- parametres de l'app
        RawImages/ (ou RawData/)        <- images .tif + export .xml
"""

from pathlib import Path
import json
import traceback

import torch

from ocularrigidity.pipeline_config import RegistrationConfig
from ocularrigidity.segmentation.utils import get_choroid_segmentation_model
from ocularrigidity.registration.export import (
    export_registered_video,
    DEFAULT_OUTPUT_SUBDIR,
)


# --------------------------------------------------------------------------- #
# Parametres
# --------------------------------------------------------------------------- #
PATH_GENERAL = Path("E:/SANSORI")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Segmentation de la choroide : meme echelle que first_cc_registration.py.
SEG_SCALE_FACTOR = 2.0
SEG_BATCH_SIZE = 8

OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
OVERWRITE = False  # re-traiter une condition deja recalee


# --------------------------------------------------------------------------- #
# Resolution des chemins (arborescence SANSORI + experiences de l'app)
# --------------------------------------------------------------------------- #
def find_raw_dir(condition_dir: Path) -> Path | None:
    """Sous-dossier contenant les .tif bruts + l'export XML Spectralis."""
    for name in ("RawImages", "RawData"):
        d = condition_dir / name
        if d.is_dir():
            return d
    return None


def latest_experiment(condition_dir: Path) -> Path | None:
    """Dernier ``experiment_*.json`` enregistre par first_cc_registration.py.

    Les fichiers sont horodates, donc le tri lexicographique met le plus recent
    en derniere position.
    """
    exp_dir = condition_dir / "experiments"
    if not exp_dir.is_dir():
        return None
    files = sorted(exp_dir.glob("experiment_*.json"))
    return files[-1] if files else None


# --------------------------------------------------------------------------- #
# Traitement d'une condition
# --------------------------------------------------------------------------- #
def process_condition(path_condi: Path, model) -> bool:
    """Recale une condition (via le module partage). True si une video est ecrite."""
    raw_dir = find_raw_dir(path_condi)
    if raw_dir is None:
        print(f"  [skip] aucun dossier RawImages/RawData : {path_condi}")
        return False

    exp_path = latest_experiment(path_condi)
    if exp_path is None:
        print(f"  [skip] aucun experiment_*.json : {path_condi}")
        return False
    experiment = json.loads(exp_path.read_text(encoding="utf-8"))
    cfg = RegistrationConfig(**experiment["registration_config"])

    result = export_registered_video(
        raw_dir,
        cfg,
        model,
        device=DEVICE,
        out_subdir=OUTPUT_SUBDIR,
        overwrite=OVERWRITE,
        scale_factor=SEG_SCALE_FACTOR,
        seg_batch_size=SEG_BATCH_SIZE,
        verbose=True,
        extra_meta={"source_experiment": exp_path.name, "experiment": experiment},
    )

    if result["status"] != "ok":
        print(f"  [skip] {result.get('reason')} : {raw_dir}")
        return False
    print(f"  -> {result['n_frames']} frames recalees @ {result['fps']:.1f} fps "
          f"(lateral={cfg.lateral_method}, flatten={cfg.flatten}, "
          f"hx={cfg.horizontal_alignment}) : {result['video']}")
    return True


# --------------------------------------------------------------------------- #
# Boucle principale
# --------------------------------------------------------------------------- #
def main():
    if not PATH_GENERAL.exists():
        print(f"Dossier introuvable : {PATH_GENERAL}")
        return

    model = get_choroid_segmentation_model()  # telecharge au 1er appel
    n_done = 0

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
                    if process_condition(path_condi, model):
                        n_done += 1
                except Exception as e:  # noqa: BLE001
                    print(f"  [erreur] {path_condi} : {e}")
                    traceback.print_exc()
                    continue

    print(f"\n{n_done} condition(s) recalee(s).")


if __name__ == "__main__":
    main()
