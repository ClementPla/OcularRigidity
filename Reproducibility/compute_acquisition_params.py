# -*- coding: utf-8 -*-
"""
compute_acquisition_params.py

Inventaire des PARAMETRES D'ACQUISITION des acquisitions de l'etude de
REPETABILITE, lus dans les exports XML HEYEX qui accompagnent les .tif bruts.

Meme moteur que la cohorte SANS -- ``ocularrigidity.scripts.acquisition_params``
-- avec trois differences, et trois seulement :

  - OU sont les conditions      ``E:/SANSORI/Reproducibility/<NOM_PRENOM_OEilN>/``
                                (pas de niveau ``<...>_rigidity`` intermediaire :
                                un participant peut avoir plusieurs acquisitions
                                du meme oeil, et c'est le suffixe du dossier qui
                                les distingue, pas un dossier de moment)
  - COMMENT elles s'identifient participant / oeil / rang du replicat, la ou la
                                cohorte SANS a sujet / moment / oeil
  - OU vont les CSV             ``E:/NASA_Rigidity/Reproducibility/AcquisitionParameters/``

Les dossiers sont ceux ecrits par ``Reproducibility/migrate_repeatability.py`` ;
leur nom est la seule source du participant et du rang, et il est verifie contre
la lateralite du XML par la colonne ``laterality_matches_folder``.

Lancer (kernel pyOR, depuis la racine du depot) :
    C:/Users/transformer/anaconda3/envs/pyOR/python.exe \
        Reproducibility/compute_acquisition_params.py
"""

from __future__ import annotations

import re
from pathlib import Path

from ocularrigidity.scripts.acquisition_params import eye_of, run_batch

# --------------------------------------------------------------------------- #
# Parametres
# --------------------------------------------------------------------------- #
PATH_GENERAL = Path("E:/SANSORI/Reproducibility")
OUT_DIR = Path("E:/NASA_Rigidity/Reproducibility/AcquisitionParameters")

# Les dossiers de service ecrits a cote des acquisitions (manifeste de
# migration, et tout ce qu'on y ajouterait) commencent par un souligne.
SERVICE_PREFIX = "_"

# ``BELANGER_CHARLES_OD1`` -> participant / oeil / rang
RE_SLUG = re.compile(r"^(?P<participant>.+)_(?P<eye>OD|OS)(?P<replicate>\d+)$")


# --------------------------------------------------------------------------- #
# Chemins
# --------------------------------------------------------------------------- #
def iter_conditions():
    """Toutes les acquisitions ``E:/SANSORI/Reproducibility/<slug>/``."""
    if not PATH_GENERAL.is_dir():
        return
    for path_condi in sorted(PATH_GENERAL.iterdir()):
        if path_condi.is_dir() and not path_condi.name.startswith(SERVICE_PREFIX):
            yield path_condi


def identity_of(path_condi: Path) -> dict:
    """Colonnes d'identite, DANS L'ORDRE ou elles ouvrent les deux tables.

    Un nom qui ne se decompose pas garde tout de meme sa ligne : le participant
    vaut alors le nom entier et le rang est absent, ce qui se voit dans la table
    plutot que de faire disparaitre l'acquisition du lot.
    """
    slug = path_condi.name
    m = RE_SLUG.match(slug)
    return {
        "slug": slug,
        "participant": m.group("participant") if m else slug,
        "eye": m.group("eye") if m else eye_of(slug),
        "replicate": int(m.group("replicate")) if m else None,
        "path": str(path_condi),
    }


# --------------------------------------------------------------------------- #
# Lot
# --------------------------------------------------------------------------- #
def main():
    run_batch(
        conditions=list(iter_conditions()),
        identity_of=identity_of,
        out_dir=OUT_DIR,
        label_of=lambda p: p.name,
    )


if __name__ == "__main__":
    main()
