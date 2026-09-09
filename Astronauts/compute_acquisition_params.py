# -*- coding: utf-8 -*-
"""
compute_acquisition_params.py

Inventaire des PARAMETRES D'ACQUISITION de toutes les conditions SANSORI, lus
dans les exports XML HEYEX qui accompagnent les .tif bruts.

Le moteur est partage : il vit dans
``ocularrigidity.scripts.acquisition_params`` et sert aussi a l'etude de
repetabilite (``Reproducibility/compute_acquisition_params.py``). Ce fichier ne
dit que trois choses, les seules qui different d'un lot a l'autre :

  - OU sont les conditions           -> ``iter_conditions``
  - COMMENT elles s'identifient      -> ``identity_of``
  - OU vont les CSV                  -> ``OUT_DIR``

Ce que « une condition » veut dire ici
--------------------------------------
Un dossier ``E:/SANSORI/<NN_id>/<...>_rigidity/<..._OD|OS...>/`` : un sujet, un
moment (before / post / R+...), un oeil, une acquisition.

Ou vont les resultats
---------------------
    E:/NASA_Rigidity/AcquisitionParameters/

Contrairement aux autres scripts de lot, la sortie n'est PAS rangee sous une
variante de segmentation : ce script ne consomme aucun masque, aucune image
recalee, aucun modele -- seulement le XML brut. La ranger sous
``SegmentationVariations/<variante>/`` suggererait une tracabilite qui n'existe
pas ici.

Lancer (kernel pyOR, depuis la racine du depot) :
    C:/Users/transformer/anaconda3/envs/pyOR/python.exe \
        Astronauts/compute_acquisition_params.py
"""

from __future__ import annotations

from pathlib import Path

from ocularrigidity.scripts.acquisition_params import eye_of, run_batch

# --------------------------------------------------------------------------- #
# Parametres
# --------------------------------------------------------------------------- #
PATH_GENERAL = Path("E:/SANSORI")
OUT_DIR = Path("E:/NASA_Rigidity/AcquisitionParameters")


# --------------------------------------------------------------------------- #
# Chemins
# --------------------------------------------------------------------------- #
def iter_conditions():
    """Toutes les conditions ``E:/SANSORI/<astro>/<...rigidity>/<condition>``.

    Le filtre ``*rigidity`` est ce qui garde l'etude de repetabilite
    (``E:/SANSORI/Reproducibility/``) hors de ce lot : elle n'a pas ce niveau
    intermediaire.
    """
    for path_astro in sorted(PATH_GENERAL.iterdir()):
        if not path_astro.is_dir():
            continue
        for path_moment in sorted(path_astro.iterdir()):
            if not path_moment.is_dir() or not path_moment.match("*rigidity"):
                continue
            for path_condi in sorted(path_moment.iterdir()):
                if path_condi.is_dir():
                    yield path_condi


def slug_of(astro: str, condition: str) -> str:
    """Meme convention que les autres scripts de lot (``pulse_from_data``)."""
    return f"{astro}__{condition}"


def identity_of(path_condi: Path) -> dict:
    """Colonnes d'identite, DANS L'ORDRE ou elles ouvrent les deux tables."""
    astro = path_condi.parent.parent.name
    condition = path_condi.name
    return {
        "slug": slug_of(astro, condition),
        "astro": astro,
        "moment": path_condi.parent.name,
        "condition": condition,
        "eye": eye_of(condition),
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
        label_of=lambda p: f"{p.parent.parent.name} / {p.name}",
    )


if __name__ == "__main__":
    main()
