# -*- coding: utf-8 -*-
"""
run_batch.py

Lance sur l'etude de REPETABILITE les memes lots que sur la cohorte SANS, sans
en dupliquer une ligne.

    python Reproducibility/run_batch.py registration   # segmente + recale
    python Reproducibility/run_batch.py pulse          # pouls + phases (SVD)
    python Reproducibility/run_batch.py strain         # demons -> strain retinien
    python Reproducibility/run_batch.py ct_pulse       # courbes CT et pouls (figures)
    python Reproducibility/run_batch.py all            # les quatre, dans l'ordre

Pourquoi un lanceur et non trois copies des scripts
---------------------------------------------------
Les trois lots de ``Astronauts/`` font DEJA exactement ce qu'il faut, avec les
memes reglages que l'experience SANS -- ce qui est le seul moyen de comparer
les deux etudes. Ce qui differe tient en quatre valeurs :

    OR_LAYOUT        "flat"  : un dossier par acquisition, pas trois niveaux
    OR_PATH_GENERAL  E:/SANSORI/Reproducibility
    OR_SEGVAR_ROOT   E:/NASA_Rigidity/Reproducibility/SegmentationVariations
    OR_VARIANT       la meme variante de segmentation que la cohorte SANS

Elles passent par l'ENVIRONNEMENT et non par des arguments, pour une raison
concrete : ``compute_demons_strain`` repartit les conditions sur dix processus,
et un processus fils re-importe le module sans rien savoir de ce que le parent
aurait change en memoire. L'environnement, lui, est herite. C'est aussi ce qui
garantit qu'un lot lance SANS ce fichier retombe exactement sur la cohorte SANS.

Les sorties sont sous une racine SEPAREE (``E:/NASA_Rigidity/Reproducibility/``)
plutot que melangees a celles de SANS : les deux etudes n'ont ni le meme
effectif, ni les memes conditions, et les tables de l'une ne doivent jamais
etre lues par erreur pour l'autre.

Duree indicative, mesuree sur la cohorte SANS : le recalage et le pouls sont
rapides, le strain par demons coute ~1 h de CPU par acquisition et tourne sur
dix processus -- comptez plusieurs heures pour ``strain`` sur 46 acquisitions.
Les trois lots sont INTERRUPTIBLES et reprennent ou ils se sont arretes.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ASTRO = REPO / "Astronauts"

# La variante de segmentation est celle de l'experience SANS : c'est ce qui rend
# les deux etudes comparables. La changer ici sans la changer la-bas romprait la
# seule chose que cette etude est censee borner.
ENV = {
    "OR_LAYOUT": "flat",
    "OR_PATH_GENERAL": "E:/SANSORI/Reproducibility",
    "OR_SEGVAR_ROOT": "E:/NASA_Rigidity/Reproducibility/SegmentationVariations",
    "OR_VARIANT": "model1_scale_1.0_flatten_choroid_xcorr",
}

STAGES = {
    "registration": "compute_registration.py",
    "pulse": "compute_pulse_from_data.py",
    "strain": "compute_demons_strain.py",
    "ct_pulse": "compute_ct_pulse_traces.py",
}
# ``ct_pulse`` vient APRES ``strain`` : il lit ``demons_strain/conditions.csv``
# pour savoir quelles conditions ont abouti, et ses fenetres temporelles.
ORDER = ["registration", "pulse", "strain", "ct_pulse"]


def run(stage: str) -> None:
    script = ASTRO / STAGES[stage]
    print(f"\n{'=' * 78}\n  {stage}  ->  {script.name}\n{'=' * 78}", flush=True)
    # ``runpy`` plutot qu'un sous-processus : le script s'execute avec
    # ``__name__ == '__main__'``, donc son ``main()`` part, et l'environnement
    # pose ci-dessous est deja en place au moment ou le module lit ses
    # constantes -- ce qu'un import classique ne garantirait pas.
    sys.argv = [str(script)]
    runpy.run_path(str(script), run_name="__main__")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1 or (argv[0] not in STAGES and argv[0] != "all"):
        print(__doc__)
        print("etapes : " + ", ".join(ORDER) + ", all")
        return 2

    os.environ.update(ENV)
    print("Etude de repetabilite -- lots de Astronauts/ avec :")
    for k, v in ENV.items():
        print(f"  {k:18} {v}")

    for stage in (ORDER if argv[0] == "all" else [argv[0]]):
        run(stage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
