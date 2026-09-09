# -*- coding: utf-8 -*-
"""
Ou sont les conditions, et sous quel nom -- pour les scripts de lot.

Deux arborescences coexistent et les memes lots doivent tourner sur les deux :

    SANSORI      E:/SANSORI/<astro>/<...>_rigidity/<...OD|OS...>/     3 niveaux
    plate        E:/SANSORI/Reproducibility/<NOM_PRENOM_OEilN>/       1 niveau

Les scripts de lot (``compute_registration``, ``compute_pulse_from_data``,
``compute_demons_strain``) identifient une condition par un TRIPLET
``(astro, moment, condition)``. Ce triplet sert a deux choses tres differentes
qu'il faut cesser de confondre :

  - a RETROUVER les donnees brutes -- et la, seule l'arborescence SANSORI a
    trois niveaux ; l'arborescence plate n'en a qu'un ;
  - a RANGER les sorties et a etiqueter les lignes des CSV -- et la, un triplet
    reste utile meme quand il ne correspond a aucun chemin reel.

Ce module rend donc les deux independants : le triplet devient une ETIQUETTE
(sur l'arborescence plate : ``Reproducibility / <participant> / <slug>``), et
c'est :func:`condition_dir` qui dit ou vivent reellement les images. Les sorties
d'un lot restent ainsi rangees a trois niveaux, lisibles et separees de celles
de la cohorte SANS, sans qu'aucun script n'ait a savoir laquelle des deux
arborescences il parcourt.

Reglage par VARIABLES D'ENVIRONNEMENT, et non par argument : ``compute_demons_strain``
repartit les conditions sur dix processus, et un processus fils re-importe le
module sans rien connaitre de ce que le parent aurait modifie en memoire.
L'environnement, lui, est herite.

    OR_LAYOUT         "sansori" (defaut) ou "flat"
    OR_PATH_GENERAL   racine des donnees brutes
    OR_SEGVAR_ROOT    racine des sorties (SegmentationVariations)
    OR_VARIANT        nom de la variante de segmentation

Non definies, elles laissent EXACTEMENT le comportement d'avant : c'est ce qui
permet de relancer la cohorte SANS sans y penser.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# ``BELANGER_CHARLES_OD1`` -> participant / oeil / rang. Meme expression que dans
# ``Reproducibility/*.py`` : c'est le nom de dossier qui porte l'identite.
RE_SLUG = re.compile(r"^(?P<participant>.+)_(?P<eye>OD|OS)(?P<replicate>\d+)$")

LAYOUT_SANSORI = "sansori"
LAYOUT_FLAT = "flat"


# --------------------------------------------------------------------------- #
# Lecture de l'environnement
# --------------------------------------------------------------------------- #
def layout() -> str:
    return os.environ.get("OR_LAYOUT", LAYOUT_SANSORI).strip().lower()


def env_path(name: str, default) -> Path:
    return Path(os.environ.get(name) or default)


def env_str(name: str, default: str) -> str:
    return os.environ.get(name) or default


def is_flat() -> bool:
    return layout() == LAYOUT_FLAT


# --------------------------------------------------------------------------- #
# Parcours
# --------------------------------------------------------------------------- #
def iter_condition_dirs(path_general: Path):
    """Dossiers de conditions, dans l'arborescence en vigueur.

    Sur l'arborescence SANSORI, le filtre ``*rigidity`` est ce qui garde
    ``E:/SANSORI/Reproducibility/`` hors des lots de la cohorte SANS. Sur
    l'arborescence plate, les dossiers de service (prefixe ``_``, comme
    ``_migration``) sont ecartes.
    """
    path_general = Path(path_general)
    if not path_general.is_dir():
        return
    if is_flat():
        for d in sorted(path_general.iterdir()):
            if d.is_dir() and not d.name.startswith("_"):
                yield d
        return
    for path_astro in sorted(path_general.iterdir()):
        if not path_astro.is_dir():
            continue
        for path_moment in sorted(path_astro.iterdir()):
            if not path_moment.is_dir() or not path_moment.match("*rigidity"):
                continue
            for path_condi in sorted(path_moment.iterdir()):
                if path_condi.is_dir():
                    yield path_condi


def labels_of(path_condi: Path, path_general: Path) -> tuple[str, str, str]:
    """``(astro, moment, condition)`` -- ETIQUETTES, pas forcement des chemins.

    Sur l'arborescence plate, ``astro`` vaut le nom de la racine
    (``Reproducibility``) et ``moment`` le participant : les sorties se rangent
    donc sous ``Reproducibility/<participant>/<slug>/``, ce qui les separe
    naturellement de la cohorte SANS et reste lisible dans un CSV.
    """
    path_condi, path_general = Path(path_condi), Path(path_general)
    if not is_flat():
        return (path_condi.parent.parent.name, path_condi.parent.name,
                path_condi.name)
    slug = path_condi.name
    m = RE_SLUG.match(slug)
    return (path_general.name, m.group("participant") if m else slug, slug)


def condition_dir(astro: str, moment: str, condition: str,
                  path_general: Path) -> Path:
    """Le chemin REEL des donnees brutes, a partir des etiquettes.

    C'est l'inverse de :func:`labels_of`, et le seul endroit ou l'arborescence
    plate se distingue : ``moment`` y est un participant, pas un dossier.
    """
    path_general = Path(path_general)
    if is_flat():
        return path_general / condition
    return path_general / astro / moment / condition


def slug_of(astro: str, condition: str) -> str:
    """Identifiant d'une condition dans les CSV de lot.

    Sur l'arborescence plate le nom de dossier est deja unique et porte le
    participant : le prefixer de ``Reproducibility__`` n'ajouterait rien et
    alourdirait chaque etiquette de figure.
    """
    if is_flat():
        return condition
    return f"{astro}__{condition}"


def describe() -> str:
    """Une ligne a imprimer en tete de lot -- ce qui va etre lu et ecrit."""
    return (f"layout={layout()}  "
            f"OR_PATH_GENERAL={os.environ.get('OR_PATH_GENERAL', '(defaut)')}  "
            f"OR_SEGVAR_ROOT={os.environ.get('OR_SEGVAR_ROOT', '(defaut)')}  "
            f"OR_VARIANT={os.environ.get('OR_VARIANT', '(defaut)')}")
