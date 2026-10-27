# -*- coding: utf-8 -*-
"""
Ou sont les conditions, et sous quel nom -- pour les scripts de lot.

Deux arborescences coexistent et les memes lots doivent tourner sur les deux :

    SANSORI      E:/SANSORI/<astro>/<...>_rigidity/<...OD|OS...>/RawImages/
                 E:/SANSORI/<astro>/<...>_rigidity/<...OD|OS...>/RawImages/registered*/
    plate        E:/SANSORI/Reproducibility/RawImages/<NOM_PRENOM_OEilN>/
                 E:/SANSORI/Reproducibility/registered*/<NOM_PRENOM_OEilN>/

Sur l'arborescence plate, les donnees brutes (.tif + .xml) et chaque variante de
recalage sont donc des dossiers FRERES sous la racine, avec la meme liste de
sous-dossiers par acquisition ; sur l'arborescence SANSORI, les variantes sont
rangees DANS ``RawImages/``. Un script de lot ne compose jamais ces chemins
lui-meme : il passe par :func:`raw_dir` et :func:`data_dir`.

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

# Dossier des donnees brutes : sous la condition (SANSORI), a la racine (plate).
RAW_SUBDIR = "RawImages"

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
    l'arborescence plate, une condition EST son dossier de donnees brutes
    ``<racine>/RawImages/<slug>/`` -- cf. :func:`flat_slugs`.
    """
    path_general = Path(path_general)
    if not path_general.is_dir():
        return
    if is_flat():
        for slug in flat_slugs(path_general):
            yield flat_raw_dir(path_general, slug)
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
        return flat_raw_dir(path_general, condition)
    return path_general / astro / moment / condition


def raw_dir(path_condi: Path) -> Path | None:
    """Dossier des .tif + .xml d'une condition, ``None`` s'il n'existe pas.

    Sur l'arborescence plate, ``path_condi`` est deja ce dossier. Sur
    l'arborescence SANSORI, c'est ``RawImages/`` ou, a defaut, ``RawData/``.
    """
    path_condi = Path(path_condi)
    if is_flat():
        return path_condi if path_condi.is_dir() else None
    for name in (RAW_SUBDIR, "RawData"):
        d = path_condi / name
        if d.is_dir():
            return d
    return None


def data_dir(path_condi: Path, subdir: str) -> Path:
    """Dossier d'une variante de recalage (``registered*``) d'une condition.

    SANSORI : ``<condition>/RawImages/<subdir>/`` ; plate :
    ``<racine>/<subdir>/<slug>/``, ou ``<racine>`` est le parent de ``RawImages``.
    """
    path_condi = Path(path_condi)
    if is_flat():
        return flat_data_dir(path_condi.parent.parent, path_condi.name, subdir)
    return path_condi / RAW_SUBDIR / subdir


# --------------------------------------------------------------------------- #
# Arborescence plate, a racine explicite (independant de OR_LAYOUT)
# --------------------------------------------------------------------------- #
def flat_raw_dir(root: Path, slug: str) -> Path:
    """``<racine>/RawImages/<slug>/`` -- les .tif + le .xml d'une acquisition."""
    return Path(root) / RAW_SUBDIR / slug


def flat_data_dir(root: Path, slug: str, subdir: str) -> Path:
    """``<racine>/<subdir>/<slug>/`` -- une variante de recalage d'une acquisition."""
    return Path(root) / subdir / slug


def flat_slugs(root: Path) -> list[str]:
    """Acquisitions presentes : les sous-dossiers de ``<racine>/RawImages/``
    dont le nom a la forme ``NOM_PRENOM_OD|OS<rang>``, tries."""
    d = Path(root) / RAW_SUBDIR
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir() and RE_SLUG.match(p.name))


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
