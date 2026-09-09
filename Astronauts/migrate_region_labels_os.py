"""
migrate_region_labels_os.py

Renomme les bandes laterales des tables DEJA ECRITES par
``compute_demons_strain.py``, pour appliquer la correction du 2 septembre 2026 :
la numerotation des bandes designe desormais un cote de la RETINE et non un cote
de l'IMAGE, ce qui impose de la MIROITER pour l'oeil gauche.

    oeil droit (OD) : bande +2 a gauche de la B-scan   (inchange)
    oeil gauche (OS): bande +2 a DROITE de la B-scan   (+2 <-> -2, +1 <-> -1)

Pourquoi une migration plutot qu'une relance du lot : les libelles ne changent
RIEN au calcul. ``regions.csv`` et ``slopes.csv`` portent, a cote de ``region``,
la colonne ``k`` -- l'indice GEOMETRIQUE de la bande, 0 a gauche de l'image, que
la correction ne touche pas. Le libelle est donc entierement redeductible de
``(k, oeil)``, et le recalculer coute quelques secondes la ou relancer les
demons couterait 10,7 h pour des nombres identiques.

Consequence pratique : ce script est IDEMPOTENT. Il ne permute pas des chaines,
il REGENERE le libelle depuis ``k`` -- le relancer deux fois donne le meme
resultat. ``compute_demons_strain.column_label(k, eye)`` est la seule definition
du libelle, et c'est elle que ce script appelle : une relance du lot produira
directement les bons libelles, sans repasser ici.

Ce qui est migre (sous ``SEGVAR_ROOT/<variante>/demons_strain/``) :
  - ``regions.csv``  (192 480 lignes)
  - ``slopes.csv``   (420 000 lignes)
Les autres tables du lot ne portent pas de colonne ``region``.

Une copie horodatee de chaque fichier est ecrite a cote avant reecriture.

A relancer ensuite, dans cet ordre :
    compute_sans_predictors.py
    reveal_quarto_presentations/figures_sans_predictors/make_figures.py

Lancer (kernel pyOR, depuis la racine du depot) :
    C:/Users/transformer/anaconda3/envs/pyOR/python.exe Astronauts/migrate_region_labels_os.py
"""

from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_demons_strain import OUT_DIR as DEMONS_DIR, column_label  # noqa: E402
from compute_rigidity_time_series import parse_eye  # noqa: E402

TABLES = ("regions.csv", "slopes.csv")


def libelles(df: pd.DataFrame) -> pd.Series:
    """Le libelle de bande de chaque ligne, deduit de ``k`` et de l'oeil."""
    eyes = {slug: parse_eye(slug.split("__", 1)[1]) for slug in df["slug"].unique()}
    return pd.Series(
        [column_label(int(k), eyes[slug]) for slug, k in zip(df["slug"], df["k"])],
        index=df.index)


def main() -> None:
    horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
    for nom in TABLES:
        chemin = DEMONS_DIR / nom
        if not chemin.exists():
            print(f"{chemin} absent -- ignore")
            continue
        df = pd.read_csv(chemin)
        if "k" not in df.columns or "region" not in df.columns:
            print(f"{nom} : pas de colonnes k/region -- ignore")
            continue
        neuf = libelles(df)
        n_change = int((neuf != df["region"]).sum())
        oeils = {slug: parse_eye(slug.split("__", 1)[1]) for slug in df["slug"].unique()}
        n_os = sum(1 for v in oeils.values() if v == "OS")
        print(f"{nom} : {len(df)} lignes, {len(oeils)} conditions "
              f"({n_os} OS), {n_change} libelles changes")
        if n_change == 0:
            print("  deja a jour -- rien a ecrire")
            continue
        sauvegarde = chemin.with_name(f"{chemin.stem}_avant_miroir_{horodatage}.csv")
        shutil.copy2(chemin, sauvegarde)
        df["region"] = neuf
        df.to_csv(chemin, index=False)
        print(f"  sauvegarde : {sauvegarde.name}")
        print(f"  reecrit    : {chemin}")


if __name__ == "__main__":
    main()
