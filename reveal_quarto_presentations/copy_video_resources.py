# -*- coding: utf-8 -*-
"""
Copie dans `_site` les .mp4 des dossiers `figures_*/` -- etape `post-render`.

Pourquoi ce script existe
-------------------------
Deux mecanismes de Quarto echouent l'un apres l'autre sur ces videos :

1. Le detecteur de ressources ne suit pas les balises `<video>` (il suit `<img>`,
   `<link>`, `<script>`). Les .mp4 ecrits en HTML brut dans les fragments generes
   sont donc invisibles pour lui -- c'est deja note dans `_quarto.yml`.
2. La parade habituelle, `project.resources` avec un glob, ne marche pas ICI :
   tous les `figures_*/` sont des JONCTIONS Windows vers
   `E:/NASA_Rigidity/quarto_results/` (cf. CLAUDE.md), et le parcours d'arbre qui
   resout ces globs ne traverse pas les jonctions. D'ou un `_site` sans aucun
   .mp4, alors que les .png du meme dossier arrivent bien : eux passent par la
   syntaxe markdown, donc par le detecteur, qui les copie chemin par chemin.

Le resultat etait des balises `<video>` pointant vers des fichiers absents.

Ce script recopie donc explicitement, apres le rendu, tout `figures_*/**/*.mp4`
vers le meme chemin relatif sous `_site/`. Il est deliberement generique : la
page `svd-simple-simulation-two-gaussians.qmd` souffrait du meme probleme
(`figures_deux_gaussiennes_svd/*.mp4` declares dans `resources`, jamais copies)
et est reparee par la meme passe.

Appele par `project.post-render` dans `_quarto.yml` ; se lance aussi a la main.
"""

import os
import shutil
import sys
from pathlib import Path

RACINE = Path(__file__).parent
# `QUARTO_PROJECT_OUTPUT_DIR` est relatif a la racine du projet quand Quarto
# appelle le script ; en execution manuelle on retombe sur `_site`.
SORTIE = RACINE / os.environ.get("QUARTO_PROJECT_OUTPUT_DIR", "_site")


def main() -> int:
    if not SORTIE.is_dir():
        print(f"[copy_video_resources] {SORTIE} absent : rien a faire.")
        return 0

    n, octets = 0, 0
    for dossier in sorted(RACINE.glob("figures_*")):
        if not dossier.is_dir():
            continue
        for src in dossier.rglob("*.mp4"):
            dst = SORTIE / src.relative_to(RACINE)
            # Ne recopier que si le contenu a change : un `quarto render` complet
            # ne doit pas reecrire 34 Mo de videos a chaque fois.
            if dst.exists() and dst.stat().st_size == src.stat().st_size:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            n += 1
            octets += dst.stat().st_size

    if n:
        print(f"[copy_video_resources] {n} video(s) copiee(s) "
              f"({octets / 1e6:.1f} Mo) -> {SORTIE}")
    else:
        print("[copy_video_resources] videos deja a jour.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
