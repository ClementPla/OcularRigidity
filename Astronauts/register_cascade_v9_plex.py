# -*- coding: utf-8 -*-
"""
register_cascade_v9_plex.py

Segmentation SegFormer + recalage cascade_v9 de la cohorte SANS sur la GRILLE
PLEX, toutes variantes (moteur : ``ocularrigidity.scripts.registration.cascade_v9_plex``) :

    E:/SANSORI/<NN_id>/<...>_rigidity/<...>/RawImages/
        registered_cascade_v9_plex_w1024/          grille PLEX, 1024 colonnes (~6 mm)
        registered_cascade_v9_plex_w1024_native/   la meme passe, ramenee a 496 x 768
        registered_cascade_v9_plex_full/           grille PLEX, toute la largeur
        registered_cascade_v9_plex_full_native/    la meme passe, ramenee a 496 x 768

Suivi : ``E:/NASA_Rigidity/SegmentationVariations/cascade_v9_plex_<variante>/registration_summary.csv``.
Conditions ``before`` d'abord ; une condition deja complete est sautee.

Lancer (kernel pyOR, depuis la racine) :
    python Astronauts/register_cascade_v9_plex.py --limit 1          # essai
    python Astronauts/register_cascade_v9_plex.py --moment before    # avant vol
    python Astronauts/register_cascade_v9_plex.py                    # tout
    python Astronauts/register_cascade_v9_plex.py --variant w1024    # une seule largeur
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ordre impose par les DLL de l'env pyOR : pandas, puis torch, puis le reste.
import pandas  # noqa: F401
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "Reproducibility"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import registration_metrics as rm  # noqa: E402
from register_cascade_v9 import MOMENT_ORDER, PATH_GENERAL, RE_COND, list_conditions  # noqa: E402

from ocularrigidity.scripts.registration import cascade_v9_plex as engine  # noqa: E402

SEGVAR_ROOT = Path("E:/NASA_Rigidity/SegmentationVariations")
ID_COLS = ["condition", "id", "moment", "eye"]


def items(moments=None):
    out = []
    for rel in list_conditions():
        m = RE_COND.match(rel)
        if moments and m["moment"] not in moments:
            continue
        ident = {"condition": rel, "id": m["id"], "moment": m["moment"],
                 "eye": m["eye"] + m["rep"], "slug": rel}
        out.append((rel.rsplit("/", 1)[-1], PATH_GENERAL / rel / "RawImages", ident))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SegFormer + cascade_v9 sur grille PLEX, cohorte SANS.")
    ap.add_argument("--moment", choices=sorted(MOMENT_ORDER), action="append")
    ap.add_argument("--variant", choices=list(engine.VARIANTS), action="append")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    if not torch.cuda.is_available():
        print("CUDA requis.")
        return 2
    # Le resume couvre toujours toute la cohorte, quel que soit le filtre.
    all_items = items()
    run = items(args.moment)
    keys = {it[0] for it in run}
    order = [it for it in all_items if it[0] in keys]
    engine.run_batch(order, SEGVAR_ROOT, ID_COLS, rm.ncc_to_median, rm.summarize, REPO,
                     variants=args.variant, overwrite=args.overwrite, limit=args.limit)
    engine.write_summaries(all_items, SEGVAR_ROOT, ID_COLS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
