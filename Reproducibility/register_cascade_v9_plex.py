# -*- coding: utf-8 -*-
"""
register_cascade_v9_plex.py

Segmentation SegFormer + recalage cascade_v9 des acquisitions de repetabilite sur
la GRILLE PLEX, toutes variantes (moteur :
``ocularrigidity.scripts.registration.cascade_v9_plex``) :

    E:/SANSORI/Reproducibility/RawImages/<NOM_PRENOM_OD|OS+N>/            (entree)
    E:/SANSORI/Reproducibility/registered_cascade_v9_plex_{w1024,full}[_native]/<NOM_PRENOM_OD|OS+N>/

Suivi : ``E:/NASA_Rigidity/Reproducibility/SegmentationVariations/cascade_v9_plex_<variante>/registration_summary.csv``.

Lancer (kernel pyOR, depuis la racine) :
    python Reproducibility/register_cascade_v9_plex.py --slug MODICA_GRAZIANA_OS3
    python Reproducibility/register_cascade_v9_plex.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Ordre impose par les DLL de l'env pyOR : pandas, puis torch, puis le reste.
import pandas  # noqa: F401
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import registration_metrics as rm  # noqa: E402

from ocularrigidity.scripts.registration import cascade_v9_plex as engine  # noqa: E402
from ocularrigidity.scripts.batch_layout import flat_data_dir, flat_raw_dir, flat_slugs  # noqa: E402

PATH_GENERAL = Path("E:/SANSORI/Reproducibility")
SEGVAR_ROOT = Path("E:/NASA_Rigidity/Reproducibility/SegmentationVariations")
RE_SLUG = re.compile(r"^(?P<participant>.+)_(?P<eye>OD|OS)(?P<replicate>\d+)$")
ID_COLS = ["slug", "participant", "eye", "replicate"]


def items():
    out = []
    for slug in flat_slugs(PATH_GENERAL):
        m = RE_SLUG.match(slug)
        out.append((slug, flat_raw_dir(PATH_GENERAL, slug),
                    {"slug": slug, "participant": m["participant"], "eye": m["eye"],
                     "replicate": int(m["replicate"])}))
    return out


def out_dir_of(raw_dir: Path, subdir: str) -> Path:
    """``<racine>/<variante>/<slug>/`` : les variantes sont soeurs de ``RawImages/``."""
    return flat_data_dir(PATH_GENERAL, Path(raw_dir).name, subdir)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SegFormer + cascade_v9 sur grille PLEX, repetabilite.")
    ap.add_argument("--slug", action="append")
    ap.add_argument("--variant", choices=list(engine.VARIANTS), action="append")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    if not torch.cuda.is_available():
        print("CUDA requis.")
        return 2
    all_items = items()
    run = [it for it in all_items if not args.slug or it[0] in args.slug]
    unknown = set(args.slug or []) - {it[0] for it in all_items}
    if unknown:
        print(f"Acquisition(s) introuvable(s) : {', '.join(sorted(unknown))}")
        return 2
    engine.run_batch(run, SEGVAR_ROOT, ID_COLS, rm.ncc_to_median, rm.summarize, REPO,
                     variants=args.variant, overwrite=args.overwrite, limit=args.limit,
                     out_dir_of=out_dir_of)
    engine.write_summaries(all_items, SEGVAR_ROOT, ID_COLS, out_dir_of)
    return 0


if __name__ == "__main__":
    sys.exit(main())
