# -*- coding: utf-8 -*-
"""
compute_registration_metrics.py

Mesure, avec LA MEME definition que ``register_newmodel.py``, la qualite des
variantes de recalage deja ecrites a cote des donnees :

    E:/SANSORI/Reproducibility/<SLUG>/RawImages/<variante>/
        registered_video.mp4  mask.npz  transform.npz  registration_params.json

Critere : ``registration_metrics.ncc_to_median`` -- NCC de chaque frame recalee a
la mediane temporelle, sur la bande de choroide, dans la geometrie propre a
chaque variante. S'y ajoutent l'amplitude des transforms et les colonnes
noircies, relues de ``transform.npz``.

La variante cascade_v9 et la reference identite sont mesurees par
``register_newmodel.py`` au moment du recalage (il a le cube brut en memoire) ;
ce script ne relit que les variantes produites par d'autres lots. Il ne touche
pas au GPU et peut tourner pendant un recalage.

Sortie (interruptible, reprend ou il s'est arrete) :
    E:/NASA_Rigidity/Reproducibility/registration_comparison/metrics.csv
        une ligne par (acquisition, variante)

Lancer (depuis la racine du depot) :
    python Reproducibility/compute_registration_metrics.py
    python Reproducibility/compute_registration_metrics.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import registration_metrics as rm  # noqa: E402
from ocularrigidity.data.compression import read_gray  # noqa: E402
from ocularrigidity.data.io import load_mask  # noqa: E402

PATH_GENERAL = Path("E:/SANSORI/Reproducibility")
OUT_CSV = Path("E:/NASA_Rigidity/Reproducibility/registration_comparison/metrics.csv")
VARIANTS = (
    "registered_model1_flatten_choroid_xcorr",
    "registered_segformer2_flatten_choroid_xcorr",
    "registered_model1_fullframe",
)
RE_SLUG = re.compile(r"^(?P<participant>.+)_(?P<eye>OD|OS)(?P<replicate>\d+)$")


def list_slugs() -> list[str]:
    return sorted(
        d.name for d in PATH_GENERAL.iterdir()
        if d.is_dir() and RE_SLUG.match(d.name) and (d / "RawImages").is_dir()
    )


def measure(slug: str, variant: str) -> dict:
    d = PATH_GENERAL / slug / "RawImages" / variant
    row = {"slug": slug, "variant": variant}
    params_path = d / "registration_params.json"
    params = json.loads(params_path.read_text(encoding="utf-8")) if params_path.exists() else {}
    if params.get("status") != "ok" or not (d / "registered_video.mp4").exists():
        row.update(status=params.get("status", "absente"), reason=params.get("reason", ""))
        return row

    t0 = time.perf_counter()
    frames = read_gray(str(d / "registered_video.mp4"))
    masks = load_mask(d / "mask.npz")
    T = min(len(frames), len(masks))
    frames, masks = frames[:T], masks[:T]
    row.update(rm.summarize(rm.ncc_to_median(frames, masks), "ncc_reg"))

    area = masks.sum(axis=(1, 2)).astype(float)
    row.update(n_frames=int(T), mask_area_mean=float(area.mean()),
               mask_area_cv=float(area.std() / area.mean()) if area.mean() > 0 else np.nan,
               n_empty_mask_frames=int((area == 0).sum()))
    with np.load(d / "transform.npz") as tr:
        dx = np.asarray(tr["dx"], dtype=float)
        dy = np.asarray(tr["dy"], dtype=float)
        dy_bulk = dy.mean(axis=1) if dy.ndim == 2 else dy
        row.update(dx_ptp=float(np.ptp(dx)), dx_std=float(dx.std()),
                   dy_bulk_ptp=float(np.ptp(dy_bulk)), dy_bulk_std=float(dy_bulk.std()),
                   n_bad_columns=int(np.asarray(tr["bad_columns"]).sum())
                   if "bad_columns" in tr.files else np.nan)
    row.update(status="ok", reason="", seconds=round(time.perf_counter() - t0, 1))
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    rows = []
    if OUT_CSV.exists() and not args.overwrite:
        rows = pd.read_csv(OUT_CSV).to_dict("records")
    done = {(r["slug"], r["variant"]) for r in rows if r.get("status") == "ok"}
    rows = [r for r in rows if (r["slug"], r["variant"]) in done]

    slugs = list_slugs()
    todo = [(s, v) for s in slugs for v in variants if (s, v) not in done]
    print(f"{len(slugs)} acquisitions x {len(variants)} variantes : "
          f"{len(done)} deja mesurees, {len(todo)} a mesurer", flush=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    for i, (slug, variant) in enumerate(todo, 1):
        try:
            row = measure(slug, variant)
        except Exception as e:  # noqa: BLE001
            row = {"slug": slug, "variant": variant, "status": "error",
                   "reason": f"{type(e).__name__}: {str(e)[:200]}"}
        rows.append(row)
        pd.DataFrame(rows).sort_values(["slug", "variant"]).to_csv(OUT_CSV, index=False)
        ncc = row.get("ncc_reg_median")
        print(f"[{i:>3}/{len(todo)}] {slug:28} {variant:45} "
              f"{row['status']:>7} " + (f"NCC {ncc:.3f}" if isinstance(ncc, float) else
                                         str(row.get("reason", ""))), flush=True)
    print(f"\n{OUT_CSV}  ({len(rows)} lignes, {(time.perf_counter() - t_all) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
