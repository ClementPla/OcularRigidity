# -*- coding: utf-8 -*-
"""
register_spectralis_native.py

Segmentation + recalage de la cohorte SANS par ``ocularrigidity.data.spectralis.process_condition`` :
B-scans bruts (aucun pretraitement) -> grille PLEX -> inference (modeles Hub courants,
``pipeline_config.REGISTRATION``) -> retour a la grille native.

    E:/SANSORI/<NN_id>/<...>_rigidity/<...>/RawImages/registered_upernet_native/
        registered_video.mp4, mask.npz, prob.npz, timestamp.txt, transform.npz,
        registration_params.json

Suivi : ``E:/NASA_Rigidity/SegmentationVariations/upernet_native/registration_summary.csv``.
Le GUI PyQt lit ces dossiers sous « UPerNet natif (data.spectralis) ».

Lancer (kernel pyOR, depuis la racine) :
    python Astronauts/register_spectralis_native.py --limit 1
    python Astronauts/register_spectralis_native.py --only 210920001before_rigidity_OD2
    python Astronauts/register_spectralis_native.py
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

# Ordre impose par les DLL de l'env pyOR : pandas, puis torch, puis le reste.
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "Reproducibility"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import registration_metrics as rm  # noqa: E402
from register_cascade_v9 import PATH_GENERAL, RE_COND, list_conditions  # noqa: E402

from ocularrigidity.consts import SEGMENTATION_REVISION  # noqa: E402
from ocularrigidity.data.spectralis import process_condition  # noqa: E402
from ocularrigidity.pipeline_config import REGISTRATION  # noqa: E402
from ocularrigidity.scripts.registration.cascade_v9_plex import patch_largest_cc  # noqa: E402
from ocularrigidity.segmentation.utils import get_model, get_registration_model  # noqa: E402

SUBDIR = "registered_upernet_native"
SUMMARY = Path("E:/NASA_Rigidity/SegmentationVariations/upernet_native/registration_summary.csv")
KEYS = ["status", "reason", "n_frames", "height", "width", "fps", "ref_idx", "ncc_reg_median",
        "ncc_reg_n_lt_0.5", "iou_ref_median", "n_empty_mask_frames", "seconds", "created"]


def conditions():
    out = []
    for rel in list_conditions():
        m = RE_COND.match(rel)
        out.append({"condition": rel, "id": m["id"], "moment": m["moment"],
                    "eye": m["eye"] + m["rep"], "raw": PATH_GENERAL / rel / "RawImages"})
    return out


def params_of(raw: Path) -> dict | None:
    try:
        return json.loads((raw / SUBDIR / "registration_params.json").read_text("utf-8"))
    except Exception:  # noqa: BLE001
        return None


def write_summary(conds):
    rows = [{**{k: c[k] for k in ("condition", "id", "moment", "eye")},
             **{k: (params_of(c["raw"]) or {}).get(k) for k in KEYS}} for c in conds]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(SUMMARY, index=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--only", action="append", help="nom de dossier de condition")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    # Sous Windows, un lot trop gros deborde en memoire partagee SANS lever d'OOM
    # et le calcul rampe : 8 (defaut de REGISTRATION) a 992 x 1472 remplit les 12 Go.
    ap.add_argument("--batch", type=int, default=2, help="fused_batch_size")
    args = ap.parse_args(argv)
    if not torch.cuda.is_available():
        print("CUDA requis.")
        return 2
    conds = conditions()
    todo = [c for c in conds if (not args.only or c["condition"].rsplit("/", 1)[-1] in args.only)
            and (args.overwrite or (params_of(c["raw"]) or {}).get("status") != "ok")]
    todo = todo[: args.limit] if args.limit is not None else todo
    print(f"{len(conds)} condition(s), {len(todo)} a traiter -> RawImages/{SUBDIR}", flush=True)
    if not todo:
        write_summary(conds)
        return 0
    torch.backends.cudnn.benchmark = True
    # La sonde de reference de fused filtre toujours la plus grande composante :
    # cuCIM n'existe pas sous Windows -> cupyx.scipy.ndimage (GPU), sinon cc3d (CPU).
    print("composante connexe :", patch_largest_cc(), flush=True)
    config = replace(REGISTRATION, fused_batch_size=args.batch)
    seg = get_model("cuda")
    reg = get_registration_model(REGISTRATION.registrator_revision).to("cuda")
    for i, c in enumerate(todo, 1):
        out_dir = c["raw"] / SUBDIR
        t0 = time.perf_counter()
        try:
            r = process_condition(c["raw"], seg, reg, config=config, out_dir=out_dir)
            area = r["masks"].sum(axis=(1, 2))
            iou = r["frame_iou"]
            p = json.loads((out_dir / "registration_params.json").read_text("utf-8"))
            p.update({"condition": c["condition"],
                      "seg_revision": SEGMENTATION_REVISION,  # ceux charges plus haut
                      "reg_revision": REGISTRATION.registrator_revision,
                      **rm.summarize(rm.ncc_to_median(r["frames"], r["masks"]), "ncc_reg"),
                      "iou_ref_median": None if iou is None else float(pd.Series(iou).median()),
                      "n_empty_mask_frames": int((area == 0).sum()),
                      "seconds": round(time.perf_counter() - t0, 1)})
            (out_dir / "registration_params.json").write_text(json.dumps(p, indent=2, default=str),
                                                             encoding="utf-8")
            print(f"[{i:>3}/{len(todo)}] {c['condition']} | {p['seconds']:.0f} s | "
                  f"NCC {p['ncc_reg_median']:.3f} (<0,5 : {p['ncc_reg_n_lt_0.5']}) "
                  f"IoU {p['iou_ref_median']:.3f}", flush=True)
        except Exception as e:  # noqa: BLE001 - une condition ne doit pas tout arreter
            traceback.print_exc()
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "registration_params.json").write_text(json.dumps(
                {"status": "error", "reason": f"{type(e).__name__}: {str(e)[:300]}"}, indent=2),
                encoding="utf-8")
            print(f"[{i:>3}/{len(todo)}] {c['condition']} ERREUR {e}", flush=True)
        r = None
        gc.collect()
        torch.cuda.empty_cache()
        write_summary(conds)
    write_summary(conds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
