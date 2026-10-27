# -*- coding: utf-8 -*-
"""
register_cascade_v9.py

Recalage appris cascade_v9 sur la cohorte SANS (astronautes), avec le MEME moteur
que ``Reproducibility/register_newmodel.py`` (dont ``process`` est repris tel quel) :

    E:/SANSORI/<NN_id>/<...>_rigidity/<..._OD|OS...>/RawImages/registered_cascade_v9_1536x1024/
        registered_video.mp4  mask.npz  timestamp.txt  transform.npz
        registration_params.json

Le masque est celui de la tete de segmentation qui partage l'encodeur du recaleur
(``segment_and_register``, une seule passe) : aucun autre modele de segmentation
n'est appele. Il est reduit a la resolution native par moyenne de surface, puis
deforme par le meme transform que les frames.

Les conditions ``before_rigidity`` passent en premier, puis ``post_rigidity``.
Une condition deja complete est sautee (``--overwrite`` pour refaire).
Suivi : ``E:/NASA_Rigidity/SegmentationVariations/cascade_v9_1536x1024/registration_summary.csv``.

Lancer (kernel pyOR, depuis la racine) :
    python Astronauts/register_cascade_v9.py                    # toutes les conditions
    python Astronauts/register_cascade_v9.py --moment before     # avant vol seulement
    python Astronauts/register_cascade_v9.py --limit 1           # essai sur une condition
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
import traceback
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Reproducibility"))

import register_newmodel as rn  # noqa: E402

PATH_GENERAL = Path("E:/SANSORI")
SEGVAR_ROOT = Path("E:/NASA_Rigidity/SegmentationVariations")
RESIZE = (1536, 1024)
SUBDIR = f"{rn.OUTPUT_SUBDIR}_{RESIZE[0]}x{RESIZE[1]}"
MOMENT_ORDER = {"before": 0, "post": 1}

RE_COND = re.compile(r"^(?P<nn>\d+)_(?P<id>\d+)/(?P=id)(?P<moment>[a-z]+)_rigidity/"
                     r"(?P=id)(?P=moment)_rigidity_(?P<eye>OD|OS)(?P<rep>\d*)$")

# Le moteur resout tout par rapport a ces deux racines : on les pointe sur la cohorte SANS.
rn.PATH_GENERAL = PATH_GENERAL
rn.SEGVAR_ROOT = SEGVAR_ROOT


def list_conditions() -> list[str]:
    """Chemins relatifs ``<NN_id>/<id><moment>_rigidity/<...>`` ayant un RawImages."""
    out = []
    for raw in PATH_GENERAL.glob("*/*rigidity/*/RawImages"):
        rel = raw.parent.relative_to(PATH_GENERAL).as_posix()
        if raw.is_dir() and RE_COND.match(rel):
            out.append(rel)
    return sorted(out, key=lambda r: (MOMENT_ORDER.get(RE_COND.match(r)["moment"], 9), r))


def write_summary(conds: list[str]) -> Path:
    rows = []
    for cond in conds:
        params = rn.read_params(PATH_GENERAL / cond / "RawImages" / SUBDIR)
        if params:
            m = RE_COND.match(cond)
            ident = {"condition": cond, "id": m["id"], "moment": m["moment"],
                     "eye": m["eye"] + m["rep"]}
            rows.append(ident | {k: params.get(k) for k in rn.SUMMARY_KEYS if k != "slug"})
    cols = ["condition", "id", "moment", "eye"] + [k for k in rn.SUMMARY_KEYS if k != "slug"]
    out = SEGVAR_ROOT / SUBDIR.removeprefix("registered_") / "registration_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=cols).to_csv(out, index=False)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Recalage appris cascade_v9, cohorte SANS.")
    parser.add_argument("--moment", choices=sorted(MOMENT_ORDER), action="append",
                        help="restreint a before et/ou post (repetable)")
    parser.add_argument("--limit", type=int, default=None, help="n premieres conditions a traiter")
    parser.add_argument("--fused-batch", type=int, default=rn.RESIZE_DEFAULT_FUSED_BATCH)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("Le recalage appris exige CUDA.")
        return 2
    all_conds = list_conditions()
    conds = [c for c in all_conds
             if not args.moment or RE_COND.match(c)["moment"] in args.moment]
    todo = [c for c in conds
            if args.overwrite or not rn.is_complete(PATH_GENERAL / c / "RawImages" / SUBDIR)]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"{len(conds)} condition(s), {len(todo)} a traiter -> RawImages/{SUBDIR}/", flush=True)
    if not todo:
        print(f"suivi : {write_summary(all_conds)}")
        return 0

    config = replace(rn.REGISTRATION, fused_batch_size=args.fused_batch)
    seg_model = rn.get_choroid_segmentation_model().to(rn.DEVICE)
    reg_model = rn.get_registration_model().to(rn.DEVICE)
    trained_shape = tuple(int(v) for v in reg_model.img_shape)
    cc_backend = rn.patch_largest_cc()
    torch.backends.cudnn.benchmark = True
    print(f"modeles : {rn.SEG_REPO}@{rn.SEG_REVISION}, {rn.REG_REPO}@{rn.REG_REVISION} | "
          f"agrandi {RESIZE[0]}x{RESIZE[1]} | fused_batch_size {config.fused_batch_size} | "
          f"composante connexe : {cc_backend}\n", flush=True)

    t_batch = time.perf_counter()
    n_ok = n_err = 0
    for i, cond in enumerate(todo, 1):
        t0 = time.perf_counter()
        try:
            p = rn.process(cond, seg_model, reg_model, trained_shape, RESIZE, config,
                           cc_backend, SUBDIR)
        except Exception as e:  # noqa: BLE001 - une condition ne doit pas tout arreter
            traceback.print_exc()
            out_dir = PATH_GENERAL / cond / "RawImages" / SUBDIR
            out_dir.mkdir(parents=True, exist_ok=True)
            p = {"variant": SUBDIR, "slug": cond, "condition": cond, "status": "error",
                 "reason": f"{type(e).__name__}: {str(e)[:300]}",
                 "created": datetime.now().isoformat(timespec="seconds")}
            (out_dir / "registration_params.json").write_text(
                json.dumps(p, indent=2), encoding="utf-8")
        gc.collect()
        torch.cuda.empty_cache()
        dt = time.perf_counter() - t0
        name = cond.rsplit("/", 1)[-1]
        if p.get("status") == "ok":
            n_ok += 1
            print(f"[{i:>3}/{len(todo)}] {name:34} ok | T {p['n_frames']:4d} | "
                  f"NCC {p['ncc_reg_median']:.3f} (id {p['ncc_identity_median']:.3f}) | "
                  f"<0,5 : {p['ncc_reg_n_lt_0.5']:3d} | dy ptp {p['dy_bulk_ptp']:5.2f} | "
                  f"{dt:.0f} s", flush=True)
        else:
            n_err += 1
            print(f"[{i:>3}/{len(todo)}] {name:34} {p.get('status')} : {p.get('reason')} | "
                  f"{dt:.0f} s", flush=True)
        write_summary(all_conds)

    out = write_summary(all_conds)
    print(f"\n{n_ok} ok, {n_err} en echec ({(time.perf_counter() - t_batch) / 60:.1f} min)")
    print(f"suivi : {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
