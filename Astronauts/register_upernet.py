# -*- coding: utf-8 -*-
"""
register_upernet.py

Version script de ``notebook/register_and_segment_uppernet_test.ipynb``, sur toute la
cohorte SANS. Le calcul est celui du notebook, sans changement :

    load_video(RawImages)                         B-scans natifs, ordre du XML
    to_plex_grid(video, 0.0039, 0.0116)           grille PLEX, 1024 colonnes au centre
    segment_and_register(..., config=REGISTRATION)
        segmentation get_model()                  (consts.SEGMENTATION_REVISION)
        recalage reg_upernet_convnext_t_bmlines

APRES le calcul seulement, frames recalees et masques sont ramenes a la resolution
d'origine (``registration.plex_grid.plex_to_native`` : meme geometrie que
``to_plex_grid``, verifiee a chaque condition) ; les colonnes natives hors des 1024
colonnes PLEX sont mises a 0.

    E:/SANSORI/<NN_id>/<...>_rigidity/<...>/RawImages/upernet/
        registered_video.mp4      frames recalees, resolution d'origine
        mask.npz                  masques recales (save_mask), resolution d'origine
        timestamp.txt             horodatages (us), ordre des frames
        transform.npz             dx, dy en pixels natifs, colonnes hors recadrage, ref_idx
        registration_params.json  status "ok" ecrit en dernier

Le GUI PyQt (``Astronauts/data_visualizer_GUI``) lit ce dossier sous « UPerNet (notebook) ».

Lancer (kernel pyOR, depuis la racine) :
    python Astronauts/register_upernet.py --limit 1      # essai, donne le temps par condition
    python Astronauts/register_upernet.py                # toute la cohorte
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Ordre impose par les DLL de l'env pyOR : pandas, puis torch, puis le reste.
import pandas as pd
import torch

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ocularrigidity.consts import (  # noqa: E402
    AXIAL_PIXEL_SIZE_MM,
    PLEX_WIDTH,
    SEGMENTATION_REVISION,
    TRANVERSAL_PIXEL_SIZE_MM,
)
from ocularrigidity.data.compression import cube_to_mp4  # noqa: E402
from ocularrigidity.data.io import save_mask  # noqa: E402
from ocularrigidity.data.spectralis import load_video  # noqa: E402
from ocularrigidity.pipeline_config import REGISTRATION  # noqa: E402
from ocularrigidity.registration.fused import segment_and_register  # noqa: E402
from ocularrigidity.registration.plex_grid import (  # noqa: E402
    native_coverage,
    plex_geometry,
    plex_to_native,
    transform_to_native,
)
from ocularrigidity.segmentation.utils import get_model, get_registration_model  # noqa: E402

PATH_GENERAL = Path("E:/SANSORI")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_SUBDIR = "upernet"
REG_REVISION = "reg_upernet_convnext_t_bmlines"
# Tailles de pixel natives passees a to_plex_grid, comme dans le notebook.
AXIAL_MM, LATERAL_MM = 0.0039, 0.0116
SUMMARY = Path("E:/NASA_Rigidity/SegmentationVariations/upernet/registration_summary.csv")
NATIVE_MIN_COVERAGE = 0.999  # colonne native gardee si entierement dans le recadrage


def to_plex_grid(frames, axial_mm, lateral_mm, width=PLEX_WIDTH):
    """Resample B-scans to the PLEX pixel size the networks were trained on."""
    T, H, W = frames.shape
    new_h = int(round(H * axial_mm / AXIAL_PIXEL_SIZE_MM / 32)) * 32
    new_w = int(round(W * lateral_mm / TRANVERSAL_PIXEL_SIZE_MM))
    out = np.stack(
        [cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_CUBIC) for f in frames]
    )
    width = min(width, new_w // 32 * 32)
    x0 = (new_w - width) // 2
    out = np.ascontiguousarray(out[:, :, x0 : x0 + width])
    return out, H * axial_mm / new_h, W * lateral_mm / new_w


def conditions() -> list[Path]:
    """Meme parcours que le notebook (*rigidity), toutes les conditions."""
    out = []
    for path_astro in sorted(PATH_GENERAL.iterdir()):
        if not path_astro.is_dir():
            continue
        for path_moment in sorted(path_astro.iterdir()):
            if not path_moment.match("*rigidity"):
                continue
            for path_condi in sorted(path_moment.iterdir()):
                if path_condi.is_dir() and (path_condi / "RawImages").is_dir():
                    out.append(path_condi)
    return out


def is_done(out_dir: Path) -> bool:
    try:
        p = json.loads((out_dir / "registration_params.json").read_text("utf-8"))
    except Exception:  # noqa: BLE001
        return False
    return p.get("status") == "ok" and (out_dir / "registered_video.mp4").exists() \
        and (out_dir / "mask.npz").exists()


def process(path_condi: Path, seg_model, reg_model) -> dict:
    t0 = time.perf_counter()
    # --- calcul : celui du notebook ----------------------------------------------
    this_vid, t_stamps, df = load_video(path_condi / "RawImages", return_fundus=False)
    this_vid_plex_grid, scale_axial, scale_lat = to_plex_grid(this_vid, AXIAL_MM, LATERAL_MM)
    result = segment_and_register(this_vid_plex_grid, seg_model, reg_model, config=REGISTRATION,
                                  device=DEVICE, verbose=False)
    t_calc = time.perf_counter() - t0

    # --- retour a la resolution d'origine ------------------------------------------
    T, H, W = this_vid.shape
    g = plex_geometry(H, W, AXIAL_MM, LATERAL_MM, PLEX_WIDTH)
    if g.shape != this_vid_plex_grid.shape[1:]:
        raise RuntimeError(f"geometrie PLEX {g.shape} != to_plex_grid {this_vid_plex_grid.shape[1:]}")
    del this_vid_plex_grid
    outside = native_coverage(g) < NATIVE_MIN_COVERAGE
    frames = np.clip(np.round(plex_to_native(result.registered_frames, g)), 0, 255).astype(np.uint8)
    masks = plex_to_native(result.registered_masks, g, cv2.INTER_NEAREST) > 0
    # frames[..., outside] = 0
    # masks[..., outside] = False
    dx, dy = transform_to_native(result.transform["dx"], result.transform["dy"], g)

    # --- ecriture -----------------------------------------------------------------
    out_dir = path_condi / "RawImages" / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "registration_params.json").unlink(missing_ok=True)  # incomplet tant qu'absent
    sec = df["seconds_of_day"].to_numpy(dtype=float)
    ts_us = np.round((sec - sec[0]) * 1e6).astype(np.int64)
    dts = np.diff(sec)
    dts = dts[dts > 0]
    fps = float(1.0 / np.median(dts)) if dts.size else 30.0
    cube_to_mp4(frames, out_dir / "registered_video.mp4", fps=fps, crf=25)
    save_mask(masks, out_dir / "mask.npz")
    (out_dir / "timestamp.txt").write_text("\n".join(str(int(t)) for t in ts_us), encoding="utf-8")
    np.savez(out_dir / "transform.npz", dx=dx, dy=dy, outside_crop=outside,
             ref_idx=np.int64(result.ref_idx),
             frame_iou=result.frame_iou if result.frame_iou is not None else np.array([]),
             pixel_mm=np.array([AXIAL_MM, LATERAL_MM]))
    iou = result.frame_iou
    params = {
        "status": "ok", "grid": "native", "condition": path_condi.relative_to(PATH_GENERAL).as_posix(),
        "engine": "Astronauts/register_upernet.py (notebook register_and_segment_uppernet_test)",
        "seg_revision": SEGMENTATION_REVISION, "reg_revision": REG_REVISION,
        "to_plex_grid_mm": [AXIAL_MM, LATERAL_MM], "plex_scale_mm": [scale_axial, scale_lat],
        "xml_axial_mm": float(df["axial_resolution"].median()),
        "xml_lateral_mm": float(df["lateral_resolution"].median()),
        "plex_geometry": g.as_dict(), "native_columns_in_crop": int((~outside).sum()),
        "registration_config": {k: (list(v) if isinstance(v, tuple) else v)
                                for k, v in vars(REGISTRATION).items()},
        "n_frames": int(T), "height": int(H), "width": int(W), "fps": round(fps, 3),
        "ref_idx": int(result.ref_idx), "ref_percentile": result.ref_percentile,
        "iou_ref_median": None if iou is None else float(np.nanmedian(iou)),
        "n_empty_mask_frames": int((masks.sum(axis=(1, 2)) == 0).sum()),
        "timings_s": {k: round(float(v), 2) for k, v in result.timings.items()},
        "seconds_calc": round(t_calc, 1), "seconds_total": round(time.perf_counter() - t0, 1),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "registration_params.json").write_text(json.dumps(params, indent=2, default=str),
                                                     encoding="utf-8")
    return params


def write_summary(conds):
    rows = []
    for c in conds:
        try:
            p = json.loads((c / "RawImages" / OUTPUT_SUBDIR / "registration_params.json").read_text("utf-8"))
        except Exception:  # noqa: BLE001
            p = {}
        rows.append({"condition": c.relative_to(PATH_GENERAL).as_posix(),
                     **{k: p.get(k) for k in ("status", "reason", "n_frames", "fps", "ref_idx",
                                              "iou_ref_median", "n_empty_mask_frames",
                                              "seconds_calc", "seconds_total", "created")}})
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(SUMMARY, index=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Notebook UPerNet sur la cohorte SANS -> RawImages/upernet")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    conds = conditions()
    todo = [c for c in conds if args.overwrite or not is_done(c / "RawImages" / OUTPUT_SUBDIR)]
    todo = todo[: args.limit] if args.limit is not None else todo
    print(f"{len(conds)} condition(s), {len(todo)} a traiter -> RawImages/{OUTPUT_SUBDIR}", flush=True)
    if not todo:
        write_summary(conds)
        return 0
    seg_model = get_model()
    reg_model = get_registration_model(REG_REVISION).to("cuda")
    t_all = time.perf_counter()
    for i, c in enumerate(todo, 1):
        try:
            p = process(c, seg_model, reg_model)
            print(f"[{i:>3}/{len(todo)}] {c.name} | calcul {p['seconds_calc']:.0f} s, "
                  f"total {p['seconds_total']:.0f} s | IoU {p['iou_ref_median']:.3f}", flush=True)
        except Exception as e:  # noqa: BLE001 - une condition ne doit pas tout arreter
            traceback.print_exc()
            out_dir = c / "RawImages" / OUTPUT_SUBDIR
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "registration_params.json").write_text(json.dumps(
                {"status": "error", "reason": f"{type(e).__name__}: {str(e)[:300]}",
                 "created": datetime.now().isoformat(timespec="seconds")}, indent=2), encoding="utf-8")
            print(f"[{i:>3}/{len(todo)}] {c.name} ERREUR {e}", flush=True)
        torch.cuda.empty_cache()
        write_summary(conds)
        break
    print(f"termine en {(time.perf_counter() - t_all) / 60:.1f} min -> {SUMMARY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
