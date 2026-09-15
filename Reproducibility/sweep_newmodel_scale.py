# -*- coding: utf-8 -*-
"""
sweep_newmodel_scale.py

Balaie le PARAMETRE D'ECHELLE du recaleur appris de Clement
(``ClementP/OCTVideoRegistration@cascade_v9``) sur une acquisition, et mesure
chaque recalage avec le meme critere que la comparaison des variantes.

Ou vit l'echelle (``registration/deep_learning/models/regressor.py``)
-------------------------------------------------------------------
Deux attributs du modele convertissent ses sorties en pixels, et eux seuls
agissent en mode cascade :

    img_shape   l. 360 (defaut), config.json du Hub ; lu l. 437
                ``stride = H / f.shape[-2]`` (l. 455) puis
                ``dy += (d_bulk + d_res) * stride`` (l. 464)
    dx_step     l. 363 (defaut 4.0), 8.0 dans config.json ; lu l. 460
                ``dx += ddx * dx_step``

``scales`` (l. 362) n'est lu que si ``cascade=False`` (l. 440) : sans effet ici.

Le balayage fixe, pour un facteur ``s``, ``img_shape = (H*s, W*s)`` et
``dx_step = 8 * s``. Ce couple multiplie dx et dy par ``s`` en laissant
invariant ce que le reseau compare : les deplacements appliques aux cartes de
caracteristiques entre deux etages sont ``dy * h / (H*s)`` et
``dx * w / (W*s)``, soit le meme nombre de cellules quel que soit ``s``. On ne
change donc que l'AMPLITUDE des deplacements, pas leur direction -- c'est
exactement la question : le modele se trompe-t-il d'echelle ou de sens ?

Chaque ``s`` est un VRAI recalage (``segment_and_register`` relance avec les
attributs modifies), pas une remise a l'echelle apres coup. S'y ajoutent :

    identite                 aucune transformation (s = 0) : les B-scans
                             Spectralis sont deja stabilises par l'appareil,
                             et un petit s peut « gagner » en recalant moins.
    code de Clement tel quel img_shape d'entrainement (1536, 1024), dx_step 8 :
                             ce que fait ``segment_and_register`` sans reglage.

Critere (le meme que la comparaison des variantes) : NCC de chaque frame recalee
a la mediane temporelle, sur la bande de choroide (pixels masques dans >= 50 %
des frames, 3/4 centraux des colonnes, hors pixels nuls).

Rien n'est ecrit dans les dossiers de donnees : une table sous
    E:/NASA_Rigidity/Reproducibility/registration_cascade_v9/scale_sweep_<SLUG>.csv

Lancer (kernel pyOR, branche portant le recalage appris, depuis la racine) :
    python Reproducibility/sweep_newmodel_scale.py --slug MODICA_GRAZIANA_OS3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ocularrigidity.pipeline_config import REGISTRATION
from ocularrigidity.registration.deep_learning.models.losses import warp
from ocularrigidity.registration.fused import segment_and_register
from ocularrigidity.registration.postprocess import filter_bad_ascans_per_bms
from ocularrigidity.scripts.registration.astronauts import (
    build_cube_and_timestamps,
    load_ordered_oct_series,
)
from ocularrigidity.segmentation.utils import (
    get_choroid_segmentation_model,
    get_registration_model,
)

PATH_GENERAL = Path("E:/SANSORI/Reproducibility")
OUT_DIR = Path("E:/NASA_Rigidity/Reproducibility/registration_cascade_v9")
DEVICE = "cuda"
SCALES = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
TRAINED_DX_STEP = 8.0


def patch_largest_cc() -> str:
    """Repli CPU de ``keep_largest_connected_component_gpu`` si cuCIM manque.

    Meme substitution que ``register_newmodel.py`` : la version cc3d est
    bit-identique selon la docstring de la version GPU.
    """
    try:
        import cucim  # noqa: F401

        return "cucim (GPU)"
    except ImportError:
        import ocularrigidity.registration.fused as fused_mod
        from ocularrigidity.segmentation.postprocess.blob import (
            keep_largest_connected_component,
        )

        def _largest_cc_cpu(masks: torch.Tensor) -> torch.Tensor:
            out = keep_largest_connected_component(masks.detach().cpu().numpy())
            return torch.from_numpy(out).to(masks.device)

        fused_mod.keep_largest_connected_component_gpu = _largest_cc_cpu
        return "cc3d (CPU, repli : cucim absent)"


def warp_identity(cube: np.ndarray, raw_masks: np.ndarray, batch: int = 16):
    """Le chemin de ``segment_and_register`` avec dx = dy = 0 : meme warp, meme
    filtre de colonnes, pour que l'identite soit mesuree comme le reste."""
    T, H, W = cube.shape
    frames = np.empty_like(cube)
    masks = np.empty_like(raw_masks)
    zdx = torch.zeros(batch, device=DEVICE)
    zdy = torch.zeros(batch, W, device=DEVICE)
    for s in range(0, T, batch):
        e = min(s + batch, T)
        data = torch.stack([
            torch.from_numpy(raw_masks[s:e]).to(DEVICE).float(),
            torch.from_numpy(cube[s:e]).to(DEVICE).float(),
        ], dim=1)
        out, _ = warp(data, zdx[: e - s], zdy[: e - s], (H, W))
        masks[s:e] = (out[:, 0] > 0.5).cpu().numpy()
        frames[s:e] = out[:, 1].clamp(0, 255).to(torch.uint8).cpu().numpy()
    bad = filter_bad_ascans_per_bms(masks)
    if bool(bad.any()):
        cols = bad.cpu().numpy()
        frames[..., cols] = 0
        masks[..., cols] = 0
    return frames, masks, int(bad.sum())


def ncc_to_median(frames: np.ndarray, masks: np.ndarray) -> np.ndarray:
    T, H, W = frames.shape
    f = frames.astype(np.float32)
    band = masks.mean(axis=0) >= 0.5
    band[:, : int(0.125 * W)] = False
    band[:, int(0.875 * W):] = False
    med = np.median(f, axis=0)
    out = np.full(T, np.nan)
    for t in range(T):
        sel = band & (f[t] > 0) & (med > 0)
        if sel.sum() < 500:
            continue
        a = f[t][sel] - f[t][sel].mean()
        b = med[sel] - med[sel].mean()
        out[t] = float((a @ b) / (np.sqrt((a @ a) * (b @ b)) + 1e-8))
    return out


def summarize(label, s, img_shape, dx_step, frames, masks, dx, dy, ref, n_bad, secs):
    ncc = ncc_to_median(frames, masks)
    dy_bulk = dy.mean(axis=1)
    return {
        "essai": label,
        "s": s,
        "img_shape": f"{img_shape[0]}x{img_shape[1]}" if img_shape else "",
        "dx_step": dx_step,
        "ref_idx": ref,
        "ncc_median": float(np.nanmedian(ncc)),
        "ncc_q1": float(np.nanpercentile(ncc, 25)),
        "ncc_q3": float(np.nanpercentile(ncc, 75)),
        "ncc_p5": float(np.nanpercentile(ncc, 5)),
        "n_frames_ncc_lt_0.5": int(np.sum(ncc < 0.5)),
        "dx_ptp_px": float(np.ptp(dx)),
        "dy_bulk_ptp_px": float(np.ptp(dy_bulk)),
        "dy_intra_frame_std_px": float(np.median(dy.std(axis=1))),
        "n_bad_columns": n_bad,
        "n_empty_masks": int((masks.sum(axis=(1, 2)) == 0).sum()),
        "seconds": round(secs, 1),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    args = ap.parse_args(argv)

    raw_dir = PATH_GENERAL / args.slug / "RawImages"
    series = load_ordered_oct_series(raw_dir)
    cube, _ = build_cube_and_timestamps(raw_dir, series)
    T, H, W = cube.shape
    print(f"{args.slug} : cube {T} x {H} x {W}", flush=True)

    backend = patch_largest_cc()
    print(f"plus grande composante connexe : {backend}", flush=True)
    seg_model = get_choroid_segmentation_model().to(DEVICE)
    reg_model = get_registration_model().to(DEVICE)
    trained_shape = tuple(int(v) for v in reg_model.img_shape)
    assert float(reg_model.dx_step) == TRAINED_DX_STEP, reg_model.dx_step
    torch.backends.cudnn.benchmark = True

    runs = [("code de Clement tel quel", None, trained_shape, TRAINED_DX_STEP)]
    runs += [(f"s = {s:g}", s, (int(round(H * s)), int(round(W * s))), TRAINED_DX_STEP * s)
             for s in SCALES]

    rows = []
    raw_masks = None
    for label, s, shape, step in runs:
        reg_model.img_shape = shape
        reg_model.dx_step = step
        t0 = time.perf_counter()
        res = segment_and_register(cube, seg_model, reg_model, REGISTRATION,
                                   device=DEVICE, verbose=False)
        secs = time.perf_counter() - t0
        raw_masks = res.raw_masks
        dx = np.asarray(res.transform["dx"], dtype=np.float64)
        dy = np.asarray(res.transform["dy"], dtype=np.float64)
        row = summarize(label, s if s is not None else np.nan, shape, step,
                        res.registered_frames, res.registered_masks, dx, dy,
                        res.ref_idx, int(np.asarray(res.transform["bad_columns"]).sum()), secs)
        rows.append(row)
        print(f"  {label:26} img_shape {row['img_shape']:>9} dx_step {step:5.2f} | "
              f"NCC {row['ncc_median']:.3f} [{row['ncc_q1']:.3f}-{row['ncc_q3']:.3f}] "
              f"p5 {row['ncc_p5']:.3f} | <0.5 : {row['n_frames_ncc_lt_0.5']:3d} | "
              f"dx ptp {row['dx_ptp_px']:6.2f} | dy global ptp {row['dy_bulk_ptp_px']:6.2f} | "
              f"ref {row['ref_idx']}", flush=True)
        del res
        torch.cuda.empty_cache()

    # Identite : segmentation identique (le masque brut ne depend pas du recaleur).
    t0 = time.perf_counter()
    frames_id, masks_id, n_bad_id = warp_identity(cube, raw_masks)
    row = summarize("identite (s = 0)", 0.0, None, 0.0, frames_id, masks_id,
                    np.zeros(T), np.zeros((T, W)), -1, n_bad_id, time.perf_counter() - t0)
    rows.append(row)
    print(f"  {'identite (s = 0)':26} {'':>9} {'':>13} | "
          f"NCC {row['ncc_median']:.3f} [{row['ncc_q1']:.3f}-{row['ncc_q3']:.3f}] "
          f"p5 {row['ncc_p5']:.3f} | <0.5 : {row['n_frames_ncc_lt_0.5']:3d}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"scale_sweep_{args.slug}.csv"
    df = pd.DataFrame(rows)
    df["img_shape_trained"] = f"{trained_shape[0]}x{trained_shape[1]}"
    df["largest_cc_backend"] = backend
    df.to_csv(out, index=False)
    print(f"\n{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
