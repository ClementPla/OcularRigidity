# -*- coding: utf-8 -*-
"""
Videos de choroide recadrees, corrigees en illumination, pour un CNN 3D (SiNC).

Pour chaque acquisition de la piste Reproducibility :

    E:/SANSORI/Reproducibility/registered_cascade_v9_plex_full/<SLUG>/
        {registered_video.mp4, mask.npz, timestamp.txt, registration_params.json}
    ->  E:/SANSORI/Reproducibility/choroid_crops/<SLUG>/{crop.npz, crop_params.json, qc.png}

1. Frontiere BM = bord superieur du masque, MEDIANE par colonne sur les frames
   valides (une seule frontiere : la video est deja recalee). Fenetre aplatie
   ``rows(d, x) = bm[x] + d``, -FIT_ABOVE_UM <= d < ROI_DEPTH_UM (117 um), sur
   TOUTE la largeur. La partie au-dessus de la BM (RPE, photorecepteurs) ne sert
   qu'aux ajustements : sans elle le profil de la choroide seule est trop plat et
   la pente par A-scan n'est pas identifiable (454 pentes negatives sur
   BELANGER_CHARLES_OD1, contre 0 avec 117 um au-dessus).
2. Image moyenne M de la fenetre (frames valides).
3. Correction A-SCAN PAR A-SCAN : chaque colonne de M ajustee lineairement sur le
   profil median r(d), ``M[:, x] ~ a_x r + b_x`` ; Mc = (M - b) / a. Retire les
   ombres des vaisseaux retiniens et le gain lateral.
4. Correction D'UNE IMAGE A L'AUTRE : chaque frame recoit la meme correction par
   colonne, puis est ajustee lineairement sur Mc, ``F'_t ~ alpha_t Mc + beta_t`` ;
   F''_t = (F'_t - beta_t) / alpha_t. Ajustement sur la pleine largeur, pour que
   alpha/beta captent le gain global plutot que la pulsation propre au crop (le
   pouls d'intensite COMMUN a toute la bande est retire, comme le veut la regle
   « traces normalisees par frame » ; alpha_t est garde dans le npz).
5. Crop = tiers central lateral (ROI_WIDTH_MM, centre) x ROI_DEPTH_UM, la ROI de
   ``Astronauts/data_visualizer_GUI/pulse_compute.py`` (importee, pas recopiee).

``crop.npz`` : video (T, depth, n_cols) float16 ; frames invalides (noires ou sans
masque) et colonnes invalides (mises a zero par le recalage, sans masque, pente
non identifiable) a 0, signalees par ``frame_valid`` / ``col_valid``.

    python Astronauts/pulsation/sinc_intensity/preprocessing/OCT_choroid/make_dataset.py
    python .../make_dataset.py --slug BELANGER_CHARLES_OD1 --overwrite
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
import torch  # noqa: F401

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "Astronauts" / "data_visualizer_GUI"))

import pulse_compute  # noqa: E402

from ocularrigidity.data.compression import read_gray  # noqa: E402
from ocularrigidity.data.io import load_mask  # noqa: E402
from ocularrigidity.scripts.batch_layout import RE_SLUG, flat_data_dir, flat_slugs  # noqa: E402
from ocularrigidity.scripts.registration.cascade_v9_plex import git_state  # noqa: E402

PATH_GENERAL = Path("E:/SANSORI/Reproducibility")
VARIANT = "registered_cascade_v9_plex_full"
OUT_SUBDIR = "choroid_crops"
SUMMARY_DIR = Path("E:/NASA_Rigidity/Reproducibility") / OUT_SUBDIR

MIN_FRAME_MEAN = 1.0  # niveau de gris moyen sous lequel une frame est noire
MAX_ZERO_FRAC = 0.5  # colonne invalide si nulle sur plus de cette fraction des frames
FIT_ABOVE_UM = 117.0  # fenetre d'ajustement au-dessus de la BM (RPE comprise)
A_MIN = 0.05  # pente A-scan / frame sous laquelle l'ajustement n'est pas identifiable
CHUNK = 64


# --------------------------------------------------------------------------- #
# Ajustements lineaires (forme fermee, vectorises)
# --------------------------------------------------------------------------- #
def linfit(x: np.ndarray, y: np.ndarray, axis: int):
    """Pente et ordonnee de y ~ a x + b le long de ``axis`` (moindres carres)."""
    xm = x.mean(axis=axis, keepdims=True)
    ym = y.mean(axis=axis, keepdims=True)
    xc = x - xm
    a = (xc * (y - ym)).sum(axis=axis, keepdims=True) / ((xc ** 2).sum(axis=axis, keepdims=True) + 1e-12)
    b = ym - a * xm
    return np.squeeze(a, axis=axis), np.squeeze(b, axis=axis)


def fill_columns(v: np.ndarray, ok: np.ndarray) -> np.ndarray:
    x = np.arange(v.size)
    return np.interp(x, x[ok], v[ok])


# --------------------------------------------------------------------------- #
# Une acquisition
# --------------------------------------------------------------------------- #
def load(data_dir: Path):
    frames = read_gray(data_dir / "registered_video.mp4")  # (T, H, W) uint8
    masks = load_mask(data_dir / "mask.npz")
    if frames.shape != masks.shape:
        raise ValueError(f"video {frames.shape} != masque {masks.shape}")
    ts_us = np.loadtxt(data_dir / "timestamp.txt", dtype=np.float64).ravel()
    if ts_us.size != frames.shape[0]:
        raise ValueError(f"frames ({frames.shape[0]}) != horodatages ({ts_us.size})")
    params = json.loads((data_dir / "registration_params.json").read_text(encoding="utf-8"))
    return frames, masks, ts_us, params


def process(slug: str, variant: str) -> dict:
    t0 = time.perf_counter()
    data_dir = flat_data_dir(PATH_GENERAL, slug, variant)
    frames, masks, ts_us, reg = load(data_dir)
    T, H, W = frames.shape
    axial_mm = reg["plex_axial_um"] / 1e3
    lateral_mm = reg["plex_lateral_um"] / 1e3
    depth, c0, c1 = pulse_compute.roi_pixels(W, (axial_mm, lateral_mm))
    above = int(round(FIT_ABOVE_UM / 1e3 / axial_mm))

    # --- Validite des frames et des colonnes --------------------------------
    has = masks.any(axis=1)  # (T, W)
    frame_mean = np.concatenate([frames[i:i + CHUNK].mean(axis=(1, 2)) for i in range(0, T, CHUNK)])
    frame_valid = (frame_mean > MIN_FRAME_MEAN) & has.any(axis=1)
    if frame_valid.sum() < 2:
        raise ValueError("moins de 2 frames valides")
    zero = np.concatenate([frames[i:i + CHUNK].max(axis=1) == 0 for i in range(0, T, CHUNK)])
    zero_frac = zero[frame_valid].mean(axis=0)
    col_valid = (zero_frac <= MAX_ZERO_FRAC) & has[frame_valid].any(axis=0)

    # --- Frontiere BM mediane, bande aplatie pleine largeur -----------------
    top = masks.argmax(axis=1).astype(np.float64)  # (T, W)
    top[~has] = np.nan
    top[~frame_valid] = np.nan
    del masks
    bm = np.nanmedian(np.where(col_valid[None, :], top, np.nan)[frame_valid], axis=0)
    if col_valid.sum() < 2:
        raise ValueError("moins de 2 colonnes valides")
    bm = fill_columns(bm, col_valid)
    if np.any(bm + depth > H) or np.any(bm - above < 0):
        raise ValueError(f"la fenetre [-{above}, {depth}) px autour de la BM sort de l'image "
                         f"(BM entre {bm.min():.0f} et {bm.max():.0f} / {H})")
    rows = np.round(bm).astype(np.int64)[None, :] + np.arange(-above, depth)[:, None]
    band = frames[:, rows, np.arange(W)[None, :]].astype(np.float32)  # (T, above + depth, W)
    mean_bscan = (sum(frames[i:i + CHUNK][frame_valid[i:i + CHUNK]].sum(axis=0, dtype=np.float64)
                      for i in range(0, T, CHUNK)) / frame_valid.sum()).astype(np.float32)
    mid = int(np.flatnonzero(frame_valid)[frame_valid.sum() // 2])
    del frames

    # --- Correction A-scan par A-scan de l'image moyenne --------------------
    M = band[frame_valid].mean(axis=0)  # (above + depth, W)
    r = np.median(M[:, col_valid], axis=1)  # (above + depth,)
    a, b = linfit(np.broadcast_to(r[:, None], M.shape), M, axis=0)  # (W,)
    col_valid &= a > A_MIN
    a = np.where(col_valid, a, 1.0).astype(np.float32)
    b = np.where(col_valid, b, 0.0).astype(np.float32)
    Mc = (M - b) / a

    # --- Correction d'une image a l'autre -----------------------------------
    band = (band - b) / a  # meme correction par colonne pour toutes les frames
    X = Mc[:, col_valid].ravel()
    Y = band[:, :, col_valid].reshape(T, -1)
    alpha, beta = linfit(np.broadcast_to(X[None, :], Y.shape), Y, axis=1)  # (T,)
    del X, Y
    frame_valid &= alpha > A_MIN
    alpha = np.where(frame_valid, alpha, 1.0).astype(np.float32)
    beta = np.where(frame_valid, beta, 0.0).astype(np.float32)
    corrected = (band - beta[:, None, None]) / alpha[:, None, None]
    corrected[~frame_valid] = 0.0
    corrected[:, :, ~col_valid] = 0.0

    # --- Crop ----------------------------------------------------------------
    video = corrected[:, above:, c0:c1].astype(np.float16)
    if not np.isfinite(video).all():
        raise ValueError("valeurs non finies dans la video recadree")
    t_s = (ts_us - ts_us[0]) / 1e6
    fps = float(1.0 / np.median(np.diff(t_s)))

    out_dir = PATH_GENERAL / OUT_SUBDIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "crop.npz", video=video, t_s=t_s, fps=fps,
        frame_valid=frame_valid, col_valid=col_valid, crop_col_valid=col_valid[c0:c1],
        bm_row=bm.astype(np.float32), a=a, b=b, alpha=alpha, beta=beta,
        mean_before=M.astype(np.float32), mean_after=Mc.astype(np.float32),
        c0=c0, c1=c1, depth=depth, fit_above=above, axial_um=1e3 * axial_mm, lateral_um=1e3 * lateral_mm)
    qc_figure(out_dir / "qc.png", slug, mean_bscan, bm, above, depth, c0, c1, M, Mc,
              band_raw_frame=(corrected[mid] * alpha[mid] + beta[mid]) * a + b,
              band_cor_frame=corrected[mid], mid=mid)

    q = lambda v: [float(x) for x in np.quantile(v, [0.05, 0.5, 0.95])]  # noqa: E731
    m = RE_SLUG.match(slug)
    row = {
        "slug": slug, "participant": m["participant"], "eye": m["eye"],
        "replicate": int(m["replicate"]), "status": "ok",
        "T": T, "n_frames_valid": int(frame_valid.sum()),
        "W": W, "n_cols_valid": int(col_valid.sum()),
        "n_crop_cols_valid": int(col_valid[c0:c1].sum()),
        "depth_px": int(depth), "fit_above_px": above, "crop_cols": int(c1 - c0), "fps": fps,
        "a_median": float(np.median(a[col_valid])),
        "alpha_median": float(np.median(alpha[frame_valid])),
        "beta_median": float(np.median(beta[frame_valid])),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    params = {
        **row, "source": str(data_dir), "variant": variant,
        "created": datetime.now().isoformat(timespec="seconds"),
        "roi_depth_um": pulse_compute.ROI_DEPTH_UM, "fit_above_um": FIT_ABOVE_UM,
        "roi_width_mm": pulse_compute.ROI_WIDTH_MM,
        "c0": int(c0), "c1": int(c1), "H": H,
        "axial_um": 1e3 * axial_mm, "lateral_um": 1e3 * lateral_mm,
        "min_frame_mean": MIN_FRAME_MEAN, "max_zero_frac": MAX_ZERO_FRAC, "a_min": A_MIN,
        "a_q05_q50_q95": q(a[col_valid]), "b_q05_q50_q95": q(b[col_valid]),
        "alpha_q05_q50_q95": q(alpha[frame_valid]), "beta_q05_q50_q95": q(beta[frame_valid]),
        "git": git_state(REPO),
    }
    (out_dir / "crop_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    (out_dir / "error.txt").unlink(missing_ok=True)
    return row


def qc_figure(path, slug, mean_bscan, bm, above, depth, c0, c1, M, Mc, band_raw_frame, band_cor_frame, mid):
    """Panneaux d'IMAGE (moyennes, frames) : PNG statique, l'exception de CLAUDE.md."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(5, 1, figsize=(12, 13), gridspec_kw={"height_ratios": [4, 1, 1, 1, 1]})
    ax[0].imshow(mean_bscan, cmap="gray", aspect="auto")
    ax[0].plot(bm, "r", lw=0.6)
    ax[0].plot(bm + depth, "r--", lw=0.6)
    ax[0].plot(bm - above, "c:", lw=0.6)
    ax[0].axvline(c0, color="y", lw=0.8)
    ax[0].axvline(c1, color="y", lw=0.8)
    ax[0].set_title(f"{slug} : B-scan moyen, BM mediane (rouge), crop {depth} px x colonnes {c0}-{c1}, "
                    f"ajustements depuis {above} px au-dessus (cyan)")
    panels = [(M, "fenetre moyenne brute"), (Mc, "fenetre moyenne corrigee A-scan par A-scan"),
              (band_raw_frame, f"frame {mid} brute (aplatie)"),
              (band_cor_frame, f"frame {mid} corrigee")]
    for k, (img, title) in enumerate(panels, start=1):
        lo, hi = np.percentile(img, [1, 99])
        ax[k].imshow(img, cmap="gray", aspect="auto", vmin=lo, vmax=hi)
        ax[k].add_patch(Rectangle((c0 - 0.5, above - 0.5), c1 - c0, depth, fill=False, ec="y", lw=0.8))
        ax[k].set_title(title)
    for x in ax:
        x.set_xticks([])
        x.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def summary_row_from_disk(slug: str) -> dict | None:
    p = PATH_GENERAL / OUT_SUBDIR / slug / "crop_params.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    keys = ("slug", "participant", "eye", "replicate", "status", "T", "n_frames_valid", "W",
            "n_cols_valid", "n_crop_cols_valid", "depth_px", "fit_above_px", "crop_cols", "fps",
            "a_median", "alpha_median", "beta_median", "seconds")
    return {k: d.get(k) for k in keys}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slug", action="append", help="une acquisition (repetable)")
    ap.add_argument("--variant", default=VARIANT)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    slugs = args.slug or flat_slugs(PATH_GENERAL)
    rows = []
    for i, slug in enumerate(slugs, 1):
        out_dir = PATH_GENERAL / OUT_SUBDIR / slug
        if (out_dir / "crop.npz").exists() and not args.overwrite:
            print(f"[{i}/{len(slugs)}] {slug} : deja fait")
            rows.append(summary_row_from_disk(slug) or {"slug": slug, "status": "ok"})
            continue
        try:
            row = process(slug, args.variant)
            print(f"[{i}/{len(slugs)}] {slug} : {row['n_frames_valid']}/{row['T']} frames, "
                  f"{row['n_crop_cols_valid']}/{row['crop_cols']} colonnes, "
                  f"{row['depth_px']} px, a~{row['a_median']:.2f}, alpha~{row['alpha_median']:.2f} "
                  f"({row['seconds']} s)")
        except Exception as e:  # une acquisition en echec n'arrete pas le lot
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"[{i}/{len(slugs)}] {slug} : ECHEC {e!r}")
            row = {"slug": slug, "status": f"error: {e}"}
        rows.append(row)

    if not args.slug:
        SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(SUMMARY_DIR / "summary.csv", index=False)
        n_ok = sum(r.get("status") == "ok" for r in rows)
        print(f"{n_ok}/{len(rows)} ok -> {SUMMARY_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
