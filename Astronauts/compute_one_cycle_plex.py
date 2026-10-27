# -*- coding: utf-8 -*-
"""
compute_one_cycle_plex.py

One-cycles des videos ``registered_cascade_v9_plex_full`` (SegFormer + cascade_v9 sur la
grille PLEX pleine largeur, 1,95 x 5,9 um par pixel) de la cohorte SANS, pour DEUX phases
cardiaques :

    ct    pouls d'EPAISSEUR des masques (um)
    svd   pouls d'INTENSITE : SVD rang 100 + combinaison optimisee, 117 um sous la RPE x 2,9 mm
    sinc  pouls SiNC (reseau appris, ``motion/pulsation/sinc/trace_source.py``) : chaine par
          defaut du package (``pipeline_config.PULSATION``), FC trouvee par SiNC, phase IQ
          ancree sur l'epaisseur -- c'est CETTE phase qui replie, pas le Hilbert ci-dessous

Les pouls sont ceux de la page « Pouls » du GUI (``data_visualizer_GUI/pulse_compute.py``,
meme code, memes reglages : FIR FC +/- 10 %) et partagent son cache sur E: : les trois pouls
de chaque condition y sont enregistres, le GUI ne les recalcule plus.

Pour chaque pouls :
  1. phase de Hilbert du pouls FILTRE (grille uniforme), deroulee, ramenee aux horodatages des
     frames (sinc : phase du package, deroulee frame a frame) ; 1 battement exclu a chaque bout ;
  2. bin 0 CENTRE sur le maximum de l'EPAISSEUR des masques (grille fine de 60 bins) -- pour les
     DEUX phases : le signe du pouls SVD est arbitraire, ce centrage l'efface ;
  3. groupes de N_CYCLE battements consecutifs (phase deroulee) : un groupe = un one-cycle de
     N_BINS bins ; groupe de moins de N_BINS frames ecarte ;
  4. repliement SANS recalage intra-bin : frames par mediane, masques par moyenne (occupation ;
     sa somme par colonne = epaisseur moyenne des frames du bin, exactement) ;
  5. one-cycle RESEGMENTE (SegFormer ``version-2.0.0`` + plus grande composante), directement sur
     la grille PLEX -- c'est deja la taille de pixel d'entrainement, rien a agrandir.

Epaisseurs par bin (um), sur les colonnes ou le masque est present dans >= 90 % des frames (regle
du pouls d'epaisseur) : ``thickness_fold`` (masques replies), ``thickness_seg`` (resegmente), leur
fondamental ajuste par cycle (``*_fit``), et le Dice resegmente / replie (> 0,5).

Sorties :
    E:/NASA_Rigidity/SegmentationVariations/cascade_v9_plex_full/one_cycle_6beats_10bins/
        cycles/<astro>__<condition>__<pouls>.npz   lire avec one_cycle_figure.load_one_cycle
        conditions.csv   une ligne par (condition, pouls)
        bins.csv         une ligne par (condition, pouls, cycle, bin)
        params.json

Lancer (kernel pyOR, depuis la racine) :
    python Astronauts/compute_one_cycle_plex.py --limit 1
    python Astronauts/compute_one_cycle_plex.py
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Ordre impose par les DLL de l'env pyOR : pandas, puis torch, puis le reste.
import pandas as pd
import torch
import numpy as np
from scipy.signal import hilbert

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "data_visualizer_GUI"))

import catalog  # noqa: E402
import pulse_compute  # noqa: E402

from ocularrigidity.data.compression import read_gray  # noqa: E402
from ocularrigidity.data.io import load_mask  # noqa: E402
from ocularrigidity.motion.one_cycle import (  # noqa: E402
    fold_video_numba_mean,
    fold_video_numba_median,
)
from ocularrigidity.registration import fused  # noqa: E402
from ocularrigidity.scripts.registration.cascade_v9_plex import patch_largest_cc  # noqa: E402

DATA_SUBDIR = "registered_cascade_v9_plex_full"
N_CYCLE = 6  # battements par one-cycle
N_BINS = 10  # bins de phase par one-cycle
PHASE_PROBE_BINS = 60  # grille fine du maximum d'epaisseur
EDGE_BEATS = 1.0  # battements exclus a chaque bout
SEG_BATCH = 2
DEVICE = "cuda"
PULSES = ("ct", "svd", "sinc")
# Cle du cache des pouls : les libelles du GUI (catalog.REGISTRATIONS).
CACHE_REG = "cascade_v9 PLEX pleine largeur (grille PLEX)"
CACHE_SEG = "SegFormer cascade_v9 (meme passe)"

OUT_DIR = (catalog.SEGVAR_ROOT / DATA_SUBDIR.removeprefix("registered_")
           / f"one_cycle_{N_CYCLE}beats_{N_BINS}bins")
CYCLES_DIR = OUT_DIR / "cycles"
CSV_CONDITIONS = OUT_DIR / "conditions.csv"
CSV_BINS = OUT_DIR / "bins.csv"


def npz_path(cond: catalog.Condition, pulse: str) -> Path:
    return CYCLES_DIR / f"{cond.astro}__{cond.name}__{pulse}.npz"


# --------------------------------------------------------------------------- #
# Briques
# --------------------------------------------------------------------------- #
def fit_cycle(y: np.ndarray) -> np.ndarray:
    """Fondamental ajuste sur les bins renseignes, evalue partout
    (= ``compute_one_cycle_compare_methods.fit_cycle``)."""
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(y)
    n = y.size
    if ok.sum() < 4:
        return np.full(n, np.nan)
    t = np.arange(n) / n
    X = np.stack([np.ones(n), np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)], axis=1)
    coeffs, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    return X @ coeffs


def frame_phase(filtered_u: np.ndarray, u: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Phase de Hilbert deroulee (en CYCLES) du pouls filtre, aux horodatages des frames."""
    ph = np.unwrap(np.angle(hilbert(filtered_u - filtered_u.mean())))
    return np.interp(t, u, ph) / (2 * np.pi)


def thickness_curve(occ: np.ndarray, cols: np.ndarray, counts: np.ndarray, um_px: float):
    th = np.asarray(occ, np.float64)[:, :, cols].sum(axis=1).mean(axis=1) * um_px
    th[counts == 0] = np.nan
    return th


@torch.inference_mode()
def resegment(cycles_u8: np.ndarray, seg_model):
    """(n, H, W) uint8 -> masques bool + probabilites uint8, sur la grille PLEX telle quelle."""
    n = cycles_u8.shape[0]
    masks = np.zeros(cycles_u8.shape, bool)
    probs = np.zeros(cycles_u8.shape, np.uint8)
    for s in range(0, n, SEG_BATCH):
        e = min(s + SEG_BATCH, n)
        m, _, p = fused._encode_segment(seg_model, torch.from_numpy(cycles_u8[s:e]).to(DEVICE),
                                        torch.bfloat16, keep_largest_cc=True, return_probs=True)
        masks[s:e], probs[s:e] = m.cpu().numpy(), p.cpu().numpy()
    return masks, probs


def to_u8(x):
    return np.clip(np.nan_to_num(np.asarray(x, np.float32)), 0, 255).round().astype(np.uint8)


def ptp_nan(y):
    y = np.asarray(y, float)
    return float(np.nanmax(y) - np.nanmin(y)) if np.isfinite(y).any() else np.nan


# --------------------------------------------------------------------------- #
# Une condition
# --------------------------------------------------------------------------- #
def process(cond: catalog.Condition, seg_model, pulses) -> tuple[list[dict], list[dict]]:
    t_all = time.perf_counter()
    data_dir = cond.raw / DATA_SUBDIR
    for name in ("registered_video.mp4", "mask.npz", "timestamp.txt", "registration_params.json"):
        if not (data_dir / name).exists():
            raise FileNotFoundError(data_dir / name)
    frames = np.ascontiguousarray(read_gray(data_dir / "registered_video.mp4"))
    masks = np.asarray(load_mask(data_dir / "mask.npz"), dtype=bool)
    ts = np.loadtxt(data_dir / "timestamp.txt", dtype=np.int64, ndmin=1)
    T, H, W = frames.shape
    if masks.shape != frames.shape or ts.size != T:
        raise ValueError(f"tailles incoherentes : frames {frames.shape}, masques {masks.shape}, "
                         f"{ts.size} horodatages")
    t = (ts - ts[0]) / 1e6
    pixel_mm = catalog.video_pixel_mm(cond, data_dir / "registered_video.mp4", (H, W))
    um_px = 1e3 * pixel_mm[0]
    hr = catalog.read_hr(cond)

    t0 = time.perf_counter()
    res = pulse_compute.cached_compute(
        pulse_compute.cache_path(CACHE_REG, CACHE_SEG, cond.astro, cond.name),
        frames, masks, t, hr, pixel_mm)
    t_pulse = time.perf_counter() - t0
    hr = res.hr
    thick_frames = res.mask_pulse.raw  # (T,) epaisseur centree, um, sur t

    cover = masks.any(axis=1).mean(axis=0) >= pulse_compute.MIN_COL_COVERAGE
    cols = np.flatnonzero(cover)
    if cols.size == 0:
        raise ValueError("aucune colonne de choroide exploitable")
    margin = EDGE_BEATS * 60.0 / hr
    edge_ok = (t >= t[0] + margin) & (t <= t[-1] - margin)
    # Vue uint8 (0/1), sans copie : en float32 le masque PLEX ferait ~5 Go.
    masks_f = masks.view(np.uint8)

    sources = {"ct": res.mask_pulse, "svd": res.svd_pulse, "sinc": res.sinc_pulse}
    cond_rows, bin_rows = [], []
    for pulse in pulses:
        tr = sources[pulse]
        if tr is None:
            raise ValueError(f"pas de pouls {pulse}" + (f" ({res.sinc_note})" if pulse == "sinc" else ""))
        tp = time.perf_counter()
        if pulse == "sinc":
            # Phase du package (fraction [0, 1)), deroulee frame a frame pour compter les battements.
            cyc = np.unwrap(2 * np.pi * res.sinc_phase_per_frame) / (2 * np.pi)
            good = edge_ok & np.isfinite(cyc) & res.sinc_good_per_frame
            info = f"{tr.info} ; {res.sinc_note}"
        else:
            cyc = frame_phase(tr.filtered, res.u, t)  # deroulee, en cycles
            good = edge_ok & np.isfinite(cyc)
            info = tr.info
        phase_raw = np.mod(cyc, 1.0)

        # Bin 0 centre sur le maximum d'EPAISSEUR (pour les deux phases).
        pb = (phase_raw * PHASE_PROBE_BINS).astype(int) % PHASE_PROBE_BINS
        s = np.bincount(pb[good], weights=thick_frames[good], minlength=PHASE_PROBE_BINS)
        c = np.bincount(pb[good], minlength=PHASE_PROBE_BINS).astype(float)
        probe = np.where(c > 0, s / np.maximum(c, 1), np.nan)
        phase_max = (np.nanargmax(probe) + 0.5) / PHASE_PROBE_BINS
        phase_min = (np.nanargmin(probe) + 0.5) / PHASE_PROBE_BINS
        phase_f = np.mod(phase_raw - phase_max + 0.5 / N_BINS, 1.0).astype(np.float32)

        # Groupes de N_CYCLE battements sur la phase deroulee.
        pos = cyc - cyc[good].min()
        n_beats = float(pos[good].max())
        n_groups = max(int(np.floor(n_beats / N_CYCLE)), 1)
        group_of = np.clip(np.floor(pos / N_CYCLE).astype(int), 0, n_groups - 1)
        good_slot = good.copy()
        for g in range(n_groups):
            if int((good & (group_of == g)).sum()) < N_BINS:
                good_slot &= group_of != g

        n_slots = n_groups * N_BINS
        cycles = np.zeros((n_slots, H, W), np.float32)
        occupancy = np.zeros((n_slots, H, W), np.float32)
        counts = np.zeros(n_slots, np.int32)
        for g in range(n_groups):
            keep = good_slot & (group_of == g)
            if not keep.any():
                continue
            sl = slice(g * N_BINS, (g + 1) * N_BINS)
            cf, cc = fold_video_numba_median(frames, phase_f, keep, n_bins=N_BINS)
            mo, _ = fold_video_numba_mean(masks_f, phase_f, keep, n_bins=N_BINS)
            cycles[sl], occupancy[sl], counts[sl] = cf, mo, cc
        cycles_u8 = to_u8(cycles)
        cycles_u8[counts == 0] = 0
        del cycles

        seg_masks, seg_probs = resegment(cycles_u8, seg_model)
        seg_masks[counts == 0] = False
        seg_probs[counts == 0] = 0

        th_fold = thickness_curve(occupancy, cols, counts, um_px)
        th_seg = thickness_curve(seg_masks, cols, counts, um_px)
        fit_fold = np.concatenate([fit_cycle(th_fold[g * N_BINS:(g + 1) * N_BINS])
                                   for g in range(n_groups)])
        fit_seg = np.concatenate([fit_cycle(th_seg[g * N_BINS:(g + 1) * N_BINS])
                                  for g in range(n_groups)])
        a, b = seg_masks, occupancy > 0.5
        dice = 2 * (a & b).sum(axis=(1, 2)) / (a.sum(axis=(1, 2)) + b.sum(axis=(1, 2)) + 1e-12)
        dice = np.where(counts > 0, dice, np.nan)

        CYCLES_DIR.mkdir(parents=True, exist_ok=True)
        out = npz_path(cond, pulse)
        tmp = out.with_name(out.name + ".tmp.npz")
        np.savez_compressed(
            tmp, cycles=cycles_u8, occupancy=to_u8(occupancy * 255.0),
            seg_masks_packed=np.packbits(seg_masks.reshape(-1)),
            seg_shape=np.array(seg_masks.shape, np.int64), seg_probs=seg_probs, counts=counts,
            thickness_fold=th_fold, thickness_fold_fit=fit_fold,
            thickness_seg=th_seg, thickness_seg_fit=fit_seg, dice=dice, cols=cols,
            phase_f=phase_f, phase_raw=phase_raw, good_f=good, good_slot=good_slot,
            group_of=group_of, n_bins=N_BINS, n_cycle=N_CYCLE, n_groups=n_groups, hr=hr,
            pixel_um=np.array([1e3 * pixel_mm[0], 1e3 * pixel_mm[1]]),
            phase_max=phase_max, phase_min=phase_min, pulse=np.array(pulse),
            condition=np.array(cond.name), astro=np.array(cond.astro),
            data_subdir=np.array(DATA_SUBDIR), pulse_info=np.array(info),
            sinc_bpm=res.sinc_bpm, sinc_confidence=np.array(res.sinc_confidence))
        tmp.replace(out)
        del occupancy, seg_masks, seg_probs, cycles_u8
        gc.collect()

        slug = f"{cond.astro}__{cond.name}"
        for k in range(n_slots):
            bin_rows.append({
                "slug": slug, "pulse": pulse, "cycle": k // N_BINS + 1, "bin": k % N_BINS,
                "n_frames": int(counts[k]), "thickness_fold_um": th_fold[k],
                "thickness_fold_fit_um": fit_fold[k], "thickness_seg_um": th_seg[k],
                "thickness_seg_fit_um": fit_seg[k], "dice": dice[k]})
        filled = counts > 0

        def per_cycle(th):
            return [ptp_nan(th[g * N_BINS:(g + 1) * N_BINS]) for g in range(n_groups)]

        cond_rows.append({
            "slug": slug, "pulse": pulse, "astro": cond.astro, "moment": cond.moment,
            "eye": f"{cond.eye}{cond.rep}", "condition": cond.name, "status": "ok", "reason": "",
            "n_frames": T, "hr_BPM": hr, "hr_source": res.hr_source, "pixel_axial_um": um_px,
            "n_cols": int(cols.size), "n_beats": n_beats, "n_groups": n_groups,
            "n_frames_kept": int(good_slot.sum()), "n_bins_empty": int((~filled).sum()),
            "frames_per_bin_median": float(np.median(counts[filled])) if filled.any() else np.nan,
            "frames_per_bin_min": int(counts[filled].min()) if filled.any() else 0,
            "phase_max": phase_max, "phase_min": phase_min,
            "thickness_fold_um_mean": float(np.nanmean(th_fold)),
            "thickness_seg_um_mean": float(np.nanmean(th_seg)),
            "dY_fold_fit_um_med": float(np.nanmedian(per_cycle(fit_fold))),
            "dY_seg_fit_um_med": float(np.nanmedian(per_cycle(fit_seg))),
            "dY_fold_raw_um_med": float(np.nanmedian(per_cycle(th_fold))),
            "dY_seg_raw_um_med": float(np.nanmedian(per_cycle(th_seg))),
            "dice_median": float(np.nanmedian(dice)),
            "plv_svd_ct": float(res.phase.plv) if res.phase is not None else np.nan,
            "sinc_bpm": res.sinc_bpm, "sinc_confidence": res.sinc_confidence,
            "pulses_from_cache": res.from_cache,
            "seconds_pulse": round(t_pulse, 1), "seconds_fold": round(time.perf_counter() - tp, 1),
            "seconds_total": round(time.perf_counter() - t_all, 1),
            "created": datetime.now().isoformat(timespec="seconds")})
    return cond_rows, bin_rows


# --------------------------------------------------------------------------- #
# Lot
# --------------------------------------------------------------------------- #
def load_table(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def upsert(path: Path, rows: list[dict], slug: str, pulses) -> None:
    df = load_table(path)
    if not df.empty:
        df = df[~((df["slug"] == slug) & df["pulse"].isin(pulses))]
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(path, index=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="One-cycles PLEX pleine largeur, cohorte SANS.")
    ap.add_argument("--moment", choices=["before", "post"], action="append")
    ap.add_argument("--pulse", choices=PULSES, action="append")
    ap.add_argument("--condition", action="append", help="ex. 210713001before_rigidity_OD")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    pulses = tuple(args.pulse or PULSES)

    conds = [c for c in catalog.discover_conditions()
             if (not args.moment or c.moment in args.moment)
             and (not args.condition or c.name in args.condition)]
    conds.sort(key=lambda c: (c.moment != "before", c.rel))
    todo = [c for c in conds if args.overwrite or not all(npz_path(c, p).exists() for p in pulses)]
    if args.limit is not None:
        todo = todo[: args.limit]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "params.json").write_text(json.dumps({
        "data_subdir": DATA_SUBDIR, "n_cycle": N_CYCLE, "n_bins": N_BINS,
        "phase_probe_bins": PHASE_PROBE_BINS, "edge_beats": EDGE_BEATS,
        "fold": "frames mediane, masques moyenne (occupation)", "intra_bin_registration": False,
        "resegmentation": "SegFormer version-2.0.0 + plus grande composante, grille PLEX telle quelle",
        "bin0": "centre sur le maximum d'epaisseur des masques, pour toutes les phases",
        "sinc_phase": "chaine par defaut du package (PULSATION) : SiNC + IQ ancree sur l'epaisseur",
        "pulses_cache": str(pulse_compute.CACHE_ROOT),
        "pulse_compute": {"band_frac": pulse_compute.BAND_FRAC, "roi_depth_um": pulse_compute.ROI_DEPTH_UM,
                          "roi_width_mm": pulse_compute.ROI_WIDTH_MM, "rank_svd": pulse_compute.RANK_SVD},
    }, indent=2), encoding="utf-8")
    print(f"{len(conds)} condition(s), {len(todo)} a traiter, pouls {pulses} -> {OUT_DIR}", flush=True)
    if not todo:
        return 0

    from ocularrigidity.segmentation.utils import get_choroid_segmentation_model

    seg_model = get_choroid_segmentation_model().to(DEVICE)
    print(f"composante connexe : {patch_largest_cc()}\n", flush=True)

    t_batch = time.perf_counter()
    for i, cond in enumerate(todo, 1):
        slug = f"{cond.astro}__{cond.name}"
        t0 = time.perf_counter()
        # Seuls les pouls manquants (sauf --overwrite) : ajouter sinc ne refait pas ct/svd.
        need = pulses if args.overwrite else tuple(p for p in pulses if not npz_path(cond, p).exists())
        try:
            crow, brow = process(cond, seg_model, need)
        except Exception as e:  # noqa: BLE001 - une condition ne doit pas tout arreter
            traceback.print_exc()
            crow = [{"slug": slug, "pulse": p, "condition": cond.name, "status": "error",
                     "reason": f"{type(e).__name__}: {str(e)[:300]}",
                     "created": datetime.now().isoformat(timespec="seconds")} for p in need]
            brow = []
        upsert(CSV_CONDITIONS, crow, slug, need)
        if brow:
            upsert(CSV_BINS, brow, slug, need)
        torch.cuda.empty_cache()
        gc.collect()
        msg = " | ".join(
            f"{r['pulse']}: {r['n_groups']} cycles, {r['frames_per_bin_median']:.0f} fr/bin, "
            f"epaisseur {r['thickness_fold_um_mean']:.0f}/{r['thickness_seg_um_mean']:.0f} um, "
            f"dY fit {r['dY_fold_fit_um_med']:.1f}/{r['dY_seg_fit_um_med']:.1f} um, "
            f"Dice {r['dice_median']:.3f}" if r["status"] == "ok" else f"{r['pulse']}: ERREUR {r['reason']}"
            for r in crow)
        print(f"[{i:>3}/{len(todo)}] {cond.name} | {time.perf_counter() - t0:.0f} s | {msg}", flush=True)
    print(f"\ntermine en {(time.perf_counter() - t_batch) / 60:.1f} min -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
