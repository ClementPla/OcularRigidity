# -*- coding: utf-8 -*-
"""
compute_one_cycle_plex_groups.py

One-cycles des videos ``registered_cascade_v9_plex_full`` (SegFormer + cascade_v9, grille PLEX
pleine largeur) de la cohorte SANS, en un nombre FIXE de one-cycles par video plutot qu'un
nombre fixe de battements par one-cycle (``compute_one_cycle_plex.py`` : 6 battements, ce qui
ne laissait que ~4 frames par bin). Deux scenarios, chacun dans son dossier :

    --groups 1   un seul one-cycle par video : tous les battements retenus
    --groups 3   trois one-cycles par video : battements deroules coupes en trois tiers egaux

Phases (memes pouls que la page « Pouls » du GUI, lus dans son cache sur E:) :
    ct    epaisseur des masques, Hilbert du pouls filtre
    svd   intensite, SVD rang 100 + combinaison optimisee
    sinc  SiNC, chaine par defaut du package (phase IQ ancree sur l'epaisseur)

Etapes reprises de ``compute_one_cycle_plex.py`` (phase, 1 battement exclu a chaque bout, bin 0
centre sur le maximum d'epaisseur pour toutes les phases, repliement frames-mediane /
masques-moyenne sans recalage intra-bin). Ce qui change :

  * groupes : la phase deroulee est decoupee en ``G`` tranches de ``n_battements / G`` (fraction
    de battement comprise : rien n'est jete) ;
  * resegmentation : passe cascade_v9 COMPLETE (``segment_and_register`` : SegFormer
    version-2.0.0 + recalage cascade_v9, ``pipeline_config.REGISTRATION``) sur la pile des
    bins non vides de la condition -- les G one-cycles ensemble, une seule reference. On garde
    le masque resegmente brut (``seg_*``, pas de recalage) ET le one-cycle recale (``reg_*``).
    L'epaisseur par colonne ne depend pas du recalage vertical (somme par colonne) ; le
    recalage sert a regarder le one-cycle sans le tremblement du repliement.

Epaisseurs par bin (um) sur les colonnes ou le masque est present dans >= 90 % des frames :
``thickness_fold`` (masques replies), ``thickness_seg`` (resegmente brut), ``thickness_reg``
(resegmente recale, colonnes ``bad_columns`` du recalage exclues en plus), fondamental ajuste
par one-cycle (``*_fit``) et Dice resegmente / replie (> 0,5).

Sorties :
    E:/NASA_Rigidity/SegmentationVariations/cascade_v9_plex_full/one_cycle_<G>per_video_10bins/
        cycles/<astro>__<condition>__<pouls>.npz   ``one_cycle_figure.load_one_cycle`` le lit
        conditions.csv   une ligne par (condition, pouls)
        bins.csv         une ligne par (condition, pouls, cycle, bin)
        params.json

Lancer (kernel pyOR, depuis la racine) :
    python Astronauts/compute_one_cycle_plex_groups.py --limit 1
    python Astronauts/compute_one_cycle_plex_groups.py               # G = 1 et 3
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

# Ordre impose par les DLL de l'env pyOR : pandas, puis torch, puis le reste.
import pandas as pd
import torch
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "data_visualizer_GUI"))
sys.path.insert(0, str(HERE))

import catalog  # noqa: E402
import pulse_compute  # noqa: E402
from compute_one_cycle_plex import (  # noqa: E402
    CACHE_REG,
    CACHE_SEG,
    DATA_SUBDIR,
    EDGE_BEATS,
    PHASE_PROBE_BINS,
    fit_cycle,
    frame_phase,
    ptp_nan,
    thickness_curve,
    to_u8,
    upsert,
)

from ocularrigidity.data.compression import read_gray  # noqa: E402
from ocularrigidity.data.io import load_mask  # noqa: E402
from ocularrigidity.motion.one_cycle import (  # noqa: E402
    fold_video_numba_mean,
    fold_video_numba_median,
)
from ocularrigidity.pipeline_config import REGISTRATION  # noqa: E402
from ocularrigidity.scripts.registration.cascade_v9_plex import load_models, run_fused  # noqa: E402

N_BINS = 10
GROUPS = (1, 3)
PULSES = ("ct", "svd", "sinc")
FUSED_BATCH = 2  # grille PLEX pleine largeur : 2 tenait (cascade_v9_plex.VARIANTS["full"])


def out_dir(n_groups: int) -> Path:
    return (catalog.SEGVAR_ROOT / DATA_SUBDIR.removeprefix("registered_")
            / f"one_cycle_{n_groups}per_video_{N_BINS}bins")


def npz_path(n_groups: int, cond: catalog.Condition, pulse: str) -> Path:
    return out_dir(n_groups) / "cycles" / f"{cond.astro}__{cond.name}__{pulse}.npz"


def packbits(m: np.ndarray) -> np.ndarray:
    return np.packbits(np.asarray(m, bool).reshape(-1))


def resegment_cascade(cycles_u8: np.ndarray, filled: np.ndarray, seg_model, reg_model):
    """Passe cascade_v9 sur les bins non vides ; les bins vides restent a zero."""
    n, H, W = cycles_u8.shape
    idx = np.flatnonzero(filled)
    res, cfg = run_fused(cycles_u8[idx], seg_model, reg_model,
                         replace(REGISTRATION, fused_batch_size=FUSED_BATCH))

    def scatter(a, dtype):
        out = np.zeros((n, H, W), dtype)
        out[idx] = a
        return out

    dx = np.full(n, np.nan, np.float32)
    dy = np.full((n, W), np.nan, np.float32)
    dx[idx], dy[idx] = res.transform["dx"], res.transform["dy"]
    return {
        "seg_masks": scatter(res.raw_masks, bool), "seg_probs": scatter(res.raw_probs, np.uint8),
        "reg_frames": scatter(res.registered_frames, np.uint8),
        "reg_masks": scatter(res.registered_masks, bool),
        "reg_probs": scatter(res.registered_probs, np.uint8),
        "dx": dx, "dy": dy, "bad_columns": np.asarray(res.transform["bad_columns"], bool),
        "ref_slot": int(idx[res.ref_idx]), "ref_percentile": res.ref_percentile,
        "config": cfg,
    }


# --------------------------------------------------------------------------- #
# Une condition : video et pouls lus une fois, tous les (G, pouls) demandes
# --------------------------------------------------------------------------- #
def process(cond: catalog.Condition, seg_model, reg_model, jobs) -> dict[int, tuple[list, list]]:
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

    t0 = time.perf_counter()
    res = pulse_compute.cached_compute(
        pulse_compute.cache_path(CACHE_REG, CACHE_SEG, cond.astro, cond.name),
        frames, masks, t, catalog.read_hr(cond), pixel_mm)
    t_pulse = time.perf_counter() - t0
    hr = res.hr
    thick_frames = res.mask_pulse.raw

    cover = masks.any(axis=1).mean(axis=0) >= pulse_compute.MIN_COL_COVERAGE
    cols = np.flatnonzero(cover)
    if cols.size == 0:
        raise ValueError("aucune colonne de choroide exploitable")
    margin = EDGE_BEATS * 60.0 / hr
    edge_ok = (t >= t[0] + margin) & (t <= t[-1] - margin)
    masks_f = masks.view(np.uint8)
    sources = {"ct": res.mask_pulse, "svd": res.svd_pulse, "sinc": res.sinc_pulse}
    slug = f"{cond.astro}__{cond.name}"

    # Phase par pouls (independante de G).
    phases = {}
    for pulse in {p for _, ps in jobs.items() for p in ps}:
        tr = sources[pulse]
        if tr is None:
            phases[pulse] = ValueError(f"pas de pouls {pulse}"
                                       + (f" ({res.sinc_note})" if pulse == "sinc" else ""))
            continue
        if pulse == "sinc":
            cyc = np.unwrap(2 * np.pi * res.sinc_phase_per_frame) / (2 * np.pi)
            good = edge_ok & np.isfinite(cyc) & res.sinc_good_per_frame
            info = f"{tr.info} ; {res.sinc_note}"
        else:
            cyc = frame_phase(tr.filtered, res.u, t)
            good = edge_ok & np.isfinite(cyc)
            info = tr.info
        phase_raw = np.mod(cyc, 1.0)
        pb = (phase_raw * PHASE_PROBE_BINS).astype(int) % PHASE_PROBE_BINS
        s = np.bincount(pb[good], weights=thick_frames[good], minlength=PHASE_PROBE_BINS)
        c = np.bincount(pb[good], minlength=PHASE_PROBE_BINS).astype(float)
        probe = np.where(c > 0, s / np.maximum(c, 1), np.nan)
        phase_max = (np.nanargmax(probe) + 0.5) / PHASE_PROBE_BINS
        phase_min = (np.nanargmin(probe) + 0.5) / PHASE_PROBE_BINS
        phase_f = np.mod(phase_raw - phase_max + 0.5 / N_BINS, 1.0).astype(np.float32)
        pos = cyc - cyc[good].min()
        phases[pulse] = dict(cyc=cyc, good=good, info=info, phase_raw=phase_raw,
                             phase_max=phase_max, phase_min=phase_min, phase_f=phase_f,
                             pos=pos, n_beats=float(pos[good].max()))

    results = {}
    for G, pulses in jobs.items():
        cond_rows, bin_rows = [], []
        for pulse in pulses:
            ph = phases[pulse]
            if isinstance(ph, Exception):
                cond_rows.append({"slug": slug, "pulse": pulse, "condition": cond.name,
                                  "status": "error", "reason": str(ph),
                                  "created": datetime.now().isoformat(timespec="seconds")})
                continue
            tp = time.perf_counter()
            good, phase_f, n_beats = ph["good"], ph["phase_f"], ph["n_beats"]
            beats_per_group = n_beats / G
            group_of = np.clip(np.floor(ph["pos"] / beats_per_group).astype(int), 0, G - 1)
            good_slot = good.copy()
            for g in range(G):
                if int((good & (group_of == g)).sum()) < N_BINS:
                    good_slot &= group_of != g

            n_slots = G * N_BINS
            cycles = np.zeros((n_slots, H, W), np.float32)
            occupancy = np.zeros((n_slots, H, W), np.float32)
            counts = np.zeros(n_slots, np.int32)
            for g in range(G):
                keep = good_slot & (group_of == g)
                if not keep.any():
                    continue
                sl = slice(g * N_BINS, (g + 1) * N_BINS)
                cf, cc = fold_video_numba_median(frames, phase_f, keep, n_bins=N_BINS)
                mo, _ = fold_video_numba_mean(masks_f, phase_f, keep, n_bins=N_BINS)
                cycles[sl], occupancy[sl], counts[sl] = cf, mo, cc
            cycles_u8 = to_u8(cycles)
            del cycles
            filled = counts > 0
            cycles_u8[~filled] = 0
            if not filled.any():
                raise ValueError(f"G={G}, {pulse} : aucun bin rempli")

            rs = resegment_cascade(cycles_u8, filled, seg_model, reg_model)
            bad = rs["bad_columns"]
            cols_reg = cols[~bad[cols]]

            th_fold = thickness_curve(occupancy, cols, counts, um_px)
            th_seg = thickness_curve(rs["seg_masks"], cols, counts, um_px)
            th_reg = (thickness_curve(rs["reg_masks"], cols_reg, counts, um_px)
                      if cols_reg.size else np.full(n_slots, np.nan))

            def fit_all(th):
                return np.concatenate([fit_cycle(th[g * N_BINS:(g + 1) * N_BINS]) for g in range(G)])

            fit_fold, fit_seg, fit_reg = fit_all(th_fold), fit_all(th_seg), fit_all(th_reg)
            a, b = rs["seg_masks"], occupancy > 0.5
            dice = 2 * (a & b).sum(axis=(1, 2)) / (a.sum(axis=(1, 2)) + b.sum(axis=(1, 2)) + 1e-12)
            dice = np.where(filled, dice, np.nan)

            out = npz_path(G, cond, pulse)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.name + ".tmp.npz")
            np.savez_compressed(
                tmp, cycles=cycles_u8, occupancy=to_u8(occupancy * 255.0),
                seg_masks_packed=packbits(rs["seg_masks"]), seg_shape=np.array(cycles_u8.shape, np.int64),
                seg_probs=rs["seg_probs"],
                reg_frames=rs["reg_frames"], reg_masks_packed=packbits(rs["reg_masks"]),
                reg_probs=rs["reg_probs"], reg_dx=rs["dx"], reg_dy=rs["dy"],
                bad_columns=bad, ref_slot=rs["ref_slot"], cols_reg=cols_reg,
                counts=counts,
                thickness_fold=th_fold, thickness_fold_fit=fit_fold,
                thickness_seg=th_seg, thickness_seg_fit=fit_seg,
                thickness_reg=th_reg, thickness_reg_fit=fit_reg, dice=dice, cols=cols,
                phase_f=phase_f, phase_raw=ph["phase_raw"], good_f=good, good_slot=good_slot,
                group_of=group_of, n_bins=N_BINS, n_cycle=int(round(beats_per_group)),
                beats_per_group=beats_per_group, n_groups=G, hr=hr,
                pixel_um=np.array([1e3 * pixel_mm[0], 1e3 * pixel_mm[1]]),
                phase_max=ph["phase_max"], phase_min=ph["phase_min"], pulse=np.array(pulse),
                condition=np.array(cond.name), astro=np.array(cond.astro),
                data_subdir=np.array(DATA_SUBDIR), pulse_info=np.array(ph["info"]),
                resegmentation=np.array("cascade_v9 (segment_and_register)"),
                sinc_bpm=res.sinc_bpm, sinc_confidence=np.array(res.sinc_confidence))
            tmp.replace(out)

            for k in range(n_slots):
                bin_rows.append({
                    "slug": slug, "pulse": pulse, "cycle": k // N_BINS + 1, "bin": k % N_BINS,
                    "n_frames": int(counts[k]), "thickness_fold_um": th_fold[k],
                    "thickness_fold_fit_um": fit_fold[k], "thickness_seg_um": th_seg[k],
                    "thickness_seg_fit_um": fit_seg[k], "thickness_reg_um": th_reg[k],
                    "thickness_reg_fit_um": fit_reg[k], "dice": dice[k]})

            def per_cycle(th):
                return [ptp_nan(th[g * N_BINS:(g + 1) * N_BINS]) for g in range(G)]

            cond_rows.append({
                "slug": slug, "pulse": pulse, "astro": cond.astro, "moment": cond.moment,
                "eye": f"{cond.eye}{cond.rep}", "condition": cond.name, "status": "ok", "reason": "",
                "n_frames": T, "hr_BPM": hr, "hr_source": res.hr_source, "pixel_axial_um": um_px,
                "n_cols": int(cols.size), "n_cols_reg": int(cols_reg.size), "n_beats": n_beats,
                "n_groups": G, "beats_per_group": beats_per_group,
                "n_frames_kept": int(good_slot.sum()), "n_bins_empty": int((~filled).sum()),
                "frames_per_bin_median": float(np.median(counts[filled])),
                "frames_per_bin_min": int(counts[filled].min()),
                "phase_max": ph["phase_max"], "phase_min": ph["phase_min"],
                "thickness_fold_um_mean": float(np.nanmean(th_fold)),
                "thickness_seg_um_mean": float(np.nanmean(th_seg)),
                "thickness_reg_um_mean": float(np.nanmean(th_reg)),
                "dY_fold_fit_um_med": float(np.nanmedian(per_cycle(fit_fold))),
                "dY_seg_fit_um_med": float(np.nanmedian(per_cycle(fit_seg))),
                "dY_reg_fit_um_med": float(np.nanmedian(per_cycle(fit_reg))),
                "dY_fold_raw_um_med": float(np.nanmedian(per_cycle(th_fold))),
                "dY_seg_raw_um_med": float(np.nanmedian(per_cycle(th_seg))),
                "dY_reg_raw_um_med": float(np.nanmedian(per_cycle(th_reg))),
                "dice_median": float(np.nanmedian(dice)),
                "reg_ref_slot": rs["ref_slot"], "reg_dx_ptp": ptp_nan(rs["dx"]),
                "reg_n_bad_columns": int(bad.sum()),
                "plv_svd_ct": float(res.phase.plv) if res.phase is not None else np.nan,
                "sinc_bpm": res.sinc_bpm, "sinc_confidence": res.sinc_confidence,
                "pulses_from_cache": res.from_cache,
                "seconds_pulse": round(t_pulse, 1), "seconds_fold": round(time.perf_counter() - tp, 1),
                "seconds_total": round(time.perf_counter() - t_all, 1),
                "created": datetime.now().isoformat(timespec="seconds")})
            del occupancy, cycles_u8, rs
            gc.collect()
            torch.cuda.empty_cache()
        results[G] = (cond_rows, bin_rows)
    return results


# --------------------------------------------------------------------------- #
# Lot
# --------------------------------------------------------------------------- #
def write_params(G: int) -> None:
    d = out_dir(G)
    d.mkdir(parents=True, exist_ok=True)
    (d / "params.json").write_text(json.dumps({
        "data_subdir": DATA_SUBDIR, "n_groups": G, "n_bins": N_BINS,
        "grouping": "phase deroulee coupee en n_groups tranches egales de n_battements / n_groups",
        "phase_probe_bins": PHASE_PROBE_BINS, "edge_beats": EDGE_BEATS,
        "fold": "frames mediane, masques moyenne (occupation)", "intra_bin_registration": False,
        "resegmentation": "cascade_v9 segment_and_register (SegFormer version-2.0.0 + cascade_v9) "
                          "sur la pile des bins non vides, une reference pour tous les one-cycles",
        "registration_config": {**asdict(REGISTRATION), "fused_batch_size": FUSED_BATCH},
        "bin0": "centre sur le maximum d'epaisseur des masques, pour toutes les phases",
        "sinc_phase": "chaine par defaut du package (PULSATION) : SiNC + IQ ancree sur l'epaisseur",
        "pulses_cache": str(pulse_compute.CACHE_ROOT),
    }, indent=2, default=str), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="One-cycles PLEX, 1 ou 3 par video, resegmentes cascade_v9.")
    ap.add_argument("--groups", type=int, action="append", help="one-cycles par video (defaut 1 et 3)")
    ap.add_argument("--moment", choices=["before", "post"], action="append")
    ap.add_argument("--pulse", choices=PULSES, action="append")
    ap.add_argument("--condition", action="append", help="ex. 210713001before_rigidity_OD")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    groups = tuple(args.groups or GROUPS)
    pulses = tuple(args.pulse or PULSES)

    conds = [c for c in catalog.discover_conditions()
             if (not args.moment or c.moment in args.moment)
             and (not args.condition or c.name in args.condition)]
    conds.sort(key=lambda c: (c.moment != "before", c.rel))

    def need(c):
        return {G: tuple(p for p in pulses if args.overwrite or not npz_path(G, c, p).exists())
                for G in groups}

    todo = [c for c in conds if any(need(c).values())]
    if args.limit is not None:
        todo = todo[: args.limit]
    for G in groups:
        write_params(G)
    print(f"{len(conds)} condition(s), {len(todo)} a traiter, G {groups}, pouls {pulses}", flush=True)
    if not todo:
        return 0

    seg_model, reg_model, cc = load_models()
    print(f"composante connexe : {cc}\n", flush=True)

    t_batch = time.perf_counter()
    for i, cond in enumerate(todo, 1):
        slug = f"{cond.astro}__{cond.name}"
        t0 = time.perf_counter()
        jobs = {G: ps for G, ps in need(cond).items() if ps}
        try:
            results = process(cond, seg_model, reg_model, jobs)
        except Exception as e:  # noqa: BLE001 - une condition ne doit pas tout arreter
            traceback.print_exc()
            results = {G: ([{"slug": slug, "pulse": p, "condition": cond.name, "status": "error",
                             "reason": f"{type(e).__name__}: {str(e)[:300]}",
                             "created": datetime.now().isoformat(timespec="seconds")} for p in ps], [])
                       for G, ps in jobs.items()}
        msgs = []
        for G, (crow, brow) in results.items():
            d = out_dir(G)
            upsert(d / "conditions.csv", crow, slug, jobs[G])
            if brow:
                upsert(d / "bins.csv", brow, slug, jobs[G])
            msgs.append(f"G={G} " + " ".join(
                f"{r['pulse']}:{r['frames_per_bin_median']:.0f}fr/bin,dY {r['dY_fold_fit_um_med']:.1f}"
                f"/{r['dY_seg_fit_um_med']:.1f}um,Dice {r['dice_median']:.3f}"
                if r["status"] == "ok" else f"{r['pulse']}:ERREUR {r['reason'][:80]}" for r in crow))
        torch.cuda.empty_cache()
        gc.collect()
        print(f"[{i:>3}/{len(todo)}] {cond.name} | {time.perf_counter() - t0:.0f} s | "
              + " || ".join(msgs), flush=True)
    print(f"\ntermine en {(time.perf_counter() - t_batch) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
