# -*- coding: utf-8 -*-
"""
compute_one_cycle_ct.py

One-cycles des acquisitions de repetabilite, phase assignee par le POULS
D'EPAISSEUR CHOROIDIENNE (issu de la segmentation), et non plus par le pouls
d'intensite pixel. Chaque video cascade_v9 (deja recalee GLOBALEMENT) est
repliee en one-cycles de ``N_CYCLE`` battements x ``N_BINS`` bins, avec un
RECALAGE INTRA-BIN par cascade_v9 (meme modele que pour la segmentation), puis
le one-cycle replie est resegmente.

Le recalage intra-bin est ici un recalage Y GENERAL (un seul dy par frame,
``dy_bulk_only=True``) et non plus par A-scan : le decalage axial rigide est
corrige, mais pas la deformation locale de la BM le long de la largeur de
l'image.

Entrees
-------
    E:/SANSORI/Reproducibility/<SLUG>/RawImages/
        registered_cascade_v9_1536x1024/{registered_video.mp4, mask.npz}
        <hash>.xml                                  horodatages, taille du pixel
    E:/NASA_Rigidity/Reproducibility/SegmentationVariations/
        cascade_v9_1536x1024_hr_mediane/pulse_from_data/
            conditions.csv  traces/<slug>.npz        hr, t, u_time (grille de reference)
            ct_pulse_traces.npz                      POULS d'epaisseur <slug>__ct_fir_um
                                                       (ecrit par compute_ct_pulse_traces.py :
                                                       epaisseur choroidienne des masques
                                                       cascade_v9, passe-bande FIR a la FC
                                                       ancre +/- 20 %, sur la grille uniforme)

La phase n'est PAS reprise d'un .npz existant : elle est calculee ICI, par
transformee de Hilbert du pouls d'epaisseur, exactement comme
``Astronauts/compute_pulse_from_data.py::analytic_phase`` le fait pour le pouls
d'intensite (``hilbert(y - y.mean())``, angle deroule puis reduit modulo 2 pi).
Le reste du decoupage reprend celui de ``notebook/demons_sitk_variants.ipynb`` :

  - la phase est recentree pour que le bin 0 soit CENTRE sur le maximum du
    pouls d'EPAISSEUR (et non plus d'intensite) ;
  - les battements sont comptes sur la phase deroulee et regroupes par
    ``N_CYCLE`` consecutifs : un groupe = un one-cycle ; un groupe de moins de
    ``N_BINS`` frames est ecarte.

Le recalage intra-bin
----------------------
Pour chaque (one-cycle, bin), TOUTES les frames du bin partent ensemble dans UN
appel ``registration.fused.segment_and_register`` : le volume du bin est agrandi
a 1536 x 1024 (la taille d'entrainement du regresseur, cf.
``register_newmodel.py``), la reference est choisie parmi SES frames par la regle
du modele (``motion_medoid``), et le dx / dy predit est applique en UN SEUL warp
(``losses.warp``) aux frames et aux masques natifs, en flottant. ``dy_bulk_only``
force le regresseur a ne rendre que le decalage axial RIGIDE (un dy constant sur
toute la largeur de la frame, cf. ``RegistrationRegressor.forward``) : pas de
residu par A-scan.

Pas de recalage ENTRE bins : une translation rigide par colonne ne change pas
l'epaisseur.

Deux epaisseurs par bin, plus un temoin
----------------------------------------
  - ``thickness_fold``  masques par frame, recales dans le bin (recalage Y
                        general), puis replies ;
  - ``thickness_seg``   le one-cycle replie, RESEGMENTE avec le protocole des
                        masques de la video cascade_v9 (agrandi 1536 x 1024,
                        SegFormer ``version-2.0.0`` + plus grande composante,
                        reduit par moyenne de surface, seuil 0,5).
Plus un temoin ``thickness_std`` : le repliement standard, SANS recalage
intra-bin (``NCycleReconstructor``, celui de la cohorte).

Sorties
-------
    E:/NASA_Rigidity/Reproducibility/SegmentationVariations/
        cascade_v9_1536x1024_hr_mediane/one_cycle_ct_6beats_6bins/
            conditions.csv   une ligne par acquisition
            bins.csv         une ligne par (one-cycle, bin)
            params.json
            cycles/<slug>.npz

Les tables sont reecrites apres CHAQUE acquisition ; le lot reprend ou il s'est
arrete (``--overwrite`` pour tout refaire).

Lancer (kernel pyOR, depuis la racine du depot) :
    C:/Users/transformer/anaconda3/envs/pyOR/python.exe Reproducibility/compute_one_cycle_ct.py --slug BELANGER_CHARLES_OD1
    C:/Users/transformer/anaconda3/envs/pyOR/python.exe Reproducibility/compute_one_cycle_ct.py
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

import numpy as np
import pandas as pd
import torch
from scipy.signal import hilbert

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "Astronauts"))

import compute_one_cycle_compare_methods as occ  # noqa: E402
import register_newmodel as rnm  # noqa: E402
from ocularrigidity.consts import AXIAL_PIXEL_SIZE_MM  # noqa: E402
from ocularrigidity.motion.one_cycle import fold_video_numba_mean  # noqa: E402
from ocularrigidity.motion.pulsation import NCycleConfig, NCycleReconstructor  # noqa: E402
from ocularrigidity.motion.video_timeline_aligner import VideoTimelineAligner  # noqa: E402
from ocularrigidity.pipeline_config import REGISTRATION  # noqa: E402
from ocularrigidity.registration import fused  # noqa: E402
from ocularrigidity.registration.deep_learning.models.losses import (  # noqa: E402
    _resample_dy,
    warp,
)
from ocularrigidity.scripts.one_cycle.astronauts import _prepared_registrator  # noqa: E402
from ocularrigidity.scripts.registration.astronauts import load_ordered_oct_series  # noqa: E402
from ocularrigidity.segmentation.utils import (  # noqa: E402
    get_choroid_segmentation_model,
    get_registration_model,
)

# --------------------------------------------------------------------------- #
# Parametres
# --------------------------------------------------------------------------- #
PATH_GENERAL = Path("E:/SANSORI/Reproducibility")
SEGVAR_ROOT = Path("E:/NASA_Rigidity/Reproducibility/SegmentationVariations")
DATA_SUBDIR = "registered_cascade_v9_1536x1024"
PULSE_VARIANT = "cascade_v9_1536x1024_hr_mediane"
PULSE_DIR = SEGVAR_ROOT / PULSE_VARIANT / "pulse_from_data"
CT_NPZ = PULSE_DIR / "ct_pulse_traces.npz"

PULSE_METHOD = "ct_fir"  # pouls d'EPAISSEUR (segmentation), pas d'intensite
N_CYCLE = 6               # battements MOYENNES pour faire UN one-cycle
N_BINS = 6                # bins de phase par one-cycle
FOLD_METHOD = "median"    # frames ; les masques sont toujours moyennes (occupation)
PHASE_PROBE_BINS = 60     # grille fine de la phase du maximum du pouls
COL_FRAC = (0.125, 0.875)

MODEL_SHAPE = (1536, 1024)   # taille d'entrainement du regresseur cascade_v9
FUSED_BATCH = 2               # 12 Go sous Windows : cf. register_newmodel.py
DEVICE = "cuda"

OUTPUT_SUBDIR = f"one_cycle_ct_{N_CYCLE}beats_{N_BINS}bins"
OUT_DIR = SEGVAR_ROOT / PULSE_VARIANT / OUTPUT_SUBDIR
CYCLES_DIR = OUT_DIR / "cycles"
CSV_CONDITIONS = OUT_DIR / "conditions.csv"
CSV_BINS = OUT_DIR / "bins.csv"


# --------------------------------------------------------------------------- #
# Recalage intra-bin (Y general) et resegmentation
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def register_bin(frames_u8, masks, seg_model, reg_model, config):
    """Recale entre elles TOUTES les frames d'un bin, en un seul appel cascade_v9.

    ``config.dy_bulk_only`` force un dy CONSTANT sur la largeur (recalage Y
    general, pas par A-scan). Retourne (frames float32, occupation float32,
    dx natif (n,), dy natif (n, W), indice local de la reference). Un bin a une
    frame est rendu tel quel.
    """
    n, H, W = frames_u8.shape
    if n < 2:
        return (frames_u8.astype(np.float32), masks.astype(np.float32),
                np.zeros(n, np.float32), np.zeros((n, W), np.float32), 0)

    result = fused.segment_and_register(rnm.resize_frames(frames_u8, MODEL_SHAPE),
                                        seg_model, reg_model, config,
                                        device=DEVICE, verbose=False)
    dx_m = torch.from_numpy(np.asarray(result.transform["dx"], dtype=np.float32))
    dy_m = torch.from_numpy(np.asarray(result.transform["dy"], dtype=np.float32))
    ref_idx = int(result.ref_idx)
    del result

    # UN SEUL reechantillonnage, sur les images NATIVES : ``warp`` ramene lui-meme
    # le transform de la grille du modele a celle de ce qu'il deforme.
    f_out = np.empty((n, H, W), dtype=np.float32)
    o_out = np.empty((n, H, W), dtype=np.float32)
    for s in range(0, n, rnm.WARP_BATCH):
        e = min(s + rnm.WARP_BATCH, n)
        data = torch.stack([torch.from_numpy(masks[s:e]).to(DEVICE).float(),
                            torch.from_numpy(frames_u8[s:e]).to(DEVICE).float()], dim=1)
        out, _ = warp(data, dx_m[s:e].to(DEVICE), dy_m[s:e].to(DEVICE), MODEL_SHAPE)
        o_out[s:e] = out[:, 0].clamp(0, 1).cpu().numpy()
        f_out[s:e] = out[:, 1].clamp(0, 255).cpu().numpy()
    dx_native = (dx_m * (W / MODEL_SHAPE[1])).numpy()
    dy_native = (_resample_dy(dy_m, W) * (H / MODEL_SHAPE[0])).numpy()
    return f_out, o_out, dx_native, dy_native, ref_idx


@torch.inference_mode()
def resegment(cycles_u8, seg_model):
    """Le protocole des masques de la video cascade_v9 : agrandi, SegFormer +
    plus grande composante connexe, reduit par moyenne de surface, seuil 0,5."""
    n, H, W = cycles_u8.shape
    up = rnm.resize_frames(cycles_u8, MODEL_SHAPE)
    out = np.empty((n, *MODEL_SHAPE), dtype=bool)
    for s in range(0, n, FUSED_BATCH):
        e = min(s + FUSED_BATCH, n)
        mask, _ = fused._encode_segment(seg_model, torch.from_numpy(up[s:e]).to(DEVICE),
                                        torch.bfloat16, keep_largest_cc=True)
        out[s:e] = mask.cpu().numpy()
    return rnm.resize_masks(out, (H, W))


# --------------------------------------------------------------------------- #
# Une acquisition
# --------------------------------------------------------------------------- #
def to_u8(stack):
    return np.clip(np.nan_to_num(np.asarray(stack, dtype=np.float32)), 0, 255).round().astype(np.uint8)


def ptp_nan(y):
    y = np.asarray(y, dtype=float)
    return float(np.nanmax(y) - np.nanmin(y)) if np.isfinite(y).any() else np.nan


def process(slug, row, seg_model, reg_model, config, ct_traces):
    t_all = time.perf_counter()
    m = rnm.RE_SLUG.match(slug)
    raw_dir = PATH_GENERAL / slug / "RawImages"
    data_dir = raw_dir / DATA_SUBDIR
    npz_path = PULSE_DIR / "traces" / f"{slug}.npz"
    ct_key = f"{slug}__ct_fir_um"
    for p in (data_dir / "registered_video.mp4", data_dir / "mask.npz", npz_path):
        if not p.exists():
            raise FileNotFoundError(p)
    if ct_key not in ct_traces.files:
        raise KeyError(f"{CT_NPZ.name} sans {ct_key}")

    # --- Video, horodatages ---------------------------------------------------
    registrator = _prepared_registrator(data_dir / "registered_video.mp4",
                                        data_dir / "mask.npz", DEVICE, verbose=False)
    frames = np.asarray(registrator.registered_frames)
    masks = np.asarray(registrator.registered_masks, dtype=bool)
    T, H, W = frames.shape

    ts_us = occ.raw_timestamps_us(raw_dir)
    if ts_us.size != T:
        raise ValueError(f"frames ({T}) != horodatages XML ({ts_us.size})")
    aligner = VideoTimelineAligner(registrator, ts_us)
    u_time = aligner.uniform_time

    data = np.load(npz_path)
    hr = float(data["hr"])
    if (not np.allclose(aligner.timestamps_seconds, data["t"], atol=1e-6)
            or not np.allclose(u_time, data["u_time"], atol=1e-6)):
        raise ValueError("les horodatages du .npz ne sont pas ceux de la video chargee")

    series = load_ordered_oct_series(raw_dir)
    ax = [s.axial_resolution for s in series if s.axial_resolution]
    um_per_px_y = 1000.0 * float(np.median(ax)) if ax else 1000.0 * AXIAL_PIXEL_SIZE_MM

    # ROI d'epaisseur : la regle du carnet (et de compute_one_cycle_compare_methods).
    roi_all = masks.all(axis=0)
    if roi_all.sum() < 100:
        roi_all = masks.mean(axis=0) > 0.5
    c0, c1 = int(COL_FRAC[0] * W), int(COL_FRAC[1] * W)
    cols = np.flatnonzero(roi_all[:, c0:c1].any(axis=0)) + c0
    if cols.size == 0:
        raise ValueError("aucune colonne de choroide exploitable")

    # --- Pouls d'epaisseur (segmentation) et sa phase de Hilbert -------------
    pulse = np.asarray(ct_traces[ct_key], dtype=float)
    if pulse.shape != u_time.shape:
        raise ValueError("pouls d'epaisseur (grille uniforme) de taille differente de u_time")
    phase_uniform = np.mod(np.angle(hilbert(pulse - pulse.mean())), 2 * np.pi)
    good_u = occ.edge_mask(u_time, hr) & ~aligner.gap_mask(np.zeros(T, dtype=bool))
    extractor = occ.build_extractor(registrator, aligner, pulse, phase_uniform, hr, good_u)
    phase_raw = np.asarray(extractor.phase_per_frame)
    good_f = np.asarray(extractor.good_per_frame, dtype=bool)

    # --- Bin 0 centre sur le maximum du pouls d'epaisseur ---------------------
    pulse_on_frames = np.interp(extractor.timestamps_seconds, u_time, pulse)
    probe_bin = (phase_raw * PHASE_PROBE_BINS).astype(int) % PHASE_PROBE_BINS
    probe_sum = np.bincount(probe_bin[good_f], weights=pulse_on_frames[good_f],
                            minlength=PHASE_PROBE_BINS)
    probe_cnt = np.bincount(probe_bin[good_f], minlength=PHASE_PROBE_BINS).astype(float)
    probe = np.where(probe_cnt > 0, probe_sum / np.maximum(probe_cnt, 1), np.nan)
    phase_max = (np.nanargmax(probe) + 0.5) / PHASE_PROBE_BINS
    phase_min = (np.nanargmin(probe) + 0.5) / PHASE_PROBE_BINS
    phase_f = np.mod(phase_raw - phase_max + 0.5 / N_BINS, 1.0).astype(np.float32)

    # --- Groupes de N_CYCLE battements, comptes sur la phase deroulee -------
    cycle_pos = np.unwrap(2 * np.pi * phase_raw) / (2 * np.pi)
    cycle_pos = cycle_pos - cycle_pos[good_f].min()
    n_beats = float(cycle_pos[good_f].max())
    group_of = np.floor(cycle_pos / N_CYCLE).astype(int)
    group_of = np.clip(group_of, 0, max(int(np.floor(n_beats / N_CYCLE)) - 1, 0))
    n_groups = int(group_of[good_f].max()) + 1
    n_slots = n_groups * N_BINS

    bin_of = (phase_f * N_BINS).astype(np.int32) % N_BINS
    slot_of = group_of * N_BINS + bin_of
    good_slot = good_f.copy()
    for g in range(n_groups):
        sel_g = group_of == g
        if int((good_f & sel_g).sum()) < N_BINS:
            good_slot &= ~sel_g

    # --- Temoin : repliement standard, sans recalage intra-bin --------------
    reconstructor = NCycleReconstructor(
        extractor, NCycleConfig(n_bins=N_BINS, n_cycle=1, fold_method=FOLD_METHOD, verbose=False))
    cyc_std, occ_std, cnt_std = [], [], []
    masks_f = masks.astype(np.float32)
    for g in range(n_groups):
        keep_g = good_f & (group_of == g)
        if int(keep_g.sum()) < N_BINS:
            cyc_std.append(np.full((N_BINS, H, W), np.nan, np.float32))
            occ_std.append(np.zeros((N_BINS, H, W), np.float32))
            cnt_std.append(np.zeros(N_BINS, np.int32))
            continue
        cyc_g, _ = reconstructor.compute(phase_per_frame=phase_f, good_per_frame=keep_g,
                                         n_cycle=1, n_bins=N_BINS)
        mc, cc = fold_video_numba_mean(masks_f, phase_f, keep_g, n_bins=N_BINS, verbose=False)
        cyc_std.append(np.asarray(cyc_g, np.float32))
        occ_std.append(np.asarray(mc, np.float32))
        cnt_std.append(np.asarray(cc, np.int32))
    del masks_f
    cycles_std = np.concatenate(cyc_std, axis=0)
    occupancy_std = np.concatenate(occ_std, axis=0)
    counts = np.concatenate(cnt_std, axis=0)
    counts_slot = np.bincount(slot_of[good_slot], minlength=n_slots)[:n_slots]
    if not np.array_equal(counts, counts_slot):
        raise ValueError(f"occupation des bins incoherente : repliement {counts.tolist()} "
                         f"contre affectation {counts_slot.tolist()}")

    # --- Recalage intra-bin cascade_v9 (Y general), puis repliement ----------
    t0 = time.perf_counter()
    cycles = np.full((n_slots, H, W), np.nan, dtype=np.float32)
    occupancy = np.zeros((n_slots, H, W), dtype=np.float32)
    dx_frame = np.full(T, np.nan, dtype=np.float32)
    dy_frame = np.full((T, W), np.nan, dtype=np.float32)
    ref_frame = np.full(n_slots, -1, dtype=np.int64)
    dx_ptp = np.full(n_slots, np.nan)
    dy_bulk_ptp = np.full(n_slots, np.nan)
    dy_intra_std = np.full(n_slots, np.nan)
    for s in range(n_slots):
        idx = np.flatnonzero(good_slot & (slot_of == s))
        if idx.size == 0:
            continue
        f_s, o_s, dx_s, dy_s, ref_local = register_bin(frames[idx], masks[idx],
                                                       seg_model, reg_model, config)
        cycles[s] = np.median(f_s, axis=0) if FOLD_METHOD == "median" else f_s.mean(axis=0)
        occupancy[s] = o_s.mean(axis=0)
        dx_frame[idx], dy_frame[idx] = dx_s, dy_s
        ref_frame[s] = int(idx[ref_local])
        dx_ptp[s] = float(np.ptp(dx_s))
        dy_bulk_ptp[s] = float(np.ptp(dy_s.mean(axis=1)))
        dy_intra_std[s] = float(np.median(dy_s.std(axis=1)))
    t_reg = time.perf_counter() - t0

    # --- Epaisseurs ----------------------------------------------------------
    def fit_by_cycle(th):
        return np.concatenate([occ.fit_cycle(th[g * N_BINS:(g + 1) * N_BINS])
                               for g in range(n_groups)])

    thickness_fold = occ.thickness_curve(occupancy, cols, counts)
    thickness_std = occ.thickness_curve(occupancy_std, cols, counts)
    cycles_u8 = to_u8(cycles)
    t0 = time.perf_counter()
    seg_masks = resegment(cycles_u8, seg_model)
    t_seg = time.perf_counter() - t0
    thickness_seg = occ.thickness_curve(seg_masks.astype(np.float32), cols, counts)
    fit_fold, fit_seg, fit_std = (fit_by_cycle(thickness_fold), fit_by_cycle(thickness_seg),
                                  fit_by_cycle(thickness_std))
    a, b = seg_masks, occupancy > 0.5
    dice = 2 * (a & b).sum(axis=(1, 2)) / (a.sum(axis=(1, 2)) + b.sum(axis=(1, 2)) + 1e-12)
    dice = np.where(counts > 0, dice, np.nan)

    # --- Ecriture ------------------------------------------------------------
    CYCLES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CYCLES_DIR / f"{slug}.npz",
        cycles=cycles_u8, cycles_std=to_u8(cycles_std),
        occupancy=occupancy.astype(np.float16), occupancy_std=occupancy_std.astype(np.float16),
        seg_masks=seg_masks, counts=counts,
        thickness_fold=thickness_fold, thickness_fold_fit=fit_fold,
        thickness_seg=thickness_seg, thickness_seg_fit=fit_seg,
        thickness_std=thickness_std, thickness_std_fit=fit_std, dice=dice,
        cols=cols, phase_f=phase_f, phase_raw=phase_raw, good_f=good_f, good_slot=good_slot,
        group_of=group_of, slot_of=slot_of, dx_frame=dx_frame, dy_frame=dy_frame.astype(np.float16),
        ref_frame=ref_frame, dx_ptp=dx_ptp, dy_bulk_ptp=dy_bulk_ptp, dy_intra_std=dy_intra_std,
        n_bins=N_BINS, n_cycle=N_CYCLE, n_groups=n_groups, hr=hr, um_per_px_y=um_per_px_y,
        phase_max=phase_max, phase_min=phase_min, pulse_method=np.array(PULSE_METHOD),
        fold_method=np.array(FOLD_METHOD),
    )

    bins_rows = []
    for s in range(n_slots):
        bins_rows.append({
            "slug": slug, "cycle": s // N_BINS + 1, "bin": s % N_BINS, "slot": s,
            "n_frames": int(counts[s]),
            "thickness_fold_px": thickness_fold[s], "thickness_fold_fit_px": fit_fold[s],
            "thickness_seg_px": thickness_seg[s], "thickness_seg_fit_px": fit_seg[s],
            "thickness_std_px": thickness_std[s], "thickness_std_fit_px": fit_std[s],
            "dice": dice[s], "dx_ptp_px": dx_ptp[s], "dy_bulk_ptp_px": dy_bulk_ptp[s],
            "dy_intra_std_px": dy_intra_std[s], "ref_frame": int(ref_frame[s]),
        })

    def per_cycle_ptp(th):
        return [ptp_nan(th[g * N_BINS:(g + 1) * N_BINS]) for g in range(n_groups)]

    filled = counts > 0
    cond = {
        "slug": slug, "participant": m["participant"], "eye": m["eye"],
        "replicate": int(m["replicate"]), "status": "ok", "reason": "",
        "n_frames": T, "fs_Hz": float(row.get("fs_Hz", np.nan)), "hr_BPM": hr,
        "um_per_px_y": um_per_px_y, "n_cols": int(cols.size),
        "n_beats": n_beats, "n_groups": n_groups, "n_frames_kept": int(good_slot.sum()),
        "n_bins_filled": int(filled.sum()), "n_bins_empty": int((~filled).sum()),
        "frames_per_bin_min": int(counts[filled].min()) if filled.any() else 0,
        "frames_per_bin_median": float(np.median(counts[filled])) if filled.any() else np.nan,
        "frames_per_bin_max": int(counts.max()),
        "n_bins_lt5": int((filled & (counts < 5)).sum()),
        "phase_max": phase_max, "phase_min": phase_min,
        "dY_fold_raw_px_med": float(np.nanmedian(per_cycle_ptp(thickness_fold))),
        "dY_fold_fit_px_med": float(np.nanmedian(per_cycle_ptp(fit_fold))),
        "dY_seg_raw_px_med": float(np.nanmedian(per_cycle_ptp(thickness_seg))),
        "dY_seg_fit_px_med": float(np.nanmedian(per_cycle_ptp(fit_seg))),
        "dY_std_fit_px_med": float(np.nanmedian(per_cycle_ptp(fit_std))),
        "thickness_fold_px_mean": float(np.nanmean(thickness_fold)),
        "thickness_seg_px_mean": float(np.nanmean(thickness_seg)),
        "dice_median": float(np.nanmedian(dice)),
        "dx_ptp_px_median": float(np.nanmedian(dx_ptp)),
        "dx_ptp_px_max": float(np.nanmax(dx_ptp)),
        "dy_bulk_ptp_px_median": float(np.nanmedian(dy_bulk_ptp)),
        "seconds_registration": round(t_reg, 1), "seconds_resegmentation": round(t_seg, 1),
        "seconds_total": round(time.perf_counter() - t_all, 1),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    return cond, bins_rows


# --------------------------------------------------------------------------- #
# Lot
# --------------------------------------------------------------------------- #
def load_table(path):
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:  # table ecrite vide (premiere acquisition en echec)
        return pd.DataFrame()


def upsert(table, rows, slug):
    table = table[table["slug"] != slug] if not table.empty else table
    return pd.concat([table, pd.DataFrame(rows)], ignore_index=True).sort_values(
        [c for c in ("slug", "slot") if c in (table.columns if not table.empty else rows[0])])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="One-cycles des replicats, phase du pouls d'epaisseur, recalage intra-bin Y general.")
    parser.add_argument("--slug", action="append", help="ex. BELANGER_CHARLES_OD1 (repetable)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("Le recalage appris exige CUDA.")
        return 2
    if not CT_NPZ.exists():
        print(f"{CT_NPZ} introuvable : lancer d'abord compute_ct_pulse_traces.py.")
        return 2
    pulse = pd.read_csv(PULSE_DIR / "conditions.csv")
    pulse = pulse[pulse["status"] == "ok"].set_index("slug")
    ct_traces = np.load(CT_NPZ)
    slugs = sorted(pulse.index) if not args.slug else args.slug
    unknown = [s for s in slugs if s not in pulse.index]
    if unknown:
        print(f"Sans pouls exploitable : {', '.join(unknown)}")
        return 2
    no_ct = [s for s in slugs if f"{s}__ct_fir_um" not in ct_traces.files]
    if no_ct:
        print(f"Sans pouls d'epaisseur : {', '.join(no_ct)}")
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conditions, bins = load_table(CSV_CONDITIONS), load_table(CSV_BINS)
    done = set() if conditions.empty else set(conditions.loc[conditions["status"] == "ok", "slug"])
    todo = [s for s in slugs
            if args.overwrite or s not in done or not (CYCLES_DIR / f"{s}.npz").exists()]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(slugs)} acquisition(s), {len(todo)} a traiter -> {OUT_DIR}", flush=True)
    if not todo:
        return 0

    # dy_bulk_only=True : recalage Y GENERAL (un seul dy par frame), pas par A-scan.
    config = replace(REGISTRATION, fused_batch_size=FUSED_BATCH, filter_bad_columns=False,
                     reference_selection="motion_medoid", dy_bulk_only=True)
    seg_model = get_choroid_segmentation_model().to(DEVICE)
    reg_model = get_registration_model().to(DEVICE)
    reg_model.img_shape = MODEL_SHAPE
    cc_backend = rnm.patch_largest_cc()
    torch.backends.cudnn.benchmark = True
    params = {
        "pulse_variant": PULSE_VARIANT, "pulse_method": PULSE_METHOD,
        "pulse_source": "choroid thickness (segmentation), band-passed at HR +/- 20%",
        "ct_npz": str(CT_NPZ), "data_subdir": DATA_SUBDIR,
        "n_cycle": N_CYCLE, "n_bins": N_BINS, "fold_method": FOLD_METHOD,
        "phase_probe_bins": PHASE_PROBE_BINS, "edge_cycles": occ.EDGE_CYCLES,
        "col_frac": list(COL_FRAC), "model_shape": list(MODEL_SHAPE),
        "seg_repo": rnm.SEG_REPO, "seg_revision": rnm.SEG_REVISION,
        "seg_commit_sha": rnm.hub_sha(rnm.SEG_REPO, rnm.SEG_REVISION),
        "reg_repo": rnm.REG_REPO, "reg_revision": rnm.REG_REVISION,
        "reg_commit_sha": rnm.hub_sha(rnm.REG_REPO, rnm.REG_REVISION),
        "registration_config": asdict(config), "largest_cc_backend": cc_backend,
        "git": rnm.git_state(), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    (OUT_DIR / "params.json").write_text(json.dumps(params, indent=2, default=str),
                                         encoding="utf-8")
    print(f"modeles : {rnm.SEG_REPO}@{rnm.SEG_REVISION}, {rnm.REG_REPO}@{rnm.REG_REVISION} | "
          f"dy_bulk_only=True | composante connexe : {cc_backend}\n", flush=True)

    t_batch = time.perf_counter()
    n_ok = n_err = 0
    for i, slug in enumerate(todo, 1):
        t0 = time.perf_counter()
        try:
            cond, bin_rows = process(slug, pulse.loc[slug], seg_model, reg_model, config, ct_traces)
            bins = upsert(bins, bin_rows, slug)
            n_ok += 1
            print(f"[{i:>2}/{len(todo)}] {slug:28} ok | {cond['n_groups']:2d} one-cycles | "
                  f"frames/bin {cond['frames_per_bin_min']}-{cond['frames_per_bin_max']} | "
                  f"dx intra-bin ptp med {cond['dx_ptp_px_median']:.2f} px | "
                  f"dY harm. replie {cond['dY_fold_fit_px_med']:.2f} / resegmente "
                  f"{cond['dY_seg_fit_px_med']:.2f} / standard {cond['dY_std_fit_px_med']:.2f} px | "
                  f"Dice {cond['dice_median']:.3f} | {time.perf_counter() - t0:.0f} s", flush=True)
        except Exception as e:  # noqa: BLE001 - une acquisition ne doit pas tout arreter
            traceback.print_exc()
            n_err += 1
            cond = {"slug": slug, "status": "error",
                    "reason": f"{type(e).__name__}: {str(e)[:300]}",
                    "created": datetime.now().isoformat(timespec="seconds")}
            if not bins.empty:
                bins = bins[bins["slug"] != slug]
            print(f"[{i:>2}/{len(todo)}] {slug:28} ERREUR : {cond['reason']}", flush=True)
        conditions = upsert(conditions, [cond], slug)
        conditions.to_csv(CSV_CONDITIONS, index=False)
        bins.to_csv(CSV_BINS, index=False)
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{n_ok} ok, {n_err} en echec ({(time.perf_counter() - t_batch) / 60:.1f} min)")
    print(f"tables : {CSV_CONDITIONS}\n         {CSV_BINS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
