# -*- coding: utf-8 -*-
"""
compute_svd_first_mode.py

Pouls d'INTENSITE des replicats tire d'UN SEUL mode SVD -- le premier qui pulse --
au lieu de la combinaison optimisee (``y_comb``) du lot de pouls.

Meme video, memes traces pixel, meme SVD de rang 100 que le lot
``Astronauts/compute_pulse_from_data.py`` (dont on reutilise les fonctions : ce
script l'importe et doit donc tourner avec le MEME environnement que le lot, voir
plus bas). Seul change le choix de la composante :

    premier mode qui pulse = le mode de plus petit indice (plus grande valeur
    singuliere) dont le pic Lomb-Scargle, sur la grille large 20-240 BPM, tombe
    dans FC +/- 20 % autour de l'ancre du lot.

Ce mode est ensuite traite comme ``y_comb`` pour la methode ``1_fir`` : signe fixe
par le phaseur a f0 de la trace moyenne (``make_sign_fixer``), interpolation sur
``u_time``, FIR ``bandpass_pulse``.

Controle : le spectre de chaque mode recalcule ici est compare a ``peak_comp``
sauvegarde par le lot (``pulse_from_data/traces/<slug>.npz``). S'ils different,
la SVD n'est pas la meme et l'acquisition est marquee ``status = svd_differente``.

Sorties, a cote du lot (``<OR_VARIANT>/pulse_from_data/``)
    svd_first_mode.csv          1 ligne / acquisition : indice du mode, sa
                                valeur singuliere relative, son pic, sa fraction
                                en bande, accord de peak_comp avec le lot
    svd_first_mode_traces.npz   ``<slug>__pulse_mode_fir`` (sur ``u_time``),
                                ``<slug>__mode_brut`` (sur ``t``)

Lancer comme le lot A (kernel pyOR, depuis la racine du depot) :
    OR_PATH_GENERAL=E:/SANSORI/Reproducibility
    OR_SEGVAR_ROOT=E:/NASA_Rigidity/Reproducibility/SegmentationVariations
    OR_VARIANT=cascade_v9_1536x1024_hr_mediane
    OR_DATA_SUBDIR=registered_cascade_v9_1536x1024
    OR_HR_PRIOR=E:/NASA_Rigidity/Reproducibility/hr_prior_mediane_bandpass_50_85_cascade_v9.csv
    python Reproducibility/compute_svd_first_mode.py
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Astronauts"))

import compute_pulse_from_data as cpd  # noqa: E402

OUT_CSV = cpd.OUT_DIR / "svd_first_mode.csv"
OUT_NPZ = cpd.OUT_DIR / "svd_first_mode_traces.npz"


def charger(path_condi: Path) -> dict:
    """Traces pixel du lot pour une acquisition : memes video, ROI, normalisation
    et ancre de FC que ``compute_pulse_from_data.py``. Partage avec
    ``compute_costs_pulse.py``.

    Renvoie ``slug``, ``t``, ``u_time``, ``fs``, ``hr``, ``X`` (T, N) centre,
    ``ctx`` et ``fix_sign`` (convention de signe du lot), ``peak_lot``."""
    astro, moment, condition = cpd.labels_of(path_condi)
    slug = cpd.slug_of(astro, moment, condition)
    lot = cpd.TRACES_DIR / f"{slug}.npz"
    if not lot.exists():
        raise FileNotFoundError(f"pas de trace du lot : {lot}")
    with np.load(lot) as z:
        peak_lot = np.asarray(z["peak_comp"], dtype=float)
        u_lot, hr = np.asarray(z["u_time"], dtype=float), float(z["hr"])

    data_dir = path_condi / "RawImages" / cpd.DATA_SUBDIR
    registrator = cpd._prepared_registrator(data_dir / "registered_video.mp4",
                                            data_dir / "mask.npz", cpd.DEVICE,
                                            verbose=False)
    ts_us = cpd.raw_timestamps_us(cpd.find_raw_dir(path_condi))
    aligner = cpd.VideoTimelineAligner(registrator, ts_us)
    t, u_time = aligner.timestamps_seconds, aligner.uniform_time
    if not np.allclose(u_time, u_lot, atol=1e-4):
        raise ValueError("u_time differente de celle du lot")

    band = cpd.CardiacBand(expected_bpm=hr, expected_bpm_band_frac=cpd.BAND_FRAC)
    source = cpd.PixelTraceSource(
        registrator, aligner,
        cpd.PixelTraceConfig(band=band, col_frac=cpd.COL_FRAC, row_frac=cpd.ROW_FRAC,
                             block_sizes=cpd.BLOCK_SIZES, verbose=False))
    sig0 = source.raw_signal().astype(np.float64)
    sig0n = source.normalized_signal().astype(np.float64)
    X = sig0n - sig0n.mean(axis=0, keepdims=True)

    ctx = cpd.build_ctx(t, u_time, aligner.fs, hr)
    ref_t = (sig0 - sig0.mean(axis=0, keepdims=True)).mean(axis=1)
    fix_sign = cpd.make_sign_fixer(ref_t, np.interp(u_time, t, ref_t), ctx)
    return {"slug": slug, "t": t, "u_time": u_time, "fs": aligner.fs, "hr": hr, "X": X,
            "ctx": ctx, "fix_sign": fix_sign, "peak_lot": peak_lot}


def process(path_condi: Path) -> tuple[dict, dict]:
    d = charger(path_condi)
    slug, t, hr, X, ctx = d["slug"], d["t"], d["hr"], d["X"], d["ctx"]
    peak_lot, fix_sign = d["peak_lot"], d["fix_sign"]
    T, N = X.shape
    U, _ = cpd.project_into_separable_components(
        X, method="svd", n_components=min(cpd.RANK_SVD, T, N),
        normalize=False, random_state=0)
    S = np.linalg.norm(U, axis=0)

    P = np.stack([cpd.spectrum(U[:, i], t, ctx) for i in range(U.shape[1])], axis=1)
    peak = ctx.bpm_axis[P.argmax(axis=0)]
    frac = P[ctx.in_band].sum(axis=0) / (P.sum(axis=0) + 1e-12)
    accord = float(np.mean(np.abs(peak - peak_lot) < 0.5)) if peak.size == peak_lot.size else 0.0

    en_bande = np.flatnonzero(np.abs(peak - hr) <= cpd.BAND_FRAC * hr)
    if en_bande.size == 0:
        raise ValueError("aucun mode ne pique dans FC +/- 20 %")
    k = int(en_bande[0])

    mode = fix_sign(U[:, k], t)
    pulse = cpd.bandpass_pulse(mode, t, ctx)

    ligne = {"slug": slug, "hr_BPM": hr, "mode": k, "n_modes_en_bande": int(en_bande.size),
             "modes_en_bande_5": " ".join(str(i) for i in en_bande[:5]),
             "s_rel": float(S[k] / S[0]), "var_frac": float(S[k] ** 2 / np.sum(S ** 2)),
             "pic_BPM": float(peak[k]), "frac_bande": float(frac[k]),
             "accord_peak_comp_lot": accord,
             "status": "ok" if accord > 0.95 else "svd_differente"}
    payload = {f"{slug}__pulse_mode_fir": pulse.astype(np.float32),
               f"{slug}__mode_brut": mode.astype(np.float32)}
    return ligne, payload


def main() -> None:
    conds = sorted(cpd.iter_conditions())
    if cpd.LIMIT:
        conds = conds[:cpd.LIMIT]
    print(f"{len(conds)} acquisition(s) -> {OUT_CSV}")
    lignes, payload = [], {}
    for i, p in enumerate(conds):
        t0 = time.perf_counter()
        try:
            ligne, pl = process(p)
            payload.update(pl)
        except Exception as exc:  # noqa: BLE001 - une acquisition ne doit pas tout arreter
            ligne = {"slug": p.name, "status": f"echec : {exc}"}
            traceback.print_exc()
        lignes.append(ligne)
        print(f"[{i + 1:>2d}/{len(conds)}] {ligne['slug']}  mode {ligne.get('mode', '-')}"
              f"  accord {ligne.get('accord_peak_comp_lot', float('nan')):.2f}"
              f"  {ligne['status']}  ({time.perf_counter() - t0:.0f} s)", flush=True)
        pd.DataFrame(lignes).to_csv(OUT_CSV, index=False)
        np.savez_compressed(OUT_NPZ, **payload)


if __name__ == "__main__":
    main()
