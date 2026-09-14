# -*- coding: utf-8 -*-
"""
compute_hr_bandpass.py

Frequence cardiaque des acquisitions de repetabilite, estimee SANS A PRIORI, sur
deux signaux independants passes au meme passe-bande FIXE (``BANDE_BPM``, 50-85
BPM par defaut ; ``OR_BANDE_BPM=lo,hi`` pour un autre essai).

Pourquoi un nouveau test
------------------------
La FC du lot de pouls est ANCREE sur ``hr_prior.csv`` (consensus par
participant, ``estimate_consensus_hr.py``). Tout ce qui en descend -- le pouls
d'intensite ``y_comb`` (combinaison optimisee a FC +/- 20 %), le FIR de
``compute_ct_pulse_traces.py`` (FC +/- 20 %), la DMD (bornee a FC +/- 20 %) --
depend donc de la valeur qu'on voudrait valider. Aucun de ces nombres ne peut
la confirmer sans tourner en rond.

Ici, AUCUNE FC n'entre dans le calcul :

  - INTENSITE : memes traces pixel et meme SVD rang 100 que le lot de pouls,
    mais la composante retenue est celle dont la puissance Lomb-Scargle est la
    plus CONCENTREE dans la bande -- pas la combinaison optimisee autour de
    l'a priori ;
  - EPAISSEUR : epaisseur choroidienne par frame lue sur les masques recales,
    meme definition que ``compute_ct_pulse_traces.py`` ;
  - les deux passent le MEME FIR a phase lineaire (``spatio_temporal_filter``,
    celui de la methode ``1_fir``) aux bornes FIXES de la bande, puis la FC est
    le pic Lomb-Scargle du signal filtre dans la bande, bords exclus.

``hr_prior.csv`` n'est lu que pour etre RECOPIE dans la table (colonne
``hr_prior_BPM``), comme reference de lecture pour les figures.

Entrees (variante de reference, celle de la page Strain repeatability)
---------------------------------------------------------------------
    E:/NASA_Rigidity/Reproducibility/SegmentationVariations/<variante>/
        pulse_from_data/conditions.csv          conditions ok + chemin brut
        registered_frames/<astro>/<moment>/<condition>/cube.mp4
        registered_masks/<astro>/<moment>/<condition>/mask.npz
        demons_strain/conditions.csv            um_per_px_y
    E:/SANSORI/Reproducibility/<SLUG>/RawImages/  horodatages (XML)

Sorties (sous ``<variante>/hr_bandpass_<lo>_<hi>/``, ex. ``hr_bandpass_50_85``)
------------------------------------------------------------------------------
    conditions.csv   1 ligne / acquisition
    traces.npz       ``<slug>__t``, ``__u_time``, ``__int_brut``, ``__int_filt``,
                     ``__ct_brut_um``, ``__ct_filt_um``, ``__core``,
                     ``__spec_int``, ``__spec_ct`` ; plus ``slugs`` et ``bpm_axis``

Interruptible : ``traces/<slug>.npz`` et la table sont reecrits apres chaque
acquisition ; ``OR_OVERWRITE=1`` pour tout refaire, ``OR_LIMIT=N`` pour un essai.

Lancer (kernel pyOR, depuis la racine du depot) :
    C:/Users/transformer/anaconda3/envs/pyOR/python.exe \\
        Reproducibility/compute_hr_bandpass.py
"""

from __future__ import annotations

import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ocularrigidity.data.io import load_mask
from ocularrigidity.motion.filters._1d import spatio_temporal_filter
from ocularrigidity.motion.projection._1d import project_into_separable_components
from ocularrigidity.motion.pulsation import (
    CardiacBand,
    PixelTraceConfig,
    PixelTraceSource,
)
from ocularrigidity.motion.pulsation.rate import lomb_scargle_power
from ocularrigidity.motion.video_timeline_aligner import VideoTimelineAligner
from ocularrigidity.scripts.one_cycle.astronauts import _prepared_registrator
from ocularrigidity.scripts.registration.astronauts import load_ordered_oct_series

# --------------------------------------------------------------------------- #
# Parametres
# --------------------------------------------------------------------------- #
SEGVAR_ROOT = Path(os.environ.get(
    "OR_SEGVAR_ROOT", "E:/NASA_Rigidity/Reproducibility/SegmentationVariations"))
MASK_VARIANT = os.environ.get("OR_VARIANT", "model1_scale_1.0_flatten_choroid_xcorr")
VARIANT_ROOT = SEGVAR_ROOT / MASK_VARIANT
HR_PRIOR_CSV = Path("E:/NASA_Rigidity/Reproducibility/hr_prior.csv")

OVERWRITE = bool(os.environ.get("OR_OVERWRITE"))
LIMIT = int(os.environ["OR_LIMIT"]) if os.environ.get("OR_LIMIT") else None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LA bande du test : fixe, identique pour toutes les acquisitions. Surchargeable
# (``OR_BANDE_BPM=30,100``) ; le dossier de sortie porte les bornes, si bien que
# deux essais de bande ne s'ecrasent jamais l'un l'autre.
#   30-100 : premier essai (2026-09-14) -- trop large, sous-harmoniques et derive
#   50-85  : bande resserree, adulte au repos
BANDE_BPM = tuple(float(v) for v in os.environ.get("OR_BANDE_BPM", "50,85").split(","))
OUTPUT_SUBDIR = f"hr_bandpass_{BANDE_BPM[0]:.0f}_{BANDE_BPM[1]:.0f}"

# Memes reglages d'entree que ``Astronauts/compute_pulse_from_data.py`` : la
# seule chose qui change est la facon de choisir la composante et la bande.
COL_FRAC = (0.125, 0.875)
ROW_FRAC = (0.0, 0.5)
BLOCK_SIZES = (1,)
RANK_SVD = 100
# Grille LARGE pour juger la concentration d'une composante : une fraction en
# bande ne veut rien dire si la grille s'arrete aux bornes de la bande.
DIAG_BPM_RANGE = (20.0, 240.0)
EDGE_FRAC = 0.2  # bords exclus du pic (transitoire du FIR)
UM_PER_PX_Y_DEFAUT = 3.9

OUT_DIR = VARIANT_ROOT / OUTPUT_SUBDIR
TRACES_DIR = OUT_DIR / "traces"
CSV_OUT = OUT_DIR / "conditions.csv"
NPZ_OUT = OUT_DIR / "traces.npz"

RE_SLUG = re.compile(r"^(?P<participant>.+)_(?P<eye>OD|OS)(?P<replicate>\d+)$")


# --------------------------------------------------------------------------- #
# Outils
# --------------------------------------------------------------------------- #
def raw_timestamps_us(raw_dir: Path) -> np.ndarray:
    """Horodatages bruts (us), MEME ORDRE que les frames recalees."""
    series = load_ordered_oct_series(raw_dir)
    return np.array(
        [int(round(s.acquisition_time.seconds_of_day * 1e6)) for s in series],
        dtype=np.int64,
    )


def find_raw_dir(condition_dir: Path) -> Path | None:
    for name in ("RawImages", "RawData"):
        d = condition_dir / name
        if d.is_dir():
            return d
    return None


def grille(duree_s: float, bpm_range) -> np.ndarray:
    """Frequences (Hz) a pas 1 / (5 x duree), comme ``build_ctx`` du lot de pouls."""
    df = 1.0 / (5.0 * duree_s)
    return np.arange(bpm_range[0] / 60.0, bpm_range[1] / 60.0 + df, df)


def spectre(tt, y, freqs) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 8:
        return np.zeros_like(freqs)
    return lomb_scargle_power(np.asarray(tt)[ok], y[ok] - y[ok].mean(), freqs)


def passe_bande(y_u: np.ndarray, fs: float) -> np.ndarray:
    """FIR a phase lineaire aux bornes FIXES, sur la grille uniforme."""
    nyq = 0.5 * fs
    lo = (BANDE_BPM[0] / 60.0) / nyq
    hi = min((BANDE_BPM[1] / 60.0) / nyq, 0.99)
    return spatio_temporal_filter(np.asarray(y_u, dtype=float)[:, None], 0.0,
                                  lo, hi, fs, None)[:, 0]


def pic(tt, y, freqs_bande):
    """(BPM du pic, part de la puissance en bande portee par le pic, spectre)."""
    P = spectre(tt, y, freqs_bande)
    k = int(np.argmax(P))
    return float(freqs_bande[k] * 60.0), float(P[k] / (P.sum() + 1e-12)), P


def signe_sur(y, ref, tt, f_hz) -> np.ndarray:
    """Polarite de ``y`` alignee sur ``ref`` par leurs phaseurs a ``f_hz``.

    Le signe d'un vecteur singulier est arbitraire. On l'aligne sur la trace
    moyenne de la ROI, en BANDE ETROITE a la frequence du pic -- une correlation
    large bande avec cette moyenne est de l'ordre du bruit (meme convention que
    ``make_sign_fixer`` du lot de pouls). Sans effet sur la FC ; utile seulement
    pour que les courbes affichees ne s'inversent pas d'une acquisition a l'autre.
    """
    c, s = np.cos(2 * np.pi * f_hz * tt), np.sin(2 * np.pi * f_hz * tt)
    ph = lambda v: ((v - v.mean()) @ c) + 1j * ((v - v.mean()) @ s)  # noqa: E731
    return -y if np.real(ph(y) * np.conj(ph(ref))) < 0 else y


# --------------------------------------------------------------------------- #
# Une acquisition
# --------------------------------------------------------------------------- #
def process(row, um_y: float, hr_prior: float) -> tuple[dict, dict]:
    slug = row["slug"]
    astro, moment, condition = row["astro"], row["moment"], row["condition"]
    frames_path = VARIANT_ROOT / "registered_frames" / astro / moment / condition / "cube.mp4"
    mask_path = VARIANT_ROOT / "registered_masks" / astro / moment / condition / "mask.npz"
    for p in (frames_path, mask_path):
        if not p.exists():
            raise FileNotFoundError(p)
    raw_dir = find_raw_dir(Path(row["path"]))
    if raw_dir is None:
        raise FileNotFoundError(f"RawImages absent : {row['path']}")

    reg = _prepared_registrator(frames_path, mask_path, DEVICE, verbose=False)
    ts_us = raw_timestamps_us(raw_dir)
    n_frames = reg.registered_frames.shape[0]
    if n_frames != ts_us.size:
        raise ValueError(f"frames ({n_frames}) != horodatages ({ts_us.size})")
    aligner = VideoTimelineAligner(reg, ts_us)
    t = np.asarray(aligner.timestamps_seconds, dtype=float)
    u = np.asarray(aligner.uniform_time, dtype=float)
    fs = float(aligner.fs)
    duree = float(t[-1] - t[0])

    m = int(EDGE_FRAC * u.size)
    core = np.zeros(u.size, dtype=bool)
    core[m:u.size - m] = True

    f_large = grille(duree, DIAG_BPM_RANGE)
    bpm_large = f_large * 60.0
    in_band = (bpm_large >= BANDE_BPM[0]) & (bpm_large <= BANDE_BPM[1])
    f_bande = grille(duree, BANDE_BPM)

    # --- Intensite : SVD, composante la plus concentree dans la bande ---------
    source = PixelTraceSource(
        reg, aligner,
        PixelTraceConfig(band=CardiacBand(bpm_range=BANDE_BPM), col_frac=COL_FRAC,
                         row_frac=ROW_FRAC, block_sizes=BLOCK_SIZES, verbose=False),
    )
    sig0 = source.raw_signal().astype(np.float64)
    sig0n = source.normalized_signal().astype(np.float64)
    X = sig0n - sig0n.mean(axis=0, keepdims=True)
    T, N = X.shape
    U, _ = project_into_separable_components(
        X, method="svd", n_components=min(RANK_SVD, T, N),
        normalize=False, random_state=0)
    fracs = np.array([
        (lambda P: P[in_band].sum() / (P.sum() + 1e-12))(spectre(t, U[:, i], f_large))
        for i in range(U.shape[1])
    ])
    k = int(np.argmax(fracs))
    int_brut = np.asarray(U[:, k], dtype=float)

    int_filt = passe_bande(np.interp(u, t, int_brut), fs)
    hr_int, pk_int, spec_int = pic(u[core], int_filt[core], f_bande)
    ref = (sig0 - sig0.mean(axis=0, keepdims=True)).mean(axis=1)
    int_brut = signe_sur(int_brut, ref, t, hr_int / 60.0)
    int_filt = passe_bande(np.interp(u, t, int_brut), fs)

    # --- Epaisseur : meme definition que compute_ct_pulse_traces.py -----------
    masks = np.asarray(load_mask(mask_path), dtype=bool)
    W = masks.shape[2]
    roi_all = masks.all(axis=0)
    if roi_all.sum() < 100:
        roi_all = masks.mean(axis=0) > 0.5
    c0, c1 = int(COL_FRAC[0] * W), int(COL_FRAC[1] * W)
    cols = np.flatnonzero(roi_all[:, c0:c1].any(axis=0)) + c0
    if cols.size == 0:
        raise ValueError("aucune colonne exploitable pour l'epaisseur")
    ct_brut = masks[:, :, cols].sum(axis=1).mean(axis=1).astype(np.float64) * um_y
    ct_filt = passe_bande(np.interp(u, t, ct_brut), fs)
    hr_ct, pk_ct, spec_ct = pic(u[core], ct_filt[core], f_bande)

    parts = RE_SLUG.match(slug)
    ligne = {
        "slug": slug,
        "participant": parts["participant"] if parts else moment,
        "eye": parts["eye"] if parts else "",
        "replicate": int(parts["replicate"]) if parts else -1,
        "n_frames": int(T), "n_pixels": int(N),
        "fs_Hz": fs, "duree_s": duree, "n_uniform": int(u.size),
        "bande_lo_BPM": BANDE_BPM[0], "bande_hi_BPM": BANDE_BPM[1],
        "svd_comp": k, "svd_frac_bande": float(fracs[k]),
        "hr_intensite_BPM": hr_int, "pic_frac_intensite": pk_int,
        "hr_epaisseur_BPM": hr_ct, "pic_frac_epaisseur": pk_ct,
        "ecart_int_ct_BPM": hr_int - hr_ct,
        "ct_moyenne_um": float(np.mean(ct_brut)),
        "hr_prior_BPM": hr_prior,
        "status": "ok",
    }
    payload = {
        "t": t.astype(np.float32), "u_time": u.astype(np.float32),
        "int_brut": int_brut.astype(np.float32), "int_filt": int_filt.astype(np.float32),
        "ct_brut_um": ct_brut.astype(np.float32), "ct_filt_um": ct_filt.astype(np.float32),
        "core": core, "bpm_bande": (f_bande * 60.0).astype(np.float32),
        "spec_int": spec_int.astype(np.float32), "spec_ct": spec_ct.astype(np.float32),
    }
    return ligne, payload


# --------------------------------------------------------------------------- #
# Lot
# --------------------------------------------------------------------------- #
def main() -> int:
    csv_pulse = VARIANT_ROOT / "pulse_from_data" / "conditions.csv"
    if not csv_pulse.exists():
        print(f"{csv_pulse} absent -- lancer d'abord run_batch.py pulse")
        return 2
    conditions = pd.read_csv(csv_pulse)
    conditions = conditions[conditions["status"] == "ok"].sort_values("slug")
    conditions = conditions.reset_index(drop=True)
    if LIMIT is not None:
        conditions = conditions.head(LIMIT)

    um = {}
    csv_strain = VARIANT_ROOT / "demons_strain" / "conditions.csv"
    if csv_strain.exists():
        d = pd.read_csv(csv_strain)
        um = dict(zip(d["slug"], pd.to_numeric(d["um_per_px_y"], errors="coerce")))
    prior = {}
    if HR_PRIOR_CSV.exists():
        d = pd.read_csv(HR_PRIOR_CSV)
        prior = dict(zip(d["slug"], pd.to_numeric(d["hr_BPM"], errors="coerce")))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    lignes = []
    if CSV_OUT.exists() and not OVERWRITE:
        lignes = [r for r in pd.read_csv(CSV_OUT).to_dict("records")
                  if r.get("status") == "ok" and (TRACES_DIR / f"{r['slug']}.npz").exists()]
    faites = {r["slug"] for r in lignes}
    print(f"sortie : {OUT_DIR}\nbande : {BANDE_BPM[0]:.0f}-{BANDE_BPM[1]:.0f} BPM, "
          f"device {DEVICE}\n{len(conditions)} acquisition(s), {len(faites)} deja faite(s)\n")

    t0_lot = time.perf_counter()
    n_ok = n_err = 0
    for i, row in conditions.iterrows():
        slug = row["slug"]
        if slug in faites:
            continue
        t0 = time.perf_counter()
        um_y = um.get(slug, np.nan)
        um_y = float(um_y) if np.isfinite(um_y) else UM_PER_PX_Y_DEFAUT
        try:
            ligne, payload = process(row, um_y, float(prior.get(slug, np.nan)))
        except Exception as exc:  # noqa: BLE001 - une acquisition ne doit pas tout arreter
            n_err += 1
            print(f"[{i + 1:>2d}/{len(conditions)}] {slug}  ECHEC : {exc}", flush=True)
            traceback.print_exc()
            lignes = [r for r in lignes if r["slug"] != slug]
            lignes.append({"slug": slug, "status": f"echec : {exc}"})
        else:
            n_ok += 1
            np.savez_compressed(TRACES_DIR / f"{slug}.npz", **payload)
            lignes = [r for r in lignes if r["slug"] != slug]
            lignes.append(ligne)
            print(f"[{i + 1:>2d}/{len(conditions)}] {slug}  "
                  f"intensite {ligne['hr_intensite_BPM']:5.1f} (comp {ligne['svd_comp']}), "
                  f"epaisseur {ligne['hr_epaisseur_BPM']:5.1f}, "
                  f"a priori {ligne['hr_prior_BPM']:5.1f} BPM  "
                  f"[{time.perf_counter() - t0:.0f} s]", flush=True)
        pd.DataFrame(lignes).sort_values("slug").to_csv(CSV_OUT, index=False)

    # Recueil unique pour les figures : une seule lecture de disque cote Quarto.
    ok = pd.DataFrame(lignes)
    ok = ok[ok["status"] == "ok"].sort_values("slug")
    recueil = {}
    for slug in ok["slug"]:
        with np.load(TRACES_DIR / f"{slug}.npz") as d:
            for key in d.files:
                recueil[f"{slug}__{key}"] = d[key]
    recueil["slugs"] = ok["slug"].to_numpy().astype(str)
    np.savez_compressed(NPZ_OUT, **recueil)

    print(f"\n{n_ok} ok, {n_err} en echec, {len(faites)} deja faites "
          f"({(time.perf_counter() - t0_lot) / 60:.1f} min)")
    print(f"  {CSV_OUT}  ({len(ok)} acquisitions ok)")
    print(f"  {NPZ_OUT}  ({NPZ_OUT.stat().st_size / 1e6:.1f} Mo)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
