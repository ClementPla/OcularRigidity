"""
compute_pulse_from_data.py

Version "batch" des deux notebooks de comparaison de methodes d'extraction du
pouls choroidien, appliquee a TOUTES les conditions de TOUS les sujets :

  - ``notebook/compare_pipeline_pydmd_bopdmd.ipynb``
      pipeline actuel (SVD rang 100 + combinaison optimisee) vs BOPDMD (pyDMD)
      au rang 6, avec les trois initialisations B1/B2/B3.
  - ``notebook/compare_pulse_hankel_ssa.ipynb``
      cinq facons d'aller des vecteurs singuliers a la phase : temoin sans
      filtre, passe-bande FIR, SSA sur le pouls, M-SSA multicanal, SSA par
      canal puis combinaison.

Les deux analyses partagent EXACTEMENT la meme entree -- meme ROI, meme
normalisation, meme SVD rang 100, memes canaux cardiaques -- si bien qu'une
seule passe de chargement/decomposition sert aux deux. C'est tout l'interet de
les fusionner dans un seul script plutot que d'en faire deux.

Chaine par condition
--------------------
  1. traces pixel (ROI = 3/4 centraux x moitie superieure du masque, un pixel =
     une trace), normalisation par frame, moyenne temporelle retiree ;
  2. SVD rang 100 + periodogramme Lomb-Scargle de chaque vecteur singulier ;
  3. ``OptimizedSpectralCombination`` -> pouls de reference ET liste des canaux
     cardiaques (ses candidates) ;
  4. BOPDMD rang 6, trois initialisations, mode retenu, pouls modele et projete ;
  5. cinq pouls (0, 1, 2, 3a, 3b) + phase de Hilbert + HR instantanee ;
  6. diagnostic de legitimite de la phase (part de frequence instantanee
     negative, hors bande) pour chaque methode ;
  7. balayage de la fenetre SSA ``L``.

HR de reference : ``Data Files/visit_data.csv``. Quand elle manque, elle est
estimee sur la SVD (colonne ``hr_source``) au lieu de faire echouer la
condition -- meme choix que ``compute_phase_shift_pixel_svd.py``.

Arborescence lue (identique aux autres scripts pixel-SVD)
---------------------------------------------------------
    E:/NASA_Rigidity/SegmentationVariations/<variante>/
        registered_frames/<NN_id>/<...>_rigidity/<..._OD|OS...>/cube.mp4
        registered_masks/<NN_id>/<...>_rigidity/<..._OD|OS...>/mask.npz
    E:/SANSORI/<NN_id>/<...>_rigidity/<..._OD|OS...>/
        RawImages/ (ou RawData/)     <- .tif + export .xml, pour les horodatages
        Data Files/visit_data.csv    <- HR

Sorties (sous ``SEGVAR_ROOT/<variante>/pulse_from_data/``)
----------------------------------------------------------
  - ``conditions.csv``   1 ligne / condition -- metadonnees + resultats de tete
  - ``methods.csv``      1 ligne / (condition, methode) -- les 7 pouls compares
  - ``dmd_eigs.csv``     1 ligne / (condition, variante BOPDMD, paire propre)
  - ``ssa_sweep.csv``    1 ligne / (condition, approche SSA, L)
  - ``traces/<slug>.npz`` par condition -- ce qu'il faut pour tracer les figures

Les quatre tables sont reecrites apres CHAQUE condition et relues au demarrage :
le script est interruptible et reprend ou il s'est arrete (``OVERWRITE`` pour
tout refaire, ``LIMIT`` pour un essai sur les N premieres conditions).

Lancer (kernel pyOR, depuis la racine du depot) :
    C:/Users/transformer/anaconda3/envs/pyOR/python.exe \
        Astronauts/compute_pulse_from_data.py
"""

from __future__ import annotations

import csv
import time
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import hilbert, medfilt

from pydmd import BOPDMD

from ocularrigidity.motion.filters._1d import spatio_temporal_filter
from ocularrigidity.motion.projection._1d import project_into_separable_components
from ocularrigidity.motion.pulsation import (
    CardiacBand,
    PixelTraceConfig,
    PixelTraceSource,
)
from ocularrigidity.motion.pulsation.phase import (
    OptimizedSpectralCombination,
    SpectralCombinationConfig,
)
from ocularrigidity.motion.pulsation.rate import (
    LombScargleConfig,
    LombScargleRateEstimator,
    lomb_scargle_power,
)
from ocularrigidity.motion.pulsation.traces import Traces
from ocularrigidity.motion.video_timeline_aligner import VideoTimelineAligner
from ocularrigidity.scripts.one_cycle.astronauts import _prepared_registrator
from ocularrigidity.scripts.registration.astronauts import load_ordered_oct_series

# --------------------------------------------------------------------------- #
# Parametres
# --------------------------------------------------------------------------- #
PATH_GENERAL = Path("E:/SANSORI")
SEGVAR_ROOT = Path("E:/NASA_Rigidity/SegmentationVariations")
MASK_VARIANT = "model1_scale_1.0"
FRAMES_SUBDIR = "registered_frames"
MASKS_SUBDIR = "registered_masks"
OUTPUT_SUBDIR = "pulse_from_data"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OVERWRITE = False  # True = retraiter les conditions deja presentes dans les CSV
LIMIT = None  # int = ne traiter que les N premieres conditions (essai)
SAVE_TRACES = True  # .npz par condition, consomme par le generateur de figures

# --- Entree : la ROI des deux notebooks -------------------------------------
# 3/4 CENTRAUX des A-scans (les bords lateraux sont ceux ou le recalage est le
# moins fiable) et MOITIE SUPERIEURE de l'epaisseur locale (ligne 0 = haut de
# frame, donc moitie interne, cote Bruch/choriocapillaire).
COL_FRAC = (0.125, 0.875)
ROW_FRAC = (0.0, 0.5)
BLOCK_SIZES = (1,)  # un pixel = une trace, pas de multiresolution

RANK_SVD = 100  # rang du pipeline
RANK_DMD = 6  # rang de BOPDMD

# --- Bande cardiaque ---------------------------------------------------------
DIAG_BPM_RANGE = (20.0, 240.0)  # grille LARGE, pour les periodogrammes
BAND_FRAC = 0.2  # bande de SCORE : HR +/- 20 %

# --- Combinaison optimisee ---------------------------------------------------
OPT_MIN_CONCENTRATION = 0.15
OPT_MAX_CANDIDATES = 12
OPT_N_RESTARTS = 8
OPT_MAXITER = 200
OPT_WINDOW_N_CYCLES = 6
OPT_N_WINDOWS = 5
# Paliers de repli si aucune candidate ne passe le seuil nominal. Le palier
# effectivement utilise est reporte (``min_conc_used``) : une condition qui a
# du descendre est une condition a regarder de pres.
MIN_CONC_LADDER = (0.15, 0.10, 0.05, 0.0)

# --- SSA ---------------------------------------------------------------------
SSA_CYCLES = 3.0  # fenetre de plongement, en cycles cardiaques
SSA_N_COMP = 20  # composantes SSA examinees pour le regroupement
SSA_CYCLES_SWEEP = (1.0, 2.0, 3.0, 5.0, 8.0)

# --- Phase -------------------------------------------------------------------
EDGE_FRAC = 0.2  # bords exclus des statistiques (transitoire du FIR)

# --- Sorties -----------------------------------------------------------------
OUT_DIR = SEGVAR_ROOT / MASK_VARIANT / OUTPUT_SUBDIR
TRACES_DIR = OUT_DIR / "traces"
CSV_CONDITIONS = OUT_DIR / "conditions.csv"
CSV_METHODS = OUT_DIR / "methods.csv"
CSV_DMD = OUT_DIR / "dmd_eigs.csv"
CSV_SWEEP = OUT_DIR / "ssa_sweep.csv"

METHODS = ("0_sans_filtre", "1_fir", "2_ssa", "3a_mssa", "3b_ssa_canal",
           "dmd_modele", "dmd_projete")


# --------------------------------------------------------------------------- #
# Resolution des chemins
# --------------------------------------------------------------------------- #
def find_raw_dir(condition_dir: Path) -> Path | None:
    for name in ("RawImages", "RawData"):
        d = condition_dir / name
        if d.is_dir():
            return d
    return None


def raw_timestamps_us(raw_dir: Path) -> np.ndarray:
    """Horodatages bruts (us), MEME ORDRE que les frames de SegmentationVariations."""
    series = load_ordered_oct_series(raw_dir)
    return np.array(
        [int(round(s.acquisition_time.seconds_of_day * 1e6)) for s in series],
        dtype=np.int64,
    )


def read_hr(path_condi: Path) -> float:
    path_heartbeat = path_condi / "Data Files" / "visit_data.csv"
    if not path_heartbeat.exists():
        return float("nan")
    df = pd.read_csv(path_heartbeat, quoting=csv.QUOTE_NONE)
    if "HR" not in df.columns:
        return float("nan")
    return float(np.nanmean(pd.to_numeric(df["HR"], errors="coerce")))


def iter_conditions():
    """Toutes les conditions ``E:/SANSORI/<astro>/<...rigidity>/<condition>``."""
    for path_astro in sorted(PATH_GENERAL.iterdir()):
        if not path_astro.is_dir():
            continue
        for path_moment in sorted(path_astro.iterdir()):
            if not path_moment.is_dir() or not path_moment.match("*rigidity"):
                continue
            for path_condi in sorted(path_moment.iterdir()):
                if path_condi.is_dir():
                    yield path_condi


def slug_of(astro: str, moment: str, condition: str) -> str:
    return f"{astro}__{condition}"


# --------------------------------------------------------------------------- #
# Contexte spectral
# --------------------------------------------------------------------------- #
@dataclass
class Ctx:
    """Ce qui, dans les notebooks, vivait en variable globale de cellule."""

    t: np.ndarray  # (T,) horodatages reels, NON uniformes
    u_time: np.ndarray  # (Tu,) grille uniforme -- obligatoire pour SSA/Hilbert
    fs: float
    hr: float
    f0: float
    freqs: np.ndarray
    bpm_axis: np.ndarray
    in_band: np.ndarray
    band: CardiacBand
    low_cut: float
    high_cut: float
    core: np.ndarray  # (Tu,) bool : hors transitoire de bord


def build_ctx(t, u_time, fs, hr) -> Ctx:
    f0 = hr / 60.0
    df = 1.0 / (5.0 * (t[-1] - t[0]))
    freqs = np.arange(DIAG_BPM_RANGE[0] / 60.0, DIAG_BPM_RANGE[1] / 60.0 + df, df)
    bpm_axis = freqs * 60.0
    in_band = np.abs(bpm_axis - hr) <= BAND_FRAC * hr
    if not in_band.any():
        # HR hors grille : on garde le bin le plus proche plutot que de renvoyer
        # une fraction en bande identiquement nulle.
        in_band = np.zeros_like(in_band)
        in_band[int(np.argmin(np.abs(bpm_axis - hr)))] = True

    band = CardiacBand(expected_bpm=hr, expected_bpm_band_frac=BAND_FRAC)
    nyq = 0.5 * fs
    lo_bpm, hi_bpm = band.effective_bpm_range
    low = (lo_bpm / 60.0) / nyq
    high = min((hi_bpm / 60.0) / nyq, 0.99)

    m = int(EDGE_FRAC * u_time.size)
    core = np.zeros(u_time.size, dtype=bool)
    core[m:u_time.size - m] = True

    return Ctx(t=t, u_time=u_time, fs=fs, hr=hr, f0=f0, freqs=freqs,
               bpm_axis=bpm_axis, in_band=in_band, band=band,
               low_cut=low, high_cut=high, core=core)


def spectrum(y, tt, ctx: Ctx) -> np.ndarray:
    """Lomb-Scargle d'une trace sur SA propre base de temps.

    La meme grille ``freqs`` sert pour les traces vivant sur ``t`` (non
    uniforme) et sur ``u_time`` : les puissances restent comparables.
    """
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 8:
        return np.zeros_like(ctx.freqs)
    return lomb_scargle_power(np.asarray(tt)[ok], y[ok] - y[ok].mean(), ctx.freqs)


def band_score(y, tt, ctx: Ctx) -> tuple[float, float]:
    """(fraction de puissance en bande, BPM du pic)."""
    P = spectrum(y, tt, ctx)
    return (float(P[ctx.in_band].sum() / (P.sum() + 1e-12)),
            float(ctx.bpm_axis[P.argmax()]))


# --------------------------------------------------------------------------- #
# Signe : phaseur a f0
# --------------------------------------------------------------------------- #
def phasor_at_f0(y, tt, ctx: Ctx) -> complex:
    y = np.asarray(y, dtype=float)
    y = y - y.mean()
    tt = np.asarray(tt, dtype=float)
    return (y @ np.cos(2 * np.pi * ctx.f0 * tt)) + 1j * (
        y @ np.sin(2 * np.pi * ctx.f0 * tt))


def make_sign_fixer(ref_t, ref_u, ctx: Ctx):
    """Fixe le signe d'un pouls en BANDE ETROITE, sur le phaseur a f0.

    La correlation large bande avec la moyenne des traces brutes est de l'ordre
    de 0.05 -- du bruit -- et ne suffit pas a trancher une polarite. Meme
    convention que ``OptimizedSpectralCombination._initial_weights``.
    """
    ref = {len(ctx.t): phasor_at_f0(ref_t, ctx.t, ctx),
           len(ctx.u_time): phasor_at_f0(ref_u, ctx.u_time, ctx)}

    def fix_sign(y, tt):
        aligned = float(np.real(phasor_at_f0(y, tt, ctx) * np.conj(ref[len(tt)])))
        return (-1.0 if aligned < 0 else 1.0) * np.asarray(y, dtype=float)

    return fix_sign


# --------------------------------------------------------------------------- #
# Passe-bande et phase
# --------------------------------------------------------------------------- #
def bandpass_pulse(y, tt, ctx: Ctx) -> np.ndarray:
    """Passe-bande cardiaque : interpolation sur ``u_time`` puis FIR, comme
    ``AbstractUniformTraceSource.filtered_signal``."""
    yu = np.interp(ctx.u_time, tt, np.asarray(y, dtype=float))
    return spatio_temporal_filter(yu[:, None], 0.0, ctx.low_cut, ctx.high_cut,
                                  ctx.fs, None)[:, 0]


def analytic_phase(y, ctx: Ctx) -> tuple[np.ndarray, np.ndarray]:
    """(phase repliee, BPM instantane filtre par mediane sur un cycle)."""
    y = np.asarray(y, dtype=float)
    an = hilbert(y - y.mean())
    ph = np.unwrap(np.angle(an))
    f_inst = np.gradient(ph, ctx.u_time) / (2 * np.pi)
    win = int(round(ctx.fs / ctx.f0)) | 1
    if 1 < win < f_inst.size:
        f_inst = medfilt(f_inst, win)
    return np.mod(ph, 2 * np.pi), f_inst * 60.0


def inst_freq_raw(y, ctx: Ctx) -> np.ndarray:
    """Frequence instantanee BRUTE (BPM), sans filtre median.

    C'est elle qui revele les violations de la condition de bande etroite
    (Bedrosian/Nuttall) : une frequence negative signifie que la phase RECULE,
    donc que la phase de Hilbert n'est pas interpretable sur ce signal.
    """
    y = np.asarray(y, dtype=float)
    ph = np.unwrap(np.angle(hilbert(y - y.mean())))
    return np.gradient(ph, ctx.u_time) / (2 * np.pi) * 60.0


# --------------------------------------------------------------------------- #
# SSA / Hankel
# --------------------------------------------------------------------------- #
def hankel_matrix(x, L) -> np.ndarray:
    """Matrice trajectoire (L, K), K = N - L + 1 : H[i, j] = x[i + j]."""
    x = np.asarray(x, dtype=float)
    K = x.size - L + 1
    if K < L:
        raise ValueError(f"L = {L} trop grand pour N = {x.size} (K = {K} < L)")
    return np.lib.stride_tricks.sliding_window_view(x, K)


def diagonal_average(M) -> np.ndarray:
    """Retour Hankel -> 1-D : moyenne de chaque anti-diagonale."""
    L, K = M.shape
    out = np.zeros(L + K - 1)
    cnt = np.zeros(L + K - 1)
    for i in range(L):
        out[i:i + K] += M[i]
        cnt[i:i + K] += 1.0
    return out / cnt


def ssa_decompose(x, L, n_comp):
    H = hankel_matrix(x, L)
    Uh, sh, Vth = np.linalg.svd(H, full_matrices=False)
    n = int(min(n_comp, sh.size))
    recon = np.stack([
        diagonal_average(sh[k] * np.outer(Uh[:, k], Vth[k])) for k in range(n)
    ])
    return sh, recon


def _cardiac_keep(recon_1d_list, ctx: Ctx, tol_bpm):
    """Indices des composantes dont le pic tombe a moins de ``tol_bpm`` de la HR.

    A defaut, la paire de plus forte puissance en bande : mieux vaut un groupe
    explicitement degrade qu'une exception au milieu d'un batch.
    """
    peaks, fracs = [], []
    for r in recon_1d_list:
        fr, pk = band_score(r, ctx.u_time, ctx)
        peaks.append(pk)
        fracs.append(fr)
    peaks = np.asarray(peaks)
    fracs = np.asarray(fracs)
    keep = np.where(np.abs(peaks - ctx.hr) <= tol_bpm)[0]
    if keep.size == 0:
        keep = np.argsort(fracs)[::-1][:2]
    return np.sort(keep), peaks, fracs


def ssa_denoise(x, L, ctx: Ctx, n_comp=SSA_N_COMP):
    """Somme des composantes SSA cardiaques."""
    tol = BAND_FRAC * ctx.hr
    sh, recon = ssa_decompose(x, L, n_comp)
    keep, peaks, fracs = _cardiac_keep(list(recon), ctx, tol)
    return recon[keep].sum(axis=0), {"sv": sh, "keep": keep, "peaks": peaks,
                                     "fracs": fracs}


def mssa_denoise(chan, L, ctx: Ctx, n_comp=SSA_N_COMP):
    """M-SSA : une seule matrice trajectoire empilant les canaux.

    Les Hankel des canaux sont empiles VERTICALEMENT, si bien que la SVD voit
    un sous-espace temporel PARTAGE -- l'hypothese physique du probleme : un
    meme pouls vu par plusieurs canaux, a des amplitudes et des retards
    differents.
    """
    tol = BAND_FRAC * ctx.hr
    n_ch, N = chan.shape
    H = np.vstack([hankel_matrix(c, L) for c in chan])
    Uh, sh, Vth = np.linalg.svd(H, full_matrices=False)
    n = int(min(n_comp, sh.size))

    recon = np.empty((n, n_ch, N))
    for k in range(n):
        Mk = sh[k] * np.outer(Uh[:, k], Vth[k])
        for c in range(n_ch):
            recon[k, c] = diagonal_average(Mk[c * L:(c + 1) * L])

    # Regroupement sur la MOYENNE des canaux : une composante M-SSA est
    # cardiaque ou ne l'est pas, c'est une propriete du mode partage.
    keep, peaks, fracs = _cardiac_keep([recon[k].mean(axis=0) for k in range(n)],
                                       ctx, tol)
    return recon[keep].sum(axis=0), {"sv": sh, "keep": keep, "peaks": peaks,
                                     "fracs": fracs, "H_shape": H.shape}


def mssa_pulse(chan, L, ctx: Ctx):
    """M-SSA puis 1re composante principale des canaux debruites.

    Les canaux debruites sont, par construction, quasi de rang 2 : leur
    premiere PC EST l'oscillation partagee.
    """
    R, diag = mssa_denoise(chan, L, ctx)
    _, Sc, Vtc = np.linalg.svd(R - R.mean(axis=1, keepdims=True),
                               full_matrices=False)
    diag["pc1_var"] = float(Sc[0] ** 2 / (Sc ** 2).sum())
    return Vtc[0] * Sc[0], diag


# --------------------------------------------------------------------------- #
# BOPDMD
# --------------------------------------------------------------------------- #
def harmonic_init(rank, f0_hz, damping=-0.05) -> np.ndarray:
    """Valeurs propres de depart : f0 et ses harmoniques, en paires conjuguees."""
    pairs = []
    for k in range(1, rank // 2 + 1):
        w = 2j * np.pi * f0_hz * k
        pairs += [damping + w, damping - w]
    if rank % 2:
        pairs.append(damping + 0j)
    return np.array(pairs[:rank])


def fit_bopdmd(Xd, t, **kwargs):
    """``eigs`` est en TEMPS CONTINU : la frequence est Im(lambda)/2pi, PAS
    log(lambda).imag/(2 pi dt) comme pour une DMD a temps discret."""
    kwargs.setdefault("eig_constraints", {"conjugate_pairs"})
    kwargs.setdefault("varpro_opts_dict", {"maxiter": 200})
    bop = BOPDMD(svd_rank=RANK_DMD, num_trials=0, **kwargs)
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bop.fit(Xd, t)
        failed = any("converge" in str(c.message) for c in caught)
    return {
        "dmd": bop,
        "f_bpm": bop.eigs.imag / (2 * np.pi) * 60.0,
        "growth": bop.eigs.real,
        "elapsed": time.perf_counter() - t0,
        "converged": not failed,
    }


# --------------------------------------------------------------------------- #
# Combinaison optimisee
# --------------------------------------------------------------------------- #
def make_rate_estimator(hr):
    """Bande LARGE : la grille ``freqs`` du rate estimator sert aussi de grille
    hors-bande dans l'objectif de l'optimiseur. Avec la bande etroite, le
    rapport a maximiser serait degenere."""
    return LombScargleRateEstimator(
        LombScargleConfig(band=CardiacBand(bpm_range=DIAG_BPM_RANGE),
                          verbose=False),
        override_bpm=float(hr),
    )


def optimized_pulse(values, mixing, tt, rate_est, ctx: Ctx,
                    ladder=MIN_CONC_LADDER):
    """Combinaison optimisee. Renvoie (pouls, resultat, seuil utilise, duree)."""
    traces = Traces(
        values=np.asarray(values, dtype=np.float64),
        uniform_time=np.asarray(tt, dtype=float),
        kept_mask=np.ones(len(tt), dtype=bool),
        gap_mask=np.zeros(len(tt), dtype=bool),
        timestamps_seconds=np.asarray(tt, dtype=float),
        mixing=None if mixing is None else np.asarray(mixing, dtype=np.float64),
    )
    rate = rate_est.estimate(traces)
    last_exc = None
    for min_conc in ladder:
        agg = OptimizedSpectralCombination(SpectralCombinationConfig(
            accept_tol_bpm=BAND_FRAC * ctx.hr,
            min_concentration=min_conc,
            max_candidates=min(OPT_MAX_CANDIDATES, np.shape(values)[1]),
            n_restarts=OPT_N_RESTARTS, maxiter=OPT_MAXITER, random_state=0,
            window_n_cycles=OPT_WINDOW_N_CYCLES, n_windows=OPT_N_WINDOWS,
        ))
        t0 = time.perf_counter()
        try:
            y = agg.aggregate(traces, rate)
        except ValueError as exc:
            last_exc = exc
            continue
        return y, agg.last_result, min_conc, time.perf_counter() - t0
    raise RuntimeError(f"aucune combinaison possible : {last_exc}")


# --------------------------------------------------------------------------- #
# Traitement d'une condition
# --------------------------------------------------------------------------- #
def process_condition(path_condi: Path) -> dict:
    """Renvoie {'condition': row, 'methods': [...], 'dmd': [...], 'sweep': [...]}."""
    astro = path_condi.parent.parent.name
    moment = path_condi.parent.name
    condition = path_condi.name
    slug = slug_of(astro, moment, condition)

    variant_root = SEGVAR_ROOT / MASK_VARIANT
    frames_path = variant_root / FRAMES_SUBDIR / astro / moment / condition / "cube.mp4"
    mask_path = variant_root / MASKS_SUBDIR / astro / moment / condition / "mask.npz"
    for p in (frames_path, mask_path):
        if not p.exists():
            raise FileNotFoundError(p)
    raw_dir = find_raw_dir(path_condi)
    if raw_dir is None:
        raise FileNotFoundError(f"RawImages/RawData absent : {path_condi}")

    # --- 1. Chargement --------------------------------------------------------
    t_load = time.perf_counter()
    registrator = _prepared_registrator(frames_path, mask_path, DEVICE, verbose=False)
    ts_us = raw_timestamps_us(raw_dir)
    n_frames = registrator.registered_frames.shape[0]
    if n_frames != ts_us.size:
        raise ValueError(f"frames ({n_frames}) != horodatages ({ts_us.size})")
    aligner = VideoTimelineAligner(registrator, ts_us)
    t = aligner.timestamps_seconds
    u_time = aligner.uniform_time
    t_load = time.perf_counter() - t_load

    hr_measured = read_hr(path_condi)
    hr_source = "visit_data"
    hr = hr_measured

    # --- 2. Traces + SVD ------------------------------------------------------
    # La HR n'entre pas dans la construction des traces (seule `band` en depend,
    # et elle ne sert qu'au passe-bande) : on peut donc decomposer AVANT de
    # trancher la HR, ce qui permet de l'estimer sur la SVD si elle manque.
    band_tmp = (CardiacBand(expected_bpm=hr, expected_bpm_band_frac=BAND_FRAC)
                if np.isfinite(hr) else CardiacBand(bpm_range=DIAG_BPM_RANGE))
    source = PixelTraceSource(
        registrator, aligner,
        PixelTraceConfig(band=band_tmp, col_frac=COL_FRAC, row_frac=ROW_FRAC,
                         block_sizes=BLOCK_SIZES, verbose=False),
    )
    sig0 = source.raw_signal().astype(np.float64)
    sig0n = source.normalized_signal().astype(np.float64)
    X = sig0n - sig0n.mean(axis=0, keepdims=True)
    roi = source.base_roi
    T, N = X.shape

    t0 = time.perf_counter()
    U, V = project_into_separable_components(
        X, method="svd", n_components=min(RANK_SVD, T, N),
        normalize=False, random_state=0)
    S = np.linalg.norm(U, axis=0)
    t_svd = time.perf_counter() - t0

    if not np.isfinite(hr):
        # HR absente : on la lit sur la SVD plutot que d'abandonner la condition.
        probe = Traces(values=np.asarray(U, dtype=np.float64), uniform_time=t,
                       kept_mask=np.ones(t.size, bool),
                       gap_mask=np.zeros(t.size, bool),
                       timestamps_seconds=t, mixing=None)
        est = LombScargleRateEstimator(
            LombScargleConfig(band=CardiacBand(bpm_range=(30.0, 180.0)),
                              verbose=False)).estimate(probe)
        hr = float(est.freq * 60.0)
        hr_source = "svd"

    ctx = build_ctx(t, u_time, aligner.fs, hr)
    t0 = time.perf_counter()
    P_svd = np.stack([spectrum(U[:, i], t, ctx) for i in range(U.shape[1])], axis=1)
    t_per = time.perf_counter() - t0
    peak_comp = ctx.bpm_axis[P_svd.argmax(axis=0)]
    frac_comp = P_svd[ctx.in_band].sum(axis=0) / (P_svd.sum(axis=0) + 1e-12)

    # --- 3. Combinaison optimisee : le pouls ET les canaux --------------------
    rate_est = make_rate_estimator(hr)
    y_comb, res_comb, min_conc_used, t_opt = optimized_pulse(
        U, V, t, rate_est, ctx)
    channels = np.asarray(res_comb.selected_indices, dtype=int)
    n_ch = channels.size

    ref_t = (sig0 - sig0.mean(axis=0, keepdims=True)).mean(axis=1)
    ref_u = np.interp(u_time, t, ref_t)
    fix_sign = make_sign_fixer(ref_t, ref_u, ctx)
    y_comb = fix_sign(y_comb, t)
    y_comb_u = np.interp(u_time, t, y_comb)

    # Cout de l'interpolation exigee par SSA et Hilbert : le pipeline actuel
    # l'evite jusqu'au FIR, les methodes SSA la subissent d'entree.
    corr_interp = float(np.corrcoef(np.interp(t, u_time, y_comb_u), y_comb)[0, 1])

    # --- 4. BOPDMD ------------------------------------------------------------
    Xd = X.T  # convention pyDMD : (espace, temps)
    init_alpha = harmonic_init(RANK_DMD, ctx.f0)
    variants = {}
    dmd_rows = []
    for key, kw in (
        ("B1", {}),
        ("B2", {"init_alpha": init_alpha}),
        ("B3", {"init_alpha": init_alpha,
                "eig_constraints": {"conjugate_pairs", "imag"}}),
    ):
        try:
            v = fit_bopdmd(Xd, t, **kw)
        except Exception as exc:  # noqa: BLE001 - une variante peut echouer seule
            variants[key] = None
            dmd_rows.append({"slug": slug, "variante": key, "status": f"echec : {exc}"})
            continue
        variants[key] = v
        for i in np.argsort(-np.abs(v["dmd"].amplitudes)):
            if v["f_bpm"][i] < 0:
                continue  # une ligne par paire conjuguee
            dmd_rows.append({
                "slug": slug, "variante": key, "status": "ok",
                "f_BPM": float(v["f_bpm"][i]),
                "croissance_1s": float(v["growth"][i]),
                "amplitude": float(abs(v["dmd"].amplitudes[i])),
                "ecart_HR_BPM": float(v["f_bpm"][i] - hr),
                "ratio_HR": float(v["f_bpm"][i] / hr),
                "converged": bool(v["converged"]),
                "t_fit_s": float(v["elapsed"]),
            })

    def closest_to_hr(v):
        if v is None:
            return np.inf
        pos = v["f_bpm"] > 0
        return float(np.min(np.abs(v["f_bpm"][pos] - hr))) if pos.any() else np.inf

    retained = min(variants, key=lambda k: closest_to_hr(variants[k]))
    V_DMD = variants[retained]
    y_dmd_model = y_dmd_proj = None
    hr_dmd = np.nan
    dmd_status = "ok"
    if V_DMD is None or not np.isfinite(closest_to_hr(V_DMD)):
        dmd_status = "aucune valeur propre exploitable"
    else:
        dmd = V_DMD["dmd"]
        f_bpm = V_DMD["f_bpm"]
        cand = np.where(f_bpm > 0)[0]
        k_dmd = int(cand[np.argmin(np.abs(f_bpm[cand] - hr))])
        hr_dmd = float(f_bpm[k_dmd])
        coords = np.linalg.pinv(dmd.modes) @ Xd
        y_dmd_model = fix_sign(
            2 * np.real(dmd.amplitudes[k_dmd] * np.exp(dmd.eigs[k_dmd] * t)), t)
        y_dmd_proj = fix_sign(2 * np.real(coords[k_dmd]), t)

    # --- 5. Les cinq pouls du second notebook ---------------------------------
    L_ssa = int(round(SSA_CYCLES * ctx.fs / ctx.f0)) | 1
    if u_time.size - L_ssa + 1 < L_ssa:
        raise ValueError(f"enregistrement trop court pour L = {L_ssa} "
                         f"(Tu = {u_time.size})")

    costs = {}
    t0 = time.perf_counter()
    y0 = fix_sign(y_comb_u, u_time)
    costs["0_sans_filtre"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    y1 = fix_sign(bandpass_pulse(y_comb, t, ctx), u_time)
    costs["1_fir"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    y2_raw, d2 = ssa_denoise(y_comb_u, L_ssa, ctx)
    y2 = fix_sign(y2_raw, u_time)
    costs["2_ssa"] = time.perf_counter() - t0

    chan_u = np.stack([np.interp(u_time, t, U[:, i]) for i in channels])
    t0 = time.perf_counter()
    y3a_raw, d3a = mssa_pulse(chan_u, L_ssa, ctx)
    y3a = fix_sign(y3a_raw, u_time)
    costs["3a_mssa"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    chan_den = np.stack([ssa_denoise(c, L_ssa, ctx)[0] for c in chan_u])
    y3b_raw, res3b, _, _ = optimized_pulse(chan_den.T, None, u_time, rate_est, ctx)
    y3b = fix_sign(y3b_raw, u_time)
    costs["3b_ssa_canal"] = time.perf_counter() - t0

    pulses = {"0_sans_filtre": y0, "1_fir": y1, "2_ssa": y2,
              "3a_mssa": y3a, "3b_ssa_canal": y3b}
    if y_dmd_model is not None:
        # Les pouls DMD vivent sur `t` : on les amene sur `u_time` pour que
        # toutes les phases soient comparables echantillon par echantillon.
        pulses["dmd_modele"] = np.interp(u_time, t, y_dmd_model)
        pulses["dmd_projete"] = np.interp(u_time, t, y_dmd_proj)
        costs["dmd_modele"] = costs["dmd_projete"] = float(V_DMD["elapsed"])

    # --- 6. Phase, HR instantanee, legitimite ---------------------------------
    lo_b, hi_b = ctx.band.effective_bpm_range
    method_rows = []
    phases, bpms = {}, {}
    for name, y in pulses.items():
        ph, bpm = analytic_phase(y, ctx)
        phases[name], bpms[name] = ph, bpm
        fr, pk = band_score(y, u_time, ctx)
        b = bpm[ctx.core]
        raw = inst_freq_raw(y, ctx)[ctx.core]
        method_rows.append({
            "slug": slug, "astro": astro, "condition": condition, "methode": name,
            "pic_LS_BPM": pk, "en_bande_frac": fr,
            "HR_med_BPM": float(np.median(b)),
            "HR_IQR_BPM": float(np.percentile(b, 75) - np.percentile(b, 25)),
            "HR_min_BPM": float(b.min()), "HR_max_BPM": float(b.max()),
            "ecart_HR_ref_BPM": float(np.median(b) - hr),
            "f_neg_frac": float((raw < 0).mean()),
            "hors_bande_frac": float(((raw < lo_b) | (raw > hi_b)).mean()),
            "corr_fir": float(np.corrcoef(pulses["1_fir"][ctx.core],
                                          y[ctx.core])[0, 1]),
            "cout_s": float(costs.get(name, np.nan)),
        })

    # --- 7. Balayage de L ------------------------------------------------------
    sweep_rows = []
    for cyc in SSA_CYCLES_SWEEP:
        L = int(round(cyc * ctx.fs / ctx.f0)) | 1
        if u_time.size - L + 1 < L:
            continue
        for tag, fn in (("2_ssa", lambda: ssa_denoise(y_comb_u, L, ctx)),
                        ("3a_mssa", lambda: mssa_pulse(chan_u, L, ctx))):
            try:
                y_c, d_c = fn()
            except Exception as exc:  # noqa: BLE001
                sweep_rows.append({"slug": slug, "approche": tag, "cycles": cyc,
                                   "L": L, "status": f"echec : {exc}"})
                continue
            y_c = fix_sign(y_c, u_time)
            fr, pk = band_score(y_c, u_time, ctx)
            _, bpm_c = analytic_phase(y_c, ctx)
            b = bpm_c[ctx.core]
            ref_pulse = pulses["2_ssa"] if tag == "2_ssa" else pulses["3a_mssa"]
            sweep_rows.append({
                "slug": slug, "approche": tag, "cycles": cyc, "L": L,
                "K": int(u_time.size - L + 1), "status": "ok",
                "n_comp_groupe": int(len(d_c["keep"])),
                "pic_BPM": pk, "en_bande_frac": fr,
                "HR_med_BPM": float(np.median(b)),
                "HR_IQR_BPM": float(np.percentile(b, 75) - np.percentile(b, 25)),
                "corr_L_defaut": float(np.corrcoef(y_c[ctx.core],
                                                   ref_pulse[ctx.core])[0, 1]),
            })

    # --- 8. Ligne de synthese --------------------------------------------------
    by_method = {r["methode"]: r for r in method_rows}
    f_pos = (np.sort(V_DMD["f_bpm"][V_DMD["f_bpm"] > 0])
             if V_DMD is not None else np.array([]))
    row = {
        "slug": slug, "astro": astro, "moment": moment, "condition": condition,
        "n_frames": int(T), "n_pixels": int(N),
        "duree_s": float(t[-1] - t[0]), "fs_Hz": float(ctx.fs),
        "n_uniform": int(u_time.size),
        "dt_max_s": float(np.diff(t).max()),
        "hr_BPM": float(hr), "hr_source": hr_source,
        "hr_measured_BPM": float(hr_measured),
        # -- pipeline --
        "rank_svd": int(U.shape[1]),
        "min_conc_used": float(min_conc_used),
        "n_channels": int(n_ch),
        "channels": " ".join(str(c) for c in channels),
        "channel_min": int(channels.min()) if n_ch else -1,
        "channel_max": int(channels.max()) if n_ch else -1,
        "var_1er_triplet": float(S[0] ** 2 / (S ** 2).sum()),
        "n_comp_pic_en_bande": int(((peak_comp >= lo_b) & (peak_comp <= hi_b)).sum()),
        "objectif_opt": float(res_comb.objective),
        "corr_interp": corr_interp,
        # -- BOPDMD --
        "dmd_variante_retenue": retained,
        "dmd_status": dmd_status,
        "dmd_hr_BPM": hr_dmd,
        "dmd_ecart_HR_BPM": float(hr_dmd - hr) if np.isfinite(hr_dmd) else np.nan,
        "dmd_f_positives": " ".join(f"{x:.2f}" for x in f_pos),
        "dmd_b1_ok": bool(variants["B1"] is not None
                          and closest_to_hr(variants["B1"]) <= BAND_FRAC * hr),
        "dmd_b2_ok": bool(variants["B2"] is not None
                          and closest_to_hr(variants["B2"]) <= BAND_FRAC * hr),
        "dmd_b3_ok": bool(variants["B3"] is not None
                          and closest_to_hr(variants["B3"]) <= BAND_FRAC * hr),
        # -- SSA --
        "L_ssa": int(L_ssa),
        "ssa_n_comp_groupe": int(len(d2["keep"])),
        "mssa_n_comp_groupe": int(len(d3a["keep"])),
        "mssa_pc1_var": float(d3a["pc1_var"]),
        # -- resultats de tete, par methode --
        **{f"HRmed_{m}": by_method[m]["HR_med_BPM"] for m in by_method},
        **{f"HRiqr_{m}": by_method[m]["HR_IQR_BPM"] for m in by_method},
        **{f"fneg_{m}": by_method[m]["f_neg_frac"] for m in by_method},
        **{f"corrfir_{m}": by_method[m]["corr_fir"] for m in by_method},
        "t_load_s": t_load, "t_svd_s": t_svd, "t_periodo_s": t_per,
        "t_opt_s": t_opt,
        "status": "ok",
    }

    if SAVE_TRACES:
        TRACES_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "t": t, "u_time": u_time, "hr": hr, "fs": ctx.fs,
            "bpm_axis": ctx.bpm_axis, "in_band": ctx.in_band, "core": ctx.core,
            "S": S, "peak_comp": peak_comp, "frac_comp": frac_comp,
            "channels": channels, "weights": np.asarray(res_comb.weights),
            "y_comb": y_comb,
            "P_svd": P_svd.astype(np.float32),
            "ssa_sv": d2["sv"][:SSA_N_COMP], "ssa_keep": d2["keep"],
            "ssa_peaks": d2["peaks"], "mssa_sv": d3a["sv"][:SSA_N_COMP],
            "mssa_keep": d3a["keep"],
            "chan_u": chan_u.astype(np.float32),
            "L_ssa": L_ssa,
        }
        for name, y in pulses.items():
            payload[f"pulse_{name}"] = y.astype(np.float32)
            payload[f"phase_{name}"] = phases[name].astype(np.float32)
            payload[f"bpm_{name}"] = bpms[name].astype(np.float32)
            payload[f"spec_{name}"] = spectrum(y, u_time, ctx).astype(np.float32)
        for key, v in variants.items():
            if v is not None:
                payload[f"dmd_eigs_{key}"] = v["dmd"].eigs
                payload[f"dmd_amp_{key}"] = v["dmd"].amplitudes
        np.savez_compressed(TRACES_DIR / f"{slug}.npz", **payload)

    return {"condition": row, "methods": method_rows, "dmd": dmd_rows,
            "sweep": sweep_rows}


# --------------------------------------------------------------------------- #
# Persistance : reecriture complete apres chaque condition (reprise sur erreur)
# --------------------------------------------------------------------------- #
def load_table(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def save_table(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"sortie : {OUT_DIR}")
    print(f"device : {DEVICE}   variante de masques : {MASK_VARIANT}")

    tables = {
        "condition": (load_table(CSV_CONDITIONS), CSV_CONDITIONS),
        "methods": (load_table(CSV_METHODS), CSV_METHODS),
        "dmd": (load_table(CSV_DMD), CSV_DMD),
        "sweep": (load_table(CSV_SWEEP), CSV_SWEEP),
    }
    rows = {k: (df.to_dict("records") if not df.empty else [])
            for k, (df, _) in tables.items()}
    # Seules les conditions REUSSIES comptent comme faites : une ligne en echec
    # est un travail a refaire, pas un resultat. Elle est retiree de la table et
    # la condition est retentee -- sinon un correctif du pipeline ne profite
    # jamais aux conditions qu'il etait cense debloquer, et il faut passer par
    # OVERWRITE, donc tout recalculer. Les conditions en echec n'ont de ligne
    # dans aucune des trois autres tables (l'exception survient avant), il n'y a
    # donc rien a nettoyer ailleurs.
    if OVERWRITE:
        done = set()
    else:
        done = {r["slug"] for r in rows["condition"] if r.get("status") == "ok"}
        n_retry = len(rows["condition"]) - len(done)
        rows["condition"] = [r for r in rows["condition"]
                             if r.get("status") == "ok"]
        if n_retry:
            print(f"{n_retry} condition(s) en echec seront retentees")
    if done:
        print(f"{len(done)} condition(s) deja reussie(s), reprise")

    conditions = list(iter_conditions())
    if LIMIT is not None:
        conditions = conditions[:LIMIT]
    print(f"{len(conditions)} condition(s) a parcourir\n")

    t_start = time.perf_counter()
    n_ok = n_skip = n_fail = 0
    for i, path_condi in enumerate(conditions, 1):
        astro = path_condi.parent.parent.name
        slug = slug_of(astro, path_condi.parent.name, path_condi.name)
        if slug in done:
            n_skip += 1
            continue

        print(f"[{i:>3d}/{len(conditions)}] {slug}", flush=True)
        t0 = time.perf_counter()
        try:
            out = process_condition(path_condi)
        except Exception as exc:  # noqa: BLE001 - une condition ne doit pas tout arreter
            n_fail += 1
            print(f"        ECHEC : {exc}")
            traceback.print_exc()
            rows["condition"].append({
                "slug": slug, "astro": astro, "moment": path_condi.parent.name,
                "condition": path_condi.name, "status": f"echec : {exc}",
            })
        else:
            n_ok += 1
            for key in ("condition", "methods", "dmd", "sweep"):
                payload = out[key]
                rows[key].extend([payload] if key == "condition" else payload)
            c = out["condition"]
            print(f"        HR {c['hr_BPM']:.0f} ({c['hr_source']}), "
                  f"{c['n_channels']} canaux, DMD {c['dmd_variante_retenue']} "
                  f"-> {c['dmd_hr_BPM']:.1f} BPM, "
                  f"f<0 sans filtre {100 * c['fneg_0_sans_filtre']:.0f} % "
                  f"vs FIR {100 * c['fneg_1_fir']:.0f} %  "
                  f"[{time.perf_counter() - t0:.0f} s]")

        for key, (_, path) in tables.items():
            save_table(rows[key], path)

    dt = time.perf_counter() - t_start
    print(f"\ntermine {datetime.now():%Y-%m-%d %H:%M} : {n_ok} ok, {n_skip} "
          f"deja faites, {n_fail} en echec  ({dt / 60:.1f} min)")
    for key, (_, path) in tables.items():
        print(f"  {path}  ({len(rows[key])} lignes)")


if __name__ == "__main__":
    main()
