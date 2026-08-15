"""
compute_phase_shift_pixel_svd.py

Version "batch" du notebook ``notebook/test_norm_int.ipynb`` : la meme chaine
(traces pixel multiresolution -> normalisation par frame -> SVD -> combinaison
optimisee des vecteurs singuliers droits) est appliquee a TOUTES les conditions
de TOUS les sujets, pour repondre a deux questions :

  1. DEPHASAGE ENTRE REGIONS DE LA CHOROIDE. La ROI est decoupee en
     ``N_LAYERS x N_COLS_REG`` tuiles (couches = tiers de l'epaisseur LOCALE,
     colonnes = parts egales de l'etendue d'A-scans). Un pouls est reconstruit
     par tuile, plus un pouls "choroide entiere" sur toutes les traces d'un
     seul bloc. Les dephasages sont mesures par correlation croisee de |xc| :
       - tuile <-> choroide entiere (ce que demande l'analyse principale),
       - tuile <-> tuile (matrice complete, choix de pic lisse spatialement).
  2. NOMBRE DE VECTEURS SINGULIERS NECESSAIRES. Pour chaque groupe (global et
     par tuile) on enregistre, composante par composante : sa valeur
     singuliere, son pic Lomb-Scargle, sa concentration, si elle etait
     candidate, si elle a ete retenue, son poids dans la combinaison, et la
     COURBE DE RECONSTRUCTION (correlation de la combinaison partielle -- les
     ``m`` composantes de plus grand |poids| -- avec la combinaison complete).
     Les seuils ``n_sv_corr90/95/99`` en decoulent directement.

Differences assumees avec le notebook :
  - ``N_COLS_REG = 5`` (au lieu de 9) et ``N_LAYERS = 3``, comme demande.
  - Optimisation spectrale sur TOUTE la longueur du signal et UNE SEULE
    fenetre (``n_windows = 1``, cf. ``_window_slices`` : l'objectif redevient
    exactement l'objectif global d'origine).
  - HR absente de ``visit_data.csv`` : la frequence cardiaque est alors
    estimee sur la SVD globale (colonne ``hr_source``) au lieu de faire
    echouer la condition.
  - Aucune figure n'est tracee ni enregistree : uniquement des CSV.

Arborescence lue (identique a ``compute_one_cycle_pixel_svd.py``) :
    E:/NASA_Rigidity/SegmentationVariations/<variante>/
        registered_frames/<NN_id>/<...>_rigidity/<..._OD|OS...>/cube.mp4
        registered_masks/<NN_id>/<...>_rigidity/<..._OD|OS...>/mask.npz
    E:/SANSORI/<NN_id>/<...>_rigidity/<..._OD|OS...>/
        RawImages/ (ou RawData/)     <- .tif + export .xml, pour les horodatages
        Data Files/visit_data.csv    <- HR (prior physiologique, optionnel)

Sorties (sous ``SEGVAR_ROOT/<variante>/phase_shift_pixel_svd/``) :
  - ``conditions.csv``   1 ligne / condition  (choroide entiere)
  - ``regions.csv``      1 ligne / (condition, tuile) -- dont le dephasage
                         tuile <-> choroide entiere
  - ``region_pairs.csv`` 1 ligne / (condition, tuile i, tuile j) -- dephasage
                         tuile <-> tuile
  - ``components.csv``   1 ligne / (condition, groupe, composante SVD)

Les quatre tables sont reecrites apres CHAQUE condition et relues au demarrage :
le script est interruptible et reprend ou il s'est arrete (``OVERWRITE`` pour
tout refaire).
"""

from __future__ import annotations

import csv
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
PATH_GENERAL = Path("E:/SANSORI")  # arborescence brute (horodatages, HR)
SEGVAR_ROOT = Path("E:/NASA_Rigidity/SegmentationVariations")
MASK_VARIANT = "model1_scale_1.0"
FRAMES_SUBDIR = "registered_frames"
MASKS_SUBDIR = "registered_masks"
OUTPUT_SUBDIR = "phase_shift_pixel_svd"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OVERWRITE = False  # True = retraiter les conditions deja presentes dans les CSV

# --- Traces pixel (memes valeurs que le notebook) ---
# ROW_FRAC couvre TOUTE l'epaisseur : c'est le decoupage regional qui la
# redivise en N_LAYERS tiers. La restreindre ici reviendrait a n'analyser
# qu'une partie de la choroide.
COL_FRAC = (0.0, 1.0)
ROW_FRAC = (0.0, 1.0)
BLOCK_SIZES = (1, 2, 3, 4, 5)

# --- Decoupage regional ---
N_LAYERS = 3
N_COLS_REG = 5
N_REGIONS = N_LAYERS * N_COLS_REG

# --- Bande cardiaque ---
# Large pour le DIAGNOSTIC (grille de frequences des periodogrammes, et grille
# hors-bande de l'objectif d'optimisation), etroite (+/- BAND_FRAC de la HR)
# pour le SCORE "puissance en bande".
DIAG_BPM_RANGE = (20.0, 240.0)
BAND_FRAC = 0.2

# --- SVD ---
N_SVD_GLOBAL = 100  # composantes gardees sur la choroide entiere
N_SVD_REGION = 40  # composantes gardees par tuile

# --- Optimisation de la combinaison des vecteurs singuliers droits ---
# n_windows = 1 -> l'energie en bande est evaluee sur TOUTE la longueur du
# signal, en une seule fenetre (cf. `OptimizedSpectralCombination._window_slices`).
N_WINDOWS = 1
WINDOW_N_CYCLES = 1e6  # ignore des lors que n_windows <= 1
# 0.15 (valeur du notebook et defaut du package) rejetait 46 % des conditions et
# 70 % des tuiles : dans les cas en echec, ~19 composantes sur 100 piquent bien a
# la bonne frequence mais aucune n'atteint 0.15 de concentration (max median
# 0.12, contre 0.21 chez celles qui passent). A 0.10, 89 conditions sur 100 ont
# au moins une candidate. Le seuil est reporte dans `conditions.csv`
# (`min_concentration`) : deux runs a seuils differents restent distinguables.
MIN_CONCENTRATION = 0.10
MAX_CANDIDATES = 12  # plafonne mecaniquement `n_selected` (cf. components.csv)
N_RESTARTS = 4
MAXITER = 100
RANDOM_STATE = 0

# --- Correlations croisees ---
LAG_MAX_CYCLES = 2.0  # demi-largeur de la fenetre de lag, en cycles cardiaques
N_PEAK_CANDIDATES = 16  # pics de |xc| gardes comme candidats par paire
# Seuils de la courbe de reconstruction -> colonnes n_sv_corrXX
CORR_THRESHOLDS = (0.90, 0.95, 0.99)

OUT_DIR = SEGVAR_ROOT / MASK_VARIANT / OUTPUT_SUBDIR
CSV_CONDITIONS = OUT_DIR / "conditions.csv"
CSV_REGIONS = OUT_DIR / "regions.csv"
CSV_PAIRS = OUT_DIR / "region_pairs.csv"
CSV_COMPONENTS = OUT_DIR / "components.csv"


# --------------------------------------------------------------------------- #
# Resolution des chemins (arborescence SANSORI, cf. compute_one_cycle_pixel_svd.py)
# --------------------------------------------------------------------------- #
def find_raw_dir(condition_dir: Path) -> Path | None:
    """Sous-dossier contenant les .tif bruts + l'export XML Spectralis."""
    for name in ("RawImages", "RawData"):
        d = condition_dir / name
        if d.is_dir():
            return d
    return None


def raw_timestamps_us(raw_dir: Path) -> np.ndarray:
    """Horodatages bruts (us), un par frame, MEME ORDRE que les frames/masques
    de ``SegmentationVariations``."""
    series = load_ordered_oct_series(raw_dir)
    return np.array(
        [int(round(s.acquisition_time.seconds_of_day * 1e6)) for s in series],
        dtype=np.int64,
    )


def read_hr(path_condi: Path) -> float:
    """HR moyenne (BPM) depuis ``Data Files/visit_data.csv``, NaN si absente."""
    path_heartbeat = path_condi / "Data Files" / "visit_data.csv"
    if not path_heartbeat.exists():
        return float("nan")
    df = pd.read_csv(path_heartbeat, quoting=csv.QUOTE_NONE)
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


# --------------------------------------------------------------------------- #
# Contexte spectral d'une condition
# --------------------------------------------------------------------------- #
@dataclass
class Ctx:
    """Tout ce qui, dans le notebook, vivait en variable globale de cellule."""

    t: np.ndarray  # (T,) horodatages reels, NON uniformes
    u_time: np.ndarray  # (Tu,) grille uniforme de l'aligner
    fs: float
    hr: float  # BPM de reference (mesuree ou estimee)
    freqs: np.ndarray  # (F,) grille de diagnostic, en Hz
    bpm_axis: np.ndarray  # (F,) la meme, en BPM
    in_band: np.ndarray  # (F,) bool, bande de score autour de hr
    band: CardiacBand  # bande etroite, sert au passe-bande du pouls
    low_cut: float  # cutoffs normalises par Nyquist
    high_cut: float
    lags: np.ndarray  # (L,) lags de la fenetre de correlation, en s
    win: np.ndarray  # masque de `lags` dans le `full` de np.correlate
    i_zero: int  # indice du lag 0 dans `lags`
    dt_u: float
    lag_max: float


def diagnostic_freqs(t: np.ndarray) -> np.ndarray:
    """Grille de frequences du notebook : bande DIAG large, sur-echantillonnage
    x5, exactement celle que ``LombScargleRateEstimator.score`` construit pour
    ``CardiacBand(bpm_range=DIAG_BPM_RANGE)`` -- les puissances par composante
    peuvent donc etre relues telles quelles depuis ``rate.diagnostics``."""
    df = 1.0 / (5.0 * (t[-1] - t[0]))
    return np.arange(DIAG_BPM_RANGE[0] / 60.0, DIAG_BPM_RANGE[1] / 60.0 + df, df)


def build_ctx(t: np.ndarray, u_time: np.ndarray, fs: float, hr: float) -> Ctx:
    freqs = diagnostic_freqs(t)
    bpm_axis = freqs * 60.0
    in_band = np.abs(bpm_axis - hr) <= BAND_FRAC * hr
    if not in_band.any():
        # HR hors de la grille de diagnostic : on garde le bin le plus proche
        # plutot que de renvoyer une fraction en bande identiquement nulle.
        in_band = np.zeros_like(in_band)
        in_band[int(np.abs(bpm_axis - hr).argmin())] = True

    # Bande etroite : celle du passe-bande applique aux pouls avant correlation
    # croisee (meme fonction/cutoffs que `AbstractUniformTraceSource.filtered_signal`).
    band = CardiacBand(expected_bpm=float(hr), expected_bpm_band_frac=BAND_FRAC)
    nyq = 0.5 * fs
    lo_bpm, hi_bpm = band.effective_bpm_range

    lag_max = LAG_MAX_CYCLES * 60.0 / hr
    tu = u_time.size
    dt_u = float(u_time[1] - u_time[0])
    lags_all = np.arange(-tu + 1, tu) * dt_u
    win = np.abs(lags_all) <= lag_max
    lags = lags_all[win]

    return Ctx(
        t=t,
        u_time=u_time,
        fs=fs,
        hr=float(hr),
        freqs=freqs,
        bpm_axis=bpm_axis,
        in_band=in_band,
        band=band,
        low_cut=(lo_bpm / 60.0) / nyq,
        high_cut=min((hi_bpm / 60.0) / nyq, 0.99),
        lags=lags,
        win=win,
        i_zero=int(np.abs(lags).argmin()),
        dt_u=dt_u,
        lag_max=float(lag_max),
    )


def spectrum(y: np.ndarray, ctx: Ctx) -> np.ndarray:
    """Lomb-Scargle d'une trace unique (centree), sur les horodatages reels."""
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 8:
        return np.zeros_like(ctx.freqs)
    return lomb_scargle_power(ctx.t[ok], y[ok] - y[ok].mean(), ctx.freqs)


def frac_in_band(power: np.ndarray, freqs: np.ndarray, hr: float) -> np.ndarray:
    """Part de la puissance dans la bande de score. ``power`` est (F,) ou (F, K)."""
    mask = np.abs(freqs * 60.0 - hr) <= BAND_FRAC * hr
    p = np.asarray(power, dtype=float)
    return p[mask].sum(axis=0) / (p.sum(axis=0) + 1e-12)


def bandpass_pulse(y: np.ndarray, ctx: Ctx) -> np.ndarray:
    """Passe-bande cardiaque d'une trace unique, sur la grille UNIFORME.

    Le FIR suppose un echantillonnage uniforme ; ``y`` vit sur ``ctx.t``, qui ne
    l'est pas -- on l'interpole d'abord, exactement comme ``filtered_signal``
    filtre ``interpolated_signal`` et non le signal brut.
    """
    yu = np.interp(ctx.u_time, ctx.t, y)
    return spatio_temporal_filter(
        yu[:, None],
        spatial_sigma=0.0,
        temporal_low_freq=ctx.low_cut,
        temporal_high_freq=ctx.high_cut,
        fs=ctx.fs,
        validity_mask=None,
    )[:, 0]


# --------------------------------------------------------------------------- #
# Decoupage regional de la ROI
# --------------------------------------------------------------------------- #
@dataclass
class RegionGrid:
    region_of_pixel: np.ndarray  # (H, W) int, -1 hors ROI
    region_of_trace: np.ndarray  # (N,) int, une entree par colonne de sig0
    cols: np.ndarray  # A-scans gardes
    thickness_px: np.ndarray  # epaisseur locale par A-scan garde


def build_region_grid(roi: np.ndarray, n_traces: int) -> RegionGrid:
    """Decoupe la ROI en ``N_LAYERS x N_COLS_REG`` tuiles.

    Numerotation : ``region = couche * N_COLS_REG + colonne`` (0 = couche
    superieure, gauche -> droite). Les couches sont des tiers de l'epaisseur
    LOCALE, recalculee A-scan par A-scan : la choroide n'a ni la meme epaisseur
    ni la meme profondeur d'un A-scan a l'autre, un decoupage en lignes fixes
    melangerait les couches d'une colonne a l'autre.
    """
    cols = np.where(roi.any(axis=0))[0]

    col_group_of_x = np.full(roi.shape[1], -1, dtype=int)
    rank = np.arange(cols.size) / cols.size
    col_group_of_x[cols] = np.minimum((rank * N_COLS_REG).astype(int), N_COLS_REG - 1)

    region_of_pixel = np.full(roi.shape, -1, dtype=int)
    thickness_px = np.zeros(cols.size)
    for j, x in enumerate(cols):
        rows = np.where(roi[:, x])[0]
        span = rows[-1] - rows[0] + 1
        thickness_px[j] = span
        layer = np.minimum(
            ((rows - rows[0]) / span * N_LAYERS).astype(int), N_LAYERS - 1
        )
        region_of_pixel[rows, x] = layer * N_COLS_REG + col_group_of_x[x]

    # Region de CHAQUE colonne de sig0, toutes echelles de bloc confondues.
    # `raw_signal` concatene les echelles dans l'ordre de BLOCK_SIZES, un bloc
    # n'etant garde que s'il est entierement dans la ROI -> son pixel central
    # est toujours dans la ROI, sa region est donc bien definie.
    parts = []
    for b in BLOCK_SIZES:
        roi_b = PixelTraceSource._block_all(roi, b)
        if int(roi_b.sum()) == 0:
            continue  # meme saut que raw_signal()
        if b == 1:
            reg_b = region_of_pixel
        else:
            h, w = roi.shape
            hc, wc = h - h % b, w - w % b
            reg_b = region_of_pixel[b // 2 : hc : b, b // 2 : wc : b]  # pixel central
        parts.append(reg_b[roi_b])
    region_of_trace = np.concatenate(parts)

    if region_of_trace.size != n_traces:
        raise ValueError(
            f"desalignement traces <-> regions : {region_of_trace.size} != {n_traces}"
        )
    if not (region_of_trace >= 0).all():
        raise ValueError("des traces sont hors region")

    return RegionGrid(region_of_pixel, region_of_trace, cols, thickness_px)


# --------------------------------------------------------------------------- #
# SVD + combinaison optimisee, pour un groupe de traces
# --------------------------------------------------------------------------- #
def svd_traces(X: np.ndarray, n_components: int) -> tuple:
    """SVD tronquee de ``X`` (T, N) : ``U * S`` (composantes temporelles) et
    ``V`` (motif spatial de chaque composante)."""
    k = int(min(n_components, X.shape[0], X.shape[1]))
    return project_into_separable_components(
        X,
        method="svd",
        n_components=k,
        normalize=False,  # True = centrerait chaque colonne avant la SVD
        random_state=RANDOM_STATE,
    )


def resolution_contributions(
    X: np.ndarray, pattern: np.ndarray, scale: np.ndarray, y: np.ndarray
) -> dict[int, dict]:
    """Part de chaque echelle de bloc dans le pouls reconstruit.

    La SVD donne ``U * S = X @ V``, donc la combinaison optimisee vaut
    exactement ``y = X @ pattern`` (``pattern = V[:, candidates] @ w``). Elle se
    decompose donc ADDITIVEMENT par echelle : ``y = somme_b y_b`` avec
    ``y_b = X[:, scale == b] @ pattern[scale == b]``.

    - ``var_share = cov(y_b, y) / var(y)`` : somme a 1 exactement. C'est la
      mesure d'importance -- elle attribue a chaque echelle sa contribution
      SIGNEE au pouls final (une echelle qui travaille a contre-courant sort
      negative).
    - ``std_frac = std(y_b) / std(y)`` : amplitude propre de l'echelle. Ne somme
      pas a 1 (les ``y_b`` sont correles entre eux), a lire avec ``corr``.

    Le signe de la SVD etant arbitraire, retourner ``y`` et ``pattern``
    ensemble laisse ces trois quantites inchangees : elles peuvent etre
    calculees avant le recalage de signe.
    """
    yc = np.asarray(y, dtype=float)
    yc = yc - yc.mean()
    var = float(yc @ yc)
    out: dict[int, dict] = {}
    for b in BLOCK_SIZES:
        m = scale == b
        n = int(m.sum())
        if n == 0 or var <= 0:
            out[b] = dict(n=n, var_share=np.nan, std_frac=np.nan, corr=np.nan,
                          abs_p_share=np.nan, abs_p_mean=np.nan)
            continue
        p_b = np.asarray(pattern, dtype=float)[m]
        y_b = np.asarray(X[:, m], dtype=float) @ p_b
        y_bc = y_b - y_b.mean()
        sd = float(np.sqrt(y_bc @ y_bc))
        out[b] = dict(
            n=n,
            var_share=float((y_bc @ yc) / var),
            std_frac=sd / float(np.sqrt(var)),
            corr=float((y_bc @ yc) / (sd * np.sqrt(var))) if sd > 0 else np.nan,
            abs_p_share=float(np.abs(p_b).sum() / (np.abs(pattern).sum() + 1e-30)),
            abs_p_mean=float(np.abs(p_b).mean()),
        )
    return out


def analyze_group(
    X: np.ndarray,
    ctx: Ctx,
    rate_estimator: LombScargleRateEstimator,
    opt_cfg: SpectralCombinationConfig,
    n_components: int,
    decomposition: tuple | None = None,
    scale_of_trace: np.ndarray | None = None,
) -> dict:
    """SVD -> vecteurs singuliers -> combinaison optimisee, sur ``X`` (T, N).

    ``decomposition`` evite de refaire une SVD deja calculee (cas de la
    choroide entiere, dont la SVD sert aussi a estimer la HR quand
    ``visit_data.csv`` ne la donne pas). ``failed`` est renseigne -- au lieu de
    lever -- quand l'optimiseur n'a aucune composante candidate (aucun pic
    Lomb-Scargle assez proche de la HR) : la condition reste enregistree.
    """
    U, V = (
        decomposition if decomposition is not None else svd_traces(X, n_components)
    )
    k = U.shape[1]

    traces = Traces(
        values=np.asarray(U, dtype=np.float64),
        uniform_time=ctx.t,
        kept_mask=np.ones(ctx.t.size, dtype=bool),
        gap_mask=np.zeros(ctx.t.size, dtype=bool),
        timestamps_seconds=ctx.t,
        mixing=np.asarray(V, dtype=np.float64),
    )
    rate = rate_estimator.estimate(traces)
    diag = rate.diagnostics

    # `rate.diagnostics["power"]` est (F, K) sur la MEME grille que ctx.freqs
    # (meme bande, meme sur-echantillonnage) : inutile de recalculer K
    # periodogrammes.
    power = np.asarray(diag["power"], dtype=float)
    d_freqs = np.asarray(diag["freqs"], dtype=float)
    peak_bpm = np.asarray(diag["peak_freq"], dtype=float) * 60.0
    concentration = np.asarray(diag["concentration"], dtype=float)
    quality = np.asarray(diag["quality"], dtype=float)
    comp_frac = frac_in_band(power, d_freqs, ctx.hr)

    # Meme critere que `OptimizedSpectralCombination`, mais SANS le plafond
    # `max_candidates` : combien de composantes auraient PU entrer.
    is_candidate = (np.abs(peak_bpm - ctx.hr) <= opt_cfg.accept_tol_bpm) & (
        concentration >= opt_cfg.min_concentration
    )

    aggregator = OptimizedSpectralCombination(opt_cfg)
    try:
        y = aggregator.aggregate(traces, rate)
    except ValueError as exc:
        return {
            "failed": str(exc),
            "n_components": k,
            "n_traces": int(X.shape[1]),
            "singular_values": np.linalg.norm(U, axis=0),
            "peak_bpm": peak_bpm,
            "concentration": concentration,
            "quality": quality,
            "comp_frac": comp_frac,
            "is_candidate": is_candidate,
        }
    res = aggregator.last_result
    contrib = (
        resolution_contributions(X, res.spatial_pattern, scale_of_trace, y)
        if scale_of_trace is not None and res.spatial_pattern is not None
        else None
    )

    return {
        "failed": None,
        "res_contrib": contrib,
        "n_components": k,
        "n_traces": int(X.shape[1]),
        "U": U,
        "V": V,
        "rate": rate,
        "y": y,  # sur ctx.t
        "res": res,
        "pattern": res.spatial_pattern,
        "singular_values": np.linalg.norm(U, axis=0),  # U porte deja S
        "peak_bpm": peak_bpm,
        "concentration": concentration,
        "quality": quality,
        "comp_frac": comp_frac,
        "is_candidate": is_candidate,
    }


def reconstruction_curve(group: dict, ctx: Ctx) -> dict:
    """Combien de vecteurs singuliers faut-il pour retrouver le pouls ?

    Les composantes retenues sont ajoutees par |poids| decroissant ; a chaque
    rang ``m`` on mesure la correlation de la combinaison partielle avec la
    combinaison complete, et sa fraction de puissance en bande. Les colonnes
    ``n_sv_corrXX`` sont le premier ``m`` qui atteint le seuil.

    Le resultat est memorise sur le groupe : il est relu par ``regions.csv``
    ET par ``components.csv``, et chaque rang coute un periodogramme.
    """
    if "curve" in group:
        return group["curve"]
    res = group["res"]
    w = np.asarray(res.weights, dtype=float)
    sel = np.asarray(res.selected_indices, dtype=int)
    U_sel = group["U"][:, sel]
    y_full = U_sel @ w

    order = np.argsort(np.abs(w))[::-1]
    corr = np.full(w.size, np.nan)
    frac = np.full(w.size, np.nan)
    for m in range(1, w.size + 1):
        idx = order[:m]
        y_m = U_sel[:, idx] @ w[idx]
        sd = np.std(y_m)
        corr[m - 1] = (
            float(np.corrcoef(y_m, y_full)[0, 1]) if sd > 0 else np.nan
        )
        P_m = spectrum(y_m, ctx)
        frac[m - 1] = float(P_m[ctx.in_band].sum() / (P_m.sum() + 1e-12))

    n_needed = {}
    for thr in CORR_THRESHOLDS:
        hit = np.where(np.abs(corr) >= thr)[0]
        n_needed[thr] = int(hit[0] + 1) if hit.size else -1

    # rang de chaque composante retenue dans l'ordre |poids| decroissant
    rank_of_sel = np.empty(w.size, dtype=int)
    rank_of_sel[order] = np.arange(1, w.size + 1)

    group["curve"] = {
        "order": order,
        "corr": corr,  # indexee par rang - 1
        "frac": frac,
        "rank_of_sel": rank_of_sel,  # indexee comme `selected_indices`
        "n_needed": n_needed,
    }
    return group["curve"]


# --------------------------------------------------------------------------- #
# Correlations croisees et dephasages
# --------------------------------------------------------------------------- #
def _z(y: np.ndarray) -> np.ndarray:
    return (y - np.nanmean(y)) / (np.nanstd(y) + 1e-12)


def cross_corr(z_i: np.ndarray, z_j: np.ndarray, ctx: Ctx) -> np.ndarray:
    """Correlation croisee normalisee de j contre i, restreinte a la fenetre.

    Convention de signe (verifiee sur une paire synthetique decalee) :
    ``np.correlate(z_j, z_i)`` pique a un lag POSITIF quand j est EN RETARD sur i.
    Les deux traces vivent sur la grille UNIFORME : un decalage en echantillons
    ne se traduit en secondes que si le pas de temps est constant.
    """
    tu = ctx.u_time.size
    return np.correlate(
        np.nan_to_num(z_j, nan=0.0) / tu, np.nan_to_num(z_i, nan=0.0), mode="full"
    )[ctx.win]


def _local_maxima(xc: np.ndarray) -> np.ndarray:
    """Indices des maxima LOCAUX. Sur ``|xc|`` ce sont soit de vrais maxima,
    soit de vrais minima retournes -- dans les deux cas des sommets lisses ; les
    points anguleux introduits par la valeur absolue sont des passages par zero,
    donc des minima, jamais captes."""
    up = xc[1:-1] > xc[:-2]
    down = xc[1:-1] >= xc[2:]
    return np.where(up & down)[0] + 1


def _refine_peak(xc: np.ndarray, k: int, ctx: Ctx) -> tuple[float, float]:
    """Sommet raffine par ajustement parabolique sur les 3 points autour de ``k``.
    Le pas de temps est du meme ordre que les dephasages attendus : rester sur la
    grille les arrondirait a 0 ou +/- dt."""
    lag, val = float(ctx.lags[k]), float(xc[k])
    if 0 < k < xc.size - 1:
        y1, y2, y3 = xc[k - 1], xc[k], xc[k + 1]
        den = y1 - 2.0 * y2 + y3
        if den < 0:
            delta = 0.5 * (y1 - y3) / den
            if abs(delta) <= 1.0:
                lag = float(ctx.lags[k] + delta * ctx.dt_u)
                val = float(y2 - 0.25 * (y1 - y3) * delta)
    return lag, val


def nearest_peaks(xc: np.ndarray, n: int, ctx: Ctx) -> list[tuple[float, float]]:
    """Les ``n`` maxima locaux les plus proches du lag 0, tries par |lag|.

    Sur un signal quasi-periodique le maximum GLOBAL de la fenetre peut tomber
    un cycle entier plus loin : le dephasage physiologique est porte par les
    lobes adjacents a 0, pas par le plus haut. Si moins de ``n`` maxima
    existent, le dernier est repete -- la sortie garde une taille fixe.
    """
    loc = _local_maxima(xc)
    if loc.size == 0:
        loc = np.array([int(xc.argmax())])
    ks = loc[np.argsort(np.abs(ctx.lags[loc]))][:n]
    out = [_refine_peak(xc, int(k), ctx) for k in ks]
    while len(out) < n:
        out.append(out[-1])
    return out


def nearest_peak(xc: np.ndarray, ctx: Ctx) -> tuple[float, float]:
    """Le maximum local le plus proche du lag 0."""
    return nearest_peaks(xc, 1, ctx)[0]


def _neighbours() -> list[list[int]]:
    """Voisins 4-connexes de chaque region dans la grille couche x colonne."""
    nb = []
    for j in range(N_REGIONS):
        layer, col = divmod(j, N_COLS_REG)
        vois = []
        for dl, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            l2, c2 = layer + dl, col + dc
            if 0 <= l2 < N_LAYERS and 0 <= c2 < N_COLS_REG:
                vois.append(l2 * N_COLS_REG + c2)
        nb.append(vois)
    return nb


NEIGH = _neighbours()


def spatial_roughness(lag_map: np.ndarray) -> float:
    """Somme des |differences| de lag entre regions voisines (4-connexite) sur
    la grille couche x colonne. Une region invalide (NaN) ne coute rien."""
    g = np.asarray(lag_map, dtype=float).reshape(N_LAYERS, N_COLS_REG)
    dv = np.nan_to_num(np.abs(np.diff(g, axis=0)), nan=0.0).sum()
    dh = np.nan_to_num(np.abs(np.diff(g, axis=1)), nan=0.0).sum()
    return float(dv + dh)


def _unwrap_from(L_cand: np.ndarray, pin: int, usable: set[int]) -> np.ndarray:
    """Deroulement en largeur depuis l'ancre : chaque region prend le candidat le
    plus proche du lag deja fixe du voisin par lequel on l'atteint. C'est le
    depart qui fait tout le travail -- partir d'un lag constant laisse l'ICM
    dans un minimum local des que la carte a une pente."""
    lag = np.full(N_REGIONS, np.nan)
    lag[pin] = 0.0
    seen, queue = {pin}, [pin]
    while queue:
        j = queue.pop(0)
        for n in NEIGH[j]:
            if n in seen or n not in usable:
                continue
            lag[n] = L_cand[n][int(np.nanargmin(np.abs(L_cand[n] - lag[j])))]
            seen.add(n)
            queue.append(n)
    return lag


def smooth_lag_selection(
    L_cand: np.ndarray, pin: int, ctx: Ctx, max_sweeps: int = 50
) -> tuple[np.ndarray, float]:
    """Choisit UN pic par region de facon a minimiser ``spatial_roughness``.

    Depart : deroulement depuis l'ancre (``_unwrap_from``), plus une serie de
    departs a lag constant en filet de securite. Affinage : descente par
    coordonnees (ICM) -- a chaque visite d'une region on garde le candidat le
    plus proche de la MEDIANE des lags de ses voisins (la mediane minimise
    exactement une somme de |ecarts|).

    ``pin`` est la region de reference : son lag est FIXE a 0 (c'est son
    autocorrelation). Sans cette ancre le probleme est degenere -- les pics de
    |xc| se repetent toutes les demi-periodes, decaler TOUTE la tuile d'une
    demi-periode ne changerait aucune difference entre voisins.
    """
    starts = np.linspace(-ctx.lag_max, ctx.lag_max, 9)
    usable = [j for j in range(N_REGIONS) if np.isfinite(L_cand[j]).any()]
    uset = set(usable)
    best_lag, best_energy = None, np.inf
    for c0 in [None] + list(starts):
        if c0 is None:
            lag = _unwrap_from(L_cand, pin, uset)
        else:
            lag = np.full(N_REGIONS, np.nan)
            for j in usable:
                lag[j] = L_cand[j][np.nanargmin(np.abs(L_cand[j] - c0))]
        lag[pin] = 0.0
        for sweep in range(max_sweeps):
            changed = False
            for j in usable if sweep % 2 == 0 else usable[::-1]:
                if j == pin:
                    continue
                vals = [lag[n] for n in NEIGH[j] if np.isfinite(lag[n])]
                if not vals:
                    continue
                k = int(np.nanargmin(np.abs(L_cand[j] - np.median(vals))))
                if L_cand[j][k] != lag[j]:
                    lag[j] = L_cand[j][k]
                    changed = True
            if not changed:
                break
        energy = spatial_roughness(lag)
        if energy < best_energy - 1e-12:
            best_lag, best_energy = lag.copy(), energy
    return best_lag, best_energy


# --------------------------------------------------------------------------- #
# Traitement d'une condition
# --------------------------------------------------------------------------- #
def _fmt_list(values) -> str:
    """Vecteur court -> chaine ';' (les CSV restent lisibles et relisibles)."""
    return ";".join(f"{v:.6g}" for v in np.asarray(values).ravel())


def _resolution_columns(group: dict) -> dict:
    """Colonnes ``res{b}_*`` : importance de chaque echelle de la multiresolution."""
    contrib = group.get("res_contrib")
    if not contrib:
        return {}
    cols = {}
    for b, d in contrib.items():
        cols[f"res{b}_n_traces"] = d["n"]
        cols[f"res{b}_var_share"] = d["var_share"]
        cols[f"res{b}_std_frac"] = d["std_frac"]
        cols[f"res{b}_corr"] = d["corr"]
        cols[f"res{b}_abs_p_share"] = d["abs_p_share"]
        cols[f"res{b}_abs_p_mean"] = d["abs_p_mean"]
    return cols


def process_condition(path_condi: Path) -> dict:
    astro, moment, condition = (
        path_condi.parent.parent.name,
        path_condi.parent.name,
        path_condi.name,
    )
    ident = {"patient": astro, "moment": moment, "condition": condition}
    variant_root = SEGVAR_ROOT / MASK_VARIANT
    mask_path = variant_root / MASKS_SUBDIR / astro / moment / condition / "mask.npz"
    frames_path = variant_root / FRAMES_SUBDIR / astro / moment / condition / "cube.mp4"
    if not mask_path.exists() or not frames_path.exists():
        return {"skip": f"cube.mp4/mask.npz absent ({MASK_VARIANT})"}

    raw_dir = find_raw_dir(path_condi)
    if raw_dir is None:
        return {"skip": "RawImages/RawData absent"}
    ts_us = raw_timestamps_us(raw_dir)
    if ts_us.size < 2:
        return {"skip": f"pas assez d'horodatages ({ts_us.size})"}

    registrator = _prepared_registrator(frames_path, mask_path, DEVICE, verbose=False)
    n_frames = registrator.registered_frames.shape[0]
    if n_frames != ts_us.size:
        return {"skip": f"frames ({n_frames}) != horodatages ({ts_us.size})"}

    aligner = VideoTimelineAligner(registrator, ts_us)
    t = aligner.timestamps_seconds
    hr_visit = read_hr(path_condi)

    # --- 1. Traces pixel : brut puis normalisation PAR FRAME ------------------
    # `normalized_signal()` divise chaque frame par sa moyenne spatiale ; on la
    # reproduit ici a partir de `sig0` pour ne pas relire/repooler la video deux
    # fois (`normalized_signal` rappelle `raw_signal`).
    source = PixelTraceSource(
        registrator,
        aligner,
        PixelTraceConfig(
            band=CardiacBand(bpm_range=DIAG_BPM_RANGE),
            col_frac=COL_FRAC,
            row_frac=ROW_FRAC,
            block_sizes=BLOCK_SIZES,
            verbose=False,
        ),
    )
    try:
        sig0 = source.raw_signal().astype(np.float32)  # (T, N)
    except ValueError as exc:
        # ROI vide : aucun pixel n'est dans le masque sur TOUTES les frames
        # (recalage qui derive, ou choroide qui sort du champ). Rien a analyser.
        return {"skip": f"ROI vide : {exc}"}
    sig0n = (sig0 / np.nanmean(sig0, axis=1, keepdims=True)).astype(np.float32)
    roi = source.base_roi
    scale = source.scale_of_trace

    grid = build_region_grid(roi, sig0.shape[1])

    # --- 2. SVD globale (independante de la HR : elle peut la fournir) --------
    X_all = np.nan_to_num(sig0n, nan=0.0)
    t0 = time.perf_counter()
    U_all, V_all = svd_traces(X_all, N_SVD_GLOBAL)
    k_all = U_all.shape[1]
    t_svd_global = time.perf_counter() - t0

    # --- 3. Frequence cardiaque de reference ---------------------------------
    if np.isfinite(hr_visit):
        hr, hr_source = float(hr_visit), "visit_data"
    else:
        # HR absente : on la lit sur la SVD globale plutot que d'abandonner la
        # condition. Bande large, pas de prior -> pas de correction harmonique.
        traces_hr = Traces(
            values=np.asarray(U_all, dtype=np.float64),
            uniform_time=t,
            kept_mask=np.ones(t.size, dtype=bool),
            gap_mask=np.zeros(t.size, dtype=bool),
            timestamps_seconds=t,
            mixing=np.asarray(V_all, dtype=np.float64),
        )
        free_est = LombScargleRateEstimator(
            LombScargleConfig(
                band=CardiacBand(bpm_range=DIAG_BPM_RANGE),
                harmonic_correction=False,
                verbose=False,
            )
        )
        hr = float(free_est.estimate(traces_hr).freq * 60.0)
        hr_source = "svd_globale"

    ctx = build_ctx(t, aligner.uniform_time, float(aligner.fs), hr)

    # `override_bpm` fixe la cible a la HR de reference, sans re-estimation. La
    # bande LARGE du config sert aussi de grille hors-bande dans l'objectif :
    # avec la bande etroite il n'y aurait presque aucun point hors bande et le
    # rapport a maximiser serait degenere.
    rate_estimator = LombScargleRateEstimator(
        LombScargleConfig(band=CardiacBand(bpm_range=DIAG_BPM_RANGE), verbose=False),
        override_bpm=hr,
    )
    opt_cfg = SpectralCombinationConfig(
        accept_tol_bpm=BAND_FRAC * hr,  # meme bande que `ctx.in_band`
        min_concentration=MIN_CONCENTRATION,
        max_candidates=MAX_CANDIDATES,
        n_restarts=N_RESTARTS,
        maxiter=MAXITER,
        random_state=RANDOM_STATE,
        window_n_cycles=WINDOW_N_CYCLES,
        n_windows=N_WINDOWS,
    )

    # --- 4. Combinaison optimisee sur la choroide ENTIERE ---------------------
    t0 = time.perf_counter()
    glob = analyze_group(
        X_all,
        ctx,
        rate_estimator,
        opt_cfg,
        k_all,
        decomposition=(U_all, V_all),  # deja calculee ci-dessus
        scale_of_trace=scale,
    )
    t_opt_global = time.perf_counter() - t0

    # --- 5. Une combinaison par tuile ----------------------------------------
    # Signal moyen de chaque tuile : sert de reference de SIGNE (celui d'une SVD
    # est arbitraire ; sans recalage les motifs spatiaux de deux tuiles voisines
    # ne sont pas comparables).
    sig_reg = np.stack(
        [
            np.nanmean(sig0n[:, grid.region_of_trace == r], axis=1)
            for r in range(N_REGIONS)
        ],
        axis=1,
    )

    t0 = time.perf_counter()
    reg_groups: dict[int, dict] = {}
    for r in range(N_REGIONS):
        cols_r = np.where(grid.region_of_trace == r)[0]
        if cols_r.size == 0:
            reg_groups[r] = {"failed": "aucune trace", "n_traces": 0}
            continue
        g = analyze_group(
            np.nan_to_num(sig0n[:, cols_r], nan=0.0),
            ctx,
            rate_estimator,
            opt_cfg,
            N_SVD_REGION,
            scale_of_trace=scale[cols_r],
        )
        g["n_traces"] = int(cols_r.size)
        if g["failed"] is None:
            c = np.corrcoef(g["y"], sig_reg[:, r])[0, 1]
            if np.isfinite(c) and c < 0:
                g["y"] = -g["y"]
                g["pattern"] = None if g["pattern"] is None else -g["pattern"]
                g["res"].weights = -np.asarray(g["res"].weights, dtype=float)
            g["sign_corr"] = float(c)
        reg_groups[r] = g
    t_regions = time.perf_counter() - t0

    valid = [r for r in range(N_REGIONS) if reg_groups[r].get("failed") is None]

    # --- 6. Signe du pouls global --------------------------------------------
    # `normalized_signal` divise chaque frame par sa moyenne spatiale : la
    # moyenne de TOUTES les traces de sig0n vaut 1 a chaque frame, elle est
    # plate par construction et ne porte plus aucun signe. On fixe donc le signe
    # global sur la moyenne des pouls regionaux (deja recales), et on rapporte
    # la correlation avec la moyenne spatiale BRUTE comme diagnostic.
    glob_raw = np.nanmean(sig0, axis=1)
    c_ref = np.nan
    if glob["failed"] is None and valid:
        ref_sign = np.mean(
            [
                reg_groups[r]["y"] / (np.nanstd(reg_groups[r]["y"]) + 1e-9)
                for r in valid
            ],
            axis=0,
        )
        c_ref = float(np.corrcoef(glob["y"], ref_sign)[0, 1])
        if np.isfinite(c_ref) and c_ref < 0:
            glob["y"] = -glob["y"]
            glob["pattern"] = None if glob["pattern"] is None else -glob["pattern"]
            glob["res"].weights = -np.asarray(glob["res"].weights, dtype=float)
            c_ref = -c_ref
    c_raw = (
        float(np.corrcoef(glob["y"], glob_raw)[0, 1]) if glob["failed"] is None else np.nan
    )

    # --- 7. Pouls filtres / interpoles, puis dephasages ----------------------
    # Les pouls bruts vivent sur `t` (non uniforme) : ils sont interpoles sur la
    # grille uniforme, sans quoi un decalage en echantillons ne se traduit pas
    # en secondes. Les pouls filtres sont deja sur `u_time`.
    y_filt = {r: bandpass_pulse(reg_groups[r]["y"], ctx) for r in valid}
    y_raw_u = {r: np.interp(ctx.u_time, ctx.t, reg_groups[r]["y"]) for r in valid}

    lag_vs_global: dict[str, dict[int, tuple]] = {"filtres": {}, "bruts": {}}
    if glob["failed"] is None:
        z_all = {
            "filtres": _z(bandpass_pulse(glob["y"], ctx)),
            "bruts": _z(np.interp(ctx.u_time, ctx.t, glob["y"])),
        }
        sig_by_name = {"filtres": y_filt, "bruts": y_raw_u}
        for name in ("filtres", "bruts"):
            for r in valid:
                xc = cross_corr(z_all[name], _z(sig_by_name[name][r]), ctx)
                lag_near, r_near = nearest_peak(np.abs(xc), ctx)
                k_glob = int(np.abs(xc).argmax())
                lag_vs_global[name][r] = (
                    lag_near,
                    r_near,
                    float(ctx.lags[k_glob]),
                    float(np.abs(xc)[k_glob]),
                    float(xc[ctx.i_zero]),
                )

    # --- 8. Dephasages tuile <-> tuile (pouls filtres) -----------------------
    # Le signe d'un pouls regional est arbitraire (vecteur singulier recale sur
    # sa seule tuile) : on passe par |xc|, ce qui laisse une ambiguite a la
    # DEMI-periode. On ne tranche donc pas pic par pic -- tous les pics de la
    # fenetre sont candidats, et on retient dans chaque tuile la combinaison qui
    # minimise la variation de lag d'une region voisine a l'autre, en ancrant la
    # reference elle-meme a lag 0 (son autocorrelation).
    t0 = time.perf_counter()
    Z_f = {r: _z(y_filt[r]) for r in valid}
    L_cand = np.full((N_REGIONS, N_REGIONS, N_PEAK_CANDIDATES), np.nan)
    R_cand = np.full((N_REGIONS, N_REGIONS, N_PEAK_CANDIDATES), np.nan)
    lag_near_ij = np.full((N_REGIONS, N_REGIONS), np.nan)
    r_near_ij = np.full((N_REGIONS, N_REGIONS), np.nan)
    r_zero_ij = np.full((N_REGIONS, N_REGIONS), np.nan)
    n_peaks_ij = np.zeros((N_REGIONS, N_REGIONS), dtype=int)
    for i in valid:
        for j in valid:
            xc = cross_corr(Z_f[i], Z_f[j], ctx)
            r_zero_ij[i, j] = float(xc[ctx.i_zero])
            axc = np.abs(xc)
            n_peaks_ij[i, j] = _local_maxima(axc).size
            peaks = nearest_peaks(axc, N_PEAK_CANDIDATES, ctx)
            for c, (lag, val) in enumerate(peaks):
                L_cand[i, j, c], R_cand[i, j, c] = lag, val
            lag_near_ij[i, j], r_near_ij[i, j] = peaks[0]

    lag_sel = np.full((N_REGIONS, N_REGIONS), np.nan)
    r_sel = np.full((N_REGIONS, N_REGIONS), np.nan)
    not_nearest = np.zeros((N_REGIONS, N_REGIONS), dtype=bool)
    rough_before = rough_after = 0.0
    for i in valid:
        lag_i, energy = smooth_lag_selection(L_cand[i], i, ctx)
        lag_sel[i] = lag_i
        for j in valid:
            if not np.isfinite(lag_i[j]):
                continue
            k = int(np.nanargmin(np.abs(L_cand[i, j] - lag_i[j])))
            r_sel[i, j] = R_cand[i, j, k]
            not_nearest[i, j] = k != 0
        base = L_cand[i, :, 0].copy()
        base[i] = 0.0  # meme ancre pour la reference
        rough_before += spatial_roughness(base)
        rough_after += energy
    t_lags = time.perf_counter() - t0

    # ----------------------------------------------------------------------- #
    # Mise en lignes
    # ----------------------------------------------------------------------- #
    cycle_ms = 60_000.0 / hr
    common = dict(
        ident,
        mask_variant=MASK_VARIANT,
        hr_bpm=hr,
        hr_source=hr_source,
    )

    # --- conditions.csv ---
    row_cond = dict(
        common,
        path=str(path_condi),
        hr_visit_data=hr_visit,
        n_frames=int(n_frames),
        duration_s=float(t[-1] - t[0]),
        fs_hz=float(aligner.fs),
        dt_uniform_ms=ctx.dt_u * 1e3,
        cycle_ms=cycle_ms,
        n_traces=int(sig0.shape[1]),
        n_pixels_roi=int(roi.sum()),
        roi_frac_frame=float(roi.sum() / roi.size),
        block_sizes=";".join(str(b) for b in BLOCK_SIZES),
        n_traces_per_block=";".join(
            str(int((scale == b).sum())) for b in BLOCK_SIZES
        ),
        n_layers=N_LAYERS,
        n_cols_reg=N_COLS_REG,
        n_regions_valid=len(valid),
        thickness_px_median=float(np.median(grid.thickness_px)),
        thickness_px_min=float(grid.thickness_px.min()),
        thickness_px_max=float(grid.thickness_px.max()),
        n_ascans_kept=int(grid.cols.size),
        # -- SVD globale + combinaison optimisee --
        k_svd=int(k_all),
        n_candidates_available=int(glob["is_candidate"].sum()),
        max_candidates=MAX_CANDIDATES,
        n_windows=N_WINDOWS,
        accept_tol_bpm=float(opt_cfg.accept_tol_bpm),
        min_concentration=float(opt_cfg.min_concentration),
        svd_var_frac_top1=float(
            glob["singular_values"][0] ** 2 / (glob["singular_values"] ** 2).sum()
        ),
        corr_sign_ref_regions=c_ref,
        corr_with_raw_spatial_mean=c_raw,
        # -- rugosite spatiale de la carte de lag (diagnostic du lissage) --
        lag_roughness_before_ms=rough_before * 1e3,
        lag_roughness_after_ms=rough_after * 1e3,
        t_svd_global_s=t_svd_global,
        t_opt_global_s=t_opt_global,
        t_regions_s=t_regions,
        t_lags_s=t_lags,
        created=datetime.now().isoformat(timespec="seconds"),
    )
    if glob["failed"] is not None:
        row_cond.update(status=f"echec optimisation : {glob['failed']}")
    else:
        res = glob["res"]
        P_y = spectrum(glob["y"], ctx)
        curve_all = reconstruction_curve(glob, ctx)
        row_cond.update(
            status="ok",
            n_selected=int(res.selected_indices.size),
            selected_indices=_fmt_list(res.selected_indices),
            weights=_fmt_list(res.weights),
            objective=float(res.objective),
            peak_bpm=float(ctx.bpm_axis[P_y.argmax()]),
            frac_in_band=float(P_y[ctx.in_band].sum() / (P_y.sum() + 1e-12)),
            best_single_index=int(
                res.selected_indices[np.argmax(glob["comp_frac"][res.selected_indices])]
            ),
            best_single_frac=float(glob["comp_frac"][res.selected_indices].max()),
            recon_corr_curve=_fmt_list(curve_all["corr"]),
            recon_frac_curve=_fmt_list(curve_all["frac"]),
            **{
                f"n_sv_corr{int(thr * 100)}": curve_all["n_needed"][thr]
                for thr in CORR_THRESHOLDS
            },
            **_resolution_columns(glob),
        )
    rows_cond = [row_cond]

    # --- regions.csv ---
    rows_reg = []
    for r in range(N_REGIONS):
        g = reg_groups[r]
        pix = grid.region_of_pixel == r
        rc = np.argwhere(pix)
        row = dict(
            common,
            region=r,
            layer=r // N_COLS_REG,
            col_group=r % N_COLS_REG,
            n_traces=int(g.get("n_traces", 0)),
            n_pixels=int(pix.sum()),
            row_center=float(rc[:, 0].mean()) if rc.size else np.nan,
            col_center=float(rc[:, 1].mean()) if rc.size else np.nan,
            row_min=int(rc[:, 0].min()) if rc.size else -1,
            row_max=int(rc[:, 0].max()) if rc.size else -1,
            col_min=int(rc[:, 1].min()) if rc.size else -1,
            col_max=int(rc[:, 1].max()) if rc.size else -1,
        )
        if g.get("failed") is not None:
            row.update(status=f"echec : {g['failed']}")
            rows_reg.append(row)
            continue

        res = g["res"]
        P_y = spectrum(g["y"], ctx)
        P_mean = spectrum(sig_reg[:, r], ctx)
        curve = reconstruction_curve(g, ctx)
        row.update(
            status="ok",
            k_svd=int(g["n_components"]),
            n_candidates_available=int(g["is_candidate"].sum()),
            n_selected=int(res.selected_indices.size),
            selected_indices=_fmt_list(res.selected_indices),
            weights=_fmt_list(res.weights),
            objective=float(res.objective),
            peak_bpm=float(ctx.bpm_axis[P_y.argmax()]),
            frac_in_band=float(P_y[ctx.in_band].sum() / (P_y.sum() + 1e-12)),
            mean_signal_peak_bpm=float(ctx.bpm_axis[P_mean.argmax()]),
            mean_signal_frac_in_band=float(
                P_mean[ctx.in_band].sum() / (P_mean.sum() + 1e-12)
            ),
            sign_corr_vs_region_mean=g.get("sign_corr", np.nan),
            svd_var_frac_top1=float(
                g["singular_values"][0] ** 2 / (g["singular_values"] ** 2).sum()
            ),
            recon_corr_curve=_fmt_list(curve["corr"]),
            recon_frac_curve=_fmt_list(curve["frac"]),
            **{
                f"n_sv_corr{int(thr * 100)}": curve["n_needed"][thr]
                for thr in CORR_THRESHOLDS
            },
            **_resolution_columns(g),
        )
        # Dephasage tuile <-> choroide entiere (lag > 0 : la tuile est EN RETARD
        # sur le pouls global).
        for name, suffix in (("filtres", "filt"), ("bruts", "raw")):
            d = lag_vs_global[name].get(r)
            if d is None:
                continue
            lag_near, r_near, lag_peak, r_peak, r0 = d
            row.update(
                {
                    f"lag_vs_global_{suffix}_ms": lag_near * 1e3,
                    f"lag_vs_global_{suffix}_deg": 360.0 * lag_near * 1e3 / cycle_ms,
                    f"r_peak_vs_global_{suffix}": r_near,
                    f"lag_globalmax_vs_global_{suffix}_ms": lag_peak * 1e3,
                    f"r_globalmax_vs_global_{suffix}": r_peak,
                    f"r_lag0_vs_global_{suffix}": r0,
                }
            )
        rows_reg.append(row)

    # --- region_pairs.csv ---
    rows_pair = []
    for i in valid:
        for j in valid:
            rows_pair.append(
                dict(
                    common,
                    region_i=i,
                    region_j=j,
                    layer_i=i // N_COLS_REG,
                    col_i=i % N_COLS_REG,
                    layer_j=j // N_COLS_REG,
                    col_j=j % N_COLS_REG,
                    lag_sel_ms=lag_sel[i, j] * 1e3,
                    lag_sel_deg=360.0 * lag_sel[i, j] * 1e3 / cycle_ms,
                    r_sel=r_sel[i, j],
                    lag_nearest_ms=lag_near_ij[i, j] * 1e3,
                    r_nearest=r_near_ij[i, j],
                    r_lag0=r_zero_ij[i, j],
                    n_peaks=int(n_peaks_ij[i, j]),
                    peak_not_nearest=bool(not_nearest[i, j]),
                )
            )

    # --- components.csv ---
    rows_comp = []

    def _component_rows(group: dict, scope: str, region: int):
        sv = np.asarray(group["singular_values"], dtype=float)
        var_frac = sv**2 / (np.sum(sv**2) + 1e-12)
        sel = (
            np.asarray(group["res"].selected_indices, dtype=int)
            if group.get("failed") is None
            else np.array([], dtype=int)
        )
        w = (
            np.asarray(group["res"].weights, dtype=float)
            if group.get("failed") is None
            else np.array([])
        )
        curve = (
            reconstruction_curve(group, ctx) if group.get("failed") is None else None
        )
        weight_of = {int(k): float(w[m]) for m, k in enumerate(sel)}
        rank_of = (
            {int(k): int(curve["rank_of_sel"][m]) for m, k in enumerate(sel)}
            if curve
            else {}
        )
        for k in range(int(group["n_components"])):
            rank = rank_of.get(k, -1)
            rows_comp.append(
                dict(
                    common,
                    scope=scope,
                    region=region,
                    component=k,
                    singular_value=float(sv[k]),
                    var_frac=float(var_frac[k]),
                    peak_bpm=float(group["peak_bpm"][k]),
                    concentration=float(group["concentration"][k]),
                    quality=float(group["quality"][k]),
                    frac_in_band=float(group["comp_frac"][k]),
                    is_candidate=bool(group["is_candidate"][k]),
                    is_selected=k in weight_of,
                    weight=weight_of.get(k, np.nan),
                    weight_rank=rank,
                    # Courbe de reconstruction cumulee, indexee par `weight_rank`
                    # (correlation avec la combinaison complete du groupe).
                    cum_corr_with_full=(
                        float(curve["corr"][rank - 1]) if rank > 0 else np.nan
                    ),
                    cum_frac_in_band=(
                        float(curve["frac"][rank - 1]) if rank > 0 else np.nan
                    ),
                )
            )

    _component_rows(glob, "global", -1)
    for r in range(N_REGIONS):
        g = reg_groups[r]
        if "n_components" in g:
            _component_rows(g, "region", r)

    return {
        "skip": None,
        "conditions": rows_cond,
        "regions": rows_reg,
        "pairs": rows_pair,
        "components": rows_comp,
        "log": (
            f"HR = {hr:.1f} BPM ({hr_source}) · {sig0.shape[1]} traces · "
            f"{len(valid)}/{N_REGIONS} tuiles · "
            + (
                "GLOBAL echec"
                if glob["failed"] is not None
                else f"global : {row_cond['n_selected']} vecteurs singuliers, "
                f"{row_cond['frac_in_band']:.1%} en bande, "
                f"n_sv(corr>=0.9) = {row_cond['n_sv_corr90']}"
            )
            + f" · {t_svd_global + t_opt_global + t_regions + t_lags:.0f} s"
        ),
    }


# --------------------------------------------------------------------------- #
# Boucle principale
# --------------------------------------------------------------------------- #
def _load_existing(path: Path) -> list[dict]:
    if OVERWRITE or not path.exists():
        return []
    return pd.read_csv(path, keep_default_na=False, na_values=["<NA>", ""]).to_dict(
        "records"
    )


def _write(path: Path, rows: list[dict]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False, na_rep="<NA>")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = {
        "conditions": _load_existing(CSV_CONDITIONS),
        "regions": _load_existing(CSV_REGIONS),
        "pairs": _load_existing(CSV_PAIRS),
        "components": _load_existing(CSV_COMPONENTS),
    }
    done = {
        (r["patient"], r["moment"], r["condition"]) for r in rows["conditions"]
    }
    if done:
        print(f"{len(done)} condition(s) deja traitee(s) : reprise.\n")

    n_new = 0
    for path_condi in iter_conditions():
        key = (
            path_condi.parent.parent.name,
            path_condi.parent.name,
            path_condi.name,
        )
        if key in done:
            continue
        print(path_condi)
        t0 = time.perf_counter()
        try:
            result = process_condition(path_condi)
        except Exception as e:  # noqa: BLE001
            print(f"  [erreur] {e}")
            traceback.print_exc()
            continue
        if result["skip"] is not None:
            print(f"  [skip] {result['skip']}")
            continue

        for name in rows:
            rows[name].extend(result[name])
        done.add(key)
        n_new += 1
        print(f"  -> {result['log']} (total {time.perf_counter() - t0:.0f} s)")

        # Reecriture apres chaque condition : le script est interruptible.
        _write(CSV_CONDITIONS, rows["conditions"])
        _write(CSV_REGIONS, rows["regions"])
        _write(CSV_PAIRS, rows["pairs"])
        _write(CSV_COMPONENTS, rows["components"])

    if not rows["conditions"]:
        print("Aucune condition traitee : rien n'a ete ecrit.")
        return
    print(
        f"\n{n_new} nouvelle(s) condition(s), {len(rows['conditions'])} au total.\n"
        f"  {CSV_CONDITIONS}\n  {CSV_REGIONS}\n  {CSV_PAIRS}\n  {CSV_COMPONENTS}"
    )


if __name__ == "__main__":
    main()
