"""
compute_one_cycle_whole_choroid.py

Videos one-cycle repliees sur LE POULS DE LA CHOROIDE ENTIERE DEJA CALCULE par
``compute_phase_shift_pixel_svd.py``.

Le pouls n'est pas re-optimise ici : il est REPRIS de
``SEGVAR_ROOT/<variante>/phase_shift_pixel_svd/conditions.csv``, dont chaque
ligne (groupe ``scope = "global"``, c.-a-d. toute la choroide) porte deja les
``selected_indices`` et les ``weights`` de la combinaison retenue, ainsi que la
frequence cardiaque de reference. Le pouls vaut, par construction,

    y = (U[:, selected_indices]) @ weights

ou ``U`` sont les composantes temporelles de la SVD. Comme ``randomized_svd``
est deterministe a ``random_state`` fixe, il suffit de refaire A L'IDENTIQUE la
chaine amont -- traces pixel multiresolution sur TOUTE la ROI, normalisation par
frame, SVD non centree -- pour retrouver exactement ``U``, puis d'appliquer les
poids enregistres. Ni la recherche de candidates, ni l'optimisation SLSQP, ni le
recalage de signe ne sont refaits (les poids du CSV sont deja ceux d'apres
recalage de signe : ``compute_phase_shift_pixel_svd.py`` retourne
``res.weights`` avant de les ecrire).

La reconstruction est VERIFIEE avant d'etre utilisee, contre trois quantites
independantes de la meme ligne de CSV : ``n_traces`` (+ le detail par echelle de
blocs), ``peak_bpm`` et ``frac_in_band`` du pouls, et sa correlation avec la
moyenne spatiale brute (``corr_with_raw_spatial_mean``, qui fixe aussi le
signe). Un ecart au-dela des tolerances signale que le CSV et les donnees ne se
correspondent plus (autre variante de masque, autre segmentation, autre version
de scikit-learn) : la condition est alors ignoree -- ou seulement signalee si
``STRICT_REBUILD_CHECK = False``.

Sur ce pouls, la suite est celle de ``compute_one_cycle_pixel_svd.py`` :
phase de Hilbert (``HilbertPhaseEstimator``) puis repliement
(``NCycleReconstructor``, qui replie ``registered_frames`` par phase). Seule
adaptation : le pouls du CSV vit sur les horodatages BRUTS (non uniformes) et la
transformee de Hilbert suppose un echantillonnage uniforme -- il est donc
interpole sur la grille uniforme de l'aligner avant Hilbert, exactement comme
``compute_phase_shift_pixel_svd.py`` le fait avant ses correlations croisees.

Arborescence lue :
    E:/NASA_Rigidity/SegmentationVariations/<variante>/
        phase_shift_pixel_svd/conditions.csv   <- POULS (poids + indices + HR)
        registered_frames/<NN_id>/<...>_rigidity/<..._OD|OS...>/cube.mp4
        registered_masks/<NN_id>/<...>_rigidity/<..._OD|OS...>/mask.npz
    E:/SANSORI/<NN_id>/<...>_rigidity/<..._OD|OS...>/
        RawImages/ (ou RawData/)     <- .tif + export .xml, pour les horodatages

Sorties (sous ``SEGVAR_ROOT/<variante>/whole_choroid_one_cycle/...``) :
  - ``one_cycle.mp4`` + ``one_cycle_params.json``
  - ``pixel_svd_diagnostics.npz`` -- memes cles que celles ecrites par
    ``compute_one_cycle_pixel_svd.py``, donc lisible tel quel par
    ``notebook/pixel_svd_diagnostics_viewer.ipynb`` (il suffit d'y pointer ce
    dossier de sortie), plus les cles de verification de la reconstruction.
  - ``whole_choroid_one_cycle_summary.csv`` (a la racine de ``SEGVAR_ROOT``),
    reecrit apres chaque condition : le script est interruptible et reprend ou
    il s'est arrete (``OVERWRITE`` pour tout refaire).
"""

from __future__ import annotations

import json
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ocularrigidity.motion.filters._1d import spatio_temporal_filter
from ocularrigidity.motion.projection._1d import project_into_separable_components
from ocularrigidity.motion.pulsation import (
    CardiacBand,
    HilbertPhaseConfig,
    HilbertPhaseEstimator,
    LombScargleConfig,
    LombScargleRateEstimator,
    NCycleConfig,
    NCycleReconstructor,
    OptimizedSpectralCombination,
    PixelTraceConfig,
    PixelTraceSource,
    PulseExtractor,
    Traces,
)
from ocularrigidity.motion.pulsation.rate import lomb_scargle_power
from ocularrigidity.motion.pulsation.traces import AbstractTraceSource
from ocularrigidity.motion.video_timeline_aligner import VideoTimelineAligner
from ocularrigidity.scripts.one_cycle.astronauts import _prepared_registrator
from ocularrigidity.scripts.registration.astronauts import (
    load_ordered_oct_series,
    write_gray_mp4,
)

# --------------------------------------------------------------------------- #
# Parametres
# --------------------------------------------------------------------------- #
PATH_GENERAL = Path("E:/SANSORI")  # arborescence brute (horodatages)
SEGVAR_ROOT = Path("E:/NASA_Rigidity/SegmentationVariations")
MASK_VARIANT = "model1_scale_1.0"
FRAMES_SUBDIR = "registered_frames"
MASKS_SUBDIR = "registered_masks"
PULSE_SUBDIR = "phase_shift_pixel_svd"  # <- d'ou viennent les pouls
OUTPUT_SUBDIR = "whole_choroid_one_cycle"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OVERWRITE = False  # True = retraiter les conditions deja presentes dans le CSV

# --- Chaine amont, A GARDER IDENTIQUE a compute_phase_shift_pixel_svd.py ----
# Ces valeurs ne sont pas des reglages : elles doivent reproduire la matrice de
# traces sur laquelle la SVD du CSV a ete calculee, sans quoi les
# `selected_indices` / `weights` enregistres n'indexent plus la meme base. Le
# controle `n_traces` (global et par echelle) est la pour attraper un ecart.
# `k_svd`, la HR et la bande sont relus dans le CSV, pas fixes ici.
COL_FRAC = (0.0, 1.0)
ROW_FRAC = (0.0, 1.0)
BLOCK_SIZES = (1, 2, 3, 4, 5)
DIAG_BPM_RANGE = (20.0, 240.0)
RANDOM_STATE = 0

# --- Verification de la reconstruction du pouls ----------------------------
STRICT_REBUILD_CHECK = True  # False = traiter quand meme, en signalant l'ecart
TOL_FRAC_IN_BAND = 0.02  # ecart absolu tolere sur `frac_in_band`
TOL_PEAK_BPM = 1.0  # ecart tolere sur le pic du pouls (BPM)
TOL_CORR_RAW = 0.05  # ecart tolere sur `corr_with_raw_spatial_mean`
# `objective` et `svd_var_frac_top1` sont les deux verifications qui portent sur
# la BASE elle-meme et pas seulement sur l'allure du pouls : la premiere est la
# valeur que l'optimiseur avait atteinte avec ces poids-la sur ces composantes-la
# (a 1e-6 pres, les poids etant ecrits a 6 chiffres significatifs), la seconde
# la part de variance du premier vecteur singulier. Deux jeux de donnees
# differents peuvent donner un pouls d'allure identique ; ils ne donnent pas la
# meme SVD.
TOL_OBJECTIVE_REL = 0.02  # ecart RELATIF tolere sur `objective`
TOL_VAR_FRAC = 0.01  # ecart absolu tolere sur `svd_var_frac_top1`

# --- Phase de Hilbert sur le pouls -----------------------------------------
HILBERT_CONFIG = HilbertPhaseConfig()  # pas de lissage additionnel par defaut
# La combinaison optimisee concentre deja la puissance dans la bande cardiaque,
# ce qui tient lieu du passe-bande que `HilbertPhaseEstimator` suppose en amont
# (hypothese de compute_one_cycle_pixel_svd.py, reprise telle quelle). True
# applique en plus le passe-bande FIR de la chaine (memes cutoffs que
# `AbstractUniformTraceSource.filtered_signal`, bande +/- BAND_FRAC x HR) avant
# Hilbert -- c'est le meme filtre que `bandpass_pulse` du script de dephasage.
BANDPASS_BEFORE_HILBERT = False
BAND_FRAC = 0.2  # demi-largeur relative de la bande etroite autour de la HR

# --- Repliement one-cycle (valeurs de compute_one_cycle_pixel_svd.py) ------
N_BINS = 10
N_CYCLE = 3
TARGET_FRAMES_PER_BIN = 25
FOLD_METHOD = "median"
OUTPUT_FPS = 30

PULSE_CSV = SEGVAR_ROOT / MASK_VARIANT / PULSE_SUBDIR / "conditions.csv"
SUMMARY_CSV = SEGVAR_ROOT / "whole_choroid_one_cycle_summary.csv"


# --------------------------------------------------------------------------- #
# Lecture des pouls deja calcules
# --------------------------------------------------------------------------- #
def parse_list(value, dtype=float) -> np.ndarray:
    """Vecteur ecrit par ``_fmt_list`` (';' entre les valeurs) -> tableau."""
    text = str(value).strip()
    if not text or text in ("<NA>", "nan"):
        return np.empty(0, dtype=dtype)
    return np.array([dtype(float(v)) for v in text.split(";")], dtype=dtype)


def load_pulses() -> pd.DataFrame:
    """Lignes ``status == "ok"`` de ``phase_shift_pixel_svd/conditions.csv``."""
    if not PULSE_CSV.exists():
        raise FileNotFoundError(
            f"{PULSE_CSV} introuvable : lancer d'abord "
            "compute_phase_shift_pixel_svd.py (c'est lui qui calcule les pouls)."
        )
    df = pd.read_csv(PULSE_CSV, keep_default_na=False, na_values=["<NA>", ""])
    n_all = len(df)
    df = df[df["status"] == "ok"].copy()
    if n_all != len(df):
        print(
            f"{n_all - len(df)} condition(s) sans pouls exploitable "
            f"(status != 'ok') ignoree(s) sur {n_all}."
        )
    return df


def condition_dir(row) -> Path:
    """Dossier SANSORI brut de la condition (colonne ``path``, ou reconstruit)."""
    path = Path(str(row["path"]))
    if path.is_dir():
        return path
    return PATH_GENERAL / row["patient"] / row["moment"] / row["condition"]


def find_raw_dir(condition_dir: Path) -> Path | None:
    """Sous-dossier contenant les .tif bruts + l'export XML Spectralis."""
    for name in ("RawImages", "RawData"):
        d = condition_dir / name
        if d.is_dir():
            return d
    return None


def raw_timestamps_us(raw_dir: Path) -> np.ndarray:
    """Horodatages bruts (us), un par frame, MEME ORDRE que les masques/frames
    de ``SegmentationVariations``."""
    series = load_ordered_oct_series(raw_dir)
    return np.array(
        [int(round(s.acquisition_time.seconds_of_day * 1e6)) for s in series],
        dtype=np.int64,
    )


# --------------------------------------------------------------------------- #
# Reconstruction du pouls a partir des poids du CSV
# --------------------------------------------------------------------------- #
class PrecomputedTraceSource(AbstractTraceSource):
    """Expose des ``Traces`` deja construites, pour les donner a un
    ``PulseExtractor`` (qui n'a besoin que de leur contrat)."""

    def __init__(self, traces: Traces):
        super().__init__()
        self._traces = traces

    def compute(self) -> Traces:  # pragma: no cover - jamais appele
        return self._traces


def spectrum(y: np.ndarray, t: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Lomb-Scargle d'une trace unique (centree), sur les horodatages reels.
    Meme fonction que ``compute_phase_shift_pixel_svd.spectrum``."""
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 8:
        return np.zeros_like(freqs)
    return lomb_scargle_power(t[ok], y[ok] - y[ok].mean(), freqs)


def rebuild_pulse(registrator, aligner, row) -> dict:
    """Refait la chaine amont, puis applique les poids du CSV.

    Retourne le pouls sur les horodatages BRUTS, les composantes SVD et leurs
    diagnostics spectraux, et les ecarts de la verification.
    """
    hr = float(row["hr_bpm"])
    selected = parse_list(row["selected_indices"], dtype=int)
    weights = parse_list(row["weights"], dtype=float)
    if selected.size != weights.size or selected.size == 0:
        raise ValueError(
            f"CSV incoherent : {selected.size} indices pour {weights.size} poids."
        )

    t = aligner.timestamps_seconds
    diag_band = CardiacBand(bpm_range=DIAG_BPM_RANGE)

    # --- 1. Traces pixel, puis normalisation PAR FRAME --------------------
    # Copie conforme de compute_phase_shift_pixel_svd.py (dtypes compris : la
    # SVD n'est reproductible qu'a entree bit-a-bit identique).
    source = PixelTraceSource(
        registrator,
        aligner,
        PixelTraceConfig(
            band=diag_band,
            col_frac=COL_FRAC,
            row_frac=ROW_FRAC,
            block_sizes=BLOCK_SIZES,
            verbose=False,
        ),
    )
    sig0 = source.raw_signal().astype(np.float32)  # (T, N)
    sig0n = (sig0 / np.nanmean(sig0, axis=1, keepdims=True)).astype(np.float32)
    glob_raw = np.nanmean(sig0, axis=1)  # moyenne spatiale BRUTE (reference)
    scale = source.scale_of_trace
    base_roi = source.base_roi
    del sig0

    # --- 2. Meme base que le CSV ? ----------------------------------------
    n_traces_csv = int(row["n_traces"])
    per_block_csv = parse_list(row["n_traces_per_block"], dtype=int)
    per_block = np.array([int((scale == b).sum()) for b in BLOCK_SIZES])
    if sig0n.shape[1] != n_traces_csv or (
        per_block_csv.size == per_block.size and not np.array_equal(per_block, per_block_csv)
    ):
        raise ValueError(
            f"traces reconstruites ({sig0n.shape[1]} : "
            f"{'+'.join(map(str, per_block))}) != CSV ({n_traces_csv} : "
            f"{'+'.join(map(str, per_block_csv))}) ; les poids enregistres "
            "n'indexent pas cette base."
        )

    # --- 3. SVD (deterministe a random_state fixe) ------------------------
    X_all = np.nan_to_num(sig0n, nan=0.0)
    del sig0n
    k = int(row["k_svd"])
    if k > min(X_all.shape):
        raise ValueError(f"k_svd = {k} > min(shape) = {min(X_all.shape)}")
    U, V = project_into_separable_components(
        X_all,
        method="svd",
        n_components=k,
        normalize=False,  # True = centrerait chaque colonne avant la SVD
        random_state=RANDOM_STATE,
    )
    del X_all
    if selected.max() >= k:
        raise ValueError(f"indice de composante {selected.max()} >= k_svd = {k}")

    # --- 4. LE POULS : combinaison enregistree ----------------------------
    pulse_raw = np.asarray(U[:, selected] @ weights, dtype=float)  # sur t
    pattern = np.asarray(V[:, selected] @ weights, dtype=np.float32)

    # --- 5. Diagnostics spectraux (memes objets que le script de dephasage) -
    traces_svd = Traces(
        values=np.asarray(U, dtype=np.float64),
        uniform_time=t,
        kept_mask=np.ones(t.size, dtype=bool),
        gap_mask=np.zeros(t.size, dtype=bool),
        timestamps_seconds=t,
        mixing=None,  # (N, k) : trop lourd a garder, `pattern` suffit
    )
    rate_estimator = LombScargleRateEstimator(
        LombScargleConfig(band=diag_band, verbose=False), override_bpm=hr
    )
    rate = rate_estimator.estimate(traces_svd)
    freqs = np.asarray(rate.diagnostics["freqs"], dtype=float)

    # --- 6. Verification contre le CSV ------------------------------------
    power = spectrum(pulse_raw, t, freqs)
    in_band = np.abs(freqs * 60.0 - hr) <= BAND_FRAC * hr
    peak_bpm = float(freqs[power.argmax()] * 60.0)
    frac_in_band = float(power[in_band].sum() / (power.sum() + 1e-12))
    corr_raw = float(np.corrcoef(pulse_raw, glob_raw)[0, 1])

    singular_values = np.linalg.norm(U, axis=0)  # U porte deja S
    var_frac_top1 = float(
        singular_values[0] ** 2 / (np.sum(singular_values**2) + 1e-30)
    )

    checks = {
        "peak_bpm": (peak_bpm, float(row["peak_bpm"]), TOL_PEAK_BPM),
        "frac_in_band": (frac_in_band, float(row["frac_in_band"]), TOL_FRAC_IN_BAND),
        "corr_with_raw_spatial_mean": (
            corr_raw,
            float(row["corr_with_raw_spatial_mean"]),
            TOL_CORR_RAW,
        ),
        "svd_var_frac_top1": (
            var_frac_top1,
            float(row["svd_var_frac_top1"]),
            TOL_VAR_FRAC,
        ),
    }

    # L'objectif atteint par l'optimiseur : recalculable seulement si l'energie
    # avait ete evaluee en UNE fenetre (sinon il faudrait `window_n_cycles`, que
    # le CSV n'enregistre pas -- cf. `_window_slices`).
    objective_csv = float(row["objective"])
    objective = np.nan
    if int(row["n_windows"]) <= 1:
        tol_hz = float(row["accept_tol_bpm"]) / 60.0
        in_band_opt = np.abs(freqs - hr / 60.0) <= tol_hz
        if not in_band_opt.any():
            in_band_opt = np.zeros_like(in_band_opt)
            in_band_opt[int(np.argmin(np.abs(freqs - hr / 60.0)))] = True
        objective = float(
            OptimizedSpectralCombination._window_objective(
                pulse_raw, t, freqs, in_band_opt
            )
        )
        checks["objective"] = (
            objective,
            objective_csv,
            TOL_OBJECTIVE_REL * abs(objective_csv),
        )

    failed = [
        f"{name} : {got:.4g} vs CSV {ref:.4g} (tolerance {tol:.3g})"
        for name, (got, ref, tol) in checks.items()
        if not np.isfinite(ref) or abs(got - ref) > tol
    ]

    return {
        "hr": hr,
        "selected": selected,
        "weights": weights,
        "objective": objective_csv,
        "pulse_raw": pulse_raw,
        "pattern": pattern,
        "glob_raw": glob_raw,
        "traces_svd": traces_svd,
        "rate": rate,
        "rate_estimator": rate_estimator,
        "freqs": freqs,
        "in_band": in_band,
        "pulse_power": power,
        "base_roi": base_roi,
        "scale_of_trace": scale,
        "n_traces": int(n_traces_csv),
        "k_svd": k,
        "check_peak_bpm": peak_bpm,
        "check_frac_in_band": frac_in_band,
        "check_corr_raw": corr_raw,
        "check_var_frac_top1": var_frac_top1,
        "check_objective": objective,
        "rebuild_failed": failed,
    }


# --------------------------------------------------------------------------- #
# Phase de Hilbert sur le pouls repris
# --------------------------------------------------------------------------- #
def bandpass_uniform(y: np.ndarray, hr: float, fs: float) -> np.ndarray:
    """Passe-bande cardiaque d'une trace vivant sur la grille UNIFORME.

    Memes cutoffs que ``AbstractUniformTraceSource.filtered_signal`` (et que
    ``compute_phase_shift_pixel_svd.bandpass_pulse``) ; les NaN (echantillons
    dans un trou) sont interpoles par le filtre puis restitues.
    """
    nyq = 0.5 * fs
    lo_bpm, hi_bpm = CardiacBand(
        expected_bpm=hr, expected_bpm_band_frac=BAND_FRAC
    ).effective_bpm_range
    return spatio_temporal_filter(
        np.asarray(y, dtype=float)[:, None],
        spatial_sigma=0.0,
        temporal_low_freq=(lo_bpm / 60.0) / nyq,
        temporal_high_freq=min((hi_bpm / 60.0) / nyq, 0.99),
        fs=fs,
        validity_mask=None,
    )[:, 0]


def hilbert_extractor(registrator, aligner, pulse_data: dict) -> PulseExtractor:
    """Pouls (horodatages bruts) -> grille uniforme -> phase de Hilbert.

    ``_rate`` et ``_phase`` sont amorces avec les resultats deja calcules :
    ``NCycleReconstructor`` les reutilise tels quels, rien n'est re-estime.
    """
    t = aligner.timestamps_seconds
    u_time = aligner.uniform_time
    # Aucune frame n'est "mauvaise" au sens des traces pixel (la ROI est
    # l'intersection temporelle du masque : toute trace est valide sur toute
    # frame), donc les seuls trous sont les trous de TEMPS.
    gap_mask = aligner.gap_mask(np.zeros(t.size, dtype=bool))
    kept = ~gap_mask

    pulse_uniform = np.interp(u_time, t, pulse_data["pulse_raw"])
    if BANDPASS_BEFORE_HILBERT:
        pulse_uniform = bandpass_uniform(
            pulse_uniform, pulse_data["hr"], float(aligner.fs)
        )
    pulse_uniform[gap_mask] = np.nan

    traces = Traces(
        values=pulse_uniform[kept][:, None],
        uniform_time=u_time,
        kept_mask=kept,
        gap_mask=gap_mask,
        timestamps_seconds=t,
        mixing=None,
    )
    rate = pulse_data["rate"]

    phase_estimator = HilbertPhaseEstimator(HILBERT_CONFIG)
    phase_uniform, good_uniform = phase_estimator.phase_from_trace(
        pulse_uniform, traces, rate
    )

    extractor = PulseExtractor(
        trace_source=PrecomputedTraceSource(traces),
        phase_estimator=phase_estimator,
        rate_estimator=pulse_data["rate_estimator"],
        registered_video=registrator,
        aligner=aligner,
    )
    extractor._rate = rate
    extractor._phase = phase_estimator.build_track(
        phase_uniform, good_uniform, traces, rate
    )
    pulse_data["pulse_uniform"] = pulse_uniform
    return extractor


# --------------------------------------------------------------------------- #
# Donnees de validation (aucune figure ici -- cf. notebook compagnon)
# --------------------------------------------------------------------------- #
def save_diagnostics(
    out_path: Path,
    *,
    astro: str,
    moment: str,
    condition: str,
    registrator,
    extractor: PulseExtractor,
    pulse_data: dict,
    row,
) -> None:
    """Memes cles que ``compute_one_cycle_pixel_svd.py`` (donc lisible par
    ``notebook/pixel_svd_diagnostics_viewer.ipynb``), plus les cles de
    verification de la reconstruction.

    Deux nuances de lecture par rapport a ce script-la :
      - ``svd_time_values`` / ``svd_uniform_time`` vivent sur les horodatages
        BRUTS (c'est la ou la SVD du CSV a ete calculee), pas sur la grille
        uniforme ; les noms sont conserves pour le viewer.
      - ``svd_mixing`` n'est pas enregistre (N_traces x k : plusieurs dizaines
        de Mo par condition avec la choroide entiere). Seul le motif spatial de
        la combinaison retenue l'est (``spatial_pattern``, une valeur par
        trace) ; il se remet en image via ``roi_selected`` + ``scale_of_trace``
        + ``block_sizes``.
    """
    traces_svd = pulse_data["traces_svd"]
    rate = pulse_data["rate"]
    phase = extractor.phase
    freqs = pulse_data["freqs"]

    np.savez_compressed(
        out_path,
        astro=astro,
        moment=moment,
        condition=condition,
        mask_variant=MASK_VARIANT,
        # -- provenance du pouls --
        pulse_csv=str(PULSE_CSV),
        hr_bpm_visit_data=float(row["hr_visit_data"]),
        hr_bpm_used=pulse_data["hr"],
        hr_source=str(row["hr_source"]),
        # -- A. masque de selection des pixels, sur une image moyennee --
        mean_frame=registrator.registered_frames.mean(axis=0).astype(np.float32),
        roi_full_intersection=registrator.registered_masks.astype(bool).all(axis=0),
        roi_selected=pulse_data["base_roi"],
        scale_of_trace=pulse_data["scale_of_trace"],
        block_sizes=np.asarray(BLOCK_SIZES),
        col_frac=np.asarray(COL_FRAC),
        row_frac=np.asarray(ROW_FRAC),
        n_traces=pulse_data["n_traces"],
        # -- B. vecteurs singuliers (SVD), domaine temporel + frequentiel --
        svd_uniform_time=traces_svd.time.astype(np.float32),
        svd_time_values=traces_svd.values.astype(np.float32),
        freqs=freqs.astype(np.float32),
        svd_power=np.asarray(rate.diagnostics["power"], dtype=np.float32),
        svd_peak_freq=np.asarray(rate.diagnostics["peak_freq"], dtype=np.float32),
        svd_concentration=np.asarray(
            rate.diagnostics["concentration"], dtype=np.float32
        ),
        svd_quality=np.asarray(rate.diagnostics["quality"], dtype=np.float32),
        # -- C. combinaison REPRISE du CSV (pas re-optimisee ici) --
        selected_indices=pulse_data["selected"],
        weights=pulse_data["weights"],
        objective=pulse_data["objective"],
        spatial_pattern=pulse_data["pattern"],
        combined_raw_time=pulse_data["pulse_raw"].astype(np.float32),
        combined_uniform=pulse_data["pulse_uniform"].astype(np.float32),
        combined_power=pulse_data["pulse_power"].astype(np.float32),
        in_band_mask=pulse_data["in_band"],
        accept_tol_bpm=BAND_FRAC * pulse_data["hr"],
        roi_mean_per_frame=pulse_data["glob_raw"].astype(np.float32),
        cardiac_freq_hz=rate.freq,
        cardiac_bpm=rate.freq * 60.0,
        confidence=extractor.confidence,
        # -- D. verification de la reconstruction --
        check_peak_bpm=pulse_data["check_peak_bpm"],
        check_frac_in_band=pulse_data["check_frac_in_band"],
        check_corr_raw=pulse_data["check_corr_raw"],
        check_var_frac_top1=pulse_data["check_var_frac_top1"],
        check_objective=pulse_data["check_objective"],
        csv_peak_bpm=float(row["peak_bpm"]),
        csv_frac_in_band=float(row["frac_in_band"]),
        csv_corr_raw=float(row["corr_with_raw_spatial_mean"]),
        csv_var_frac_top1=float(row["svd_var_frac_top1"]),
        csv_objective=pulse_data["objective"],
        rebuild_failed=np.asarray(pulse_data["rebuild_failed"], dtype=object).astype(
            str
        ),
        # -- E. phase obtenue par Hilbert --
        bandpass_before_hilbert=BANDPASS_BEFORE_HILBERT,
        uniform_time=phase.uniform_time.astype(np.float32),
        phase_uniform=phase.phase_uniform.astype(np.float32),
        good_uniform=phase.good_uniform,
        timestamps_seconds=extractor.timestamps_seconds.astype(np.float32),
        phase_per_frame=phase.phase_per_frame.astype(np.float32),
        good_per_frame=phase.good_per_frame,
        inst_bpm=phase.inst_bpm.astype(np.float32),
        gap_fraction=extractor.gap_fraction,
    )


# --------------------------------------------------------------------------- #
# Traitement d'une condition
# --------------------------------------------------------------------------- #
def process_condition(row) -> dict | None:
    astro, moment, condition = row["patient"], row["moment"], row["condition"]
    variant_root = SEGVAR_ROOT / MASK_VARIANT
    mask_path = variant_root / MASKS_SUBDIR / astro / moment / condition / "mask.npz"
    frames_path = variant_root / FRAMES_SUBDIR / astro / moment / condition / "cube.mp4"
    if not mask_path.exists() or not frames_path.exists():
        print(f"  [skip] cube.mp4/mask.npz absent ({MASK_VARIANT})")
        return None

    out_dir = variant_root / OUTPUT_SUBDIR / astro / moment / condition
    out_video = out_dir / "one_cycle.mp4"
    if out_video.exists() and not OVERWRITE:
        print(f"  [skip] deja traite : {out_video}")
        return None

    path_condi = condition_dir(row)
    raw_dir = find_raw_dir(path_condi)
    if raw_dir is None:
        print(f"  [skip] RawImages/RawData absent : {path_condi}")
        return None
    ts_us = raw_timestamps_us(raw_dir)
    if ts_us.size < 2:
        print(f"  [skip] pas assez d'horodatages ({ts_us.size})")
        return None

    registrator = _prepared_registrator(frames_path, mask_path, DEVICE, verbose=False)
    n_frames = registrator.registered_frames.shape[0]
    if n_frames != ts_us.size:
        print(f"  [skip] frames ({n_frames}) != horodatages ({ts_us.size})")
        return None
    if n_frames != int(row["n_frames"]):
        print(f"  [skip] frames ({n_frames}) != CSV ({int(row['n_frames'])})")
        return None

    aligner = VideoTimelineAligner(registrator, ts_us)

    # --- Pouls repris du CSV, verifie, puis phase de Hilbert --------------
    t0 = time.perf_counter()
    pulse_data = rebuild_pulse(registrator, aligner, row)
    t_pulse = time.perf_counter() - t0

    failed = pulse_data["rebuild_failed"]
    if failed:
        message = "reconstruction du pouls non conforme au CSV : " + " ; ".join(failed)
        if STRICT_REBUILD_CHECK:
            print(f"  [skip] {message}")
            return None
        print(f"  ATTENTION: {message}")

    extractor = hilbert_extractor(registrator, aligner, pulse_data)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_diagnostics(
        out_dir / "pixel_svd_diagnostics.npz",
        astro=astro,
        moment=moment,
        condition=condition,
        registrator=registrator,
        extractor=extractor,
        pulse_data=pulse_data,
        row=row,
    )

    # --- Repliement one-cycle ---------------------------------------------
    reconstructor = NCycleReconstructor(
        extractor,
        NCycleConfig(
            n_bins=N_BINS,
            n_cycle=N_CYCLE,
            target_frames_per_bin=TARGET_FRAMES_PER_BIN,
            fold_method=FOLD_METHOD,
            verbose=False,
        ),
    )
    cycles, _counts = reconstructor.compute()
    cube = np.clip(np.nan_to_num(np.asarray(cycles), nan=0.0), 0, 255).astype(np.uint8)
    write_gray_mp4(cube, out_video, OUTPUT_FPS)

    meta = {
        "mask_variant": MASK_VARIANT,
        "pulse_csv": str(PULSE_CSV),
        "hr_bpm_visit_data": float(row["hr_visit_data"]),
        "hr_bpm_used": pulse_data["hr"],
        "hr_source": str(row["hr_source"]),
        "col_frac": list(COL_FRAC),
        "row_frac": list(ROW_FRAC),
        "block_sizes": list(BLOCK_SIZES),
        "n_traces": pulse_data["n_traces"],
        "k_svd": pulse_data["k_svd"],
        "selected_indices": pulse_data["selected"].tolist(),
        "weights": pulse_data["weights"].tolist(),
        "objective": pulse_data["objective"],
        "rebuild_check": {
            "peak_bpm": [pulse_data["check_peak_bpm"], float(row["peak_bpm"])],
            "frac_in_band": [
                pulse_data["check_frac_in_band"],
                float(row["frac_in_band"]),
            ],
            "corr_with_raw_spatial_mean": [
                pulse_data["check_corr_raw"],
                float(row["corr_with_raw_spatial_mean"]),
            ],
            "svd_var_frac_top1": [
                pulse_data["check_var_frac_top1"],
                float(row["svd_var_frac_top1"]),
            ],
            "objective": [pulse_data["check_objective"], pulse_data["objective"]],
            "failed": failed,
        },
        "bandpass_before_hilbert": BANDPASS_BEFORE_HILBERT,
        "cardiac_bpm": float(extractor.cardiac_bpm),
        "confidence": extractor.confidence,
        "gap_fraction": float(extractor.gap_fraction),
        "n_bins": N_BINS,
        "n_cycle": N_CYCLE,
        "target_frames_per_bin": TARGET_FRAMES_PER_BIN,
        "fold_method": FOLD_METHOD,
        "output_fps": OUTPUT_FPS,
        "n_frames": int(cube.shape[0]),
        "notes": list(extractor.notes) + list(reconstructor.notes),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "one_cycle_params.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )

    return {
        "patient": astro,
        "moment": moment,
        "condition": condition,
        "path": str(path_condi),
        "hr_visit_data": float(row["hr_visit_data"]),
        "hr_used": pulse_data["hr"],
        "hr_source": str(row["hr_source"]),
        "cardiac_bpm": extractor.cardiac_bpm,
        "confidence": extractor.confidence,
        "n_traces": pulse_data["n_traces"],
        "k_svd": pulse_data["k_svd"],
        "n_selected": int(pulse_data["selected"].size),
        "objective": pulse_data["objective"],
        "peak_bpm": pulse_data["check_peak_bpm"],
        "peak_bpm_csv": float(row["peak_bpm"]),
        "frac_in_band": pulse_data["check_frac_in_band"],
        "frac_in_band_csv": float(row["frac_in_band"]),
        "corr_raw": pulse_data["check_corr_raw"],
        "corr_raw_csv": float(row["corr_with_raw_spatial_mean"]),
        "var_frac_top1": pulse_data["check_var_frac_top1"],
        "var_frac_top1_csv": float(row["svd_var_frac_top1"]),
        "objective_recomputed": pulse_data["check_objective"],
        "rebuild_ok": not failed,
        "gap_fraction": extractor.gap_fraction,
        "n_frames_cycle": int(cube.shape[0]),
        "t_pulse_s": t_pulse,
        "out_dir": str(out_dir),
    }


# --------------------------------------------------------------------------- #
# Boucle principale
# --------------------------------------------------------------------------- #
def _fmt(x):
    return f"{x:.4g}" if isinstance(x, (int, float)) and np.isfinite(x) else "<NA>"


def _load_existing() -> list[dict]:
    if OVERWRITE or not SUMMARY_CSV.exists():
        return []
    return pd.read_csv(
        SUMMARY_CSV, keep_default_na=False, na_values=["<NA>", ""]
    ).to_dict("records")


def main():
    pulses = load_pulses()
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = _load_existing()
    done = {(r["patient"], r["moment"], r["condition"]) for r in rows}
    if done:
        print(f"{len(done)} condition(s) deja traitee(s) : reprise.\n")

    n_new = 0
    for _, row in pulses.iterrows():
        key = (row["patient"], row["moment"], row["condition"])
        if key in done:
            continue
        print("/".join(key))
        t0 = time.perf_counter()
        try:
            result = process_condition(row)
        except Exception as e:  # noqa: BLE001
            print(f"  [erreur] {e}")
            traceback.print_exc()
            continue
        if result is None:
            continue

        rows.append(result)
        done.add(key)
        n_new += 1
        print(
            f"  -> FC = {_fmt(result['cardiac_bpm'])} BPM "
            f"({result['hr_source']}) · pouls repris : "
            f"{result['n_selected']} composantes sur {result['k_svd']} · "
            f"pic = {_fmt(result['peak_bpm'])} BPM (CSV {_fmt(result['peak_bpm_csv'])}) · "
            f"en bande = {result['frac_in_band']:.1%} "
            f"(CSV {result['frac_in_band_csv']:.1%}) · "
            f"gap = {result['gap_fraction']:.1%} · "
            f"{time.perf_counter() - t0:.0f} s"
        )

        # Reecriture apres chaque condition : le script est interruptible.
        pd.DataFrame(rows).to_csv(SUMMARY_CSV, index=False, na_rep="<NA>")

    if not rows:
        print("Aucune condition traitee : table CSV non ecrite.")
        return
    print(f"\n{n_new} nouvelle(s) condition(s), {len(rows)} au total : {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
