"""
compute_one_cycle_pixel_svd.py

Repliement "one-cycle" (motion.pulsation) a partir des donnees DEJA recalees
sous ``E:/NASA_Rigidity/SegmentationVariations/<variante>/`` (video recalee
``registered_frames/.../cube.mp4`` + masque recale ``registered_masks/.../mask.npz``),
en utilisant une chaine d'extraction du pouls DIFFERENTE de celle des autres
scripts Astronauts (``compute_rigidity_*``, qui partent d'un signal d'epaisseur
choroidienne par A-scan) :

  1. Traces PIXEL BRUTES, multiresolution (``PixelTraceSource`` /
     ``motion.pulsation.traces.pixel``) : intensite video recalee (pas
     l'epaisseur) sur la ROI = intersection du masque dans le temps, restreinte
     au tiers central des A-scans x tiers superieur de la choroide, plus une
     pyramide de super-pixels (blocs 1x1 .. 5x5) concatenee en une seule
     matrice (T, N_total).
  2. SVD de CETTE matrice, SANS passe-bande ET SANS normalisation des donnees
     (``DecompositionConfig(method="svd", normalize=False)``) : la SVD voit
     l'intensite pixel telle quelle. (``DecomposedTraceSource`` decompose deja
     ``interpolated_signal`` -- pas ``filtered_signal`` -- donc le passe-bande
     est evite structurellement ; ``normalize=False`` desactive en plus le
     centrage par colonne.)
  3. Optimisation des vecteurs singuliers pour l'agregation
     (``OptimizedSpectralCombination`` : combinaison lineaire, norme unite,
     des composantes SVD dont le pic Lomb-Scargle est proche de la frequence
     cardiaque, maximisant la puissance dans la bande cardiaque).
  4. Phase de Hilbert (``HilbertPhaseEstimator``) sur CE signal agrege.

Le repliement lui-meme (``NCycleReconstructor``) est inchange : il replie
``registered_frames`` par phase, quelle que soit la methode qui l'a produite.

Aucune figure n'est tracee ici : chaque condition ecrit
``pixel_svd_diagnostics.npz`` (masques de selection, vecteurs singuliers en
domaine temporel ET frequentiel, combinaison optimisee + bande d'optimisation
+ pic de frequence cardiaque, phase de Hilbert) a cote de la video one-cycle
produite -- a visualiser depuis le notebook compagnon
``pixel_svd_diagnostics_viewer.ipynb`` (meme dossier). Une table CSV
recapitulative est ecrite a la fin.

Arborescence lue (voir ``compute_rigidity_compare_mask_model.py``, meme
jeu de donnees) :
    E:/NASA_Rigidity/SegmentationVariations/<variante>/
        registered_frames/<NN_id>/<...>_rigidity/<..._OD|OS...>/cube.mp4
        registered_masks/<NN_id>/<...>_rigidity/<..._OD|OS...>/mask.npz
    E:/SANSORI/<NN_id>/<...>_rigidity/<..._OD|OS...>/
        RawImages/ (ou RawData/)     <- .tif + export .xml, pour les horodatages
        Data Files/visit_data.csv    <- HR (prior physiologique, optionnel)

Les horodatages ne sont pas relus depuis ``RawImages/registered/timestamp.txt``
(absent de ``SegmentationVariations`` -- seuls ``mask.npz``/``transform.npz``/
``cube.mp4`` y sont ecrits) : ils sont recalcules depuis l'export XML
Spectralis, comme dans ``compute_rigidity_compare_mask_model.py``.

Sorties (sous ``SEGVAR_ROOT/<variante>/pixel_svd_one_cycle/...``) :
  - ``one_cycle.mp4`` + ``one_cycle_params.json``
  - ``pixel_svd_diagnostics.npz``
  - ``pixel_svd_one_cycle_summary.csv`` (a la racine de ``SEGVAR_ROOT``)
"""

from __future__ import annotations

import csv
import json
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ocularrigidity.motion.pulsation import (
    CardiacBand,
    DecompositionConfig,
    DecomposedTraceSource,
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
    SpectralCombinationConfig,
)
from ocularrigidity.motion.pulsation.rate import lomb_scargle_power
from ocularrigidity.motion.video_timeline_aligner import VideoTimelineAligner
from ocularrigidity.scripts.one_cycle.astronauts import _prepared_registrator
from ocularrigidity.scripts.registration.astronauts import (
    load_ordered_oct_series,
    write_gray_mp4,
)

# --------------------------------------------------------------------------- #
# Parametres
# --------------------------------------------------------------------------- #
PATH_GENERAL = Path("E:/SANSORI")  # arborescence brute (horodatages, HR)
SEGVAR_ROOT = Path("E:/NASA_Rigidity/SegmentationVariations")
MASK_VARIANT = "model1_scale_1.0"
FRAMES_SUBDIR = "registered_frames"
MASKS_SUBDIR = "registered_masks"
OUTPUT_SUBDIR = "pixel_svd_one_cycle"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OVERWRITE = False

# --- Prior physiologique (bande cardiaque) ---
BPM_RANGE = (30.0, 180.0)
BAND_FRAC = (0.1)  # 1 - Fraction de la bande cardiaque attendue

# --- 1) Traces pixel, multiresolution (defauts de PixelTraceConfig, rendus
# explicites ici pour pouvoir les ajuster sans aller dans le package) ---
COL_FRAC = (1 / 3, 2 / 3)
ROW_FRAC = (0.0, 1 / 3)
BLOCK_SIZES = (1, 2, 3, 4, 5)

# --- 2) SVD, sans passe-bande (structurel, cf. DecomposedTraceSource) ni
# normalisation des donnees ---
N_SVD_COMPONENTS = 20

# --- 3) Optimisation des vecteurs singuliers ---
SPECTRAL_COMBINATION_CONFIG = SpectralCombinationConfig(max_candidates=N_SVD_COMPONENTS)

# --- 4) Phase de Hilbert sur le signal agrege ---
HILBERT_CONFIG = HilbertPhaseConfig()  # pas de lissage additionnel par defaut

# --- Repliement one-cycle ---
N_BINS = 30
N_CYCLE = 3
TARGET_FRAMES_PER_BIN = 25
FOLD_METHOD = "median"
OUTPUT_FPS = 30

SUMMARY_CSV = SEGVAR_ROOT / "pixel_svd_one_cycle_summary.csv"


# --------------------------------------------------------------------------- #
# Resolution des chemins (arborescence SANSORI, cf. compute_rigidity_*.py)
# --------------------------------------------------------------------------- #
def find_raw_dir(condition_dir: Path) -> Path | None:
    """Sous-dossier contenant les .tif bruts + l'export XML Spectralis."""
    for name in ("RawImages", "RawData"):
        d = condition_dir / name
        if d.is_dir():
            return d
    return None


def raw_timestamps_us(raw_dir: Path) -> np.ndarray:
    """Horodatages bruts (us), un par frame, MEME ORDRE que les masques/frames
    de ``SegmentationVariations`` (skip=drop=0 -> toutes les frames brutes,
    triees par horodatage croissant comme ``build_cube_and_timestamps``)."""
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


# --------------------------------------------------------------------------- #
# Chaine d'extraction : pixel multiresolution -> SVD (brute) -> optimisation
# des vecteurs singuliers -> phase de Hilbert
# --------------------------------------------------------------------------- #
def build_extractor(
    registrator, aligner, band: CardiacBand
) -> tuple[PulseExtractor, PixelTraceSource, OptimizedSpectralCombination]:
    pixel_source = PixelTraceSource(
        registrator,
        aligner,
        PixelTraceConfig(
            band=band,
            col_frac=COL_FRAC,
            row_frac=ROW_FRAC,
            block_sizes=BLOCK_SIZES,
            verbose=False,
        ),
    )
    svd_source = DecomposedTraceSource(
        pixel_source,
        DecompositionConfig(
            method="svd",
            n_components=N_SVD_COMPONENTS,
            normalize=False,  # SVD sur l'intensite pixel brute, non centree
            random_state=0,
        ),
    )
    rate_estimator = LombScargleRateEstimator(
        LombScargleConfig(band=band, harmonic_correction=True, verbose=False)
    )
    aggregator = OptimizedSpectralCombination(SPECTRAL_COMBINATION_CONFIG)
    phase_estimator = HilbertPhaseEstimator(HILBERT_CONFIG, aggregator=aggregator)

    extractor = PulseExtractor(
        trace_source=svd_source,
        phase_estimator=phase_estimator,
        rate_estimator=rate_estimator,
        registered_video=registrator,
        aligner=aligner,
    )
    return extractor, pixel_source, aggregator


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
    pixel_source: PixelTraceSource,
    extractor: PulseExtractor,
    aggregator: OptimizedSpectralCombination,
    combined_uniform: np.ndarray,
    hr_bpm: float,
) -> None:
    traces = extractor.traces
    rate = extractor.rate
    phase = extractor.phase
    last = aggregator.last_result

    freqs = np.asarray(rate.diagnostics["freqs"], dtype=float)
    tol_hz = SPECTRAL_COMBINATION_CONFIG.accept_tol_bpm / 60.0
    in_band_mask = np.abs(freqs - rate.freq) <= tol_hz

    combined_kept = combined_uniform[traces.kept_mask]
    combined_power = lomb_scargle_power(
        traces.time, combined_kept - np.nanmean(combined_kept), freqs
    )

    np.savez_compressed(
        out_path,
        astro=astro,
        moment=moment,
        condition=condition,
        mask_variant=MASK_VARIANT,
        # -- pouls mesure (Data Files/visit_data.csv), prior physiologique --
        hr_bpm_visit_data=hr_bpm,
        # -- A. masque de selection des pixels, sur une image moyennee --
        mean_frame=registrator.registered_frames.mean(axis=0).astype(np.float32),
        roi_full_intersection=registrator.registered_masks.astype(bool).all(axis=0),
        roi_selected=pixel_source.base_roi,
        scale_of_trace=pixel_source.scale_of_trace,
        block_sizes=np.asarray(BLOCK_SIZES),
        col_frac=np.asarray(COL_FRAC),
        row_frac=np.asarray(ROW_FRAC),
        # -- B. vecteurs singuliers (SVD), domaine temporel + frequentiel --
        svd_uniform_time=traces.time.astype(np.float32),
        svd_time_values=traces.values.astype(np.float32),
        svd_mixing=(
            traces.mixing.astype(np.float32)
            if traces.mixing is not None
            else np.empty((0, 0), dtype=np.float32)
        ),
        freqs=freqs.astype(np.float32),
        svd_power=np.asarray(rate.diagnostics["power"], dtype=np.float32),
        svd_peak_freq=np.asarray(rate.diagnostics["peak_freq"], dtype=np.float32),
        svd_concentration=np.asarray(rate.diagnostics["concentration"], dtype=np.float32),
        svd_quality=np.asarray(rate.diagnostics["quality"], dtype=np.float32),
        # -- C. agregation optimisee, domaine temporel + frequentiel, bande
        # d'optimisation et pic de frequence cardiaque --
        selected_indices=last.selected_indices,
        weights=last.weights,
        objective=last.objective,
        combined_uniform=combined_uniform.astype(np.float32),
        combined_power=combined_power.astype(np.float32),
        in_band_mask=in_band_mask,
        accept_tol_bpm=SPECTRAL_COMBINATION_CONFIG.accept_tol_bpm,
        cardiac_freq_hz=rate.freq,
        cardiac_bpm=rate.freq * 60.0,
        confidence=extractor.confidence,
        # -- D. phase obtenue par Hilbert --
        uniform_time=traces.uniform_time.astype(np.float32),
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
def process_condition(path_condi: Path) -> dict | None:
    astro, moment, condition = (
        path_condi.parent.parent.name,
        path_condi.parent.name,
        path_condi.name,
    )
    variant_root = SEGVAR_ROOT / MASK_VARIANT
    mask_path = variant_root / MASKS_SUBDIR / astro / moment / condition / "mask.npz"
    frames_path = variant_root / FRAMES_SUBDIR / astro / moment / condition / "cube.mp4"
    if not mask_path.exists() or not frames_path.exists():
        print(f"  [skip] cube.mp4/mask.npz absent ({MASK_VARIANT}) : {path_condi}")
        return None

    out_dir = variant_root / OUTPUT_SUBDIR / astro / moment / condition
    out_video = out_dir / "one_cycle.mp4"
    if out_video.exists() and not OVERWRITE:
        print(f"  [skip] deja traite : {out_video}")
        return None

    raw_dir = find_raw_dir(path_condi)
    if raw_dir is None:
        print(f"  [skip] RawImages/RawData absent : {path_condi}")
        return None
    try:
        ts_us = raw_timestamps_us(raw_dir)
    except Exception as e:  # noqa: BLE001
        print(f"  [skip] horodatages bruts : {e} : {path_condi}")
        return None
    if ts_us.size < 2:
        print(f"  [skip] pas assez d'horodatages ({ts_us.size}) : {path_condi}")
        return None

    hr = read_hr(path_condi)

    registrator = _prepared_registrator(frames_path, mask_path, DEVICE, verbose=False)
    n_frames = registrator.registered_frames.shape[0]
    if n_frames != ts_us.size:
        print(
            f"  [skip] frames ({n_frames}) != horodatages ({ts_us.size}) : {path_condi}"
        )
        return None

    aligner = VideoTimelineAligner(registrator, ts_us)
    band = CardiacBand(
        bpm_range=BPM_RANGE,
        expected_bpm_band_frac=BAND_FRAC,
        expected_bpm=float(hr) if np.isfinite(hr) else None,
    )

    extractor, pixel_source, aggregator = build_extractor(registrator, aligner, band)

    # Traces + rate d'abord : necessaires a l'agregation optimisee elle-meme.
    traces = extractor.traces
    rate = extractor.rate
    combined_uniform = aggregator.aggregate(traces, rate)

    # Phase de Hilbert sur CE signal agrege deja calcule -- on evite de
    # relancer l'optimisation (couteuse, SLSQP multi-depart) une 2e fois en
    # passant par ``extractor.phase`` : on construit la trace directement et
    # on prime le cache de l'extracteur avec, pour que ``NCycleReconstructor``
    # (``ex.phase_per_frame``/``ex.good_per_frame``) la reutilise telle quelle.
    phase_estimator = extractor.phase_estimator
    phase_u, good_u = phase_estimator.phase_from_trace(combined_uniform, traces, rate)
    extractor._phase = phase_estimator.build_track(phase_u, good_u, traces, rate)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_diagnostics(
        out_dir / "pixel_svd_diagnostics.npz",
        astro=astro,
        moment=moment,
        condition=condition,
        registrator=registrator,
        pixel_source=pixel_source,
        extractor=extractor,
        aggregator=aggregator,
        combined_uniform=combined_uniform,
        hr_bpm=hr,
    )

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

    last = aggregator.last_result
    meta = {
        "mask_variant": MASK_VARIANT,
        "hr_bpm": hr,
        "bpm_range": list(BPM_RANGE),
        "col_frac": list(COL_FRAC),
        "row_frac": list(ROW_FRAC),
        "block_sizes": list(BLOCK_SIZES),
        "n_svd_components": N_SVD_COMPONENTS,
        "n_candidates_selected": int(last.selected_indices.size),
        "objective": last.objective,
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
        "hr": hr,
        "cardiac_bpm": extractor.cardiac_bpm,
        "confidence": extractor.confidence,
        "n_candidates_selected": int(last.selected_indices.size),
        "objective": last.objective,
        "gap_fraction": extractor.gap_fraction,
        "n_frames_cycle": int(cube.shape[0]),
        "out_dir": str(out_dir),
    }


# --------------------------------------------------------------------------- #
# Boucle principale
# --------------------------------------------------------------------------- #
def _fmt(x):
    return f"{x:.4g}" if isinstance(x, (int, float)) and np.isfinite(x) else "<NA>"


def main():
    rows = []
    for path_astro in PATH_GENERAL.iterdir():
        if not path_astro.is_dir():
            continue
        for path_moment in path_astro.iterdir():
            if not path_moment.match("*rigidity"):
                continue
            for path_condi in path_moment.iterdir():
                if not path_condi.is_dir():
                    continue
                print(path_condi)
                try:
                    result = process_condition(path_condi)
                except Exception as e:  # noqa: BLE001
                    print(f"  [erreur] {path_condi} : {e}")
                    traceback.print_exc()
                    continue
                if result is None:
                    continue
                rows.append(result)
                print(
                    f"  -> FC = {_fmt(result['cardiac_bpm'])} BPM "
                    f"(confiance = {result['confidence']}) · "
                    f"{result['n_candidates_selected']} composantes SVD retenues "
                    f"(objectif = {_fmt(result['objective'])}) · "
                    f"gap = {result['gap_fraction']:.1%}"
                )

    if not rows:
        print("Aucune condition traitee : table CSV non ecrite.")
        return

    columns = [
        "patient", "moment", "condition", "path", "hr",
        "cardiac_bpm", "confidence", "n_candidates_selected", "objective",
        "gap_fraction", "n_frames_cycle", "out_dir",
    ]
    out = pd.DataFrame(rows)[columns]
    out.to_csv(SUMMARY_CSV, index=False, na_rep="<NA>")
    print(f"\n{len(out)} conditions ecrites dans {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
