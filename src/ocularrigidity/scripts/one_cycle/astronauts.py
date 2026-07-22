"""
Generation d'une video "one-cycle" a partir d'une video OCT DEJA recalee.

Une video one-cycle est la reconstruction d'un (ou ``n_cycle``) battement(s)
cardiaque(s) moyen(s) : chaque frame recalee recoit une phase cardiaque, les
frames sont rangees par phase en ``n_bins`` casiers puis moyennees. Le SNR est
fortement ameliore, ce qui permet de mesurer proprement la pulsation
choroidienne (ΔY -> rigidite).

Entree : le dossier ``registered/`` produit par
``ocularrigidity.registration.export`` / ``Astronauts/register_files.py``
(``registered_video.mp4`` + ``mask.npz`` + ``timestamp.txt``). Cette video etant
DEJA recalee et rognee (skip/drop deja appliques), on la charge telle quelle
(aucun re-recalage : on renseigne directement les frames/masques recales de
``RegisteredVideo``, comme le fait son cache) et on la replie via
``PulseExtractor`` + ``NCycleReconstructor``.

Sortie : ``one_cycle.mp4`` (+ ``one_cycle_params.json``) dans le meme dossier.
"""

from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime

import numpy as np
import torch

from ocularrigidity.data.compression import read_gray
from ocularrigidity.data.io import load_mask
from ocularrigidity.registration.registration_engine import VideoRegistrator
from ocularrigidity.registration.config import RegistrationConfig
from ocularrigidity.motion.pulsation import (
    CardiacBand,
    DecompositionConfig,
    DecomposedTraceSource,
    IQDemodPhaseEstimator,
    IQPhaseConfig,
    LombScargleConfig,
    LombScargleRateEstimator,
    MaskThicknessTraceSource,
    MaskTraceConfig,
    NCycleConfig,
    NCycleReconstructor,
    PeakLockedPhaseEstimator,
    PulseExtractor,
    SelectBestComponent,
)
from ocularrigidity.motion.video_timeline_aligner import VideoTimelineAligner
from ocularrigidity.segmentation.postprocess.interfaces import (
    clean_boundaries,
    extract_boundaries_fast,
)
from ocularrigidity.scripts.registration.astronauts import write_gray_mp4

DEFAULT_ONE_CYCLE_NAME = "one_cycle.mp4"


def _prepared_registrator(
    video_path: Path, mask_path: Path, device: str, verbose: bool
) -> VideoRegistrator:
    """RegisteredVideo dont les frames/masques recales sont pre-charges.

    La video est deja recalee : on renseigne directement ``_registered_*`` (comme
    ``RegisteredVideo._load_from_cache``) pour NE PAS re-recaler ni re-echantillonner.
    """
    frames = read_gray(str(video_path))  # (T, H, W) uint8
    masks = np.asarray(load_mask(mask_path), dtype=bool)  # (T, H, W) bool
    if frames.shape[0] != masks.shape[0]:
        raise ValueError(
            f"video ({frames.shape[0]} frames) et masques ({masks.shape[0]}) "
            f"de tailles differentes : {video_path}"
        )

    reg = VideoRegistrator(
        video=Path(video_path).name,
        root_data=Path(video_path).parent,
        root_masks=Path(mask_path).parent,
        config=RegistrationConfig(
            skip_first_n_frames=0,
            drop_last_n_frames=0,
            correct_transversal=False,
            flatten_rpe=False,
        ),
        verbose=verbose,
        device=device,
        cache_dir=None,
    )
    reg._registered_frames = frames
    reg._registered_masks = masks
    # Frontieres (BM, CSI) -> thickness, exactement comme compute_registration.
    bm, csi = extract_boundaries_fast(masks)
    bm, csi = clean_boundaries(bm, csi)
    reg._registered_lines = torch.stack(
        [torch.tensor(bm), torch.tensor(csi)], dim=1
    ).cpu()
    T, _, W = frames.shape
    reg._transform = {
        "dx": np.zeros(T, np.float32),
        "dy": np.zeros((T, W), np.float32),
    }
    return reg


def export_one_cycle_video(
    registered_dir,
    *,
    suffix: str = "",
    output_name: str | None = None,
    overwrite: bool = False,
    device: str = "cuda",
    verbose: bool = True,
    # Prior physiologique
    expected_bpm: float | None = None,
    override_bpm: float | None = None,
    bpm_range: tuple = (30.0, 180.0),
    expected_bpm_band_frac: float = 0.3,
    # Extraction du signal cardiaque
    ICA_or_PCA: str = "ICA",
    sigma_col: float = 5.0,
    col_slice: tuple | None = (100, 924),
    n_separable_components: int = 16,
    phase_smoother_cycles: float = 2.0,
    harmonic_correction: bool = True,
    # Repliement (fold)
    phase_method_for_fold: str = "peak_locked",
    n_bins: int = 30,
    n_cycle: int = 3,
    target_frames_per_bin: int = 25,
    one_cycle_fold_method: str = "median",
    output_fps: int = 30,
    extra_meta: dict | None = None,
) -> dict:
    """Genere la video one-cycle a partir du dossier ``registered/``.

    Returns un dict de statut : ``{"status": "ok", "video", "n_frames",
    "cardiac_bpm", "confidence", ...}`` ou ``{"status": "skipped", "reason", ...}``.
    Leve sur erreur dure (ex. tous les chunks rejetes faute de frames valides).
    """
    registered_dir = Path(registered_dir)
    # ``suffix`` selectionne la variante de video recalee en entree :
    # "" -> registered_video.mp4 (sans A-scan) ; "_ascan" -> registered_video_ascan.mp4.
    # La sortie one-cycle et son JSON de parametres reprennent le meme suffixe.
    video_path = registered_dir / f"registered_video{suffix}.mp4"
    mask_path = registered_dir / f"mask{suffix}.npz"
    ts_path = registered_dir / f"timestamp{suffix}.txt"
    out_name = output_name or f"one_cycle{suffix}.mp4"
    out_path = registered_dir / out_name

    for p in (video_path, mask_path, ts_path):
        if not p.exists():
            return {
                "status": "skipped",
                "reason": f"manquant: {p.name}",
                "out_dir": registered_dir,
            }
    if out_path.exists() and not overwrite:
        return {
            "status": "skipped",
            "reason": "exists",
            "video": out_path,
            "out_dir": registered_dir,
        }

    reg = _prepared_registrator(video_path, mask_path, device, verbose)

    cslice = slice(col_slice[0], col_slice[1]) if col_slice is not None else None
    aligner = VideoTimelineAligner(reg, str(ts_path))

    # Un seul ``band``, partage par le filtre passe-bande des traces et la
    # recherche frequentielle : les desynchroniser est la seule erreur facile.
    band = CardiacBand(
        bpm_range=tuple(bpm_range),
        expected_bpm=expected_bpm,
        expected_bpm_band_frac=expected_bpm_band_frac,
    )

    # Le choix de la phase est desormais le choix d'un composant, pas une
    # chaine passee au repliement : un ``PulseExtractor`` *est* une methode.
    if phase_method_for_fold == "peak_locked":
        phase_estimator = PeakLockedPhaseEstimator(
            aggregator=SelectBestComponent(), band=band
        )
    elif phase_method_for_fold == "iq":
        phase_estimator = IQDemodPhaseEstimator(
            IQPhaseConfig(smoother_cycles=phase_smoother_cycles),
            aggregator=SelectBestComponent(),
        )
    else:
        raise ValueError(
            f"phase_method_for_fold inconnu: {phase_method_for_fold!r} "
            "(attendu 'peak_locked' ou 'iq')."
        )

    extractor = PulseExtractor(
        trace_source=DecomposedTraceSource(
            MaskThicknessTraceSource(
                reg,
                aligner,
                MaskTraceConfig(
                    band=band,
                    sigma_col=sigma_col,
                    col_slice=cslice,
                    verbose=verbose,
                ),
            ),
            DecompositionConfig(
                method=ICA_or_PCA, n_components=n_separable_components
            ),
        ),
        rate_estimator=LombScargleRateEstimator(
            LombScargleConfig(
                band=band,
                harmonic_correction=harmonic_correction,
                verbose=verbose,
            ),
            override_bpm=override_bpm,
        ),
        phase_estimator=phase_estimator,
    )

    reconstructor = NCycleReconstructor(
        extractor,
        NCycleConfig(
            n_bins=n_bins,
            n_cycle=n_cycle,
            target_frames_per_bin=target_frames_per_bin,
            fold_method=one_cycle_fold_method,
            verbose=verbose,
        ),
    )
    cycles, _counts = reconstructor.compute()

    # cycles est en float32 (moyenne/mediane) et peut contenir des NaN
    # (chunks rejetes) : on nettoie avant l'encodage uint8.
    cube = np.clip(np.nan_to_num(np.asarray(cycles), nan=0.0), 0, 255).astype(np.uint8)
    write_gray_mp4(cube, out_path, output_fps)

    meta = {
        "expected_bpm": expected_bpm,
        "override_bpm": override_bpm,
        "cardiac_bpm": float(extractor.cardiac_bpm),
        "confidence": extractor.confidence,
        "gap_fraction": float(extractor.gap_fraction),
        "bpm_range": list(bpm_range),
        "expected_bpm_band_frac": expected_bpm_band_frac,
        "ICA_or_PCA": ICA_or_PCA,
        "sigma_col": sigma_col,
        "col_slice": list(col_slice) if col_slice is not None else None,
        "n_separable_components": n_separable_components,
        "phase_smoother_cycles": phase_smoother_cycles,
        "harmonic_correction": harmonic_correction,
        "phase_method_for_fold": phase_method_for_fold,
        "n_bins": n_bins,
        "n_cycle": n_cycle,
        "target_frames_per_bin": target_frames_per_bin,
        "one_cycle_fold_method": one_cycle_fold_method,
        "output_fps": output_fps,
        "n_frames": int(cube.shape[0]),
        "notes": list(extractor.notes) + list(reconstructor.notes),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    if extra_meta:
        meta.update(extra_meta)
    (registered_dir / f"one_cycle_params{suffix}.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )

    return {
        "status": "ok",
        "video": out_path,
        "out_dir": registered_dir,
        "n_frames": int(cube.shape[0]),
        "cardiac_bpm": float(extractor.cardiac_bpm),
        "confidence": extractor.confidence,
        "n_bins": n_bins,
        "n_cycle": n_cycle,
    }
