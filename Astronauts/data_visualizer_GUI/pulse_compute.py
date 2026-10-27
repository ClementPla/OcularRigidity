# -*- coding: utf-8 -*-
"""
Pouls reconstruits a la volee, a partir d'une video recalee + son masque.

* Pouls des MASQUES : epaisseur moyenne de la choroide par frame (um), sur les
  colonnes ou le masque est present dans >= 90 % des frames.
* Pouls de la SVD (intensites) : SVD de rang 100 de traces prises dans le TIERS
  CENTRAL lateral (ROI_WIDTH_MM = 2,9 mm, centre de l'image), ROI_DEPTH_UM = 117 um
  par A-scan a partir du bord superieur du masque
  (RPE/BM) vers le bas, ce bord etant suivi frame par frame
  (``rpe_band_traces``) ; traces NORMALISEES PAR FRAME (chaque frame divisee par son
  intensite moyenne sur la ROI, ``PixelTraceSource.normalized_trace``) puis chaque trace
  centree-reduite (``motion.projection._1d.project_into_separable_components``),
  puis COMBINAISON OPTIMISEE des composantes par le pipeline du package
  (``motion.pulsation.phase.OptimizedSpectralCombination``, portage de
  ``detectPulseUsingSVD.m``) : candidates = composantes dont le pic tombe a
  HR +/- 10 % et assez concentre, poids ||w|| = 1 maximisant la puissance en
  bande sur 5 fenetres de 6 cycles. Memes reglages et meme echelle de seuils de
  concentration (0,15 -> 0) que ``Astronauts/compute_pulse_from_data.py``, a deux
  differences pres : la tolerance (+/- 10 % ici, +/- 20 % la-bas) et la region
  (ici la bande sous la RPE, la-bas un rectangle fixe).
  Le pouls n'a pas de signe propre : on le fixe pour qu'il soit correle
  POSITIVEMENT a l'intensite moyenne brute de ces pixels -- un critere
  independant du pouls des masques, sans quoi le dephasage entre les deux serait
  force a |dphi| <= 90 deg. Exception : si >= 20 % du coeur du signal est en
  opposition de phase (|dphi| >= 170 deg), le pouls est INVERSE (``ANTIPHASE_*``).
  Le dephasage affiche n'est alors plus independant de ce choix de signe.
* Pouls SiNC : reseau appris du package (``motion/pulsation/sinc/trace_source.py``), chaine
  par defaut de ``pipeline_config.PULSATION`` sur les frontieres BM/CSI des masques : FC
  trouvee par SiNC sans a priori, phase IQ ancree sur l'epaisseur (``sinc_pulse``).
* Cache : ``cached_compute`` lit/ecrit ``CACHE_ROOT/<recalage>/<segmentation>/<cond>.npz``,
  invalide si un reglage change (``params_key``).
* Dephasage : signaux analytiques (Hilbert) des deux pouls FILTRES,
  dphi(t) = arg(z_svd conj(z_mask)) ; moyenne CIRCULAIRE et PLV (phase locking
  value, 0 = aucun lien, 1 = verrouilles) sur le coeur du signal (bords 20 %
  exclus : transitoire du FIR). dphi > 0 : le pouls SVD est en avance.

Filtrage : interpolation sur une grille uniforme puis le FIR passe-bande du
pipeline (``motion.filters._1d.spatio_temporal_filter``), bande HR +/- 10 %
(``CardiacBand``) ; ``compute_pulse_from_data.py`` prend +/- 20 %. Spectres : Lomb-Scargle
(normalisation « standard »), le brut sur ses horodatages reels, le filtre sur
la grille uniforme.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

BAND_FRAC = 0.1  # FIR, bande des spectres ET tolerance de l'optimisation SVD (compute_pulse_from_data.py : 0,2)
BPM_GRID = (20.0, 240.0)
HR_SEARCH = (40.0, 150.0)  # recherche du pic quand visit_data.csv n'a pas de HR
RANK_SVD = 100
# Combinaison optimisee : les constantes de compute_pulse_from_data.py.
OPT_MAX_CANDIDATES = 12
OPT_N_RESTARTS = 8
OPT_MAXITER = 200
OPT_WINDOW_N_CYCLES = 6
OPT_N_WINDOWS = 5
MIN_CONC_LADDER = (0.15, 0.10, 0.05, 0.0)
MIN_COL_COVERAGE = 0.9
# Region des traces SVD, en unites PHYSIQUES pour qu'elle soit la meme sur toutes
# les grilles (Spectralis natif 3,9 x ~11,5 um/px, PLEX 1,95 x 5,9 um/px). Valeurs
# = l'ancienne region en pixels natifs : 30 px x 3,9 um, et le tiers central d'un
# scan de ~8,8 mm. Le recadrage PLEX a 1024 colonnes (6,04 mm, centre) la contient.
ROI_DEPTH_UM = 117.0  # sous le bord superieur du masque (RPE/BM)
ROI_WIDTH_MM = 2.9  # centree sur le milieu de l'image
EDGE_FRAC = 0.2
# Inversion du pouls d'intensite : si au moins ANTIPHASE_MIN_FRAC du coeur du
# signal est a |dphi| >= 180 - ANTIPHASE_TOL_DEG (opposition de phase ; 5,6 % si
# dphi etait uniforme), le signe du pouls SVD est inverse.
ANTIPHASE_TOL_DEG = 10.0
ANTIPHASE_MIN_FRAC = 0.2
# Regions « en phase » signalees sur le graphique : |dphi| <= IN_PHASE_TOL_DEG.
IN_PHASE_TOL_DEG = 10.0


@dataclass
class PulseTrace:
    raw: np.ndarray  # sur t
    filtered: np.ndarray  # sur u
    spec_raw: np.ndarray
    spec_filtered: np.ndarray
    unit: str
    info: str
    # Phase instantanee (Hilbert), rad dans (-pi, pi], sur la grille uniforme u :
    # du pouls brut (interpole sur u) et du pouls filtre. Renseignees par compute().
    phase_raw: np.ndarray | None = None
    phase_filtered: np.ndarray | None = None


def instantaneous_phase(y: np.ndarray) -> np.ndarray:
    """Phase de Hilbert, rad dans (-pi, pi]."""
    from scipy.signal import hilbert

    y = np.asarray(y, dtype=float)
    return np.angle(hilbert(y - np.nanmean(y)))


def _set_phases(tr: PulseTrace, t: np.ndarray, u: np.ndarray) -> None:
    tr.phase_raw = instantaneous_phase(np.interp(u, t, tr.raw))
    tr.phase_filtered = instantaneous_phase(tr.filtered)


@dataclass
class PulseResult:
    t: np.ndarray
    u: np.ndarray
    fs: float
    bpm: np.ndarray
    hr: float
    hr_source: str
    band_bpm: tuple[float, float]
    mask_pulse: PulseTrace
    svd_pulse: PulseTrace | None
    svd_bandfrac: np.ndarray | None  # (RANK_SVD,) fraction en bande de chaque composante
    svd_channels: np.ndarray | None = None  # composantes retenues (indices 0-based)
    svd_weights: np.ndarray | None = None  # leurs poids, ||w|| = 1
    phase: PhaseDiff | None = None
    # Pouls SiNC (reseau appris, chaine par defaut du package) : forme d'onde, FC trouvee
    # par SiNC sans a priori, phase IQ ancree sur l'epaisseur, par frame (fraction [0, 1)).
    sinc_pulse: PulseTrace | None = None
    sinc_bpm: float = float("nan")
    sinc_confidence: str = ""
    sinc_phase_per_frame: np.ndarray | None = None
    sinc_good_per_frame: np.ndarray | None = None
    sinc_note: str = ""
    from_cache: bool = False


@dataclass
class PhaseDiff:
    dphi: np.ndarray  # (Tu,) rad, dans (-pi, pi], sur u
    core: np.ndarray  # (Tu,) bool : hors bords
    mean: float  # rad, moyenne circulaire sur le coeur
    plv: float
    dt_ms: float  # mean converti en temps a la FC


def antiphase_fraction(ph: PhaseDiff) -> float:
    """Fraction du coeur du signal en opposition de phase (|dphi| >= 180 - tol)."""
    d = np.degrees(np.abs(ph.dphi[ph.core]))
    return float((d >= 180.0 - ANTIPHASE_TOL_DEG).mean()) if d.size else 0.0


def in_phase_runs(u: np.ndarray, ph: PhaseDiff, tol_deg: float = IN_PHASE_TOL_DEG):
    """[(t_debut, t_fin), ...] des plages contigues ou |dphi| <= tol_deg."""
    ok = np.degrees(np.abs(ph.dphi)) <= tol_deg
    if not ok.any():
        return []
    edges = np.diff(np.r_[0, ok.astype(np.int8), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    half = 0.5 * (u[1] - u[0]) if u.size > 1 else 0.0
    return [(u[s] - half, u[e] + half) for s, e in zip(starts, ends)]


def phase_difference(a_filt, b_filt, f0) -> PhaseDiff:
    """Dephasage instantane de b par rapport a a (Hilbert)."""
    from scipy.signal import hilbert

    z = hilbert(b_filt - b_filt.mean()) * np.conj(hilbert(a_filt - a_filt.mean()))
    n = z.size
    core = np.zeros(n, dtype=bool)
    core[int(EDGE_FRAC * n):n - int(EDGE_FRAC * n)] = True
    unit = z[core] / (np.abs(z[core]) + 1e-12)
    m = unit.mean()
    mean = float(np.angle(m))
    return PhaseDiff(dphi=np.angle(z), core=core, mean=mean, plv=float(np.abs(m)),
                     dt_ms=1e3 * mean / (2 * np.pi * f0))


def _ls(t, y, freqs_hz):
    # Meme normalisation que ``motion.pulsation.rate.lomb_scargle_power``, sans
    # importer tout ``motion.pulsation`` (qui tire torch et le moteur de recalage).
    from astropy.timeseries import LombScargle

    ok = np.isfinite(y)
    if ok.sum() < 8:
        return np.zeros_like(freqs_hz)
    return LombScargle(t[ok], y[ok] - y[ok].mean(), normalization="standard").power(freqs_hz)


def _fir(y, t, u, fs, band_bpm):
    from ocularrigidity.motion.filters._1d import spatio_temporal_filter

    nyq = 0.5 * fs
    lo = (band_bpm[0] / 60.0) / nyq
    hi = min((band_bpm[1] / 60.0) / nyq, 0.99)
    yu = np.interp(u, t, y)
    return spatio_temporal_filter(yu[:, None], 0.0, lo, hi, fs, None)[:, 0]


def mask_thickness(mask: np.ndarray, axial_mm: float) -> np.ndarray:
    """(T,) epaisseur moyenne (um) sur les colonnes couvertes ; NaN si masque vide."""
    cover = mask.any(axis=1).mean(axis=0) >= MIN_COL_COVERAGE  # (W,)
    if not cover.any():
        cover = mask.any(axis=1).any(axis=0)
    thick = mask[:, :, cover].sum(axis=1, dtype=np.float64).mean(axis=1) * axial_mm * 1e3
    thick[mask.sum(axis=(1, 2)) == 0] = np.nan
    return thick


def svd_components(frames: np.ndarray, mask: np.ndarray, pixel_mm, n: int = RANK_SVD):
    """(U (T, n) cours temporels ponderes par la valeur singuliere, V (N, n),
    intensite moyenne brute des pixels retenus (T,), n pixels).

    Les traces sont normalisees par frame AVANT la SVD : chaque frame divisee par son
    intensite moyenne sur la ROI (meme operation que ``PixelTraceSource.normalized_trace``),
    ce qui retire le gain / l'eclairage commun a toute la ROI. L'intensite moyenne BRUTE
    est renvoyee a part : apres normalisation la moyenne spatiale vaut 1 a chaque frame,
    elle ne peut plus fixer le signe du pouls."""
    from ocularrigidity.motion.projection._1d import project_into_separable_components
    from ocularrigidity.motion.pulsation.traces.pixel import _normalize_per_frame

    X = rpe_band_traces(frames, mask, pixel_mm)
    if X is None:
        return None, None, None, 0
    mean_intensity = X.mean(axis=1).astype(np.float64)
    X = _normalize_per_frame(X, axis=1).astype(np.float32)
    X -= X.mean(axis=0, keepdims=True)
    X /= X.std(axis=0, keepdims=True) + 1e-6
    U, V = project_into_separable_components(X, method="svd", n_components=min(n, *X.shape))
    return U, V, mean_intensity, X.shape[1]


def roi_pixels(W: int, pixel_mm) -> tuple[int, int, int]:
    """(profondeur en px, premiere colonne, derniere colonne + 1) de la region SVD."""
    axial_mm, lateral_mm = pixel_mm
    depth = max(1, int(round(ROI_DEPTH_UM / 1e3 / axial_mm)))
    n = min(W, max(2, int(round(ROI_WIDTH_MM / lateral_mm))))
    c0 = int(round(W / 2 - n / 2))
    return depth, c0, c0 + n


def rpe_band_traces(frames: np.ndarray, mask: np.ndarray, pixel_mm) -> np.ndarray | None:
    """(T, profondeur * n_colonnes) float32 : ROI_DEPTH_UM sous le bord superieur du
    masque (RPE/BM), sur ROI_WIDTH_MM au centre de l'image (``pixel_mm`` =
    (axial, lateral) en mm par pixel de CETTE video).

    Le bord est repris A CHAQUE FRAME : une trace = (colonne, profondeur sous la
    RPE), si bien qu'un residu de mouvement axial ne fait pas defiler les couches
    devant la trace. Frame ou le masque manque dans une colonne : bord median de
    cette colonne. Colonne sans masque a aucune frame : exclue.
    """
    T, H, W = frames.shape
    depth, c0, c1 = roi_pixels(W, pixel_mm)
    sub_m = mask[:, :, c0:c1]
    has = sub_m.any(axis=1)  # (T, nc)
    keep = has.any(axis=0)
    if keep.sum() < 2:
        return None
    sub_m, has = sub_m[:, :, keep], has[:, keep]
    top = np.where(has, sub_m.argmax(axis=1), -1).astype(np.float64)
    top[~has] = np.nan
    med = np.nanmedian(top, axis=0)
    top = np.where(has, top, med[None, :])
    rows = top.round().astype(np.int64)[:, None, :] + np.arange(depth)[None, :, None]
    rows = np.clip(rows, 0, H - 1)  # (T, depth, nc)
    sub_f = frames[:, :, c0:c1][:, :, keep]
    X = np.take_along_axis(sub_f, rows, axis=1)
    return X.reshape(T, -1).astype(np.float32)


def optimized_combination(U, V, t, hr):
    """Pipeline du package, comme ``compute_pulse_from_data.optimized_pulse``.

    Renvoie (pouls (T,), SpectralCombinationResult, seuil de concentration utilise).
    """
    from ocularrigidity.motion.pulsation import CardiacBand
    from ocularrigidity.motion.pulsation.phase import (
        OptimizedSpectralCombination,
        SpectralCombinationConfig,
    )
    from ocularrigidity.motion.pulsation.rate import LombScargleConfig, LombScargleRateEstimator
    from ocularrigidity.motion.pulsation.traces import Traces

    traces = Traces(values=np.asarray(U, dtype=np.float64), uniform_time=t,
                    kept_mask=np.ones(t.size, bool), gap_mask=np.zeros(t.size, bool),
                    timestamps_seconds=t, mixing=np.asarray(V, dtype=np.float64))
    # Bande LARGE pour l'estimateur : sa grille sert de hors-bande a l'objectif.
    rate = LombScargleRateEstimator(
        LombScargleConfig(band=CardiacBand(bpm_range=BPM_GRID), verbose=False),
        override_bpm=float(hr)).estimate(traces)
    last = None
    for min_conc in MIN_CONC_LADDER:
        agg = OptimizedSpectralCombination(SpectralCombinationConfig(
            accept_tol_bpm=BAND_FRAC * hr, min_concentration=min_conc,
            max_candidates=min(OPT_MAX_CANDIDATES, U.shape[1]),
            n_restarts=OPT_N_RESTARTS, maxiter=OPT_MAXITER, random_state=0,
            window_n_cycles=OPT_WINDOW_N_CYCLES, n_windows=OPT_N_WINDOWS))
        try:
            y = agg.aggregate(traces, rate)
        except ValueError as exc:
            last = exc
            continue
        return np.asarray(y, dtype=np.float64), agg.last_result, min_conc
    raise RuntimeError(f"aucune combinaison possible : {last}")


class _BoundaryStub:
    """Juste assez d'un ``VideoRegistrator`` pour l'aligner et les sources de traces
    (cf. ``scripts/one_cycle/astronauts.py::_prepared_registrator``)."""

    skip_first_n_frames = 0
    drop_last_n_frames = 0
    registered_frames = None
    registered_masks = None

    def __init__(self, mask):
        import torch

        from ocularrigidity.segmentation.postprocess.interfaces import (
            clean_boundaries,
            extract_boundaries_fast,
        )

        bm, csi = clean_boundaries(*extract_boundaries_fast(np.asarray(mask, dtype=bool)))
        self.registered_lines = torch.stack([torch.as_tensor(bm), torch.as_tensor(csi)], dim=1)
        self.thickness = (np.asarray(csi) - np.asarray(bm)).astype(np.float32)


def sinc_pulse(mask, t, u, fs, band, freqs):
    """Pouls SiNC par la chaine du package (``motion.pulsation.pipeline.build_extractor`` avec
    ``pipeline_config.PULSATION``) : (PulseTrace, bpm, confiance, phase par frame, bons, note)."""
    from ocularrigidity.motion.pulsation.pipeline import build_extractor
    from ocularrigidity.motion.video_timeline_aligner import TimeUnits, VideoTimelineAligner
    from ocularrigidity.pipeline_config import PULSATION

    stub = _BoundaryStub(mask)
    aligner = VideoTimelineAligner(stub, t - t[0], units_in_timestamps=TimeUnits.SECONDS)
    stages, _ = PULSATION.chain_for_video(expected_bpm=None, verbose=False)
    ex = build_extractor(stub, aligner, stages)
    tr = ex.traces
    y_u = tr.source_map[:, 0]
    ok = np.isfinite(y_u)
    raw = np.interp(t - t[0], aligner.uniform_time[ok], y_u[ok])
    raw = raw - raw.mean()
    filt = _fir(raw, t, u, fs, band)
    trace = PulseTrace(
        raw=raw, filtered=filt, spec_raw=_ls(t, raw, freqs), spec_filtered=_ls(u, filt, freqs),
        unit="u.a.",
        info=(f"SiNC {stages['sinc']} (forme d'onde apprise sur BM/CSI) ; FC SiNC "
              f"{ex.cardiac_bpm:.1f} BPM ({ex.confidence}) ; phase IQ ancree sur l'epaisseur"))
    return (trace, float(ex.cardiac_bpm), str(ex.confidence),
            np.asarray(ex.phase_per_frame, dtype=np.float64),
            np.asarray(ex.good_per_frame, dtype=bool), " ; ".join(ex.notes))


def compute(frames, mask, t, hr_bpm, pixel_mm, with_sinc: bool = True) -> PulseResult:
    """``pixel_mm`` = (axial, lateral) en mm par pixel de la video recalee."""
    t = np.asarray(t, dtype=np.float64)
    fs = 1.0 / float(np.median(np.diff(t)))
    u = np.arange(t[0], t[-1], 1.0 / fs)
    df = 1.0 / (5.0 * (t[-1] - t[0]))
    freqs = np.arange(BPM_GRID[0] / 60.0, BPM_GRID[1] / 60.0 + df, df)
    bpm = freqs * 60.0

    thick = mask_thickness(mask, pixel_mm[0])
    ok = np.isfinite(thick)
    thick_f = np.interp(t, t[ok], thick[ok]) if (~ok).any() else thick
    spec_mask_raw = _ls(t, thick_f, freqs)

    hr, hr_source = float(hr_bpm), "visit_data.csv"
    if not np.isfinite(hr):
        sel = (bpm >= HR_SEARCH[0]) & (bpm <= HR_SEARCH[1])
        hr = float(bpm[sel][spec_mask_raw[sel].argmax()])
        hr_source = f"pic du pouls des masques ({HR_SEARCH[0]:.0f}-{HR_SEARCH[1]:.0f} BPM)"
    f0 = hr / 60.0
    band = ((1 - BAND_FRAC) * hr, (1 + BAND_FRAC) * hr)
    in_band = (bpm >= band[0]) & (bpm <= band[1])

    raw_mask = thick_f - thick_f.mean()
    filt_mask = _fir(raw_mask, t, u, fs, band)
    mask_pulse = PulseTrace(
        raw=raw_mask, filtered=filt_mask, spec_raw=spec_mask_raw,
        spec_filtered=_ls(u, filt_mask, freqs), unit="um",
        info=f"{int((~ok).sum())} frame(s) a masque vide interpolee(s)")

    U, V, mean_int, n_pix = svd_components(frames, mask, pixel_mm)
    depth, c0, c1 = roi_pixels(frames.shape[2], pixel_mm)
    svd_pulse, bandfrac, phase, channels, weights = None, None, None, None, None
    if U is not None:
        P = np.stack([_ls(t, U[:, i], freqs) for i in range(U.shape[1])], axis=1)
        bandfrac = P[in_band].sum(axis=0) / (P.sum(axis=0) + 1e-12)
        y, res, min_conc = optimized_combination(U, V, t, hr)
        channels = np.asarray(res.selected_indices, dtype=int)
        weights = np.asarray(res.weights, dtype=float)
        if np.corrcoef(y, mean_int)[0, 1] < 0:
            y, weights = -y, -weights
        y = y - y.mean()
        filt = _fir(y, t, u, fs, band)
        sign_note = "correle + a l'intensite moyenne"
        phase = phase_difference(filt_mask, filt, f0)
        anti = antiphase_fraction(phase)
        if anti >= ANTIPHASE_MIN_FRAC:
            # Le pouls d'intensite est surtout en opposition avec celui du masque :
            # on l'inverse, le dephasage se recentre alors pres de 0.
            y, filt, weights = -y, -filt, -weights
            phase = phase_difference(filt_mask, filt, f0)
            sign_note = (f"INVERSE ({anti:.0%} du temps a |dphi| >= {180 - ANTIPHASE_TOL_DEG} deg, "
                         f"seuil {ANTIPHASE_MIN_FRAC:.0%})")
        spec = _ls(t, y, freqs)
        frac = spec[in_band].sum() / (spec.sum() + 1e-12)
        top = channels[np.argsort(-np.abs(weights))][:5] + 1
        svd_pulse = PulseTrace(
            raw=y, filtered=filt, spec_raw=spec, spec_filtered=_ls(u, filt, freqs),
            unit="u.a.",
            info=(f"combinaison optimisee de {channels.size} composante(s) sur {U.shape[1]} "
                  f"(seuil de concentration {min_conc:.2f} ; poids forts : "
                  f"{', '.join(map(str, top))}) -- {frac:.0%} de la puissance en bande, "
                  f"contre {bandfrac.max():.0%} pour la meilleure composante seule ; "
                  f"{n_pix} traces normalisees par frame ({ROI_DEPTH_UM:.0f} um = {depth} px sous la RPE x {ROI_WIDTH_MM} mm = "
                  f"{c1 - c0} colonnes au centre) ; signe : {sign_note}"))

    sinc = dict(sinc_note="non demande")
    if with_sinc:
        try:
            tr_s, s_bpm, s_conf, s_ph, s_good, s_note = sinc_pulse(mask, t, u, fs, band, freqs)
            sinc = dict(sinc_pulse=tr_s, sinc_bpm=s_bpm, sinc_confidence=s_conf,
                        sinc_phase_per_frame=s_ph, sinc_good_per_frame=s_good, sinc_note=s_note)
        except Exception as exc:  # noqa: BLE001 - SiNC ne doit pas priver des deux autres pouls
            sinc = dict(sinc_note=f"echec SiNC : {type(exc).__name__}: {exc}")

    for tr in (mask_pulse, svd_pulse, sinc.get("sinc_pulse")):
        if tr is not None:
            _set_phases(tr, t, u)
    return PulseResult(t=t, u=u, fs=fs, bpm=bpm, hr=hr, hr_source=hr_source,
                       band_bpm=band, mask_pulse=mask_pulse, svd_pulse=svd_pulse,
                       svd_bandfrac=bandfrac, svd_channels=channels, svd_weights=weights,
                       phase=phase, **sinc)


# --------------------------------------------------------------------------- #
# Cache sur E: -- les pouls ne sont calcules qu'une fois par (recalage, segmentation, condition)
# --------------------------------------------------------------------------- #
CACHE_ROOT = Path("E:/NASA_Rigidity/SegmentationVariations/pulses_cache")
CACHE_VERSION = 2  # 2 : traces SVD normalisees par frame (normalized_trace)
_TRACES = ("mask_pulse", "svd_pulse", "sinc_pulse")
_TRACE_ARRAYS = ("raw", "filtered", "spec_raw", "spec_filtered", "phase_raw", "phase_filtered")


def params_key() -> str:
    """Tout reglage qui change un pouls ; un cache d'une autre cle est recalcule."""
    from ocularrigidity.consts import SINC_REVISION

    return json.dumps({"v": CACHE_VERSION, "band_frac": BAND_FRAC, "bpm_grid": BPM_GRID,
                       "rank": RANK_SVD, "roi_um": ROI_DEPTH_UM, "roi_mm": ROI_WIDTH_MM,
                       "cover": MIN_COL_COVERAGE, "antiphase": [ANTIPHASE_TOL_DEG, ANTIPHASE_MIN_FRAC],
                       "sinc": SINC_REVISION}, sort_keys=True)


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def cache_path(reg: str, seg: str, astro: str, condition: str) -> Path:
    return CACHE_ROOT / _slug(reg) / _slug(seg) / f"{astro}__{condition}.npz"


def save_result(res: PulseResult, path: Path) -> None:
    out = {"params_key": np.array(params_key())}
    for f in fields(PulseResult):
        v = getattr(res, f.name)
        if f.name in _TRACES or f.name in ("phase", "from_cache") or v is None:
            continue
        out[f.name] = np.asarray(v)
    for name in _TRACES:
        tr = getattr(res, name)
        if tr is None:
            continue
        for a in _TRACE_ARRAYS:
            if getattr(tr, a) is not None:
                out[f"{name}__{a}"] = np.asarray(getattr(tr, a))
        out[f"{name}__unit"], out[f"{name}__info"] = np.array(tr.unit), np.array(tr.info)
    if res.phase is not None:
        for f in fields(PhaseDiff):
            out[f"phase__{f.name}"] = np.asarray(getattr(res.phase, f.name))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **out)
    tmp.replace(path)


def load_result(path: Path) -> PulseResult | None:
    """None si le fichier manque ou a ete calcule avec d'autres reglages."""
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        if str(z["params_key"]) != params_key():
            return None
        d = {k: z[k] for k in z.files}

    def scalar(v):
        v = v.item()
        return v

    kw = {}
    for f in fields(PulseResult):
        if f.name in d:
            v = d[f.name]
            kw[f.name] = scalar(v) if v.ndim == 0 else v
    kw["band_bpm"] = tuple(float(x) for x in kw["band_bpm"])
    for name in _TRACES:
        if f"{name}__unit" not in d:
            kw[name] = None
            continue
        kw[name] = PulseTrace(**{a: d.get(f"{name}__{a}") for a in _TRACE_ARRAYS},
                              unit=str(d[f"{name}__unit"]), info=str(d[f"{name}__info"]))
    if "phase__dphi" in d:
        kw["phase"] = PhaseDiff(**{f.name: (scalar(d[f"phase__{f.name}"])
                                            if d[f"phase__{f.name}"].ndim == 0
                                            else d[f"phase__{f.name}"])
                                   for f in fields(PhaseDiff)})
    kw["svd_bandfrac"] = kw.get("svd_bandfrac")
    kw["from_cache"] = True
    return PulseResult(**kw)


def cached_compute(path: Path, frames, mask, t, hr_bpm, pixel_mm, refresh: bool = False):
    """Le pouls du cache s'il existe (memes reglages), sinon calcule puis enregistre."""
    if not refresh:
        try:
            res = load_result(path)
        except Exception:  # noqa: BLE001 - cache illisible : on recalcule
            res = None
        if res is not None:
            return res
    res = compute(frames, mask, t, hr_bpm, pixel_mm, with_sinc=True)
    try:
        save_result(res, path)
    except OSError:
        pass
    return res
