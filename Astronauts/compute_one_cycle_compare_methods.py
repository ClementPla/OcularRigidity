"""
compute_one_cycle_compare_methods.py

Videos one-cycle repliees avec DEUX phases concurrentes, pour toutes les
conditions deja traitees par ``compute_pulse_from_data.py`` :

  - ``1_fir``     passe-bande FIR a phase lineaire sur HR +/- 20 %
  - ``3a_mssa``   M-SSA multicanal (aucune bande imposee)

plus un TEMOIN NEGATIF (``null_shuffle``, la phase FIR permutee au hasard) qui
subit exactement le meme traitement et donne l'echelle des metriques : un score
n'est un resultat que compare a ce que le hasard produit sur les memes donnees.

RIEN n'est recalcule ici. Les pouls ET les phases sont REPRIS tels quels de
``SEGVAR_ROOT/<variante>/pulse_from_data/traces/<slug>.npz``, ou
``compute_pulse_from_data.py`` a ecrit, pour chacune des sept methodes,
``pulse_<methode>`` et ``phase_<methode>`` sur la grille uniforme de l'aligner.
Le seul travail restant est de porter cette phase sur les horodatages reels
(``AbstractPhaseEstimator.build_track``, resampling sur le cercle unite) et de
replier la video (``NCycleReconstructor``).

Comme les deux methodes recoivent EXACTEMENT les memes frames, la meme grille
temporelle, le meme nombre de bins et le meme masque de frames retenues, tout
ecart entre les deux videos est imputable a la phase, et a rien d'autre.

Ce que la comparaison mesure
----------------------------
Une video repliee est belle des que la phase est *lisse* : ce n'est donc pas un
critere. Trois quantites le sont :

  - ``split_half_r_*`` -- la premiere moitie de l'enregistrement et la seconde
    sont repliees separement, puis les deux cycles sont correles APRES retrait de
    leur moyenne temporelle (donc sur la seule partie qui depend de la phase,
    l'anatomie statique etant retiree). Une phase juste donne le meme cycle sur
    les deux moities ; une phase fausse donne deux bruits decorreles. C'est le
    seul critere qui ne suppose pas de connaitre la bonne reponse. Il est calcule
    sur TROIS representations du meme cycle, de la plus brute a la plus reduite,
    parce qu'elles n'ont pas le meme rapport signal/bruit :
      * ``_pix``   pixel par pixel (30 x ~1e5 points) -- estimateur stable mais
                   domine par le speckle, donc de faible amplitude ;
      * ``_kymo``  profil de profondeur moyenne sur les colonnes de la ROI
                   (30 x H) -- le speckle est moyenne sur ~700 colonnes ;
      * ``_thick`` la seule courbe d'epaisseur (30 points) -- la grandeur
                   physiologique, mais 30 points ne disent rien a eux seuls.
    Aucun des trois n'est concluant sur UNE condition ; c'est leur distribution
    sur la cohorte qui l'est.
  - ``deltaY_px`` -- amplitude crete-a-crete de l'epaisseur choroidienne le long
    du cycle, obtenue en repliant LES MASQUES avec la meme phase. C'est la
    grandeur dont depend le coefficient de rigidite : une phase qui se trompe
    d'instant moyenne des epaisseurs de phases differentes et RABOTE deltaY.
  - ``mod_depth`` -- ecart-type inter-bin des intensites sur la ROI, rapporte a
    l'intensite moyenne : ce qui reste de modulation apres repliement.

Plus, entre methodes : ``phase_offset_deg`` (moyenne circulaire de la difference
de phase image par image), ``phase_plv`` (longueur de la resultante, donc la
constance de ce decalage) et ``cycle_corr`` (correlation des deux cubes replies).

Arborescence lue
----------------
    E:/NASA_Rigidity/SegmentationVariations/<variante>/
        pulse_from_data/conditions.csv          <- conditions status == ok
        pulse_from_data/traces/<slug>.npz       <- POULS + PHASES
        registered_frames/<NN_id>/<...>_rigidity/<..._OD|OS...>/cube.mp4
        registered_masks/<NN_id>/<...>_rigidity/<..._OD|OS...>/mask.npz
    E:/SANSORI/<NN_id>/<...>_rigidity/<..._OD|OS...>/
        RawImages/ (ou RawData/)                <- horodatages (export XML)

Sorties (sous ``SEGVAR_ROOT/<variante>/one_cycle_compare/``)
-----------------------------------------------------------
  - ``<astro>/<moment>/<condition>/one_cycle_1_fir.mp4``
  - ``<astro>/<moment>/<condition>/one_cycle_3a_mssa.mp4``
  - ``<astro>/<moment>/<condition>/one_cycle_compare.mp4``  (FIR au-dessus,
    M-SSA en dessous, separes par une barre claire -- c'est cette video-la que
    le site Quarto embarque)
  - ``<astro>/<moment>/<condition>/one_cycle_compare.npz`` (courbes d'epaisseur,
    occupation des bins, ROI, phases par frame) + ``.json`` des parametres
  - ``conditions.csv`` / ``methods.csv`` / ``bins.csv`` a la racine

Les trois tables sont reecrites apres CHAQUE condition : le script est
interruptible et reprend ou il s'est arrete (``OVERWRITE`` pour tout refaire,
``LIMIT`` pour un essai sur les N premieres conditions).

Lancer (kernel pyOR, depuis la racine du depot) :
    C:/Users/transformer/anaconda3/envs/pyOR/python.exe \
        Astronauts/compute_one_cycle_compare_methods.py
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

from ocularrigidity.motion.one_cycle import (
    fit_cardiac_amplitude,
    fold_video_numba_mean,
    fold_video_numba_median,
)
from ocularrigidity.motion.pulsation import (
    HilbertPhaseConfig,
    HilbertPhaseEstimator,
    NCycleConfig,
    NCycleReconstructor,
    PulseExtractor,
    Traces,
)
from ocularrigidity.motion.pulsation.rate import FixedRateEstimator
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
PATH_GENERAL = Path("E:/SANSORI")
SEGVAR_ROOT = Path("E:/NASA_Rigidity/SegmentationVariations")
MASK_VARIANT = "model1_scale_1.0"
FRAMES_SUBDIR = "registered_frames"
MASKS_SUBDIR = "registered_masks"
PULSE_SUBDIR = "pulse_from_data"  # <- d'ou viennent pouls ET phases
OUTPUT_SUBDIR = "one_cycle_compare"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OVERWRITE = False
LIMIT = None

# --- Les deux methodes comparees -------------------------------------------
METHODS = ("1_fir", "3a_mssa")
LABELS = {"1_fir": "FIR band-pass", "3a_mssa": "M-SSA"}

# --- Temoin negatif ---------------------------------------------------------
# La phase FIR, PERMUTEE au hasard sur la grille uniforme : meme distribution de
# phases, meme nombre de frames, meme remplissage attendu des bins -- mais plus
# aucun lien avec l'instant d'acquisition. Sans lui, un ``split_half_r`` de 0,04
# ne veut rien dire : on ne sait pas si c'est peu ou si c'est tout ce que le
# bruit permet. Il subit exactement le meme traitement que les deux autres, et
# n'a ni video ni comparaison croisee.
NULL_METHOD = "null_shuffle"
LABELS[NULL_METHOD] = "shuffled phase (null)"
NULL_SEED = 0

# --- Repliement -------------------------------------------------------------
N_BINS = 30  # demande : 30 bins
N_CYCLE = 1  # UN cycle moyen sur tout l'enregistrement
FOLD_METHOD = "median"  # median : insensible aux frames aberrantes
OUTPUT_FPS = 30.0  # 30 bins a 30 fps = 1 s, soit ~la duree d'un vrai cycle
LOOP_REPEATS = 3  # le cycle est ecrit 3x dans le fichier (lecture en boucle)
CRF_SINGLE = 18  # videos par methode (archive, restent sur E:)
CRF_COMPARE = 23  # video cote a cote (embarquee dans le site)
SEPARATOR_PX = 6  # barre claire entre les deux methodes

# --- Frames retenues --------------------------------------------------------
# Le FIR comme la M-SSA ont un transitoire de bord ; il est retire de FACON
# IDENTIQUE pour les deux, sinon la comparaison porterait sur des jeux de frames
# differents. Un cycle cardiaque de chaque cote suffit : c'est l'echelle du
# transitoire d'un FIR dont la bande fait +/- 20 % de la porteuse, la ou les
# 20 % que `compute_pulse_from_data.py` ecarte de ses STATISTIQUES coutent ici
# 40 % des frames du repliement.
EDGE_CYCLES = 1.0

# --- ROI des metriques ------------------------------------------------------
# Meme fenetre laterale que la ROI des traces pixel (les bords sont ceux ou le
# recalage est le moins fiable) ; verticalement, toute l'epaisseur du masque.
COL_FRAC = (0.125, 0.875)

# --- Sorties ----------------------------------------------------------------
OUT_DIR = SEGVAR_ROOT / MASK_VARIANT / OUTPUT_SUBDIR
PULSE_DIR = SEGVAR_ROOT / MASK_VARIANT / PULSE_SUBDIR
TRACES_DIR = PULSE_DIR / "traces"
CSV_PULSE = PULSE_DIR / "conditions.csv"
CSV_CONDITIONS = OUT_DIR / "conditions.csv"
CSV_METHODS = OUT_DIR / "methods.csv"
CSV_BINS = OUT_DIR / "bins.csv"


# --------------------------------------------------------------------------- #
# Chemins
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


class PrecomputedTraceSource(AbstractTraceSource):
    """Expose des ``Traces`` deja construites (contrat suffisant pour
    ``PulseExtractor``, qui n'a besoin de rien d'autre)."""

    def __init__(self, traces: Traces):
        super().__init__()
        self._traces = traces

    def compute(self) -> Traces:  # pragma: no cover - jamais appele
        return self._traces


# --------------------------------------------------------------------------- #
# Phase reprise du .npz -> phase par frame
# --------------------------------------------------------------------------- #
def build_extractor(registrator, aligner, pulse, phase_uniform, hr, good_uniform):
    """``PulseExtractor`` amorce avec un pouls et une phase DEJA calcules.

    ``_rate`` et ``_phase`` sont renseignes directement : ni le pouls, ni la
    phase, ni la frequence ne sont re-estimes. Seul ``build_track`` tourne, pour
    le passage grille uniforme -> horodatages reels.
    """
    t = aligner.timestamps_seconds
    u_time = aligner.uniform_time
    gap_mask = aligner.gap_mask(np.zeros(t.size, dtype=bool))
    kept = ~gap_mask

    traces = Traces(
        values=np.asarray(pulse, dtype=float)[kept][:, None],
        uniform_time=u_time,
        kept_mask=kept,
        gap_mask=gap_mask,
        timestamps_seconds=t,
        mixing=None,
    )
    rate_estimator = FixedRateEstimator(bpm=float(hr))
    rate = rate_estimator.estimate(traces)

    estimator = HilbertPhaseEstimator(HilbertPhaseConfig())
    extractor = PulseExtractor(
        trace_source=PrecomputedTraceSource(traces),
        phase_estimator=estimator,
        rate_estimator=rate_estimator,
        registered_video=registrator,
        aligner=aligner,
    )
    extractor._rate = rate
    extractor._phase = estimator.build_track(
        np.mod(np.asarray(phase_uniform, dtype=float), 2 * np.pi),
        np.asarray(good_uniform, dtype=bool),
        traces,
        rate,
    )
    return extractor


def edge_mask(u_time: np.ndarray, hr: float) -> np.ndarray:
    """True au coeur de l'enregistrement, False sur ``EDGE_CYCLES`` a chaque bout."""
    good = np.ones(u_time.size, dtype=bool)
    if EDGE_CYCLES <= 0 or not np.isfinite(hr) or hr <= 0:
        return good
    margin = EDGE_CYCLES * 60.0 / hr  # secondes
    if 2 * margin >= (u_time[-1] - u_time[0]):
        return good  # enregistrement trop court : on garde tout
    good &= (u_time >= u_time[0] + margin) & (u_time <= u_time[-1] - margin)
    return good


# --------------------------------------------------------------------------- #
# Metriques sur les cubes replies
# --------------------------------------------------------------------------- #
def fold(frames, phase_per_frame, good_per_frame):
    fn = fold_video_numba_median if FOLD_METHOD == "median" else fold_video_numba_mean
    return fn(frames, phase_per_frame, good_per_frame, n_bins=N_BINS, verbose=False)


def centered_roi(cube: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """Pixels de la ROI, moyenne temporelle retiree -- soit exactement la part
    du cube qui DEPEND de la phase (l'anatomie statique est identique quelle que
    soit la methode et gonflerait n'importe quelle correlation vers 1)."""
    x = np.asarray(cube, dtype=np.float64)[:, roi]
    return x - x.mean(axis=0, keepdims=True)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.ravel(a)
    b = np.ravel(b)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 8 or a[ok].std() == 0 or b[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def kymograph(cube: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Profil de profondeur (n_bins, H), moyenne sur les colonnes de la ROI,
    moyenne temporelle retiree. Moyenner ~700 colonnes retire l'essentiel du
    speckle, qui domine la comparaison pixel a pixel."""
    x = np.asarray(cube, dtype=np.float64)[:, :, cols].mean(axis=2)
    return x - x.mean(axis=0, keepdims=True)


def thickness_curve(mask_cycle: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Epaisseur moyenne (px) par bin, a partir des MASQUES replies.

    Le repliement par moyenne d'un masque booleen donne, pixel par pixel, sa
    frequence d'occupation dans le bin ; sa somme sur une colonne est donc la
    moyenne des epaisseurs des frames de ce bin -- pas une approximation.
    """
    return np.asarray(mask_cycle, dtype=np.float64)[:, :, cols].sum(axis=1).mean(axis=1)


def stack_videos(cube_a: np.ndarray, cube_b: np.ndarray) -> np.ndarray:
    """Empilement vertical [A / barre / B] -- les A-scans restent alignes en
    colonnes, ce qui est la lecture utile pour deux B-scans du meme oeil."""
    T, H, W = cube_a.shape
    bar = np.full((T, SEPARATOR_PX, W), 255, dtype=np.uint8)
    out = np.concatenate([cube_a, bar, cube_b], axis=1)
    if out.shape[1] % 2:  # yuv420p exige des dimensions paires
        out = np.concatenate([out, np.zeros((T, 1, W), np.uint8)], axis=1)
    if out.shape[2] % 2:
        out = np.concatenate([out, np.zeros((out.shape[0], out.shape[1], 1), np.uint8)],
                             axis=2)
    return out


def to_uint8(cycles: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(np.asarray(cycles), nan=0.0), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Traitement d'une condition
# --------------------------------------------------------------------------- #
def process_condition(row) -> dict:
    astro = str(row["astro"])
    moment = str(row["moment"])
    condition = str(row["condition"])
    slug = str(row["slug"])

    variant_root = SEGVAR_ROOT / MASK_VARIANT
    frames_path = variant_root / FRAMES_SUBDIR / astro / moment / condition / "cube.mp4"
    mask_path = variant_root / MASKS_SUBDIR / astro / moment / condition / "mask.npz"
    npz_path = TRACES_DIR / f"{slug}.npz"
    for p in (frames_path, mask_path, npz_path):
        if not p.exists():
            raise FileNotFoundError(p)

    path_condi = PATH_GENERAL / astro / moment / condition
    raw_dir = find_raw_dir(path_condi)
    if raw_dir is None:
        raise FileNotFoundError(f"RawImages/RawData absent : {path_condi}")

    # --- 1. Pouls et phases deja calcules ------------------------------------
    data = np.load(npz_path)
    hr = float(data["hr"])
    t_npz = np.asarray(data["t"], dtype=float)
    u_npz = np.asarray(data["u_time"], dtype=float)
    for name in METHODS:
        if f"phase_{name}" not in data.files:
            raise KeyError(f"{npz_path.name} sans phase_{name}")

    # --- 2. Video, masques, base de temps ------------------------------------
    registrator = _prepared_registrator(frames_path, mask_path, DEVICE, verbose=False)
    frames = registrator.registered_frames
    masks = np.asarray(registrator.registered_masks, dtype=bool)
    ts_us = raw_timestamps_us(raw_dir)
    if frames.shape[0] != ts_us.size:
        raise ValueError(f"frames ({frames.shape[0]}) != horodatages ({ts_us.size})")

    aligner = VideoTimelineAligner(registrator, ts_us)
    t = aligner.timestamps_seconds
    u_time = aligner.uniform_time
    # La phase reprise vit sur CETTE grille-la : si elle a change (autre variante
    # de masques, autre export XML), les indices ne designent plus les memes
    # instants et le repliement serait faux sans jamais lever d'erreur.
    if t.size != t_npz.size or not np.allclose(t, t_npz, atol=1e-6):
        raise ValueError(f"horodatages != .npz ({t.size} vs {t_npz.size})")
    if u_time.size != u_npz.size or not np.allclose(u_time, u_npz, atol=1e-6):
        raise ValueError(f"grille uniforme != .npz ({u_time.size} vs {u_npz.size})")

    T, H, W = frames.shape

    # --- 3. ROI des metriques -------------------------------------------------
    roi = masks.all(axis=0)
    if roi.sum() < 100:  # intersection vide : repli sur la choroide majoritaire
        roi = masks.mean(axis=0) > 0.5
    if roi.sum() < 100:
        raise ValueError(f"ROI trop petite ({int(roi.sum())} pixels)")
    c0, c1 = int(COL_FRAC[0] * W), int(COL_FRAC[1] * W)
    cols = np.flatnonzero(roi[:, c0:c1].any(axis=0)) + c0
    if cols.size < 10:
        raise ValueError(f"trop peu de colonnes exploitables ({cols.size})")

    # --- 4. Un repliement par methode -----------------------------------------
    good_u = edge_mask(u_time, hr) & ~aligner.gap_mask(np.zeros(T, dtype=bool))
    masks_f32 = masks.astype(np.float32)
    half = t[0] + 0.5 * (t[-1] - t[0])
    first_half, second_half = t < half, t >= half

    rng = np.random.default_rng(NULL_SEED)
    per_method: dict[str, dict] = {}
    for name in (*METHODS, NULL_METHOD):
        if name == NULL_METHOD:
            pulse_in = data[f"pulse_{METHODS[0]}"]
            phase_in = rng.permutation(np.asarray(data[f"phase_{METHODS[0]}"]))
        else:
            pulse_in, phase_in = data[f"pulse_{name}"], data[f"phase_{name}"]
        extractor = build_extractor(
            registrator, aligner, pulse_in, phase_in, hr, good_u,
        )
        phase_f = extractor.phase_per_frame
        good_f = extractor.good_per_frame

        # La video livree passe par NCycleReconstructor -- meme chemin de code
        # que tous les autres one-cycle du depot.
        reconstructor = NCycleReconstructor(
            extractor,
            NCycleConfig(n_bins=N_BINS, n_cycle=N_CYCLE, fold_method=FOLD_METHOD,
                         verbose=False),
        )
        cycles, counts = reconstructor.compute()

        # Les diagnostics demandent des repliements que le reconstructeur
        # n'expose pas (les MASQUES, et chaque moitie separement) : ils passent
        # par la meme fonction de repliement, appelee directement.
        mask_cycle, _ = fold_video_numba_mean(
            masks_f32, phase_f, good_f, n_bins=N_BINS, verbose=False)
        cycle_a, counts_a = fold(frames, phase_f, good_f & first_half)
        cycle_b, counts_b = fold(frames, phase_f, good_f & second_half)
        mask_a, _ = fold_video_numba_mean(
            masks_f32, phase_f, good_f & first_half, n_bins=N_BINS, verbose=False)
        mask_b, _ = fold_video_numba_mean(
            masks_f32, phase_f, good_f & second_half, n_bins=N_BINS, verbose=False)

        thick = thickness_curve(mask_cycle, cols)
        thick_a = thickness_curve(mask_a, cols)
        thick_b = thickness_curve(mask_b, cols)
        fit, _ = fit_cardiac_amplitude(thick, n_harmonics=1)
        cent = centered_roi(cycles, roi)
        per_method[name] = {
            "extractor": extractor,
            "cycles": cycles,
            "counts": counts,
            "centered": cent,
            "thickness": thick,
            "thickness_half_a": thick_a,
            "thickness_half_b": thick_b,
            "thickness_fit": fit,
            "phase_per_frame": phase_f,
            "good_per_frame": good_f,
            "split_half_r_pix": corr(centered_roi(cycle_a, roi),
                                     centered_roi(cycle_b, roi)),
            "split_half_r_kymo": corr(kymograph(cycle_a, cols),
                                      kymograph(cycle_b, cols)),
            "split_half_r_thick": corr(thick_a - thick_a.mean(),
                                       thick_b - thick_b.mean()),
            "split_counts_min": int(min(counts_a.min(), counts_b.min())),
            "deltaY_px": float(thick.max() - thick.min()),
            "deltaY_fit_px": float(fit.max() - fit.min()),
            "mod_depth": float(
                np.asarray(cycles, np.float64)[:, roi].std(axis=0).mean()
                / (np.asarray(cycles, np.float64)[:, roi].mean() + 1e-12)
            ),
            "n_good": int(good_f.sum()),
            "notes": list(extractor.notes) + list(reconstructor.notes),
        }

    # --- 5. Entre les deux methodes -------------------------------------------
    a, b = METHODS
    dphi = 2 * np.pi * (per_method[a]["phase_per_frame"]
                        - per_method[b]["phase_per_frame"])
    both = per_method[a]["good_per_frame"] & per_method[b]["good_per_frame"]
    z = np.exp(1j * dphi[both]).mean() if both.any() else np.nan + 0j
    cycle_corr = corr(per_method[a]["centered"], per_method[b]["centered"])

    # --- 6. Videos --------------------------------------------------------------
    out_dir = OUT_DIR / astro / moment / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    cubes = {}
    for name in METHODS:
        cube = np.tile(to_uint8(per_method[name]["cycles"]), (LOOP_REPEATS, 1, 1))
        cubes[name] = cube
        write_gray_mp4(cube, out_dir / f"one_cycle_{name}.mp4", OUTPUT_FPS,
                       crf=CRF_SINGLE)
    write_gray_mp4(stack_videos(cubes[a], cubes[b]),
                   out_dir / "one_cycle_compare.mp4", OUTPUT_FPS, crf=CRF_COMPARE)

    # --- 7. Diagnostics ---------------------------------------------------------
    payload = {
        "astro": astro, "moment": moment, "condition": condition, "slug": slug,
        "mask_variant": MASK_VARIANT, "hr_bpm": hr,
        "n_bins": N_BINS, "n_cycle": N_CYCLE, "fold_method": FOLD_METHOD,
        "edge_cycles": EDGE_CYCLES,
        "roi": roi, "cols": cols,
        "timestamps_seconds": t.astype(np.float32),
        "uniform_time": u_time.astype(np.float32),
        "good_uniform": good_u,
        "mean_frame": frames.mean(axis=0).astype(np.float32),
        "phase_offset_deg": float(np.degrees(np.angle(z))),
        "phase_plv": float(np.abs(z)),
        "cycle_corr": cycle_corr,
    }
    for name in METHODS:
        m = per_method[name]
        payload[f"counts_{name}"] = m["counts"]
        payload[f"thickness_{name}"] = m["thickness"].astype(np.float32)
        payload[f"thickness_half_a_{name}"] = m["thickness_half_a"].astype(np.float32)
        payload[f"thickness_half_b_{name}"] = m["thickness_half_b"].astype(np.float32)
        payload[f"thickness_fit_{name}"] = np.asarray(m["thickness_fit"], np.float32)
        payload[f"phase_per_frame_{name}"] = m["phase_per_frame"].astype(np.float32)
        payload[f"good_per_frame_{name}"] = m["good_per_frame"]
        payload[f"pulse_{name}"] = np.asarray(data[f"pulse_{name}"], np.float32)
        payload[f"phase_uniform_{name}"] = np.asarray(data[f"phase_{name}"], np.float32)
    null = per_method[NULL_METHOD]
    payload[f"counts_{NULL_METHOD}"] = null["counts"]
    payload[f"thickness_{NULL_METHOD}"] = null["thickness"].astype(np.float32)
    np.savez_compressed(out_dir / "one_cycle_compare.npz", **payload)

    meta = {
        "mask_variant": MASK_VARIANT,
        "pulse_npz": str(npz_path),
        "methods": list(METHODS),
        "hr_bpm": hr,
        "n_bins": N_BINS,
        "n_cycle": N_CYCLE,
        "fold_method": FOLD_METHOD,
        "edge_cycles": EDGE_CYCLES,
        "output_fps": OUTPUT_FPS,
        "loop_repeats": LOOP_REPEATS,
        "n_frames": int(T),
        "frame_shape": [int(H), int(W)],
        "roi_pixels": int(roi.sum()),
        "notes": {name: per_method[name]["notes"] for name in METHODS},
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "one_cycle_compare_params.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8")

    # --- 8. Tables ----------------------------------------------------------------
    rel = f"{astro}/{moment}/{condition}"
    cond_row = {
        "slug": slug, "astro": astro, "moment": moment, "condition": condition,
        "hr_BPM": hr, "hr_source": str(row.get("hr_source", "")),
        "n_frames": int(T), "duree_s": float(t[-1] - t[0]),
        "fs_Hz": float(aligner.fs), "n_uniform": int(u_time.size),
        "frame_h": int(H), "frame_w": int(W), "roi_pixels": int(roi.sum()),
        "n_cols": int(cols.size),
        "phase_offset_deg": payload["phase_offset_deg"],
        "phase_plv": payload["phase_plv"],
        "cycle_corr": cycle_corr,
        "corrfir_3a_mssa": float(row.get("corrfir_3a_mssa", np.nan)),
        "fneg_1_fir": float(row.get("fneg_1_fir", np.nan)),
        "fneg_3a_mssa": float(row.get("fneg_3a_mssa", np.nan)),
        "out_rel": rel, "status": "ok",
    }
    metric_keys = ("split_half_r_pix", "split_half_r_kymo", "split_half_r_thick",
                   "deltaY_px", "deltaY_fit_px", "mod_depth", "n_good")
    for name in (*METHODS, NULL_METHOD):
        m = per_method[name]
        for key in metric_keys:
            cond_row[f"{key}_{name}"] = m[key]
    cond_row["split_half_gain_fir"] = (per_method[a]["split_half_r_kymo"]
                                       - per_method[b]["split_half_r_kymo"])
    cond_row["deltaY_ratio_fir_mssa"] = (per_method[a]["deltaY_fit_px"]
                                         / (per_method[b]["deltaY_fit_px"] + 1e-12))

    method_rows, bin_rows = [], []
    for name in (*METHODS, NULL_METHOD):
        m = per_method[name]
        method_rows.append({
            "slug": slug, "astro": astro, "condition": condition,
            "methode": name, "label": LABELS[name], "hr_BPM": hr,
            "split_half_r_pix": m["split_half_r_pix"],
            "split_half_r_kymo": m["split_half_r_kymo"],
            "split_half_r_thick": m["split_half_r_thick"],
            "deltaY_px": m["deltaY_px"], "deltaY_fit_px": m["deltaY_fit_px"],
            "mod_depth": m["mod_depth"],
            "n_good": m["n_good"], "n_frames": int(T),
            "good_frac": m["n_good"] / T,
            "bin_min": int(m["counts"].min()), "bin_max": int(m["counts"].max()),
            "bin_cv": float(m["counts"].std() / (m["counts"].mean() + 1e-12)),
            "split_counts_min": m["split_counts_min"],
            "out_rel": rel,
        })
        for b_i in range(N_BINS):
            bin_rows.append({
                "slug": slug, "methode": name, "bin": b_i,
                "count": int(m["counts"][b_i]),
                "thickness_px": float(m["thickness"][b_i]),
                "thickness_fit_px": float(m["thickness_fit"][b_i]),
            })

    return {"condition": cond_row, "methods": method_rows, "bins": bin_rows}


# --------------------------------------------------------------------------- #
# Boucle principale
# --------------------------------------------------------------------------- #
def load_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def save_table(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    if not CSV_PULSE.exists():
        raise FileNotFoundError(
            f"{CSV_PULSE} introuvable : lancer d'abord compute_pulse_from_data.py "
            "(c'est lui qui calcule les pouls et les phases)."
        )
    pulses = pd.read_csv(CSV_PULSE)
    n_all = len(pulses)
    pulses = pulses[pulses["status"] == "ok"].copy()
    print(f"sortie : {OUT_DIR}")
    print(f"device : {DEVICE}   variante de masques : {MASK_VARIANT}")
    print(f"{len(pulses)} condition(s) exploitable(s) sur {n_all}\n")

    tables = {
        "condition": (load_table(CSV_CONDITIONS), CSV_CONDITIONS),
        "methods": (load_table(CSV_METHODS), CSV_METHODS),
        "bins": (load_table(CSV_BINS), CSV_BINS),
    }
    rows = {k: (df.to_dict("records") if not df.empty else [])
            for k, (df, _) in tables.items()}
    if OVERWRITE:
        done = set()
        rows = {k: [] for k in rows}
    else:
        # Seules les conditions REUSSIES comptent comme faites : une ligne en
        # echec est un travail a refaire, pas un resultat.
        done = {r["slug"] for r in rows["condition"] if r.get("status") == "ok"}
        n_retry = len(rows["condition"]) - len(done)
        rows["condition"] = [r for r in rows["condition"] if r.get("status") == "ok"]
        if n_retry:
            print(f"{n_retry} condition(s) en echec seront retentees")
    if done:
        print(f"{len(done)} condition(s) deja faite(s), reprise")

    records = pulses.to_dict("records")
    if LIMIT is not None:
        records = records[:LIMIT]

    t_start = time.perf_counter()
    n_ok = n_skip = n_fail = 0
    for i, row in enumerate(records, 1):
        slug = str(row["slug"])
        if slug in done:
            n_skip += 1
            continue

        print(f"[{i:>3d}/{len(records)}] {slug}", flush=True)
        t0 = time.perf_counter()
        try:
            out = process_condition(row)
        except Exception as exc:  # noqa: BLE001 - une condition ne doit pas tout arreter
            n_fail += 1
            print(f"        ECHEC : {exc}")
            traceback.print_exc()
            rows["condition"].append({
                "slug": slug, "astro": row.get("astro"), "moment": row.get("moment"),
                "condition": row.get("condition"), "status": f"echec : {exc}",
            })
        else:
            n_ok += 1
            for key in ("condition", "methods", "bins"):
                payload = out[key]
                rows[key].extend([payload] if key == "condition" else payload)
            c = out["condition"]
            print(
                f"        moitie/moitie r (kymo) : FIR "
                f"{c['split_half_r_kymo_1_fir']:.2f} "
                f"vs M-SSA {c['split_half_r_kymo_3a_mssa']:.2f} "
                f"vs temoin {c[f'split_half_r_kymo_{NULL_METHOD}']:.2f}  ·  "
                f"deltaY {c['deltaY_fit_px_1_fir']:.2f} vs "
                f"{c['deltaY_fit_px_3a_mssa']:.2f} px  ·  "
                f"dephasage {c['phase_offset_deg']:+.0f} deg "
                f"(PLV {c['phase_plv']:.2f})  [{time.perf_counter() - t0:.0f} s]"
            )

        for key, (_, path) in tables.items():
            save_table(rows[key], path)

    dt = time.perf_counter() - t_start
    print(f"\ntermine {datetime.now():%Y-%m-%d %H:%M} : {n_ok} ok, {n_skip} "
          f"deja faites, {n_fail} en echec  ({dt / 60:.1f} min)")
    for key, (_, path) in tables.items():
        print(f"  {path}  ({len(rows[key])} lignes)")


if __name__ == "__main__":
    main()
