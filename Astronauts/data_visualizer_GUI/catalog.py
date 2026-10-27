# -*- coding: utf-8 -*-
"""
Ce que le GUI sait trouver sur disque : les conditions de la cohorte SANS, et
pour chacune, quelles combinaisons (recalage, segmentation) existent.

Une combinaison = une video recalee + un masque de choroide + (si possible) les
horodatages des frames. Le menu « recalage » choisit d'ou vient la video, le menu
« segmentation » le masque parmi ceux calcules POUR CE recalage : un masque n'est
jamais applique a une video recalee autrement (les pixels ne correspondraient pas).

Pour ajouter une methode : une entree de plus dans ``REGISTRATIONS``.
"""

from __future__ import annotations

import csv
import json
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

PATH_GENERAL = Path("E:/SANSORI")
SEGVAR_ROOT = Path("E:/NASA_Rigidity/SegmentationVariations")

MOMENT_LABELS = {"before": "avant vol (before)", "post": "apres vol (post)"}

_RE_COND = re.compile(r"^(?P<nn>\d+)_(?P<id>\d+)/(?P=id)(?P<moment>[a-z]+)_rigidity/"
                      r"(?P=id)(?P=moment)_rigidity_(?P<eye>OD|OS)(?P<rep>\d*)$")


# --------------------------------------------------------------------------- #
# Conditions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Condition:
    astro: str  # "01_210713001"
    moment: str  # "before" | "post"
    eye: str  # "OD" | "OS"
    rep: int  # 1 quand le dossier ne porte pas de numero ("..._OD")
    rel: str  # chemin relatif a PATH_GENERAL

    @property
    def path(self) -> Path:
        return PATH_GENERAL / self.rel

    @property
    def raw(self) -> Path:
        return self.path / "RawImages"

    @property
    def parts(self) -> tuple[str, str, str]:
        """(astro, dossier moment, dossier condition) -- l'arborescence de SegmentationVariations."""
        a, m, c = self.rel.split("/")
        return a, m, c

    @property
    def name(self) -> str:
        return self.parts[2]


def discover_conditions() -> list[Condition]:
    out = []
    for raw in PATH_GENERAL.glob("*/*rigidity/*/RawImages"):
        rel = raw.parent.relative_to(PATH_GENERAL).as_posix()
        m = _RE_COND.match(rel)
        if m and raw.is_dir():
            out.append(Condition(astro=f"{m['nn']}_{m['id']}", moment=m["moment"],
                                 eye=m["eye"], rep=int(m["rep"] or 1), rel=rel))
    return sorted(out, key=lambda c: (c.astro, c.eye, c.rep, c.moment))


# --------------------------------------------------------------------------- #
# Methodes de recalage / segmentation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Source:
    """Une combinaison concrete : ou lire frames, masque et horodatages."""

    frames: Callable[[Condition], Path]
    mask: Callable[[Condition], Path]
    # Fichier texte d'horodatages (us), ou None : on relit alors le XML HEYEX.
    timestamps: Callable[[Condition], Path | None] = lambda c: None
    # False quand les frames ne correspondent pas une a une aux B-scans du XML
    # (MATLAB : frames aberrantes retirees, sans trace exploitable des indices).
    xml_timeline: bool = True


@dataclass(frozen=True)
class Registration:
    label: str
    segmentations: dict[str, Source] = field(default_factory=dict)


def _segvar(variant: str) -> Source:
    def frames(c):
        return SEGVAR_ROOT / variant / "registered_frames" / Path(*c.parts) / "cube.mp4"

    def mask(c):
        return SEGVAR_ROOT / variant / "registered_masks" / Path(*c.parts) / "mask.npz"

    return Source(frames=frames, mask=mask)


def _in_raw(sub: str, video: str, mask: str, ts: str | None) -> Source:
    return Source(frames=lambda c: c.raw / sub / video,
                  mask=lambda c: c.raw / sub / mask,
                  timestamps=(lambda c: c.raw / sub / ts) if ts else (lambda c: None))


def _plex(sub: str) -> Source:
    return _in_raw(sub, "registered_video.mp4", "mask.npz", "timestamp.txt")


REGISTRATIONS: list[Registration] = [
    # Grille PLEX (scripts/registration/cascade_v9_plex.py) : 1,95 x 5,9 um par
    # pixel sur les variantes « grille PLEX » -- l'epaisseur en px n'y a pas la
    # meme valeur qu'en natif (3,9 um par pixel axial).
    Registration("cascade_v9 PLEX 1024 col. (grille PLEX)", {
        "SegFormer cascade_v9 (meme passe)": _plex("registered_cascade_v9_plex_w1024")}),
    Registration("cascade_v9 PLEX 1024 col. (ramene en natif)", {
        "SegFormer cascade_v9 (meme passe)": _plex("registered_cascade_v9_plex_w1024_native")}),
    Registration("cascade_v9 PLEX pleine largeur (grille PLEX)", {
        "SegFormer cascade_v9 (meme passe)": _plex("registered_cascade_v9_plex_full")}),
    Registration("cascade_v9 PLEX pleine largeur (ramene en natif)", {
        "SegFormer cascade_v9 (meme passe)": _plex("registered_cascade_v9_plex_full_native")}),
    Registration("cascade_v9 (appris, 1536x1024)", {
        "SegFormer cascade_v9 (meme passe)": _in_raw(
            "registered_cascade_v9_1536x1024", "registered_video.mp4", "mask.npz", "timestamp.txt"),
    }),
    Registration("Python aplati (flatten_choroid_xcorr)", {
        "U-Net model1, echelle 1.0": _segvar("model1_scale_1.0_flatten_choroid_xcorr"),
    }),
    Registration("Python (SegmentationVariations)", {
        "U-Net model1, echelle 1.0": _segvar("model1_scale_1.0"),
        "U-Net model1, echelle 2.0": _segvar("model1_scale_2.0"),
        "U-Net model2, echelle 1.0": _segvar("model2_scale_1.0"),
        "U-Net model2, echelle 2.0": _segvar("model2_scale_2.0"),
        "Clement": _segvar("Clement"),
    }),
    Registration("Python (RawImages/registered)", {
        "U-Net": _in_raw("registered", "registered_video.mp4", "mask.npz", "timestamp.txt"),
        "U-Net resegmente": _in_raw("registered", "registered_video.mp4",
                                    "mask_resegmented.npz", "timestamp.txt"),
    }),
    Registration("Python + passe A-scan (registered/*_ascan)", {
        "U-Net (A-scan)": _in_raw("registered", "registered_video_ascan.mp4",
                                  "mask_ascan.npz", "timestamp_ascan.txt"),
    }),
    Registration("MATLAB (registeredBscans, ancien)", {
        "MATLAB": Source(
            frames=lambda c: c.raw / "registeredBscans" / "compressedOCT_woOutliers_median_threshold.mj2",
            mask=lambda c: c.raw / "registeredBscans" / "mask.npz",
            xml_timeline=False),
    }),
]
REG_BY_LABEL = {r.label: r for r in REGISTRATIONS}


def available(cond: Condition) -> dict[str, list[str]]:
    """{recalage: [segmentations dont la video ET le masque existent]}."""
    out = {}
    for reg in REGISTRATIONS:
        segs = [name for name, src in reg.segmentations.items()
                if src.frames(cond).exists() and src.mask(cond).exists()]
        if segs:
            out[reg.label] = segs
    return out


# --------------------------------------------------------------------------- #
# Chargement
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    cond: Condition
    reg: str
    seg: str
    frames: np.ndarray  # (T, H, W) uint8
    mask: np.ndarray | None  # (T, H, W) bool, None si incompatible avec les frames
    t: np.ndarray | None  # (T,) secondes depuis la 1re frame, None si inconnu
    hr_bpm: float  # visit_data.csv, NaN si absent
    pixel_mm: tuple[float, float]  # (axial, lateral) mm par pixel de CETTE video
    frames_path: Path
    mask_path: Path
    notes: list[str] = field(default_factory=list)


def _read_video(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".mp4":
        from ocularrigidity.data.compression import read_gray

        return np.ascontiguousarray(read_gray(path))
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img[..., 0] if img.ndim == 3 else img)
    cap.release()
    if not frames:
        raise IOError(f"video illisible : {path}")
    return np.stack(frames).astype(np.uint8)


def _xml_timestamps_us(cond: Condition) -> np.ndarray:
    from ocularrigidity.scripts.registration.astronauts import load_ordered_oct_series

    series = load_ordered_oct_series(cond.raw)
    return np.array([round(s.acquisition_time.seconds_of_day * 1e6) for s in series],
                    dtype=np.int64)


ACQ_PARAMS = Path("E:/NASA_Rigidity/AcquisitionParameters/conditions.csv")
_SCALES: dict | None = None


def native_pixel_mm(cond: Condition) -> tuple[float, float]:
    """(ScaleY, ScaleX) du B-scan Spectralis : table d'acquisition, sinon le XML."""
    global _SCALES
    if _SCALES is None:
        try:
            df = pd.read_csv(ACQ_PARAMS, usecols=["condition", "oct_scale_y_mm", "oct_scale_x_mm"])
            _SCALES = {r.condition: (float(r.oct_scale_y_mm), float(r.oct_scale_x_mm))
                       for r in df.itertuples()}
        except Exception:  # noqa: BLE001
            _SCALES = {}
    if cond.name in _SCALES:
        return _SCALES[cond.name]
    from ocularrigidity.scripts.registration.astronauts import load_ordered_oct_series

    series = load_ordered_oct_series(cond.raw)
    return (float(np.median([s.oct.scale_y for s in series])),
            float(np.median([s.oct.scale_x for s in series])))


def video_pixel_mm(cond: Condition, frames_path: Path, shape) -> tuple[float, float]:
    """Taille de pixel de la video recalee : grille PLEX si son registration_params.json
    le dit, sinon la grille native (seules autres tailles presentes : 496 x 768)."""
    try:
        params = json.loads((frames_path.parent / "registration_params.json").read_text("utf-8"))
        if params.get("grid") == "plex":
            return params["plex_axial_um"] / 1e3, params["plex_lateral_um"] / 1e3
    except Exception:  # noqa: BLE001
        pass
    ax, lat = native_pixel_mm(cond)
    H, W = shape
    if (H, W) != (496, 768):
        raise ValueError(f"taille de pixel inconnue pour une video {H}x{W} : {frames_path}")
    return ax, lat


def read_hr(cond: Condition) -> float:
    p = cond.path / "Data Files" / "visit_data.csv"
    if not p.exists():
        return float("nan")
    try:
        df = pd.read_csv(p, quoting=csv.QUOTE_NONE)
        return float(np.nanmean(pd.to_numeric(df["HR"], errors="coerce")))
    except Exception:  # noqa: BLE001
        return float("nan")


class DataStore:
    """Cache des dernieres combinaisons chargees (une video ~ 150-350 Mo)."""

    def __init__(self, capacity: int = 3):
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self.capacity = capacity

    def load(self, cond: Condition, reg: str, seg: str) -> Dataset:
        key = (cond.rel, reg, seg)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        ds = self._load(cond, reg, seg)
        with self._lock:
            self._cache[key] = ds
            while len(self._cache) > self.capacity:
                self._cache.popitem(last=False)
        return ds

    @staticmethod
    def _load(cond: Condition, reg: str, seg: str) -> Dataset:
        from ocularrigidity.data.io import load_mask

        src = REG_BY_LABEL[reg].segmentations[seg]
        notes = []
        frames = _read_video(src.frames(cond))
        mask = np.asarray(load_mask(src.mask(cond)), dtype=bool)
        if mask.shape != frames.shape:
            notes.append(f"masque {mask.shape} incompatible avec la video {frames.shape} : non affiche")
            mask = None

        t = None
        ts_file = src.timestamps(cond)
        if ts_file is not None and ts_file.exists():
            ts = np.loadtxt(ts_file, dtype=np.int64, ndmin=1)
        elif src.xml_timeline:
            ts = _xml_timestamps_us(cond)
        else:
            ts = None
            notes.append("horodatages inconnus pour ce recalage (frames retirees)")
        if ts is not None:
            if ts.size == frames.shape[0]:
                t = (ts - ts[0]) / 1e6
            else:
                notes.append(f"{ts.size} horodatages pour {frames.shape[0]} frames : ignores")
        return Dataset(cond=cond, reg=reg, seg=seg, frames=frames, mask=mask, t=t,
                       hr_bpm=read_hr(cond),
                       pixel_mm=video_pixel_mm(cond, src.frames(cond), frames.shape[1:]),
                       frames_path=src.frames(cond),
                       mask_path=src.mask(cond), notes=notes)
