# -*- coding: utf-8 -*-
"""
registration_metrics.py

Mesure de qualite de recalage commune a TOUTES les variantes comparees, pour
qu'aucune ne soit jugee avec une definition differente :

    NCC de chaque frame recalee a la mediane temporelle de la video, sur la
    bande de choroide -- pixels masques dans >= 50 % des frames, 3/4 centraux
    des colonnes, hors pixels nuls (padding du warp, colonnes noircies).

Chaque variante est mesuree dans SA propre geometrie : une variante qui aplatit
la RPE n'a pas la choroide au meme endroit qu'une variante qui ne l'aplatit pas.
L'aplatissement rend la bande plus homogene d'une frame a l'autre, ce qui
avantage mecaniquement ces variantes-la sur ce critere : a garder en tete a la
lecture.

Utilise par ``register_newmodel.py`` (identite et cascade_v9, en memoire) et par
``compute_registration_metrics.py`` (variantes deja ecrites, relues sur disque).
"""

from __future__ import annotations

import numpy as np

COL_FRAC = (0.125, 0.875)
BAND_MIN_FRAC = 0.5
MIN_PIXELS = 500


def ncc_to_median(frames: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """(T,) NCC de chaque frame a la mediane temporelle, sur la bande de choroide."""
    T, H, W = frames.shape
    f = frames.astype(np.float32, copy=False)
    band = masks.mean(axis=0) >= BAND_MIN_FRAC
    band[:, : int(COL_FRAC[0] * W)] = False
    band[:, int(COL_FRAC[1] * W):] = False
    med = np.median(f, axis=0)
    out = np.full(T, np.nan)
    for t in range(T):
        sel = band & (f[t] > 0) & (med > 0)
        if sel.sum() < MIN_PIXELS:
            continue
        a = f[t][sel] - f[t][sel].mean()
        b = med[sel] - med[sel].mean()
        out[t] = float((a @ b) / (np.sqrt((a @ a) * (b @ b)) + 1e-8))
    return out


def summarize(ncc: np.ndarray, prefix: str) -> dict:
    """Resume d'une serie de NCC, cles prefixees (``ncc_reg_median``, ...)."""
    v = np.asarray(ncc, dtype=float)
    ok = np.isfinite(v)
    if not ok.any():
        return {f"{prefix}_median": np.nan, f"{prefix}_q1": np.nan, f"{prefix}_q3": np.nan,
                f"{prefix}_p5": np.nan, f"{prefix}_n_lt_0.5": 0, f"{prefix}_n_valid": 0}
    return {
        f"{prefix}_median": float(np.median(v[ok])),
        f"{prefix}_q1": float(np.percentile(v[ok], 25)),
        f"{prefix}_q3": float(np.percentile(v[ok], 75)),
        f"{prefix}_p5": float(np.percentile(v[ok], 5)),
        f"{prefix}_n_lt_0.5": int((v[ok] < 0.5).sum()),
        f"{prefix}_n_valid": int(ok.sum()),
    }
