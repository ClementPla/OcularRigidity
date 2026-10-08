# -*- coding: utf-8 -*-
"""
compute_ct_pulse_traces.py

Pouls d'EPAISSEUR choroidienne des replicats, passe au MEME filtre FIR que les
pouls d'INTENSITE SVD des lots cascade_v9, pour les comparer. Deux pouls
d'intensite sont evalues (colonne ``intensite``) :

    combinaison   ``pulse_1_fir`` du lot : combinaison optimisee de modes SVD ;
    mode          le PREMIER mode SVD qui pulse, seul
                  (``Reproducibility/compute_svd_first_mode.py``) ;
    costs         bande cardiaque de COSTS, BOP-DMD sur fenetre glissante
                  (``Reproducibility/compute_costs_pulse.py``).

Pendant de ``Astronauts/compute_ct_pulse_traces.py`` (cohorte SANS), sans masque
a relire : l'epaisseur par frame, tiree des masques cascade_v9 sur la meme ROI de
colonnes (1/8-7/8) que le lot, est deja dans ``hr_bandpass_50_85/traces`` sous
``ct_brut_um``. Mais son ``ct_filt_um`` y est filtre a 50-85 BPM FIXE ; ici on la
refiltre a FC +/- 20 % autour de l'ancre du lot, comme ``bandpass_pulse`` de
``Astronauts/compute_pulse_from_data.py``. C'est la condition pour que la
comparaison intensite / epaisseur veuille dire quelque chose.

Entrees (E:/NASA_Rigidity/Reproducibility/SegmentationVariations/)
    cascade_v9_1536x1024/hr_bandpass_50_85/traces/<slug>.npz      t, u_time, ct_brut_um
    cascade_v9_1536x1024_hr_<ancre>/pulse_from_data/traces/<slug>.npz
                                                    t, u_time, fs, hr, core, pulse_1_fir
    cascade_v9_1536x1024_hr_<ancre>/pulse_from_data/svd_first_mode_traces.npz
                                                    <slug>__pulse_mode_fir
    cascade_v9_1536x1024_hr_<ancre>/pulse_from_data/costs_pulse_traces.npz
                                                    <slug>__pulse_costs_fir

Sorties (a cote des traces de chaque lot, ``pulse_from_data/``)
    ct_pulse.csv            1 ligne / (acquisition, pouls d'intensite)
    ct_pulse_traces.npz     ``<slug>__ct_fir_um``

Indicateurs, sur ``core`` (bords du FIR exclus) :
    r         Pearson signe intensite / epaisseur ;
    dphi_deg  angle moyen de hilbert(int) * conj(hilbert(ct)), dans [-180, 180].
              Le signe du pouls SVD est une CONVENTION (phaseur de la trace
              moyenne) : une simple inversion apparait a +/- 180 deg ;
    plv       |moyenne de exp(i dphi(t))| : stabilite de la relation de phase,
              insensible au signe ;
    ct_amp_um ecart-type du pouls d'epaisseur filtre (um) ;
    plv_null95  95e centile de la PLV sous l'hypothese nulle : ``N_NULL``
              decalages circulaires de l'epaisseur (>= 2 cycles), qui gardent les
              deux spectres mais cassent leur relation de phase. Deux signaux
              filtres dans la meme bande etroite se verrouillent un peu par
              hasard : la PLV brute seule ne dit rien.

Lancer (kernel pyOR, depuis la racine du depot) :
    C:/Users/transformer/anaconda3/envs/pyOR/python.exe \\
        Reproducibility/compute_ct_pulse_traces.py
"""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import hilbert

from ocularrigidity.motion.filters._1d import spatio_temporal_filter
from ocularrigidity.motion.pulsation import CardiacBand

SEGVAR = Path("E:/NASA_Rigidity/Reproducibility/SegmentationVariations")
CT_TRACES = SEGVAR / "cascade_v9_1536x1024" / "hr_bandpass_50_85" / "traces"
LOTS = {
    "A": SEGVAR / "cascade_v9_1536x1024_hr_mediane" / "pulse_from_data",
    "B": SEGVAR / "cascade_v9_1536x1024_hr_consensus" / "pulse_from_data",
}
BAND_FRAC = 0.2  # celui de compute_pulse_from_data.py
PULSE = "pulse_1_fir"
N_NULL = 500
# Pouls d'intensite optionnels : (fichier a cote du lot, suffixe de cle).
AUTRES = {"mode": ("svd_first_mode_traces.npz", "__pulse_mode_fir"),
          "costs": ("costs_pulse_traces.npz", "__pulse_costs_fir")}
RE_PART = r"^(?P<participant>.+)_(?P<eye>OD|OS)(?P<replicate>\d+)$"


def passe_bande(y_u, fs, lo_bpm, hi_bpm):
    nyq = 0.5 * fs
    lo = (lo_bpm / 60.0) / nyq
    hi = min((hi_bpm / 60.0) / nyq, 0.99)
    return spatio_temporal_filter(np.asarray(y_u, dtype=float)[:, None], 0.0,
                                  lo, hi, fs, None)[:, 0]


def indicateurs(a, b):
    a = a - a.mean()
    b = b - b.mean()
    r = float(np.corrcoef(a, b)[0, 1])
    z = hilbert(a) * np.conj(hilbert(b))
    u = z / np.maximum(np.abs(z), 1e-12)
    m = u.mean()
    return r, float(np.degrees(np.angle(m))), float(np.abs(m))


def plv_nul(a, b, fs, hr, graine: str):
    """PLV sous decalages circulaires de ``b`` d'au moins 2 cycles.

    Graine propre a (acquisition, pouls) : le seuil d'une ligne ne depend pas
    des autres lignes calculees avant elle."""
    rng = np.random.default_rng(zlib.crc32(graine.encode()))
    za = hilbert(a - a.mean())
    zb = hilbert(b - b.mean())
    n = len(a)
    mini = int(round(2 * fs * 60.0 / hr))
    if n <= 2 * mini:
        return np.nan
    out = []
    for k in rng.integers(mini, n - mini, N_NULL):
        z = za * np.conj(np.roll(zb, k))
        out.append(np.abs((z / np.maximum(np.abs(z), 1e-12)).mean()))
    return float(np.quantile(out, 0.95))


def main() -> None:
    controle = []
    for lot, racine in LOTS.items():
        lignes, payload = [], {}
        sources = {nom: np.load(racine / f) for nom, (f, _) in AUTRES.items()
                   if (racine / f).exists()}
        for f in sorted((racine / "traces").glob("*.npz")):
            slug = f.stem
            fct = CT_TRACES / f.name
            if not fct.exists():
                print(f"[{lot}] {slug}  pas d'epaisseur")
                continue
            with np.load(f) as z, np.load(fct) as zc:
                t, u = z["t"].astype(float), z["u_time"].astype(float)
                if not (np.allclose(t, zc["t"], atol=1e-4)
                        and np.allclose(u, zc["u_time"], atol=1e-4)):
                    print(f"[{lot}] {slug}  bases de temps differentes, ignoree")
                    continue
                fs, hr = float(z["fs"]), float(z["hr"])
                core = z["core"].astype(bool)
                pouls = z[PULSE].astype(float)
                ct_brut = zc["ct_brut_um"].astype(float)
                ct_filt_50_85 = zc["ct_filt_um"].astype(float)
            ct_u = np.interp(u, t, ct_brut)
            lo_bpm, hi_bpm = CardiacBand(
                expected_bpm=hr, expected_bpm_band_frac=BAND_FRAC).effective_bpm_range
            ct_fir = passe_bande(ct_u, fs, lo_bpm, hi_bpm)
            if lot == "A":
                # Controle : le meme filtre a 50-85 fixe doit rendre ct_filt_um.
                ref = passe_bande(ct_u, fs, 50.0, 85.0)
                controle.append(float(np.corrcoef(ref, ct_filt_50_85)[0, 1]))
            intensites = {"combinaison": pouls}
            for nom, z_src in sources.items():
                cle = slug + AUTRES[nom][1]
                if cle in z_src.files:
                    intensites[nom] = z_src[cle].astype(float)
            for nom, y in intensites.items():
                r, dphi, plv = indicateurs(y[core], ct_fir[core])
                lignes.append({"slug": slug, "intensite": nom, "hr_BPM": hr,
                               "bande_lo_BPM": lo_bpm, "bande_hi_BPM": hi_bpm,
                               "r": r, "abs_r": abs(r), "dphi_deg": dphi, "plv": plv,
                               "plv_null95": plv_nul(y[core], ct_fir[core], fs, hr,
                                                     f"{slug}/{nom}"),
                               "ct_amp_um": float(np.std(ct_fir[core])),
                               "ct_moyenne_um": float(np.mean(ct_brut)),
                               "n_core": int(core.sum())})
            payload[f"{slug}__ct_fir_um"] = ct_fir.astype(np.float32)
        df = pd.DataFrame(lignes)
        df = pd.concat([df, df["slug"].str.extract(RE_PART)], axis=1)
        df.to_csv(racine / "ct_pulse.csv", index=False)
        np.savez_compressed(racine / "ct_pulse_traces.npz", **payload)
        print(f"[{lot}] {df['slug'].nunique()} acquisitions -> {racine / 'ct_pulse.csv'}")
        for nom, g in df.groupby("intensite"):
            print(f"      {nom:<12s} |r| mediane {g['abs_r'].median():.2f}, "
                  f"PLV mediane {g['plv'].median():.2f}, "
                  f"PLV > nul95 : {(g['plv'] > g['plv_null95']).sum()}/{len(g)}")
    print(f"controle 50-85 : r min {min(controle):.6f} sur {len(controle)}")


if __name__ == "__main__":
    main()
