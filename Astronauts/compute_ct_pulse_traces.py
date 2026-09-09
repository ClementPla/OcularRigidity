"""
compute_ct_pulse_traces.py

Traces temporelles du pouls MASQUE (epaisseur choroidienne par frame) et du
pouls INTENSITE (`pulse_1_fir`), passees au MEME filtre FIR, pour les 104
conditions.

``compute_demons_strain.py`` calcule deja ces deux signaux, mais n'en garde que
la correlation et la covariance par one-cycle : les courbes elles-memes ne sont
nulle part, et la page Quarto en a besoin pour les superposer. Les recalculer ne
coute rien -- il n'y a ni video, ni recalage, ni repliement a refaire :

  - les MASQUES suffisent pour l'epaisseur par frame (``mask.npz``) ;
  - la base de temps, la frequence d'echantillonnage, la FC et le pouls
    d'intensite sont deja dans ``pulse_from_data/traces/<slug>.npz``.

C'est pourquoi ce script ne passe PAS par ``_prepared_registrator`` : charger le
``cube.mp4`` prendrait dix fois plus longtemps pour une information dont on n'a
pas besoin ici.

Le filtre est celui de la methode ``1_fir`` du lot d'extraction -- bande
FC +/- 20 %, phase lineaire, meme grille uniforme -- applique aux DEUX signaux.
C'est la condition pour que leur comparaison veuille dire quelque chose.

Arborescence lue
----------------
    E:/NASA_Rigidity/SegmentationVariations/<variante>/
        demons_strain/conditions.csv            <- conditions status == ok
        demons_strain/one_cycles.csv            <- fenetres temporelles
        registered_masks/<NN_id>/<...>/mask.npz
        pulse_from_data/traces/<slug>.npz       <- t, u_time, fs, hr, pouls

Sortie
------
    demons_strain/ct_pulse_traces.npz  (un jeu de tableaux par condition,
    prefixe par le slug : ``<slug>__u_time``, ``__ct_fir``, ``__pulse_fir``,
    ``__ct_brut``, ``__core``, plus ``slugs`` et ``hr``)

Lancer (kernel pyOR, depuis la racine du depot) :
    C:/Users/transformer/anaconda3/envs/pyOR/python.exe Astronauts/compute_ct_pulse_traces.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ocularrigidity.data.io import load_mask  # noqa: E402
from ocularrigidity.motion.filters._1d import spatio_temporal_filter  # noqa: E402
from ocularrigidity.motion.pulsation import CardiacBand  # noqa: E402

import compute_demons_strain as cds  # noqa: E402

OUT_NPZ = cds.OUT_DIR / "ct_pulse_traces.npz"


def main() -> None:
    t_start = time.perf_counter()
    conditions = pd.read_csv(cds.CSV_CONDITIONS)
    conditions = conditions[conditions["status"] == "ok"].reset_index(drop=True)
    print(f"{len(conditions)} condition(s) exploitable(s)")

    payload: dict = {}
    slugs, hrs = [], []
    n_fail = 0
    for i, row in conditions.iterrows():
        slug = row["slug"]
        astro, moment, condition = row["astro"], row["moment"], row["condition"]
        try:
            mask_path = (cds.VARIANT_ROOT / cds.MASKS_SUBDIR / astro / moment
                         / condition / "mask.npz")
            npz_path = cds.TRACES_DIR / f"{slug}.npz"
            masks = np.asarray(load_mask(mask_path), dtype=bool)
            data = np.load(npz_path)
            t = np.asarray(data["t"], dtype=float)
            u_time = np.asarray(data["u_time"], dtype=float)
            fs = float(data["fs"])
            hr = float(data["hr"])
            if masks.shape[0] != t.size:
                raise ValueError(f"{masks.shape[0]} masques != {t.size} horodatages")

            # Memes colonnes exploitables que le lot : sans quoi l'epaisseur
            # calculee ici ne serait pas la meme grandeur.
            W = masks.shape[2]
            roi_all = masks.all(axis=0)
            if roi_all.sum() < 100:
                roi_all = masks.mean(axis=0) > 0.5
            c0, c1 = int(cds.COL_FRAC[0] * W), int(cds.COL_FRAC[1] * W)
            cols = np.flatnonzero(roi_all[:, c0:c1].any(axis=0)) + c0
            if cols.size == 0:
                raise ValueError("aucune colonne exploitable")

            ct_brut = masks[:, :, cols].sum(axis=1).mean(axis=1).astype(np.float64)
            band = CardiacBand(expected_bpm=hr, expected_bpm_band_frac=cds.BAND_FRAC)
            lo_bpm, hi_bpm = band.effective_bpm_range
            nyq = 0.5 * fs
            low_cut = (lo_bpm / 60.0) / nyq
            high_cut = min((hi_bpm / 60.0) / nyq, 0.99)
            ct_u = np.interp(u_time, t, ct_brut)
            um_y = float(row["um_per_px_y"])
            ct_fir = spatio_temporal_filter(ct_u[:, None], 0.0, low_cut, high_cut,
                                            fs, None)[:, 0] * um_y

            payload[f"{slug}__u_time"] = u_time.astype(np.float32)
            payload[f"{slug}__ct_fir"] = ct_fir.astype(np.float32)
            payload[f"{slug}__pulse_fir"] = np.asarray(
                data[f"pulse_{cds.PULSE_METHOD}"], dtype=np.float32)
            payload[f"{slug}__ct_brut_um"] = (ct_brut * um_y).astype(np.float32)
            payload[f"{slug}__t"] = t.astype(np.float32)
            payload[f"{slug}__core"] = np.asarray(
                cds.occ.edge_mask(u_time, hr), dtype=bool)
            slugs.append(slug)
            hrs.append(hr)
            print(f"[{i + 1:>3d}/{len(conditions)}] {slug}  "
                  f"{u_time.size} points, CT {ct_brut.mean() * um_y:.0f} um", flush=True)
        except Exception as exc:  # noqa: BLE001 - une condition ne doit pas tout arreter
            n_fail += 1
            print(f"[{i + 1:>3d}/{len(conditions)}] {slug}  ECHEC : {exc}", flush=True)

    payload["slugs"] = np.asarray(slugs)
    payload["hr"] = np.asarray(hrs, dtype=np.float32)
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ, **payload)
    print(f"\n{len(slugs)} condition(s), {n_fail} en echec  "
          f"({(time.perf_counter() - t_start) / 60:.1f} min)")
    print(f"  {OUT_NPZ}  ({OUT_NPZ.stat().st_size / 1e6:.1f} Mo)")


if __name__ == "__main__":
    main()
