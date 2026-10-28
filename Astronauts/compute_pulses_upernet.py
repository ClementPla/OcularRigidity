# -*- coding: utf-8 -*-
"""
compute_pulses_upernet.py

Pouls de la page « Pouls » du GUI (``data_visualizer_GUI/pulse_compute.py``) sur les
videos ``RawImages/upernet`` (``register_upernet.py``), masque brut ET masque lisse
dans le temps (``mask_smooth.npz``, notebook ``register_and_segment_smooth_mask``) :

    masques  epaisseur moyenne de la choroide par frame
    SVD      rang 100 (RANK_SVD) de traces normalisees par frame, bande de 117 um sous
             la RPE x 2,9 mm au centre, COMBINAISON OPTIMISEE autour de la FC de
             l'astronaute (visit_data.csv, +/- BAND_FRAC)
    SiNC     chaine PULSATION du package sur les frontieres BM/CSI du masque

Enregistrement : le cache du GUI, ``pulse_compute.cache_path(recalage, segmentation, ...)``
(``E:/NASA_Rigidity/SegmentationVariations/pulses_cache/...``) : la page « Pouls » les
affiche ensuite sans recalcul. Un pouls deja en cache (memes reglages) est saute.
Synthese : ``E:/NASA_Rigidity/SegmentationVariations/upernet/pulses_summary.csv``.

Lancer (kernel pyOR, depuis la racine) :
    python Astronauts/compute_pulses_upernet.py --limit 1
    python Astronauts/compute_pulses_upernet.py
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

# Ordre impose par les DLL de l'env pyOR : pandas, puis torch, puis le reste.
import pandas as pd
import torch  # noqa: F401

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "data_visualizer_GUI"))

import catalog  # noqa: E402
import pulse_compute  # noqa: E402

REG = "UPerNet (notebook, resolution d'origine)"
SEGS = {"brut": "UPerNet ConvNeXt-t (meme passe)",
        "lisse": "UPerNet ConvNeXt-t, lisse dans le temps"}
SUMMARY = catalog.SEGVAR_ROOT / "upernet" / "pulses_summary.csv"


def peak_bpm(res, spec) -> float:
    """Pic du spectre brut dans HR_SEARCH."""
    sel = (res.bpm >= pulse_compute.HR_SEARCH[0]) & (res.bpm <= pulse_compute.HR_SEARCH[1])
    return float(res.bpm[sel][np.asarray(spec)[sel].argmax()])


def row_of(cond, mask_kind, res, seconds) -> dict:
    svd, ph = res.svd_pulse, res.phase
    return {
        "astro": cond.astro, "condition": cond.name, "moment": cond.moment, "eye": cond.eye,
        "rep": cond.rep, "mask": mask_kind, "hr": res.hr, "hr_source": res.hr_source,
        "mask_peak_bpm": peak_bpm(res, res.mask_pulse.spec_raw),
        "svd_peak_bpm": peak_bpm(res, svd.spec_raw) if svd is not None else np.nan,
        "svd_n_components": int(res.svd_channels.size) if res.svd_channels is not None else 0,
        "sinc_bpm": res.sinc_bpm, "sinc_confidence": res.sinc_confidence,
        "dphi_svd_mask_deg": float(np.degrees(ph.mean)) if ph is not None else np.nan,
        "plv_svd_mask": ph.plv if ph is not None else np.nan,
        "sinc_note": res.sinc_note, "from_cache": res.from_cache, "seconds": round(seconds, 1),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pouls pulse_compute sur upernet, masque brut et lisse")
    ap.add_argument("--limit", type=int, default=None, help="nombre de conditions")
    ap.add_argument("--refresh", action="store_true", help="recalculer meme si en cache")
    args = ap.parse_args(argv)
    conds = [c for c in catalog.discover_conditions() if catalog.available(c).get(REG)]
    conds = conds[: args.limit] if args.limit is not None else conds
    print(f"{len(conds)} condition(s) x {len(SEGS)} masques -> {pulse_compute.CACHE_ROOT}", flush=True)
    rows = []
    if SUMMARY.exists():  # garder les lignes des conditions hors de ce passage
        old = pd.read_csv(SUMMARY)
        done = {(c.name, k) for c in conds for k in SEGS}
        rows = [r for r in old.to_dict("records") if (r["condition"], r["mask"]) not in done]
    t_all = time.perf_counter()
    for i, cond in enumerate(conds, 1):
        avail = catalog.available(cond).get(REG, [])
        for kind, seg in SEGS.items():
            if seg not in avail:
                print(f"[{i:>3}/{len(conds)}] {cond.name} {kind} : masque absent", flush=True)
                continue
            t0 = time.perf_counter()
            try:
                path = pulse_compute.cache_path(REG, seg, cond.astro, cond.name)
                res = None if args.refresh else pulse_compute.load_result(path)
                if res is None:
                    ds = catalog.DataStore._load(cond, REG, seg)
                    res = pulse_compute.cached_compute(path, ds.frames, ds.mask, ds.t, ds.hr_bpm,
                                                       ds.pixel_mm, refresh=True)
                r = row_of(cond, kind, res, time.perf_counter() - t0)
                rows.append(r)
                took = "cache" if res.from_cache else f"{r['seconds']:.0f} s"
                print(f"[{i:>3}/{len(conds)}] {cond.name} {kind:5} | FC {r['hr']:.1f} | pics masque "
                      f"{r['mask_peak_bpm']:.1f}, SVD {r['svd_peak_bpm']:.1f} ({r['svd_n_components']} comp.), "
                      f"SiNC {r['sinc_bpm']:.1f} | PLV {r['plv_svd_mask']:.2f} | {took}", flush=True)
            except Exception as e:  # noqa: BLE001 - une condition ne doit pas tout arreter
                traceback.print_exc()
                print(f"[{i:>3}/{len(conds)}] {cond.name} {kind} ERREUR {e}", flush=True)
        SUMMARY.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(SUMMARY, index=False)
    print(f"termine en {(time.perf_counter() - t_all) / 60:.1f} min -> {SUMMARY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
