# -*- coding: utf-8 -*-
"""
compute_one_cycle_strain.py

Strain retinien par demons sur les ONE-CYCLES cascade_v9 des replicats, puis
repetabilite du profil lateral (ICC regions x replicats).

Aucune ligne de la chaine demons n'est recopiee : ``Astronauts/compute_demons_strain.py``
est importe (``cds``) et recoit, a la place de son propre repliement, les
one-cycles deja calcules par ``compute_one_cycle.py`` :

    E:/NASA_Rigidity/Reproducibility/SegmentationVariations/
        cascade_v9_1536x1024_hr_mediane/one_cycle_6beats_6bins/cycles/<slug>.npz

Chaine par replicat
-------------------
  1. one-cycles (frames repliees uint8, occupation des masques, affectation des
     frames) relus tels quels ;
  2. video recalee, horodatages XML et pouls du lot A relus pour ce que
     ``cds._prepare_regions`` demande (CT/pouls, echelles) ;
  3. egalisation par A-scan des bins sur l'image moyenne de la video
     (``cds.match_ascans``, comme le lot SANS) ;
  4. ``cds._prepare_regions`` : CT par bin, bin de reference = CT MINIMUM de
     chaque one-cycle, retine par Otsu, crop, 5 bandes laterales ;
  5. ``cds.run_condition_demons`` : FastSymmetricForces, reference LOCALE,
     ``displacement_sigma = 15``, les 5 bins mobiles contre le bin mince.

Le MARQUEUR d'un one-cycle est la paire bin MINCE -> bin EPAIS (amplitude
complete de la pulsation), comme dans ``compute_repeatability.py``. Les autres
bins restent dans ``bins.csv`` / ``regions.csv``. Seuls les champs de la paire
marqueur sont conserves.

ICC
---
Un seul oeil : le sujet ne peut pas etre l'oeil. Les SUJETS sont les 5 bandes
laterales, les EVALUATEURS les replicats, la valeur la MEDIANE des one-cycles.
La question tranchee est « le profil lateral se repete-t-il d'un replicat a
l'autre », avec 5 sujets seulement -- des IC tres larges. Temoin : le meme ICC
sur ``dCT_region_um`` de la meme paire.

Sorties (sous ``.../one_cycle_6beats_6bins/demons_strain_s15/``)
----------------------------------------------------------------
    bins.csv  regions.csv  markers.csv  icc.csv  params.json  fields/<slug>.npz
    retina_bins.csv   epaisseur retinienne ET choroidienne par bin (cf. ``retina_by_bin``)
    retina_bands.csv  epaisseur retinienne par bin ET par bande laterale (les 5 du strain)

Lancer (kernel pyOR, depuis la racine du depot) :
    C:/Users/transformer/anaconda3/envs/pyOR/python.exe Reproducibility/compute_one_cycle_strain.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "Astronauts"))

import compute_demons_strain as cds  # noqa: E402  (pose CUDA_VISIBLE_DEVICES="" avant torch)
import pingouin as pg  # noqa: E402
import SimpleITK as sitk  # noqa: E402

from ocularrigidity.motion.video_timeline_aligner import VideoTimelineAligner  # noqa: E402
from ocularrigidity.scripts.one_cycle.astronauts import _prepared_registrator  # noqa: E402
from ocularrigidity.scripts.registration.astronauts import load_ordered_oct_series  # noqa: E402
from ocularrigidity.thickness.features import compute_deltaY_masks  # noqa: E402

occ = cds.occ

# --------------------------------------------------------------------------- #
# Parametres
# --------------------------------------------------------------------------- #
PATH_GENERAL = Path("E:/SANSORI/Reproducibility")
SEGVAR = Path("E:/NASA_Rigidity/Reproducibility/SegmentationVariations")
DATA_SUBDIR = "registered_cascade_v9_1536x1024"
PULSE_DIR = SEGVAR / "cascade_v9_1536x1024_hr_mediane" / "pulse_from_data"
ONE_CYCLE_DIR = SEGVAR / "cascade_v9_1536x1024_hr_mediane" / "one_cycle_6beats_6bins"
SIGMA = 15.0
OUT_DIR = ONE_CYCLE_DIR / f"demons_strain_s{SIGMA:g}"
DEFAULT_SLUGS = ("MODICA_GRAZIANA_OD1", "MODICA_GRAZIANA_OD2", "MODICA_GRAZIANA_OD3")

MARKERS = {
    "strain_retine": "strain rétinien moyen",
    "strain_retine_med": "strain rétinien médian",
    "strain_retine_p95": "strain rétinien p95 signé",
    "dCT_region_um": "témoin : dCT de la bande (µm)",
}
ICC_TYPES = {"ICC1": "ICC(1,1)", "ICC2": "ICC(A,1)", "ICC3": "ICC(C,1)",
             "ICC(1,1)": "ICC(1,1)", "ICC(A,1)": "ICC(A,1)", "ICC(C,1)": "ICC(C,1)"}
RE_SLUG = re.compile(r"^(?P<participant>.+)_(?P<eye>OD|OS)(?P<replicate>\d+)$")

# La chaine demons tourne dans CE processus : reference locale, un seul sigma.
cds.REF_CASES = ("local",)
cds.SIGMAS = (SIGMA,)


# --------------------------------------------------------------------------- #
# Un replicat
# --------------------------------------------------------------------------- #
def prepare(slug: str, n_beats: float):
    oc = np.load(ONE_CYCLE_DIR / "cycles" / f"{slug}.npz")
    raw_dir = PATH_GENERAL / slug / "RawImages"
    data_dir = raw_dir / DATA_SUBDIR

    registrator = _prepared_registrator(data_dir / "registered_video.mp4",
                                        data_dir / "mask.npz", "cpu", verbose=False)
    frames = np.asarray(registrator.registered_frames)
    masks = np.asarray(registrator.registered_masks, dtype=bool)
    ts_us = occ.raw_timestamps_us(raw_dir)
    aligner = VideoTimelineAligner(registrator, ts_us)
    t, u_time, fs = aligner.timestamps_seconds, aligner.uniform_time, float(aligner.fs)

    data = np.load(PULSE_DIR / "traces" / f"{slug}.npz")
    if not np.allclose(t, data["t"], atol=1e-6):
        raise ValueError("les horodatages du .npz de pouls ne sont pas ceux de la video")
    pulse_on_frames = np.interp(t, u_time, np.asarray(data[f"pulse_{cds.PULSE_METHOD}"], float))

    series = load_ordered_oct_series(raw_dir)
    lat = [s.lateral_resolution for s in series if s.lateral_resolution]
    um_x = 1000.0 * float(np.median(lat)) if lat else float("nan")

    counts = np.asarray(oc["counts"])
    cycles = cds.match_ascans(np.asarray(oc["cycles"], np.float32),
                              frames.mean(axis=0).astype(np.float32))
    n_groups = int(oc["n_groups"])
    prep = cds._prepare_regions(
        {}, slug, "Reproducibility", RE_SLUG.match(slug)["participant"], slug,
        frames, masks, data, t, u_time, fs, float(oc["hr"]), np.asarray(oc["cols"]),
        float(oc["um_per_px_y"]), um_x, cycles,
        np.asarray(oc["occupancy"], np.float32), counts, n_groups,
        np.asarray(oc["group_of"]), np.asarray(oc["good_f"], bool),
        np.asarray(oc["good_slot"], bool), np.asarray(oc["slot_of"]), pulse_on_frames,
        float(oc["phase_max"]), float(oc["phase_min"]), float(n_beats), [], {})
    return prep


def process(slug: str, n_beats: float):
    t0 = time.perf_counter()
    m = RE_SLUG.match(slug)
    prep = prepare(slug, n_beats)
    bin_rows, region_rows, fields = cds.run_condition_demons(prep, keep_fields=True)

    thick = np.full(prep.n_groups, -1, dtype=int)
    for g in range(prep.n_groups):
        if prep.ref_bins[g] >= 0 and np.isfinite(prep.th_grid[g]).sum() >= 2:
            thick[g] = int(np.nanargmax(prep.th_grid[g]))

    extra = {"replicate": int(m["replicate"]), "participant": m["participant"], "eye": m["eye"]}
    for r in bin_rows + region_rows:
        r.update(extra, is_thick=bool(r["bin"] == thick[r["one_cycle"] - 1]))

    payload = {
        "slug": np.array(slug), "um_y": prep.um_y, "um_x": prep.um_x,
        "crop": np.asarray(prep.crop), "roi": prep.roi, "retina": prep.retina,
        "tissue": prep.tissue, "column_masks": np.stack(prep.column_masks),
        "column_labels": np.array([cds.column_label(k, prep.eye) for k in range(cds.N_COLUMNS)]),
        "ref_bins": prep.ref_bins, "thick_bins": thick, "th_grid_um": prep.th_grid,
        "n_groups": prep.n_groups, "sigma": SIGMA,
    }
    for g in range(prep.n_groups):
        key = ("local", SIGMA, g, int(thick[g]))
        if thick[g] < 0 or thick[g] == prep.ref_bins[g] or key not in fields:
            continue
        f = fields[key]
        payload[f"c{g}__u_y"] = f["u_y"].astype(np.float32)
        payload[f"c{g}__strain"] = f["strain"].astype(np.float32)
        payload[f"c{g}__fixed"] = f["fixed"].astype(np.float32)
        payload[f"c{g}__warped"] = f["warped"].astype(np.float32)
        payload[f"c{g}__jacobian"] = f["jacobian"].astype(np.float32)
    (OUT_DIR / "fields").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_DIR / "fields" / f"{slug}.npz", **payload)

    n_thick = sum(r["is_thick"] for r in bin_rows)
    print(f"{slug}: {prep.n_groups} one-cycles, crop {prep.crop}, bins minces "
          f"{prep.ref_bins.tolist()}, epais {thick.tolist()}, {len(bin_rows)} recalages "
          f"({n_thick} paires marqueur) en {time.perf_counter() - t0:.0f} s", flush=True)
    return bin_rows, region_rows


# --------------------------------------------------------------------------- #
# Epaisseur de la retine par bin
# --------------------------------------------------------------------------- #
def retina_by_bin(slug: str) -> pd.DataFrame:
    """Epaisseur RETINIENNE de chaque bin du one-cycle, a cote de la choroidienne.

    Segmentation : ``cds.segment_tissue`` -- celle de la chaine demons (Otsu,
    de l'ILM au haut de la choroide), appliquee a CHAQUE bin egalise par A-scan
    avec la choroide de ce bin (occupation > 0,5). Epaisseur :
    ``thickness.features.compute_deltaY_masks`` (somme du masque par colonne),
    moyennee sur les MEMES colonnes ``cols`` que l'epaisseur choroidienne du lot
    de one-cycles -- les deux courbes portent donc sur les memes A-scans.
    """
    m = RE_SLUG.match(slug)
    oc = np.load(ONE_CYCLE_DIR / "cycles" / f"{slug}.npz")
    data_dir = PATH_GENERAL / slug / "RawImages" / DATA_SUBDIR
    registrator = _prepared_registrator(data_dir / "registered_video.mp4",
                                        data_dir / "mask.npz", "cpu", verbose=False)
    frames = np.asarray(registrator.registered_frames)
    cycles = cds.match_ascans(np.asarray(oc["cycles"], np.float32),
                              frames.mean(axis=0).astype(np.float32))
    occupancy = np.asarray(oc["occupancy"], np.float32)
    counts, cols = np.asarray(oc["counts"]), np.asarray(oc["cols"])
    n_groups, n_bins = int(oc["n_groups"]), int(oc["n_bins"])

    retina = np.zeros(cycles.shape, dtype=bool)
    notes = []
    for s in range(cycles.shape[0]):
        roi = occupancy[s] > 0.5
        if counts[s] == 0 or not roi.any():
            continue
        try:
            retina[s] = cds.segment_tissue(cycles[s], roi)[0]
        except ValueError as e:  # Otsu inadapte sur ce bin : epaisseur inconnue
            notes.append(f"bin {s} : {e}")
    th = compute_deltaY_masks(retina)[:, cols].mean(axis=1).astype(np.float64)
    th[(counts == 0) | ~retina.any(axis=(1, 2))] = np.nan
    fit = np.concatenate([occ.fit_cycle(th[g * n_bins:(g + 1) * n_bins]) for g in range(n_groups)])

    rows = []
    for s in range(n_groups * n_bins):
        rows.append({
            "slug": slug, "replicate": int(m["replicate"]), "slot": s,
            "cycle": s // n_bins + 1, "bin": s % n_bins, "n_frames": int(counts[s]),
            "um_per_px_y": float(oc["um_per_px_y"]),
            "retina_thickness_px": th[s], "retina_thickness_fit_px": fit[s],
            "choroid_thickness_px": float(oc["thickness_fold"][s]),
            "choroid_thickness_fit_px": float(oc["thickness_fold_fit"][s]),
            "choroid_thickness_seg_px": float(oc["thickness_seg"][s]),
        })
    # --- Par bande laterale : les MEMES 5 bandes que le strain -----------------
    # Les masques de bande de ``fields/<slug>.npz`` sont croppes en y seulement :
    # leur etendue en x vaut pour le bin entier. Colonnes = celles de ``cols``
    # tombant dans la bande, comme ``band_cols`` de ``cds._prepare_regions``.
    fz = np.load(OUT_DIR / "fields" / f"{slug}.npz")
    per_col = compute_deltaY_masks(retina).astype(np.float64)  # (slots, W)
    band_rows = []
    for k, (cm, label) in enumerate(zip(fz["column_masks"], fz["column_labels"])):
        xs = np.flatnonzero(cm.any(axis=0))
        bc = cols[(cols >= xs[0]) & (cols <= xs[-1])]
        if bc.size == 0:
            bc = xs
        th_k = per_col[:, bc].mean(axis=1)
        th_k[(counts == 0) | ~retina.any(axis=(1, 2))] = np.nan
        fit_k = np.concatenate([occ.fit_cycle(th_k[g * n_bins:(g + 1) * n_bins])
                                for g in range(n_groups)])
        # Choroide de la MEME bande et des MEMES colonnes : occupation repliee du
        # bin, sommee par A-scan (definition de ``CT_region_um`` dans ``cds``).
        ch_k = occupancy[:, :, bc].sum(axis=1).mean(axis=1).astype(np.float64)
        ch_k[counts == 0] = np.nan
        for s in range(n_groups * n_bins):
            band_rows.append({
                "slug": slug, "replicate": int(m["replicate"]), "slot": s,
                "cycle": s // n_bins + 1, "bin": s % n_bins, "k": k, "region": str(label),
                "n_cols": int(bc.size), "n_frames": int(counts[s]),
                "um_per_px_y": float(oc["um_per_px_y"]),
                "retina_thickness_px": th_k[s], "retina_thickness_fit_px": fit_k[s],
                "choroid_thickness_px": ch_k[s],
            })

    ptp = [np.nanmax(fit[g * n_bins:(g + 1) * n_bins]) - np.nanmin(fit[g * n_bins:(g + 1) * n_bins])
           for g in range(n_groups)]
    print(f"{slug}: retine {np.nanmean(th):.1f} px en moyenne, harmonique crete-a-crete "
          f"median {np.nanmedian(ptp):.2f} px{' | ' + '; '.join(notes) if notes else ''}",
          flush=True)
    return pd.DataFrame(rows), pd.DataFrame(band_rows)


# --------------------------------------------------------------------------- #
# ICC regions x replicats
# --------------------------------------------------------------------------- #
def icc_one_way(mat: np.ndarray) -> float:
    """ICC(1,1) a la main (ANOVA a un facteur), pour controler pingouin."""
    n, k = mat.shape
    grand = mat.mean()
    msb = k * ((mat.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    msw = ((mat - mat.mean(axis=1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    return float((msb - msw) / (msb + (k - 1) * msw))


def icc_table(markers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in MARKERS.items():
        med = (markers.groupby(["region", "k", "replicate"])[col].median()
               .reset_index().rename(columns={col: "value"}))
        mat = med.pivot(index="k", columns="replicate", values="value").to_numpy()
        tab = pg.intraclass_corr(data=med, targets="region", raters="replicate",
                                 ratings="value")
        for _, r in tab.iterrows():
            typ = ICC_TYPES.get(str(r["Type"]))
            if typ is None:
                continue
            ci = r["CI95"] if "CI95" in r.index else r["CI95%"]  # nom selon la version
            rows.append({"marqueur": col, "libelle": label, "type": typ,
                         "icc": float(r["ICC"]), "ci_bas": float(ci[0]), "ci_haut": float(ci[1]),
                         "F": float(r["F"]), "p": float(r["pval"]),
                         "n_regions": mat.shape[0], "n_replicats": mat.shape[1],
                         "icc11_manuel": icc_one_way(mat) if typ == "ICC(1,1)" else np.nan})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Strain demons sur les one-cycles cascade_v9.")
    parser.add_argument("--slug", action="append")
    parser.add_argument("--icc-only", action="store_true",
                        help="relit bins.csv / markers.csv sans relancer les demons")
    args = parser.parse_args(argv)
    slugs = args.slug or list(DEFAULT_SLUGS)
    t0 = time.perf_counter()

    if args.icc_only:
        bins = pd.read_csv(OUT_DIR / "bins.csv")
        markers = pd.read_csv(OUT_DIR / "markers.csv")
    else:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(os.cpu_count() or 8)
        conds = pd.read_csv(ONE_CYCLE_DIR / "conditions.csv").set_index("slug")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        all_bins, all_regions = [], []
        for slug in slugs:
            b, r = process(slug, conds.loc[slug, "n_beats"])
            all_bins += b
            all_regions += r
        bins, regions = pd.DataFrame(all_bins), pd.DataFrame(all_regions)
        bins.to_csv(OUT_DIR / "bins.csv", index=False)
        regions.to_csv(OUT_DIR / "regions.csv", index=False)
        markers = regions[regions["is_thick"]].copy()
        markers.to_csv(OUT_DIR / "markers.csv", index=False)
    icc = icc_table(markers)
    icc.to_csv(OUT_DIR / "icc.csv", index=False)
    retina = [retina_by_bin(s) for s in slugs]
    pd.concat([r[0] for r in retina], ignore_index=True).to_csv(
        OUT_DIR / "retina_bins.csv", index=False)
    pd.concat([r[1] for r in retina], ignore_index=True).to_csv(
        OUT_DIR / "retina_bands.csv", index=False)

    params = {
        "slugs": slugs, "sigma": SIGMA, "method": cds.METHOD, "ref_case": "local",
        "demons_params": vars(cds.DemonsParams(displacement_sigma=SIGMA)),
        "marker_pair": "bin de CT minimum -> bin de CT maximum, par one-cycle",
        "icc": "targets = region (5 bandes), raters = replicat, valeur = mediane des one-cycles",
        "one_cycles": str(ONE_CYCLE_DIR), "hist_match_ascan": True,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    (OUT_DIR / "params.json").write_text(json.dumps(params, indent=2, default=str),
                                         encoding="utf-8")

    pd.set_option("display.width", 200)
    print("\nrecalages marqueur :")
    print(bins[bins["is_thick"]][["slug", "one_cycle", "ref_bin", "bin", "dCT_um", "u_max_um",
                                  "jac_neg_pct", "rms_avant", "rms_apres",
                                  "strain_retine_moy", "strain_choroide_med"]].to_string(index=False))
    print("\nmediane des one-cycles, strain retinien moyen :")
    print(markers.pivot_table(index="region", columns="replicate", values="strain_retine",
                              aggfunc="median").to_string())
    print("\nICC :")
    print(icc[["marqueur", "type", "icc", "ci_bas", "ci_haut", "p", "icc11_manuel"]]
          .round(3).to_string(index=False))
    print(f"\ntermine en {(time.perf_counter() - t0) / 60:.1f} min -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
