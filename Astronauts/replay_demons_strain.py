"""
replay_demons_strain.py

Rejoue la chaine de ``compute_demons_strain.py`` sur UNE condition, en gardant
cette fois les CHAMPS DE DEPLACEMENT, les cartes de strain, le jacobien et les
images recalees.

Pourquoi un script separe : le lot produit 11 803 recalages ; conserver leurs
champs ferait ~250 Go pour une information qu'un seul recalage suffit a
reconstruire. Le lot n'ecrit donc que des agregats, et ce script est la porte
d'entree quand il faut REVOIR une condition -- les cartes de strain de la page
Quarto, ou le diagnostic d'une condition qui sort du lot.

Rien n'est recalcule differemment : ``prepare_condition`` et
``run_condition_demons`` sont IMPORTES du lot, avec les memes constantes. Le
one-cycle rejoue est donc bit a bit celui du lot.

Sortie
------
    <OUT_DIR>/<astro>/<moment>/<condition>/demons_strain_fields.npz

contenant, pour chaque (cas, sigma, one-cycle, bin) demande : ``u_x``, ``u_y``
(px), ``strain`` (e_yy, sans dimension), ``jacobian``, ``warped`` et ``fixed``,
plus la geometrie (roi, retine, depth_map, bandes laterales, crop) et les CT.

Lancer (kernel pyOR, depuis la racine du depot) :
    C:/Users/transformer/anaconda3/envs/pyOR/python.exe Astronauts/replay_demons_strain.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compute_demons_strain as cds  # noqa: E402

# --------------------------------------------------------------------------- #
# Parametres
# --------------------------------------------------------------------------- #
# La condition du carnet : c'est elle qui sert de temoin dans toutes les figures.
SLUG = "10_220215001__220215001post_rigidity_OD3"

# Restreindre ce qui est conserve : garder les quatre sigmas et les deux cas pour
# TOUS les one-cycles d'une condition longue fait deja plusieurs Go. None = tout.
KEEP_CASES = None  # ex. ("local",)
KEEP_SIGMAS = None  # ex. (12.0,)
KEEP_CYCLES = (0,)  # indices de one-cycle a conserver (0 = le premier), None = tous


def main() -> None:
    t_start = time.perf_counter()
    conditions = pd.read_csv(cds.CSV_PULSE)
    conditions = conditions[conditions["status"] == "ok"]
    if SLUG not in set(conditions["slug"]):
        raise SystemExit(f"{SLUG} absent de {cds.CSV_PULSE} (ou status != ok)")
    row = dict(conditions.set_index("slug").loc[SLUG])
    row["slug"] = SLUG

    print(f"rejeu : {SLUG}")
    prep = cds.prepare_condition(row)
    print(f"  {prep.n_groups} one-cycle(s), crop {prep.crop}, "
          f"{prep.um_y:.3f} um/px axial")
    for note in prep.notes:
        print(f"  note : {note}")

    bin_rows, region_rows, fields = cds.run_condition_demons(prep, keep_fields=True)
    print(f"  {len(bin_rows)} recalages en {time.perf_counter() - t_start:.0f} s")

    out_dir = cds.OUT_DIR / prep.astro / prep.moment / prep.condition
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "slug": prep.slug, "um_per_px_y": prep.um_y, "um_per_px_x": prep.um_x,
        "hr": prep.hr, "n_groups": prep.n_groups, "n_bins": cds.N_BINS,
        "crop": np.asarray(prep.crop, np.int32),
        "roi": prep.roi, "retina": prep.retina, "tissue": prep.tissue,
        "depth_map": prep.depth_map.astype(np.int16),
        "fixed_global": prep.fixed_global.astype(np.float32),
        "th_grid_um": prep.th_grid.astype(np.float32),
        "ref_bins": prep.ref_bins.astype(np.int16),
        "counts_mask": prep.counts_mask.astype(np.int32),
        "cols": prep.cols.astype(np.int32),
        "column_masks": np.stack(prep.column_masks),
        "cycles": prep.cycles.astype(np.float32),
        "mask_cycles": prep.mask_cycles.astype(np.float32),
    }

    kept = 0
    for (case, sigma, g, b), f in fields.items():
        if KEEP_CASES is not None and case not in KEEP_CASES:
            continue
        if KEEP_SIGMAS is not None and sigma not in KEEP_SIGMAS:
            continue
        if KEEP_CYCLES is not None and g not in KEEP_CYCLES:
            continue
        tag = f"{case}_s{sigma:g}_c{g}_b{b}"
        for name, arr in f.items():
            payload[f"{tag}__{name}"] = np.asarray(arr, np.float32)
        kept += 1

    out_path = out_dir / "demons_strain_fields.npz"
    np.savez_compressed(out_path, **payload)
    size_mb = out_path.stat().st_size / 1e6
    print(f"  {kept} champ(s) conserve(s) -> {out_path}  ({size_mb:.0f} Mo)")

    pd.DataFrame(bin_rows).to_csv(out_dir / "demons_strain_replay_bins.csv", index=False)
    pd.DataFrame(region_rows).to_csv(out_dir / "demons_strain_replay_regions.csv", index=False)
    print(f"  termine en {(time.perf_counter() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
