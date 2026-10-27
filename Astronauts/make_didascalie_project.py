# -*- coding: utf-8 -*-
"""
make_didascalie_project.py

Projet Didascalie (.dida) de controle de la segmentation de la choroide, a partir
des recalages ``registered_cascade_v9_plex_full_native`` de la cohorte SANS :

    E:/SANSORI/<NN_id>/<...>_rigidity/<...>/RawImages/
        <xxx>.xml + <xxx>.tif                       export Spectralis (OCT + localisateur)
        registered_cascade_v9_plex_full_native/
            registered_video.mp4                    T x 496 x 768, recale
            mask.npz                                choroide, T x 496 x 768 (packbits + zstd)
            timestamp.txt, registration_params.json

Pour chaque condition, ``--n-frames`` indices sont tires au hasard (graine fixe) et
donnent deux sequences :
    "<slug> | recale"   images de la video recalee + masque ``choroid``
    "<slug> | brut"     B-scans .tif natifs aux MEMES indices (ordre des horodatages
                        du XML, celui qu'a utilise le recalage)

Chaque image porte un texte ``source`` (indice, .tif, horodatage) ; le meme
contenu est ecrit dans ``<out>_frames.csv``. ``Reproducibility`` n'est pas inclus.

Lancer (kernel pyOR, depuis la racine) :
    python Astronauts/make_didascalie_project.py --limit 1 --out E:/SANSORI/_test.dida --overwrite
    python Astronauts/make_didascalie_project.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np

from didascalie import DidascalieProject, Label
from ocularrigidity.data.io import load_mask_frames
from ocularrigidity.data.spectralis import SpectralisStudy

PATH_GENERAL = Path("E:/SANSORI")
SUBDIR = "registered_cascade_v9_plex_full_native"
DEFAULT_OUT = PATH_GENERAL / "SANSORI_cascade_v9_5frames.dida"
LABEL = "choroid"
LABEL_COLOR = "#FF0000"


# --------------------------------------------------------------------------- #
# Lecture des images brutes : memes regles que
# ``ocularrigidity.scripts.registration.astronauts`` (copiees ici pour ne pas
# importer torch et le modele de segmentation qu'il charge).
def load_gray_u8(path) -> np.ndarray:
    """Charge une image .tif en niveaux de gris (H x W, uint8)."""
    img = iio.imread(path)
    if img.ndim == 3:
        img = img[..., 0]
    return img.astype(np.uint8)


def load_ordered_oct_series(raw_dir: Path) -> list:
    """Series OCT exploitables, triees par horodatage croissant."""
    xml_files = sorted(raw_dir.glob("*.xml"))
    if not xml_files:
        return []
    study = SpectralisStudy.from_file(xml_files[0])
    series = [
        s
        for s in study.series
        if s.oct is not None
        and s.oct_file_name
        and s.acquisition_time is not None
        and (raw_dir / s.oct_file_name).exists()
    ]
    series.sort(key=lambda s: s.acquisition_time.seconds_of_day)
    return series


def read_video_frames(path: Path, indices) -> dict[int, np.ndarray]:
    """Images ``indices`` de la video (canal 0), lues sequentiellement :
    un saut par CAP_PROP_POS_FRAMES n'est pas fiable en h264."""
    wanted = set(int(i) for i in indices)
    cap = cv2.VideoCapture(str(path))
    out = {}
    i = 0
    try:
        while len(out) < len(wanted):
            ok, frame = cap.read()
            if not ok:
                break
            if i in wanted:
                out[i] = np.ascontiguousarray(frame[..., 0])
            i += 1
    finally:
        cap.release()
    missing = wanted - out.keys()
    if missing:
        raise ValueError(f"{path}: images absentes de la video {sorted(missing)}")
    return out


# --------------------------------------------------------------------------- #
def list_conditions(root: Path) -> list[Path]:
    """Dossiers recales des sujets 01..14 (``Reproducibility`` exclu)."""
    return sorted(root.glob(f"[0-9][0-9]_*/*_rigidity/*/RawImages/{SUBDIR}"))


def slug_of(reg_dir: Path, root: Path) -> str:
    """``08_211209001/211209001post_rigidity/211209001post_rigidity_OS2``
    -> ``08_211209001/post/OS2``."""
    subject, moment_dir, cond_dir = reg_dir.relative_to(root).parts[:3]
    sid = subject.split("_", 1)[1]
    moment = moment_dir.removeprefix(sid).removesuffix("_rigidity")
    eye = cond_dir.rsplit("_", 1)[1]
    return f"{subject}/{moment}/{eye}"


def check_condition(reg_dir: Path) -> str | None:
    """Raison de sauter la condition, ou None."""
    for name in ("registered_video.mp4", "mask.npz", "registration_params.json"):
        if not (reg_dir / name).exists():
            return f"{name} manquant"
    params = json.loads((reg_dir / "registration_params.json").read_text())
    if params.get("status") != "ok":
        return f"status={params.get('status')!r}"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--root", type=Path, default=PATH_GENERAL)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-frames", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    conditions = list_conditions(args.root)
    if args.limit:
        conditions = conditions[: args.limit]
    print(f"{len(conditions)} conditions sous {args.root}")

    rng = np.random.default_rng(args.seed)
    rows = []
    n_done = 0
    project = DidascalieProject.create(
        args.out,
        name=f"SANSORI cascade_v9 plex full native ({args.n_frames} frames)",
        overwrite=args.overwrite,
    )
    with project:
        project.add_label(Label(name=LABEL, color=LABEL_COLOR))
        project.add_text_field("source")

        for k, reg_dir in enumerate(conditions, 1):
            raw_dir = reg_dir.parent
            slug = slug_of(reg_dir, args.root)
            reason = check_condition(reg_dir)
            if reason:
                print(f"[{k}/{len(conditions)}] {slug} : saute ({reason})")
                continue

            params = json.loads((reg_dir / "registration_params.json").read_text())
            T = int(params["n_frames"])
            series = load_ordered_oct_series(raw_dir)
            mask_shape = tuple(int(v) for v in np.load(reg_dir / "mask.npz")["shape"])
            ts_us = np.loadtxt(reg_dir / "timestamp.txt", dtype=np.int64).reshape(-1)
            if not (len(series) == T == mask_shape[0] == len(ts_us)):
                print(
                    f"[{k}/{len(conditions)}] {slug} : saute (series={len(series)}, "
                    f"n_frames={T}, mask={mask_shape[0]}, timestamps={len(ts_us)})"
                )
                continue
            H, W = mask_shape[1:]

            idx = np.sort(rng.choice(T, size=min(args.n_frames, T), replace=False))
            registered = read_video_frames(reg_dir / "registered_video.mp4", idx)
            masks = load_mask_frames(reg_dir / "mask.npz", idx)
            # Le cube recale a ete rogne a H, W pairs : on rogne le brut pareil.
            raws = [load_gray_u8(raw_dir / series[i].oct_file_name)[:H, :W] for i in idx]

            seq_reg = f"{slug} | recale"
            seq_raw = f"{slug} | brut"
            reg_ids = project.import_sequence(
                seq_reg,
                ((registered[i], {LABEL: m}) for i, m in zip(idx, masks)),
                format="PNG",
            )
            raw_ids = project.import_sequence(seq_raw, raws, format="PNG")

            with project.bulk():
                for j, i in enumerate(idx):
                    tif = series[i].oct_file_name
                    text = f"frame {i} / {tif} / t={ts_us[i]} us"
                    for seq, fid in ((seq_reg, reg_ids[j]), (seq_raw, raw_ids[j])):
                        project.add_text_description(fid, "source", text)
                        rows.append(
                            {
                                "slug": slug,
                                "sequence": seq,
                                "didascalie_index": j,
                                "source_index": int(i),
                                "tif": tif,
                                "timestamp_us": int(ts_us[i]),
                                "mask_px": int(masks[j].sum()) if seq == seq_reg else "",
                            }
                        )
            n_done += 1
            print(f"[{k}/{len(conditions)}] {slug} : frames {idx.tolist()}")

    csv_path = args.out.with_name(args.out.stem + "_frames.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["slug"])
        writer.writeheader()
        writer.writerows(rows)

    size_mb = args.out.stat().st_size / 1e6
    print(f"{n_done} conditions importees -> {args.out} ({size_mb:.0f} Mo), {csv_path}")


if __name__ == "__main__":
    main()
