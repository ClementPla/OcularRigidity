"""Patient-level train/val/test split of the choroid mask dataset.

Expected layout::

    MaskChoroid/<device>/images/<name>.png
    MaskChoroid/<device>/masks/<name>.png

Only names present in both ``images`` and ``masks`` are kept. The patient is read
from the file name (``1014711__2023-01-25__OD`` -> ``1014711``,
``01_210713001_before_OD_|_brut`` -> ``01_210713001``). PLEX and Spectralis share
patient ids, so a patient goes to the same split for every device. Patients are
stratified by the set of devices they appear in, so each split gets its share of
e.g. SANSORI patients.

    # Writes split.csv in the dataset root
    python -m ocularrigidity.scripts.segmentation.split_choroid_dataset

    # Also builds <out>/{train,val,test}/<device>/{images,masks}/ with symlinks
    python -m ocularrigidity.scripts.segmentation.split_choroid_dataset \\
        --out /path/to/MaskChoroid_split --link symlink
"""

import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/clement/Documents/data/OcularRigidityAnnotated/MaskChoroid/")
SPLITS = ("train", "val", "test")
SANSORI_ID = re.compile(r"^(\d+_\d+)_")


def patient_id(device: str, stem: str) -> str:
    if "__" in stem:
        return stem.split("__")[0]
    match = SANSORI_ID.match(stem)
    if match is None:
        raise ValueError(f"Cannot read the patient id from {device}/{stem}")
    # SANSORI ids live in their own namespace
    return f"{device}:{match.group(1)}"


def collect_pairs(root: Path) -> pd.DataFrame:
    rows = []
    for device_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        images, masks = device_dir / "images", device_dir / "masks"
        if not (images.is_dir() and masks.is_dir()):
            continue
        image_names = {p.name for p in images.glob("*.png")}
        mask_names = {p.name for p in masks.glob("*.png")}
        paired = sorted(image_names & mask_names)
        print(
            f"{device_dir.name}: {len(paired)} pairs "
            f"({len(image_names - mask_names)} images without mask, "
            f"{len(mask_names - image_names)} masks without image skipped)"
        )
        for name in paired:
            rows.append(
                {
                    "device": device_dir.name,
                    "name": name,
                    "patient": patient_id(device_dir.name, Path(name).stem),
                    "image": str(images / name),
                    "mask": str(masks / name),
                }
            )
    return pd.DataFrame(rows)


def split_patients(df: pd.DataFrame, ratios, seed: int) -> dict[str, str]:
    """Patient -> split, stratified on the set of devices of each patient."""
    rng = np.random.default_rng(seed)
    strata = df.groupby("patient")["device"].agg(lambda d: "+".join(sorted(set(d))))
    cuts = np.cumsum(ratios)[:-1]
    assignment = {}
    for _, patients in strata.groupby(strata):
        patients = np.array(sorted(patients.index))
        rng.shuffle(patients)
        bounds = np.round(cuts * len(patients)).astype(int)
        for split, chunk in zip(SPLITS, np.split(patients, bounds)):
            assignment.update({p: split for p in chunk})
    return assignment


def materialize(df: pd.DataFrame, out: Path, mode: str):
    for row in df.itertuples():
        for kind in ("image", "mask"):
            src = Path(getattr(row, kind))
            dst = out / row.split / row.device / f"{kind}s" / row.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if mode == "symlink":
                dst.symlink_to(src)
            else:
                shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.05, 0.15))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path, help="Default: <root>/split.csv")
    parser.add_argument("--out", type=Path, help="Folder for the split copies/links")
    parser.add_argument("--link", choices=("none", "symlink", "copy"), default="none")
    args = parser.parse_args()

    ratios = np.asarray(args.ratios, dtype=float)
    if not np.isclose(ratios.sum(), 1.0):
        parser.error(f"--ratios must sum to 1 (got {ratios.sum()})")
    if args.link != "none" and args.out is None:
        parser.error("--link needs --out")

    df = collect_pairs(args.root)
    assignment = split_patients(df, ratios, args.seed)
    df["split"] = df["patient"].map(assignment)

    # Sanity check: no patient in two splits
    assert (df.groupby("patient")["split"].nunique() == 1).all()

    summary = df.groupby(["split", "device"]).agg(
        patients=("patient", "nunique"), images=("name", "size")
    )
    print(summary.unstack("device").reindex(list(SPLITS)).fillna(0).astype(int))
    totals = df.groupby("split").agg(
        patients=("patient", "nunique"), images=("name", "size")
    ).reindex(list(SPLITS))
    totals["patients_%"] = (100 * totals.patients / totals.patients.sum()).round(1)
    totals["images_%"] = (100 * totals.images / totals.images.sum()).round(1)
    print(totals)

    csv = args.csv or args.root / "split.csv"
    df.to_csv(csv, index=False)
    print(f"Split written to {csv}")

    if args.link != "none":
        materialize(df, args.out, args.link)
        print(f"{args.link} tree written to {args.out}")


if __name__ == "__main__":
    main()
