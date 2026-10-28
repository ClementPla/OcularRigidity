"""Patient-level train/val/test split of the choroid mask dataset.

Expected layout::

    MaskChoroid/<device>/images/<name>.png
    MaskChoroid/<device>/masks/<name>.png

Only names present in both ``images`` and ``masks`` are kept. The patient is read
from the file name (``1014711__2023-01-25__OD`` -> ``1014711``,
``01_210713001_before_OD_|_recale_107`` -> ``01_210713001``). PLEX and Spectralis
share patient ids, so a patient goes to the same split for every device. Patients
are stratified by the set of devices they appear in, so each split gets its share
of e.g. SANSORI patients. Each stratum is shuffled with its own seed, so adding or
removing patients of one stratum never moves the patients of another.

Patients listed in ``--reference`` (a ``patient,split`` CSV) keep that split, so a
model trained on an earlier split can still be tested on images it never saw. The
default reference is ``split_v3.0.0.csv``, the split of segmentation model
``version-3.0.0``. Only new patients are drawn, and a device left without a val
patient gets one moved from train (never into test).

    # Writes split.csv in the dataset root
    python -m ocularrigidity.scripts.segmentation.split_choroid_dataset

    # Also builds <out>/{train,val,test}/<device>/{images,masks}/ with symlinks;
    # files of <out> the split does not place are removed
    python -m ocularrigidity.scripts.segmentation.split_choroid_dataset \\
        --out /path/to/MaskChoroid_split --link symlink
"""

import argparse
import re
import shutil
import zlib
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


def _rng(seed: int, key: str) -> np.random.Generator:
    """A generator of its own for ``key``, so strata never share random state."""
    return np.random.default_rng([seed, zlib.crc32(key.encode())])


def split_patients(
    df: pd.DataFrame, ratios, seed: int, reference: dict[str, str] | None = None
) -> dict[str, str]:
    """Patient -> split, stratified on the set of devices of each patient.

    Patients of ``reference`` keep their split; the others are drawn per stratum.
    """
    reference = reference or {}
    strata = df.groupby("patient")["device"].agg(lambda d: "+".join(sorted(set(d))))
    cuts = np.cumsum(ratios)[:-1]
    assignment = {p: reference[p] for p in strata.index if p in reference}
    for stratum, patients in strata.groupby(strata):
        new = np.array(sorted(p for p in patients.index if p not in reference))
        _rng(seed, stratum).shuffle(new)
        bounds = np.round(cuts * len(new)).astype(int)
        for split, chunk in zip(SPLITS, np.split(new, bounds)):
            assignment.update({p: split for p in chunk})

    # A device without any val patient gets one from train, so that validation
    # still tracks it; test is never touched.
    for device, patients in df.groupby("device")["patient"]:
        patients = sorted(set(patients))
        splits = {assignment[p] for p in patients}
        if "val" in splits or len(patients) < len(SPLITS):
            continue
        train = [p for p in patients if assignment[p] == "train"]
        # Prefer a patient no reference model was trained on
        pool = [p for p in train if p not in reference] or train
        if pool:
            moved = pool[_rng(seed, f"val:{device}").integers(len(pool))]
            assignment[moved] = "val"
            print(f"{device}: no val patient, moved {moved} from train to val")
    return assignment


def load_reference(path: Path) -> dict[str, str]:
    ref = pd.read_csv(path)
    unknown = set(ref["split"]) - set(SPLITS)
    if unknown:
        raise ValueError(f"{path}: unknown splits {sorted(unknown)}")
    return dict(zip(ref["patient"].astype(str), ref["split"]))


def materialize(df: pd.DataFrame, out: Path, mode: str):
    expected = set()
    for row in df.itertuples():
        for kind in ("image", "mask"):
            src = Path(getattr(row, kind))
            dst = out / row.split / row.device / f"{kind}s" / row.name
            expected.add(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if mode == "symlink":
                dst.symlink_to(src)
            else:
                shutil.copy2(src, dst)
    # Drop what an earlier split left behind, or a patient could sit in two splits
    stale = [
        p
        for split in SPLITS
        for p in (out / split).glob("*/*/*")
        if (p.is_file() or p.is_symlink()) and p not in expected
    ]
    for p in stale:
        p.unlink()
    if stale:
        print(f"Removed {len(stale)} stale files from {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.05, 0.15))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path, help="Default: <root>/split.csv")
    parser.add_argument("--out", type=Path, help="Folder for the split copies/links")
    parser.add_argument("--link", choices=("none", "symlink", "copy"), default="none")
    parser.add_argument(
        "--reference",
        type=Path,
        help="patient,split CSV whose patients keep their split. "
        "Default: <root>/split_v3.0.0.csv if it exists",
    )
    parser.add_argument(
        "--no-reference", action="store_true", help="Draw every patient afresh"
    )
    args = parser.parse_args()

    ratios = np.asarray(args.ratios, dtype=float)
    if not np.isclose(ratios.sum(), 1.0):
        parser.error(f"--ratios must sum to 1 (got {ratios.sum()})")
    if args.link != "none" and args.out is None:
        parser.error("--link needs --out")

    df = collect_pairs(args.root)
    reference = None
    if not args.no_reference:
        ref_path = args.reference or args.root / "split_v3.0.0.csv"
        if args.reference is not None or ref_path.exists():
            reference = load_reference(ref_path)
            kept = df["patient"].isin(reference).groupby(df["patient"]).first()
            print(f"Reference {ref_path}: {kept.sum()} patients keep their split, "
                  f"{(~kept).sum()} new")
    assignment = split_patients(df, ratios, args.seed, reference)
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
