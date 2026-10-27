"""3D nnU-Net of the choroidal vessels, trained on the en face annotations.

The network works on the grid the vessels are annotated on (see
``ocularrigidity.segmentation.vessels.enface``): the OCTA amplitude flattened on
the BM and binned along the depth, ``(slices, B-scans, A-scans)`` with voxels of
about 12 um. Only the sequences whose every frame is reviewed are used.

    # Annotations back to the native (B-scans, depth, A-scans) volumes
    python -m ocularrigidity.scripts.segmentation.choroid_vessels_nnunet remap

    # nnU-Net raw dataset, then the usual nnU-Net commands (printed at the end)
    python -m ocularrigidity.scripts.segmentation.choroid_vessels_nnunet dataset

    # Segment a new volume once a model is trained
    python -m ocularrigidity.scripts.segmentation.choroid_vessels_nnunet predict \\
        --volume "/mnt/smb/.../OCTA Macula/OD" --out vessels.npz

nnU-Net reads its folders from ``nnUNet_raw``, ``nnUNet_preprocessed`` and
``nnUNet_results``; they default to ``NNUNET_ROOT`` below.
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm.auto import tqdm

from ocularrigidity.data.io import DEPTH, load_octa, save_mask_atomic
from ocularrigidity.segmentation.vessels.enface import (
    DEPTH_BIN,
    MARGIN_TOP,
    choroid_boundaries,
    enface_volume,
    grid_depth,
    grid_spacing,
    read_dida_masks,
    to_native,
    to_native_linear,
)

NNUNET_ROOT = Path("/media/clement/HD/Santiago/OcularRigidity/nnUNet")
# nnunetv2 reads them on import
for _name in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
    os.environ.setdefault(_name, str(NNUNET_ROOT / _name))



REPO = Path(__file__).resolve().parents[4]
ANNOTATIONS = Path(
    "/home/clement/Documents/data/OcularRigidity/Choroid vessels (en face)2.dida"
)
# Written by notebooks/to_labelmed/export_choroid_vessel_volumes.py with the
# project the annotations were made on.
GEOMETRY = REPO / "choroid_vessels_enface.geometry.npz"
SELECTION = REPO / "notebooks/to_labelmed/choroid_vessel_volumes.csv"
# The project exported for annotation: its frames identify the volumes.
EXPORTED = REPO / "choroid_vessels_enface.dida"
NATIVE_MASKS = Path(
    "/media/clement/HD/Santiago/OcularRigidity/choroid_vessels_manual/annotations"
)

DATASET_ID = 501
DATASET_NAME = f"Dataset{DATASET_ID:03d}_ChoroidVessels"
CONFIGURATION = "3d_fullres"
TRAINER = "nnUNetTrainer"

PLANNER = "nnUNetPlannerResEncL"
PLANS = "nnUNetResEncUNetLPlans"


def load_annotations(path: Path = ANNOTATIONS) -> dict:
    """``{sequence: (masks, bm, z0)}`` of the reviewed sequences, checked against the export geometry."""
    masks, partial = read_dida_masks(path)
    for name, (done, total) in partial.items():
        print(f"skipped {name}: {done}/{total} frames reviewed")
    geometry = np.load(GEOMETRY)
    if int(geometry["depth_bin"]) != DEPTH_BIN:
        raise ValueError(f"{GEOMETRY} was exported with another depth bin")

    out = {}
    for name, mask in masks.items():
        bm = geometry[f"{name}/bm"]
        z0 = int(geometry[f"{name}/z0"])
        n_slices = (grid_depth(bm, geometry[f"{name}/csi"]) - z0) // DEPTH_BIN
        if mask.shape != (n_slices, *bm.shape):
            raise ValueError(f"{name}: masks {mask.shape}, grid {(n_slices, *bm.shape)}")
        out[name] = (mask, bm, z0)
    return out


def remap(args):
    """Annotations as ``(B-scans, depth, A-scans)`` masks of the native volumes."""
    NATIVE_MASKS.mkdir(parents=True, exist_ok=True)
    for name, (mask, bm, z0) in tqdm(load_annotations(args.dida).items(), desc="remap"):
        native = to_native(mask, bm, z0, DEPTH)
        save_mask_atomic(native, NATIVE_MASKS / f"{name}.npz")
        tqdm.write(f"{name}: {mask.sum()} grid voxels -> {native.sum()} native voxels")
    print(f"Wrote {NATIVE_MASKS}")


def write_nifti(array: np.ndarray, spacing: tuple, path: Path) -> None:
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(tuple(float(s) for s in spacing[::-1]))  # SimpleITK is (x, y, z)
    sitk.WriteImage(image, str(path), useCompression=True)


def dataset(args):
    """nnU-Net raw dataset: the amplitude grid of every reviewed volume and its vessels."""
    annotations = load_annotations(args.dida)
    paths = pd.read_csv(SELECTION, dtype=str).set_index("sequence")["path"]
    root = Path(os.environ["nnUNet_raw"]) / DATASET_NAME
    (root / "imagesTr").mkdir(parents=True, exist_ok=True)
    (root / "labelsTr").mkdir(parents=True, exist_ok=True)

    for name, (mask, bm, z0) in tqdm(annotations.items(), desc="dataset"):
        amp, _, _ = load_octa(Path(paths[name]))
        amp = np.ascontiguousarray(amp).transpose(0, 2, 1)  # (B-scans, depth, A-scans)
        image = enface_volume(amp, bm, len(mask), z0)
        spacing = grid_spacing(*bm.shape)
        write_nifti(image, spacing, root / "imagesTr" / f"{name}_0000.nii.gz")
        write_nifti(mask.astype(np.uint8), spacing, root / "labelsTr" / f"{name}.nii.gz")

    with open(root / "dataset.json", "w") as f:
        json.dump(
            {
                "channel_names": {"0": "OCT"},  # z-scored per volume
                "labels": {"background": 0, "vessel": 1},
                "numTraining": len(annotations),
                "file_ending": ".nii.gz",
                "name": DATASET_NAME,
                "description": "Choroidal vessels on BM-flattened OCTA amplitude, en face annotations",
            },
            f,
            indent=2,
        )
    env = " ".join(f"{k}={os.environ[k]}" for k in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"))
    print(
        f"Wrote {root} ({len(annotations)} volumes). Then:\n"
        f"  export {env}\n"
        f"  nnUNetv2_plan_and_preprocess -d {DATASET_ID} -c {CONFIGURATION} -pl {PLANNER} --verify_dataset_integrity\n"
        f"  for fold in 0 1 2 3 4; do CUDA_VISIBLE_DEVICES=0 nnUNetv2_train {DATASET_ID} {CONFIGURATION} $fold -p {PLANS} --npz; done\n"
        f"  # final model on every volume once the folds look right\n"
        f"  CUDA_VISIBLE_DEVICES=0 nnUNetv2_train {DATASET_ID} {CONFIGURATION} all -p {PLANS}"
    )


def load_predictor(folds=None, checkpoint: str | None = None, device: str = "cuda:0"):
    """nnU-Net predictor of the trained folds.

    ``folds`` defaults to every fold with a checkpoint, and ``checkpoint`` to
    ``checkpoint_final.pth``, or ``checkpoint_best.pth`` while a fold is still
    training.
    """
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    model_dir = (
        Path(os.environ["nnUNet_results"])
        / DATASET_NAME
        / f"{TRAINER}__{PLANS}__{CONFIGURATION}"
    )
    names = [checkpoint] if checkpoint else ["checkpoint_final.pth", "checkpoint_best.pth"]
    if folds is None:
        folds = sorted(
            d.name.removeprefix("fold_")
            for d in model_dir.glob("fold_*")
            if any((d / n).exists() for n in names)
        )
    folds = [int(f) if str(f) != "all" else "all" for f in folds]
    if not folds:
        raise FileNotFoundError(f"No trained fold in {model_dir}")
    # Every fold must have the same checkpoint name: the first one they all have.
    name = next(
        (n for n in names if all((model_dir / f"fold_{f}" / n).exists() for f in folds)),
        None,
    )
    if name is None:
        raise FileNotFoundError(f"Folds {folds} of {model_dir} share no checkpoint in {names}")

    predictor = nnUNetPredictor(device=torch.device(device))
    predictor.initialize_from_trained_model_folder(
        str(model_dir), use_folds=tuple(folds), checkpoint_name=name
    )
    print(f"{model_dir.name}: folds {folds}, {name}")
    return predictor


def predict_grid(predictor, image: np.ndarray, probabilities: bool = False) -> np.ndarray:
    """``(slices, T, W)`` vessels of an amplitude grid (``enface_volume``): bool,
    or the vessel probability with ``probabilities``."""
    spacing = grid_spacing(*image.shape[1:])
    out = predictor.predict_single_npy_array(
        image[None].astype(np.float32),
        {"spacing": list(spacing)},
        save_or_return_probabilities=probabilities,
    )
    return out[1][1].astype(np.float32) if probabilities else out.astype(bool)


class ExportedVolumes:
    """Finds the volume of a sequence of the exported project from its frames.

    Didascalie sends the contrast-stretched frames, while the network takes the
    amplitude grid; this rebuilds the grid from the volume the frames come from.
    """

    def __init__(self, project: Path = EXPORTED):
        import sqlite3

        import cv2

        conn = sqlite3.connect(f"file:{project}?mode=ro", uri=True)
        self.sequences = {}
        for name, data in conn.execute(
            "SELECT s.name, f.embedded_data FROM frames f JOIN sequences s ON s.id = f.sequence_id"
            " WHERE f.frame_index = 0"
        ):
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
            self.sequences[self._key(frame)] = name
        conn.close()
        self.paths = pd.read_csv(SELECTION, dtype=str).set_index("sequence")["path"]
        self.geometry = np.load(GEOMETRY)
        self._grids = {}

    @staticmethod
    def _key(frame: np.ndarray) -> bytes:
        import hashlib

        return hashlib.sha1(np.ascontiguousarray(frame).tobytes()).digest()

    def sequence(self, frames: np.ndarray) -> str | None:
        """Name of the sequence whose frames these are, ``(T, H, W)`` or ``(T, H, W, 3)``."""
        first = frames[0]
        if first.ndim == 3:
            if not (first == first[..., :1]).all():
                return None  # not a grey frame of ours
            first = first[..., 0]
        return self.sequences.get(self._key(first))

    def grid(self, name: str) -> np.ndarray:
        """Amplitude grid of an exported sequence, the frames Didascalie shows."""
        if name not in self._grids:
            bm = self.geometry[f"{name}/bm"]
            csi = self.geometry[f"{name}/csi"]
            z0 = int(self.geometry[f"{name}/z0"])
            amp, _, _ = load_octa(Path(self.paths[name]))
            amp = np.ascontiguousarray(amp).transpose(0, 2, 1)
            n_slices = (grid_depth(bm, csi) - z0) // DEPTH_BIN
            self._grids[name] = enface_volume(amp, bm, n_slices, z0)
        return self._grids[name]


def predict(args):
    """Vessels of one OCTA volume, saved as a native ``(B-scans, depth, A-scans)`` mask.

    The network predicts on the grid (11.7 um slices); its probabilities are
    interpolated along the depth to the native rows (1.95 um) before the
    threshold, so the vessel walls are not quantised to the slices.
    """
    from ocularrigidity.segmentation.utils import get_choroid_segmentation_model

    amp, _, _ = load_octa(Path(args.volume))
    amp = np.ascontiguousarray(amp).transpose(0, 2, 1)
    bm, csi = choroid_boundaries(
        amp, model=lambda: get_choroid_segmentation_model().to(args.device), device=args.device
    )
    z0 = -MARGIN_TOP
    image = enface_volume(amp, bm, (grid_depth(bm, csi) - z0) // DEPTH_BIN, z0)
    proba = predict_grid(load_predictor(args.folds, device=args.device), image, True)

    out = Path(args.out)
    native = to_native_linear(proba, bm, z0, amp.shape[1]) > 0.5
    save_mask_atomic(native, out)
    np.savez_compressed(
        out.with_suffix(".grid.npz"), proba=proba.astype(np.float16), bm=bm, csi=csi, z0=z0
    )
    print(f"Wrote {out}: {(proba > 0.5).mean():.1%} of the grid is vessel")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name, fn in (("remap", remap), ("dataset", dataset)):
        s = sub.add_parser(name, help=fn.__doc__)
        s.add_argument("--dida", type=Path, default=ANNOTATIONS)
        s.set_defaults(fn=fn)
    s = sub.add_parser("predict", help=predict.__doc__)
    s.add_argument("--volume", required=True, help="folder holding amplitude.bin")
    s.add_argument("--out", required=True, help="native mask, .npz (ocularrigidity.data.io.load_mask)")
    s.add_argument("--folds", nargs="+", help="default: every trained fold")
    s.add_argument("--device", default="cuda:0")
    s.set_defaults(fn=predict)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
