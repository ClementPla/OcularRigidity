"""Dump (frame, transform) pairs for the registration regressor.

Direct optimisation (``train_registration.py``) needs only the *images*: the
objective compares a warped moving frame against the fixed one, so any two
frames of a volume are a training sample. That changes what is worth writing:

* ``--n-frames`` is now the lever that matters. N frames per volume give
  N(N-1) ordered pairs, so 40 frames is ~1500 pairs from one acquisition
  against the 10 the supervised setup could use.
* Features are **off by default** (``--features`` to restore). They were 88 GB
  for 165 frames, and the training script re-encodes on the GPU in less time
  than reading them back costs.
* Transforms are still written, but they are now a *diagnostic* — the classical
  estimate the model is compared against, not the target it is fitted to.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from ocularrigidity.data.compression import mp4_to_cube
from ocularrigidity.scripts.cohort_analysis.segment_n_cycles import get_model

ROOT_TRANSFORMS = Path(
    "/media/clement/HD/Santiago/OcularRigidity/outputs_new_model/CardiacPipeline_V1/registered_masks/"
)
ROOT_VIDEOS = Path("/media/clement/HD/Santiago/OcularRigidity/outputs/compressed/")
ROOT_FLAGS = Path(
    "/media/clement/HD/Santiago/OcularRigidity/outputs_new_model/CardiacPipeline_V1/misregistration_flags.csv"
)
OUTPUT_DIR = Path("/home/clement/Documents/data/OcularRigidity/Registration/Dataset/")


def prepare_pairs_videos(
    video: str,
    model,
    output_dir: Path,
    n_frames: int,
    dump_features: bool,
    dump_transforms: bool,
    rng: np.random.Generator,
):
    frames = mp4_to_cube(ROOT_VIDEOS / video / "cube.mp4")
    transforms = np.load(ROOT_TRANSFORMS / video / "transform.npz", allow_pickle=True)

    for sub in ("images/fixed", "images/moving") + (
        ("features/fixed", "features/moving") if dump_features else ()
    ) + (("transforms",) if dump_transforms else ()):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    name = video.replace("/", "_")
    # The registration dropped these frames, so transform.npz is indexed on the
    # trimmed volume; trim identically or every transform is off by 20 frames.
    frames = frames[20:-10]

    # The reference frame is the one the classical registration left untouched.
    fixed_idx = np.where(transforms["dx"] == 0)[0]
    other_frames = np.where(transforms["dx"] != 0)[0]
    n = min(n_frames, len(other_frames))
    indices = np.hstack([fixed_idx[:1], rng.choice(other_frames, size=n, replace=False)])

    if dump_features:
        tensor = (
            torch.from_numpy(frames[indices])
            .unsqueeze(1)
            .to(torch.float32)
            .mul_(1.0 / 255.0)
            .sub_(0.5)
            .div_(0.5)
            .cuda()
        )
        with torch.no_grad():
            features = model.model.encoder.forward_features(tensor)
        torch.save(
            [f[0].unsqueeze(0) for f in features],
            output_dir / "features" / "fixed" / f"{name}_{indices[0]}.pt",
        )

    cv2.imwrite(
        str(output_dir / "images" / "fixed" / f"{name}_{indices[0]}.png"),
        frames[indices[0]],
    )
    for i, idx in enumerate(indices[1:]):
        cv2.imwrite(
            str(output_dir / "images" / "moving" / f"{name}_{idx}.png"), frames[idx]
        )
        if dump_features:
            torch.save(
                [f[i + 1].unsqueeze(0) for f in features],
                output_dir / "features" / "moving" / f"{name}_{idx}.pt",
            )
        if dump_transforms:
            torch.save(
                {
                    "x": torch.tensor(transforms["dx"][idx]),
                    "y": torch.tensor(transforms["dy"][idx]),
                },
                output_dir / "transforms" / f"{name}_{idx}.pt",
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--n-videos", type=int, default=60)
    ap.add_argument(
        "--n-frames",
        type=int,
        default=40,
        help="moving frames per volume; pairs grow as N(N-1)",
    )
    ap.add_argument(
        "--features",
        action="store_true",
        help="also dump the encoder pyramid (~0.5 GB/frame); the trainer does "
        "not need it and re-encodes on the GPU instead",
    )
    ap.add_argument("--no-transforms", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Cleanest volumes first: the diagnostic transform is only worth comparing
    # against where the classical registration itself was not flagged.
    flags = pd.read_csv(ROOT_FLAGS, index_col=0).sort_values(
        by="bm_jitter_max", ascending=True
    )
    model = get_model().eval() if args.features else None
    rng = np.random.default_rng(args.seed)
    for video in tqdm(flags.video[: args.n_videos], desc="Processing videos"):
        prepare_pairs_videos(
            video,
            model,
            args.output,
            args.n_frames,
            args.features,
            not args.no_transforms,
            rng,
        )


if __name__ == "__main__":
    main()
