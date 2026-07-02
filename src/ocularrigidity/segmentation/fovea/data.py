"""Dataset + DataModule for foveal-point keypoint training.

This is a scaffold to adapt to your annotation format. Annotations are expected
as one (x, y) foveal-pit click per labeled frame. The default loader reads a CSV
with columns: ``image, x, y, patient`` (``image`` = path to a grayscale frame,
``x``/``y`` = foveal coordinates in that frame's pixels, ``patient`` = id used
for a leak-free train/val split).

Augmentation note: lateral (x) translation is the dominant augmentation on
purpose. Without it the model learns the shortcut "fovea = image center"; with
it the fovea is pushed off-center so the model must localize the real anatomy.
albumentations transforms the keypoint together with the image, so labels stay
correct. Keep the x-translation range wide but not so wide that the (central)
fovea leaves the frame — an off-frame point is an invalid label.
"""

from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset

from ocularrigidity.segmentation.fovea.dsnt import pixel_to_normalized


class FoveaKeypointDataset(Dataset):
    """Records: list of dicts with keys ``image`` (path), ``x``, ``y``, ``patient``."""

    def __init__(self, records: list[dict], shape: tuple[int, int], transforms=None):
        self.records = records
        self.shape = shape  # (H, W) the images/targets are resized to
        self.transforms = transforms

    def __len__(self):
        return len(self.records)

    @property
    def patient_ids(self):
        return [r["patient"] for r in self.records]

    def __getitem__(self, index):
        rec = self.records[index]
        # --- ADAPT HERE if your frames aren't on-disk grayscale images ---
        img = cv2.imread(str(rec["image"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {rec['image']}")
        img = img[..., None]  # (H, W, 1)
        kp = [(float(rec["x"]), float(rec["y"]))]

        out = self.transforms(image=img, keypoints=kp)
        image = out["image"]  # (1, H, W) tensor after Normalize + ToTensorV2
        if len(out["keypoints"]) == 0:
            raise ValueError(
                f"Fovea fell outside the frame after augmentation for {rec['image']}; "
                "reduce the translation range."
            )
        kx, ky = out["keypoints"][0]

        H, W = self.shape
        target = torch.tensor(
            [[pixel_to_normalized(kx, W), pixel_to_normalized(ky, H)]],
            dtype=torch.float32,
        )  # (1, 2) -> batched to (B, 1, 2)
        return image, target

    def split(self, ratio: float = 0.8, seed: int = 42):
        """Patient-disjoint train/val split."""
        rng = np.random.default_rng(seed)
        patients = sorted({r["patient"] for r in self.records})
        rng.shuffle(patients)
        n_train = int(round(len(patients) * ratio))
        train_p = set(patients[:n_train])

        train_rec = [r for r in self.records if r["patient"] in train_p]
        val_rec = [r for r in self.records if r["patient"] not in train_p]
        return (
            FoveaKeypointDataset(train_rec, self.shape),
            FoveaKeypointDataset(val_rec, self.shape),
        )


class FoveaKeypointDataModule(LightningDataModule):
    def __init__(
        self,
        annotations_csv: str | Path,
        img_size: tuple[int, int] = (256, 512),  # (H, W); downscaled for speed
        batch_size: int = 32,
        num_workers: int = 4,
        max_x_translate: float = 0.35,  # dominant lateral-shift aug (fraction of W)
        val_ratio: float = 0.8,
    ):
        super().__init__()
        self.annotations_csv = Path(annotations_csv)
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_x_translate = max_x_translate
        self.val_ratio = val_ratio
        self.train: Optional[FoveaKeypointDataset] = None
        self.val: Optional[FoveaKeypointDataset] = None

    def _load_records(self) -> list[dict]:
        # --- ADAPT HERE to your annotation format ---
        df = pd.read_csv(self.annotations_csv)
        required = {"image", "x", "y", "patient"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Annotation CSV missing columns: {missing}")
        return df.to_dict("records")

    def setup(self, stage: Optional[str] = None):
        if self.train is not None and self.val is not None:
            return
        full = FoveaKeypointDataset(self._load_records(), shape=self.img_size)
        print(f"Found {len(full)} labeled frames in {self.annotations_csv}")
        self.train, self.val = full.split(self.val_ratio)
        self.train.shape = self.img_size
        self.val.shape = self.img_size
        self.train.transforms = self.train_transforms()
        self.val.transforms = self.val_transforms()

    def _keypoint_params(self):
        # remove_invisible=False keeps the point through flips; out-of-frame is
        # caught in __getitem__ so a bad augmentation range surfaces loudly.
        return A.KeypointParams(format="xy", remove_invisible=False)

    def train_transforms(self):
        t = self.max_x_translate
        return A.Compose(
            [
                A.Resize(*self.img_size),
                A.HorizontalFlip(p=0.5),  # flips the keypoint too; OD/OS invariance
                # Dominant lateral shift: break the "predict center" shortcut.
                A.Affine(
                    translate_percent={"x": (-t, t), "y": (-0.05, 0.05)},
                    scale=(0.9, 1.1),
                    rotate=(-3, 3),
                    border_mode=cv2.BORDER_REPLICATE,
                    p=0.9,
                ),
                A.GaussNoise(std_range=(0.0, 0.4), p=0.3),
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.3),
                A.Normalize(mean=0.5, std=0.5),
                ToTensorV2(),
            ],
            keypoint_params=self._keypoint_params(),
        )

    def val_transforms(self):
        return A.Compose(
            [
                A.Resize(*self.img_size),
                A.Normalize(mean=0.5, std=0.5),
                ToTensorV2(),
            ],
            keypoint_params=self._keypoint_params(),
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
