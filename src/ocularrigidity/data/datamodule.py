from pathlib import Path
from typing import Optional, Sequence

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset

from ocularrigidity.consts import CHOROID_SEGMENTATION_ROOT, OIMHS_ROOT
from ocularrigidity.data.datasets import DEVICES, OIMHS, AnnotatedChoroid

SPLITS = ("train", "val", "test")


class Database:
    OIMHS = "oimhs"
    ANNOTATED = "annotated"


class ChoroidSegmentationDataModule(LightningDataModule):
    """Choroid segmentation at the PLEX pixel size, from any mix of sources.

    Every sample is resampled to the PLEX pixel size (1.95 um axial, 5.9 um
    lateral) by its dataset, so no transform here changes the scale beyond a
    small jitter. Training draws ``crop_size`` crops that contain choroid;
    validation and test see whole B-scans padded to a multiple of 32, as
    ``segmentation.inference.infer`` does, one per batch since sizes differ.

    The annotated set comes pre-split (``split_choroid_dataset.py``); OIMHS is
    split here by patient with ``oimhs_ratios``. The splits of the sources are
    concatenated.
    """

    def __init__(
        self,
        databases: Sequence[str] = (Database.ANNOTATED, Database.OIMHS),
        annotated_root: str | Path = CHOROID_SEGMENTATION_ROOT,
        oimhs_root: str | Path = OIMHS_ROOT,
        devices: Sequence[str] = DEVICES,
        oimhs_ratios: Sequence[float] = (0.8, 0.05, 0.15),
        crop_size: tuple[int, int] = (1024, 768),
        batch_size: int = 8,
        num_workers: int = 4,
        seed: int = 42,
    ):
        super().__init__()
        unknown = set(databases) - {Database.ANNOTATED, Database.OIMHS}
        if unknown:
            raise NotImplementedError(f"Datasets {sorted(unknown)} not implemented yet.")
        self.databases = tuple(databases)
        self.annotated_root = Path(annotated_root)
        self.oimhs_root = Path(oimhs_root)
        self.devices = tuple(devices)
        self.oimhs_ratios = tuple(oimhs_ratios)
        self.crop_size = tuple(crop_size)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.train: ConcatDataset | None = None
        self.val: ConcatDataset | None = None
        self.test: ConcatDataset | None = None

    def setup(self, stage: Optional[str] = None):
        if self.train is not None:
            return
        parts = {split: [] for split in SPLITS}
        if Database.ANNOTATED in self.databases:
            for split in SPLITS:
                parts[split].append(
                    AnnotatedChoroid(self.annotated_root, split, self.devices)
                )
        if Database.OIMHS in self.databases:
            oimhs = OIMHS(self.oimhs_root).split(self.oimhs_ratios, seed=self.seed)
            for split, ds in zip(SPLITS, oimhs):
                parts[split].append(ds)

        transforms = {
            "train": self.train_transforms(),
            "val": self.val_transforms(),
            "test": self.val_transforms(),
        }
        for split in SPLITS:
            for ds in parts[split]:
                ds.transforms = transforms[split]
            sizes = ", ".join(f"{type(ds).__name__} {len(ds)}" for ds in parts[split])
            print(f"{split}: {sizes}")

        self.train, self.val, self.test = (ConcatDataset(parts[s]) for s in SPLITS)

    def train_transforms(self):
        h, w = self.crop_size
        return A.Compose(
            [
                A.PadIfNeeded(min_height=h, min_width=w, border_mode=cv2.BORDER_REFLECT_101),
                A.CropNonEmptyMaskIfExists(height=h, width=w),
                A.HorizontalFlip(p=0.5),
                A.CLAHE(p=0.25),
                A.Affine(
                    # Small jitter only: the network must learn the PLEX scale
                    scale=dict(x=(0.9, 1.1), y=(0.9, 1.1)),
                    rotate=(-5, 5),
                    translate_percent=dict(
                        x=(-2 / 100, 2 / 100), y=(-10 / 100, 10 / 100)
                    ),
                    shear=(-5, 5),
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.75,
                ),
                A.GaussNoise(std_range=(0, 0.5), p=0.3),
                A.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.5, 0.5)),
                A.RandomGamma(),
                A.Normalize(mean=0.5, std=0.5),
                ToTensorV2(),
            ]
        )

    def val_transforms(self):
        return A.Compose(
            [
                # Bottom/right reflect padding, as at inference
                A.PadIfNeeded(
                    min_height=None,
                    min_width=None,
                    pad_height_divisor=32,
                    pad_width_divisor=32,
                    position="top_left",
                    border_mode=cv2.BORDER_REFLECT_101,
                ),
                A.Normalize(mean=0.5, std=0.5),
                ToTensorV2(),
            ]
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def _eval_loader(self, ds):
        return torch.utils.data.DataLoader(
            ds, batch_size=1, shuffle=False, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return self._eval_loader(self.val)

    def test_dataloader(self):
        return self._eval_loader(self.test)
