from enum import Enum
from pathlib import Path
from pytorch_lightning import LightningDataModule
import torch
from ocularrigidity.data.datasets import OIMHS, ChoroidDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import Optional
import cv2


class Database:
    OIMHS = "oimhs"
    # Add more datasets as needed


class ChoroidSegmentationDataModule(LightningDataModule):
    def __init__(
        self,
        root: str | Path,
        database: Database,
        batch_size=32,
        num_workers=4,
    ):
        super().__init__()
        # We want to fit the pixel size: from 512
        original_img_shape = (496, 512)
        original_pixel_size_x = 8.69e-3
        original_pixel_size_y = 3.77e-3

        target_pixel_size_x = 5.9e-3
        target_pixel_size_y = 1.95e-3

        def _round_to_power_of_2(n: int) -> int:
            """Round to the nearest power of 2."""
            return 2 ** round(np.log2(n))

        target_w = int(
            round(original_img_shape[1] * original_pixel_size_x / target_pixel_size_x)
        )
        target_h = int(
            round(original_img_shape[0] * original_pixel_size_y / target_pixel_size_y)
        )
        self.img_size = (_round_to_power_of_2(target_h), _round_to_power_of_2(target_w))
        self.root = Path(root)
        self.database = database
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train: ChoroidDataset | None = None
        self.val: ChoroidDataset | None = None

    def setup(self, stage: Optional[str] = None):
        if self.train is not None and self.val is not None:
            return
        match self.database:
            case Database.OIMHS:
                dataset = OIMHS(self.root, shape=self.img_size)
                print(f"Found {len(dataset)} files in {self.root}")
                self.train, self.val = dataset.split(0.8)

            case _:
                raise NotImplementedError(
                    f"Dataset {self.database} not implemented yet."
                )

        self.train.transforms = self.train_transforms()
        self.val.transforms = self.val_transforms()

    def train_transforms(self):
        return A.Compose(
            [
                A.Resize(*self.img_size),
                A.HorizontalFlip(p=0.5),
                A.CLAHE(p=0.25),
                A.Affine(
                    scale=dict(x=(0.9, 2.0), y=(0.9, 2.0)),
                    rotate=(-5, 5),
                    translate_percent=dict(
                        x=(-2 / 100, 2 / 100), y=(-10 / 100, 10 / 100)
                    ),
                    shear=(-5, 5),
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.75,
                ),
                A.GaussNoise(std_range=(0, 0.5), p=0.3),
                A.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.5,0.5)),
                A.RandomGamma(),
                A.Normalize(mean=0.5, std=0.5),
                ToTensorV2(),
            ]
        )

    def val_transforms(self):
        return A.Compose(
            [
                A.Resize(*self.img_size),
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
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
