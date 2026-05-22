from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional
import cv2
import numpy as np
from ocularrigidity.data.datasets.generic import ChoroidDataset
import albumentations as A


class OIMHS(ChoroidDataset):
    """Process the OIMHS database.
    Each file contains two images side by side: the right one is the input image
    The left one is the groundtruth.
    We only extract the choroid, which is in yellow in the mask.
    The folder is organized as follows:
    - root_files
        - patient_id/
            - image1.png
            - image2.png
            - ...
    """

    def __init__(
        self,
        root_files: Path,
        shape: tuple = (512, 512),
        transforms: Optional[A.Compose] = None,
    ):
        super().__init__()
        root_files = Path(root_files)
        self.files = list(root_files.rglob("*/*.png"))
        self.shape = shape
        self.transforms = transforms

    @property
    def patient_ids(self):
        return list(file.parent.name for file in self.files)

    def split(self, ratio: float, seed: int = 42) -> tuple["OIMHS", "OIMHS"]:
        patient_ids = sorted(set(self.patient_ids))
        np.random.default_rng(seed).shuffle(patient_ids)
        split_idx = int(len(patient_ids) * ratio)
        train_patient_ids = set(patient_ids[:split_idx])
        val_patient_ids = set(patient_ids[split_idx:])
        train_files = [
            file for file in self.files if file.parent.name in train_patient_ids
        ]
        val_files = [file for file in self.files if file.parent.name in val_patient_ids]
        train_dataset = OIMHS(
            root_files="", shape=self.shape, transforms=self.transforms
        )
        val_dataset = OIMHS(root_files="", shape=self.shape, transforms=self.transforms)
        train_dataset.update_files(train_files)
        val_dataset.update_files(val_files)
        return train_dataset, val_dataset

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        image, mask = self.load_image(file)
        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        return image, mask

    def load_image(self, file):
        image_mask = cv2.imread(str(file))
        image = image_mask[:, :512, 0]
        mask = image_mask[:, 512:, :]
        mask = cv2.inRange(mask, (0, 255, 255), (0, 255, 255))
        return image, mask // 255
