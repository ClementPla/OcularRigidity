from pathlib import Path
from typing import Optional, Sequence

import albumentations as A
import cv2
import numpy as np

from ocularrigidity.data.datasets.generic import ChoroidDataset


class OIMHS(ChoroidDataset):
    """The OIMHS database: each PNG is the B-scan and its colour label map side by side."""

    # (axial, lateral) mm per pixel of the OIMHS B-scans
    PIXEL_SIZE_MM = (3.77e-3, 8.69e-3)
    WIDTH = 512  # width of the B-scan half of each PNG

    def __init__(
        self,
        root_files: Path | str | None = None,
        transforms: Optional[A.Compose] = None,
        files: Optional[Sequence[Path]] = None,
    ):
        super().__init__()
        if files is None:
            files = sorted(Path(root_files).rglob("*/*.png"))
        self.files = list(files)
        self.transforms = transforms

    @property
    def patient_ids(self):
        return [file.parent.name for file in self.files]

    def split(
        self, ratios: Sequence[float] = (0.8, 0.05, 0.15), seed: int = 42
    ) -> tuple["OIMHS", ...]:
        """Patient-level split, one dataset per ratio."""
        patients = sorted(set(self.patient_ids))
        np.random.default_rng(seed).shuffle(patients)
        bounds = np.round(np.cumsum(ratios)[:-1] * len(patients)).astype(int)
        parts = []
        for chunk in np.split(np.array(patients), bounds):
            keep = set(chunk)
            files = [f for f in self.files if f.parent.name in keep]
            parts.append(OIMHS(files=files, transforms=self.transforms))
        return tuple(parts)

    def __len__(self):
        return len(self.files)

    def pixel_size_mm(self, index):
        return self.PIXEL_SIZE_MM

    def load_image(self, index):
        image_mask = cv2.imread(str(self.files[index]))
        image = np.ascontiguousarray(image_mask[:, : self.WIDTH, 0])
        mask = image_mask[:, self.WIDTH :, :]
        mask = cv2.inRange(mask, (0, 255, 255), (0, 255, 255))
        return image, mask // 255
