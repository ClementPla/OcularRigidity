from pathlib import Path
from typing import Optional, Sequence

import albumentations as A
import cv2

from ocularrigidity.data.datasets.generic import PLEX_PIXEL_SIZE_MM, ChoroidDataset

DEVICES = ("PLEX", "SANSORI", "Spectralis")


class AnnotatedChoroid(ChoroidDataset):
    """Choroid masks annotated in Didascalie, one split of
    ``<root>/<split>/<device>/{images,masks}/<name>.png`` (made by
    ``scripts/segmentation/split_choroid_dataset.py``)."""

    # (axial, lateral) mm per pixel of the exported PNGs
    PIXEL_SIZE_MM = {
        "PLEX": PLEX_PIXEL_SIZE_MM,
        # Exported already resampled to the PLEX grid (scripts/pulsation/spectralis.py
        # to_plex_grid): 496 rows x 3.9 um -> 992 rows x 1.95 um
        "Spectralis": PLEX_PIXEL_SIZE_MM,
        # Raw 496x768 Spectralis B-scans (RawImages/registeredBscans). No
        # per-scan metadata here: the median of the Spectralis rigidity cohort
        # (DianeReplication/pulsation_all.csv), whose scans are the same 768 A-scans.
        "SANSORI": (3.9e-3, 11.47e-3),
    }

    def __init__(
        self,
        root: Path | str,
        split: str,
        devices: Sequence[str] = DEVICES,
        transforms: Optional[A.Compose] = None,
    ):
        super().__init__()
        root = Path(root)
        self.transforms = transforms
        self.files: list[tuple[Path, Path, str]] = []
        for device in devices:
            if device not in self.PIXEL_SIZE_MM:
                raise ValueError(f"No pixel size for device {device!r}")
            images = root / split / device / "images"
            masks = root / split / device / "masks"
            for image in sorted(images.glob("*.png")):
                mask = masks / image.name
                if mask.exists():
                    self.files.append((image, mask, device))

    @property
    def patient_ids(self):
        ids = []
        for image, _, device in self.files:
            stem = image.stem
            if "__" in stem:
                ids.append(stem.split("__")[0])
            else:  # SANSORI: 01_210713001_before_OD_|_brut
                ids.append(f"{device}:" + "_".join(stem.split("_")[:2]))
        return ids

    def __len__(self):
        return len(self.files)

    def pixel_size_mm(self, index):
        return self.PIXEL_SIZE_MM[self.files[index][2]]

    def load_image(self, index):
        image_path, mask_path, _ = self.files[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(image_path if image is None else mask_path)
        return image, (mask > 127).astype("uint8")
