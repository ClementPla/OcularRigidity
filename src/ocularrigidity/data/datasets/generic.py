from abc import ABC, abstractmethod
from typing import Optional
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

from ocularrigidity.consts import AXIAL_PIXEL_SIZE_MM, TRANVERSAL_PIXEL_SIZE_MM

# (axial, lateral) mm per pixel every segmentation sample is resampled to
PLEX_PIXEL_SIZE_MM = (AXIAL_PIXEL_SIZE_MM, TRANVERSAL_PIXEL_SIZE_MM)


def resample(
    image: np.ndarray,
    mask: np.ndarray,
    pixel_size_mm: tuple[float, float],
    target_mm: tuple[float, float] = PLEX_PIXEL_SIZE_MM,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize an (image, mask) pair from ``pixel_size_mm`` to ``target_mm`` (both (axial, lateral))."""
    fy = pixel_size_mm[0] / target_mm[0]
    fx = pixel_size_mm[1] / target_mm[1]
    if np.isclose(fy, 1.0, atol=1e-3) and np.isclose(fx, 1.0, atol=1e-3):
        return image, mask
    h, w = image.shape[:2]
    size = (int(round(w * fx)), int(round(h * fy)))  # cv2 wants (width, height)
    interp = cv2.INTER_AREA if fx < 1 and fy < 1 else cv2.INTER_CUBIC
    image = cv2.resize(image, size, interpolation=interp)
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return image, mask


class ChoroidDataset(Dataset, ABC):
    """(image, mask) pairs resampled to the PLEX pixel size, then ``transforms``.

    Subclasses give the native pair (``load_image``) and its pixel size
    (``pixel_size_mm``); this class does the resampling, so every source is seen
    by the network at the resolution the cohort is segmented at.
    """

    transforms = None
    target_pixel_size_mm = PLEX_PIXEL_SIZE_MM

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def load_image(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Native ``(image uint8 (H, W), mask uint8 (H, W) in {0, 1})``."""

    @abstractmethod
    def pixel_size_mm(self, index: int) -> tuple[float, float]:
        """Native (axial, lateral) mm per pixel of sample ``index``."""

    @property
    @abstractmethod
    def patient_ids(self):
        pass

    def __getitem__(self, index):
        image, mask = self.load_image(index)
        image, mask = resample(
            image, mask, self.pixel_size_mm(index), self.target_pixel_size_mm
        )
        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        return image, mask

    def update_files(self, new_files):
        self.files = new_files

    def copy(self):
        new_dataset = self.__class__.__new__(self.__class__)
        new_dataset.__dict__.update(self.__dict__)
        return new_dataset

    def plot_one(self, index: Optional[int] = None, with_overlay: bool = False):
        if index is None:
            index = np.random.randint(len(self))
        image, mask = self[index]

        # Check if the image is a tensor, if so convert it to numpy
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy().squeeze()
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy().squeeze()
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Image")
        if with_overlay:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                axes[0].plot(
                    contour[:, 0, 0], contour[:, 0, 1], color="red", linewidth=1
                )
                axes[0].fill(
                    contour[:, 0, 0], contour[:, 0, 1], color="red", alpha=0.25
                )

        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Mask")
        for ax in axes:
            ax.axis("off")
            # Transparent background
            ax.set_facecolor((0, 0, 0, 0))
        plt.tight_layout()
        # Transparent background for the whole figure
        fig.patch.set_alpha(0)
        plt.show()
