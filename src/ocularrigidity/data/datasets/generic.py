from abc import ABC, abstractmethod
from typing import Optional
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset


class ChoroidDataset(Dataset, ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    @property
    @abstractmethod
    def patient_ids(self):
        pass

    def update_files(self, new_files):
        self.files = new_files

    def copy(self):
        new_dataset = self.__class__.__new__(self.__class__)
        new_dataset.__dict__.update(self.__dict__)
        return new_dataset

    @abstractmethod
    def split(self, ratio: float) -> tuple["ChoroidDataset", "ChoroidDataset"]:
        pass

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
            # Draw polygon on the image, with full opacity for edges and 50% opacity for the inside
            # We use matplotlib to draw the polygon, and we use the mask as a contour
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
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
