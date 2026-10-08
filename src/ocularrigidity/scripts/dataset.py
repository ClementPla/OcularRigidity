import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ocularrigidity.consts import (
    ROOT_COMPRESSED_VIDEO,
    ROOT_DATA_MNT,
    ROOT_MASKS,
)
from ocularrigidity.data.compression import mp4_to_cube
from ocularrigidity.data.io import load_cube, load_mask


def identity_collate(batch):
    return batch[0]


def source_path(video: str | Path) -> Path:
    return ROOT_DATA_MNT / Path(video) / "cube.bin"


def load_source_frames(video: str | Path, config=None) -> torch.Tensor:
    """Decode one raw cube, trimmed as ``config`` asks."""
    frames = load_cube(source_path(video).parent)
    if config is not None:
        end = None if config.drop_last_n_frames == 0 else -config.drop_last_n_frames
        frames = frames[config.skip_first_n_frames : end]
    return torch.from_numpy(np.ascontiguousarray(frames))


class PrefetchDataset(Dataset):
    """Runs ``loader(task)`` in DataLoader worker processes, one task per item."""

    def __init__(self, tasks, loader):
        self.tasks = tasks
        self.loader = loader

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]
        try:
            return task, self.loader(task), None
        except Exception:
            return task, None, traceback.format_exc()


class VolumeDataset(Dataset):
    """Prefetches and decodes MP4/bin cubes in isolated CPU worker processes."""

    def __init__(self, tasks, return_masks=False):
        self.tasks = tasks
        self.return_masks = return_masks

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        measure_value = self.tasks[idx]
        try:
            encoded = ROOT_COMPRESSED_VIDEO / measure_value / "cube.mp4"
            if encoded.exists():
                data = mp4_to_cube(encoded)
            else:
                data = load_cube(ROOT_DATA_MNT / measure_value)

            tensor = torch.from_numpy(np.ascontiguousarray(data))
            if self.return_masks:
                root_mask = ROOT_MASKS / measure_value / "mask.npz"
                if not root_mask.exists():
                    raise FileNotFoundError(f"Missing mask for {measure_value}")

                masks = load_mask(ROOT_MASKS / measure_value / "mask.npz")
                masks = torch.from_numpy(np.ascontiguousarray(masks))
                return measure_value, (tensor, masks), None

            return measure_value, tensor, None
        except Exception:
            return measure_value, None, traceback.format_exc()
