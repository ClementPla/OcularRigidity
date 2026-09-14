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
    """Decode one raw cube, trimmed as ``config`` asks. Runs in a worker.

    Returns ``(T, H, W)`` uint8 as a tensor, so the ~4.4 GB travels through
    shared memory rather than being pickled down the worker socket.
    ``load_cube`` already returns the (T, 1536, 1024) orientation the models
    expect and re-sorts the frames by ``timestamp.txt``.
    """
    frames = load_cube(source_path(video).parent)
    if config is not None:
        end = None if config.drop_last_n_frames == 0 else -config.drop_last_n_frames
        frames = frames[config.skip_first_n_frames : end]
    return torch.from_numpy(np.ascontiguousarray(frames))


class PrefetchDataset(Dataset):
    """Runs ``loader(task)`` in DataLoader worker processes, one task per item.

    The generic form of :class:`VolumeDataset`: the batch scripts each decode
    something different (a registration cache, a folded one_cycle.mkv), but they
    all want the same two properties from it — the decode happening ahead of the
    main process, and a failure costing one item rather than the whole run.

    ``loader`` must be importable (a module-level function, or a ``partial`` of
    one) and should return torch tensors for anything volume-sized, so the array
    travels through shared memory instead of being pickled down a socket.

    Yields ``(task, payload, error)``; ``error`` is a traceback string when the
    load failed, and ``payload`` is whatever ``loader`` returned otherwise.
    """

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
            # Raising here would tear down the whole DataLoader: a missing
            # cube.bin must cost us one volume, not the rest of the cohort.
            return measure_value, None, traceback.format_exc()
