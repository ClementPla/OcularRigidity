"""This module is responsible for loading a video and its corresponding masks, performing registration, and providing access to the registered frames, masks, and computed thickness."""

import numpy as np
from pathlib import Path

import torch

from ocularrigidity.data.compression import mp4_to_cube, read_gray
from ocularrigidity.data.io import load_cube, load_mask


from ocularrigidity.registration.rigid import register_masks_by_displacement
from ocularrigidity.rigidity.features import (
    compute_deltaY_boundaries,
)
from ocularrigidity.segmentation.postprocess.smoothing import extract_boundaries_gpu


class RegisteredVideo:
    def __init__(
        self,
        video: Path,
        root_masks: Path,
        root_data: Path,
        skip_first_n_frames: int = 3,
        drop_last_n_frames: int = 0,
        flatten: bool = False,
        horizontal_alignment: bool = True,
        verbose: bool = True,
        use_encoded_video: bool = True,
        device: str = "cuda",
        batch_size: int = 256,
    ):
        self.video = video
        self.root_masks = root_masks
        self.root_data = root_data

        self.skip_first_n_frames = skip_first_n_frames
        self.drop_last_n_frames = drop_last_n_frames
        self.flatten = flatten
        self.horizontal_alignment = horizontal_alignment
        self.verbose = verbose
        self.use_encoded_video = use_encoded_video

        self._raw_frames = None
        self._raw_masks = None
        self._boundary_masks = None
        self._registration_params = None
        self._registered_frames = None
        self._registered_masks = None
        self._thickness = None
        self._boundary_masks = None
        self._csi = None
        self._registered_lines = None
        self._device = device
        self._batch_size = batch_size

    def save_cache(self, path):
        np.savez(
            path,
            registered_frames=self.registered_frames,
            registered_masks=self.registered_masks,
            thickness=self.thickness,
        )

    @property
    def _frame_slice(self) -> slice:
        """Slice applied to raw_frames / raw_masks."""
        end = None if self.drop_last_n_frames == 0 else -self.drop_last_n_frames
        return slice(self.skip_first_n_frames, end)

    @property
    def raw_frames(self):
        if self._raw_frames is None:
            if self.use_encoded_video:
                if (self.root_data / self.video).is_file():
                    try:
                        frames = mp4_to_cube(self.root_data / self.video)
                    except Exception as e:
                        frames = read_gray(self.root_data / self.video)
                else:
                    root_file = self.root_data / self.video / "cube.mp4"
                    if not root_file.exists():
                        raise FileNotFoundError(
                            f"Encoded video not found at {root_file}"
                        )
                    frames = mp4_to_cube(self.root_data / self.video / "cube.mp4")
            else:
                frames = load_cube(self.root_data / self.video)
            self._raw_frames = frames[self._frame_slice]
        return self._raw_frames

    @property
    def raw_masks(self):
        if self._raw_masks is None:
            # If self.video points to a file, take its parent directory as the video id for mask loading
            if (self.root_data / self.video).is_file():
                video_id = self.video.parent
            else:
                video_id = self.video
            raw_mask_path = self.root_masks / video_id / "mask.npz"
            if not raw_mask_path.exists():
                raise FileNotFoundError(f"Raw mask not found at {raw_mask_path}")
            masks = load_mask(raw_mask_path)
            self._raw_masks = masks[self._frame_slice]
        return self._raw_masks

    @property
    def registered_masks(self):
        if self._registered_masks is None:
            self.compute_registration()
        return self._registered_masks

    @property
    def registered_frames(self):
        if self._registered_frames is None:
            self.compute_registration()
        return self._registered_frames

    @property
    def thickness(self):
        if self._thickness is None:
            registered_lines = self.registered_lines.cpu().numpy()
            self._thickness = compute_deltaY_boundaries(
                registered_lines[:, 0], registered_lines[:, 1]
            )
        return self._thickness

    def _compute_boundaries(self):
        bm, csi = extract_boundaries_gpu(self.raw_masks, to_numpy=False)
        self._boundary_masks = bm
        self._csi = csi

    @property
    def boundary_masks(self):
        if self._boundary_masks is None:
            self._compute_boundaries()
        return self._boundary_masks

    @property
    def csi(self):
        if self._csi is None:
            self._compute_boundaries()
        return self._csi

    @property
    def registered_lines(self):
        """Registered (BM, CSI) as (T, 2, W) — BM at index 0, CSI at index 1."""
        if self._registered_lines is None:
            self.compute_registration()
        return self._registered_lines

    def compute_registration(self):
        raw_masks = self.raw_masks
        raw_frames = self.raw_frames
        registered_masks, registered_frames = register_masks_by_displacement(
            raw_masks,
            raw_frames,
            batch_size=self._batch_size,
            correct_dx=self.horizontal_alignment,
            flatten_rpe=self.flatten,
            verbose=self.verbose,
        )
        self._registered_masks = registered_masks.cpu().numpy() > 0
        self._registered_frames = registered_frames.cpu().numpy()
        csi, bm = extract_boundaries_gpu(registered_masks, to_numpy=False)
        self._registered_lines = torch.stack([csi, bm], dim=1).cpu()
