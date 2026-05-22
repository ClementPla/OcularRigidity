"""This module is responsible for loading a video and its corresponding masks, performing registration, and providing access to the registered frames, masks, and computed thickness."""

import numpy as np
from pathlib import Path

import torch

from ocularrigidity.data.compression import mp4_to_cube
from ocularrigidity.data.io import load_cube, load_mask
from ocularrigidity.registration.apply import (
    apply_registration_lines_torch,
    apply_registration_torch,
)
from ocularrigidity.registration.estimate_curve import (
    estimate_dx_from_images,
    register_curves_torch,
)
from ocularrigidity.rigidity.features import (
    compute_deltaY_boundaries,
)
from ocularrigidity.segmentation.postprocess.smoothing import extract_boundaries_gpu
from enum import Enum, StrEnum


class RegistrationTransform(StrEnum):
    """Type of transformation to apply during registration."""

    EUCLIDEAN = "euclidean"
    SIMILARITY = "similarity"
    TILT = "tilt"
    AFFINE = "affine"


class RegisteredVideo:
    def __init__(
        self,
        video: Path,
        root_masks: Path,
        root_data: Path,
        skip_first_n_frames: int = 3,
        drop_last_n_frames: int = 0,
        refine_iters: int = 2,
        min_pts: int = 10,
        transform: str = "tilt",
        flatten: bool = True,
        horizontal_scaling: bool = False,
        horizontal_alignment: bool = True,
        verbose: bool = True,
        use_encoded_video: bool = True,
        device: str = "cuda",
    ):
        self.video = video
        self.root_masks = root_masks
        self.root_data = root_data

        self.skip_first_n_frames = skip_first_n_frames
        self.drop_last_n_frames = drop_last_n_frames
        self.refine_iters = refine_iters
        self.min_pts = min_pts
        self.transform = RegistrationTransform(transform)
        self.flatten = flatten
        self.horizontal_scaling = horizontal_scaling
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
                root_file = self.root_data / self.video / "cube.mp4"
                if not root_file.exists():
                    raise FileNotFoundError(f"Encoded video not found at {root_file}")
                frames = mp4_to_cube(self.root_data / self.video / "cube.mp4")
            else:
                frames = load_cube(self.root_data / self.video)
            self._raw_frames = frames[self._frame_slice]
        return self._raw_frames

    @property
    def raw_masks(self):
        if self._raw_masks is None:
            raw_mask_path = self.root_masks / self.video / "mask.npz"
            if not raw_mask_path.exists():
                raise FileNotFoundError(f"Raw mask not found at {raw_mask_path}")
            masks = load_mask(raw_mask_path)
            self._raw_masks = masks[self._frame_slice]
        return self._raw_masks

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    @property
    def registration_params(self):
        if self._registration_params is None:
            self._registration_params = register_curves_torch(
                self.boundary_masks,
                ref_idx=0,
                refine_iters=self.refine_iters,
                min_pts=self.min_pts,
                device=self._device,
                transform=self.transform,
                horizontal_alignment=False,
                horizontal_scaling=self.horizontal_scaling,
            )
        if self.horizontal_alignment:
            dx = estimate_dx_from_images(self.raw_frames, 0, 100)
            self._registration_params[:, 0] = dx
        return self._registration_params

    @property
    def registered_masks(self):
        if self._registered_masks is None:
            self._registered_masks = (
                apply_registration_torch(
                    self.raw_masks,
                    self.registration_params,
                    mode="nearest",
                    flatten_bms=self.boundary_masks if self.flatten else None,
                    ref_idx_for_flatten=0 if self.flatten else None,
                    batch_size=256,
                    device=self._device,
                    verbose=self.verbose,
                ).numpy()
                > 0
            )
        return self._registered_masks

    @property
    def registered_frames(self):
        if self._registered_frames is None:
            self._registered_frames = apply_registration_torch(
                self.raw_frames,
                self.registration_params,
                mode="bilinear",
                flatten_bms=self.boundary_masks if self.flatten else None,
                ref_idx_for_flatten=0 if self.flatten else None,
                batch_size=256,
                device=self._device,
                verbose=self.verbose,
            ).numpy()
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
            lines = torch.stack([self.boundary_masks, self.csi], dim=1)
            self._registered_lines = apply_registration_lines_torch(
                lines,
                self.registration_params,
                flatten_bms=self.boundary_masks if self.flatten else None,
                ref_idx_for_flatten=0 if self.flatten else None,
                device=self._device,
            )
        return self._registered_lines
