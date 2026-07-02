"""This module is responsible for loading a video and its corresponding masks, performing registration, and providing access to the registered frames, masks, and computed thickness."""

import numpy as np
from pathlib import Path
from typing import Literal
import torch

from ocularrigidity.data.compression import (
    cube_to_mp4_fastest,
    mp4_to_cube,
    read_gray,
)
from ocularrigidity.data.io import load_cube, load_mask, save_mask


from ocularrigidity.registration.rigid import register_videos
from ocularrigidity.thickness.features import (
    compute_deltaY_boundaries,
)
from ocularrigidity.segmentation.postprocess.interfaces import (
    clean_boundaries,
    extract_boundaries_fast,
    extract_boundaries_gpu,
)
import matplotlib.pyplot as plt
import numpy as np


class VideoRegistrator:
    def __init__(
        self,
        video: Path,
        root_masks: Path,
        root_data: Path,
        # --- frame selection / loading ---
        skip_first_n_frames: int = 3,
        drop_last_n_frames: int = 0,
        use_encoded_video: bool = True,
        # --- what to correct ---
        correct_transversal: bool = True,
        correct_axial: bool = True,
        flatten_rpe: bool = False,
        axial_refinement: bool = False,
        fovea_correction_enabled: bool = True,
        # --- transversal (x) parameters ---
        lateral_method: Literal["xcorr", "fullframe", "both"] = "xcorr",
        max_lateral_shift: int = 16,
        smooth_transversal: bool = True,
        smooth_transversal_sigma: float = 2.0,
        crop_factor: float = 0.66,
        scale_factor: float = 1.0,
        transversal_bandpass=(0.02, 0.5),
        axial_bandpass=(0.02, 0.5),
        # --- axial (y) parameters ---
        max_axial_shift: int = 30,
        # --- general ---
        subpixel: bool = True,
        # --- runtime / cache ---
        device: str = "cuda",
        batch_size: int = 128,
        cache_dir: Path = None,
        overwrite_cache: bool = False,
        verbose: bool = True,
    ):
        self.video = video
        self.root_masks = root_masks
        self.root_data = root_data

        self.skip_first_n_frames = skip_first_n_frames
        self.drop_last_n_frames = drop_last_n_frames
        self.use_encoded_video = use_encoded_video

        # ``correct_transversal`` -> ``register_videos(correct_transversal=...)``
        # (lateral x shift); ``correct_axial`` toggles the per-column vertical BM
        # alignment; ``flatten_rpe`` aligns the BM to a constant row. See
        # rigid.register_videos.
        self.correct_transversal = correct_transversal
        self.correct_axial = correct_axial
        self.flatten_rpe = flatten_rpe
        # Second axial pass (RPE) aligning each A-scan onto the volume's temporal
        # median. ``max_axial_shift`` is its maximal tested vertical shift (px).
        self.axial_refinement = axial_refinement
        # Fovea-pit correction applied before registration (register_videos).
        self.fovea_correction_enabled = fovea_correction_enabled

        # Lateral (x) shift estimator: "xcorr" (profile cross-correlation) or
        # "fullframe" (2D phase correlation). Part of the cache key below.
        self.lateral_method = lateral_method
        # Lateral search half-window (px) and optional temporal smoothing of the
        # estimated dx (register_videos: smooth_transversal / _sigma).
        self.max_lateral_shift = max_lateral_shift
        self.smooth_transversal = smooth_transversal
        self.smooth_transversal_sigma = smooth_transversal_sigma
        self.crop_factor = crop_factor
        self.scale_factor = scale_factor
        self.transversal_bandpass = transversal_bandpass
        self.axial_bandpass = axial_bandpass
        self.max_axial_shift = max_axial_shift

        # Apply parabolic sub-pixel refinement to the shift peaks. Part of the
        # cache key so integer- and sub-pixel-registered results don't mix.
        self.subpixel = subpixel

        self.verbose = verbose

        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

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
        self._transform = None
        self._device = device
        self._batch_size = batch_size
        self._overwrite_cache = overwrite_cache

    def save_cache(self, path):
        np.savez(
            path,
            registered_frames=self.registered_frames,
            registered_masks=self.registered_masks,
            thickness=self.thickness,
        )

    @property
    def _video_id(self) -> Path:
        """Video identifier used to build cache/mask paths.

        Mirrors the logic in ``raw_masks``: if ``video`` points to a file, its
        parent directory is the id; otherwise ``video`` itself.
        """
        if (self.root_data / self.video).is_file():
            return self.video.parent
        return self.video

    def _cache_paths(self) -> dict:
        """Cache file locations for the registered frames, masks and transform."""
        vid = self._video_id
        return {
            "frames": self.cache_dir / "registered_frames" / vid / "cube.mp4",
            "masks": self.cache_dir / "registered_masks" / vid / "mask.npz",
            "transform": self.cache_dir / "registered_masks" / vid / "transform.npz",
        }

    def _cache_meta(self) -> dict:
        """Registration parameters the cache is keyed on (validated on load)."""
        return dict(
            skip_first_n_frames=self.skip_first_n_frames,
            drop_last_n_frames=self.drop_last_n_frames,
            correct_transversal=int(self.correct_transversal),
            correct_axial=int(self.correct_axial),
            flatten_rpe=int(self.flatten_rpe),
            fovea_correction_enabled=int(self.fovea_correction_enabled),
            lateral_method=self.lateral_method,
            max_lateral_shift=int(self.max_lateral_shift),
            smooth_transversal=int(self.smooth_transversal),
            smooth_transversal_sigma=float(self.smooth_transversal_sigma),
            axial_refinement=int(self.axial_refinement),
            max_axial_shift=int(self.max_axial_shift),
            subpixel=int(self.subpixel),
        )

    def _load_from_cache(self) -> bool:
        """Populate registration results from cache. Returns True on a valid hit."""
        paths = self._cache_paths()
        if not all(p.exists() for p in paths.values()) or self._overwrite_cache:
            return False
        try:
            data = np.load(paths["transform"])
            # Stale cache if it was produced with different registration params.
            # Compare as strings so int and string keys (e.g. lateral_method) work,
            # and a missing key (older cache) raises -> treated as a miss below.
            for k, v in self._cache_meta().items():
                if str(data[k]) != str(v):
                    return False
            frames = read_gray(paths["frames"])
            masks = load_mask(paths["masks"])
        except Exception as e:
            if self.verbose:
                print(f"Ignoring unreadable registration cache ({e})")
            return False
        if frames.shape[0] != masks.shape[0]:
            return False

        self._registered_frames = frames
        self._registered_masks = masks
        self._transform = {"dx": data["dx"], "dy": data["dy"]}
        bm, csi = extract_boundaries_gpu(masks, to_numpy=False)
        self._registered_lines = torch.stack([bm, csi], dim=1).cpu()
        if self.verbose:
            print(f"Loaded registration from cache: {paths['masks'].parent}")
        return True

    def _save_to_cache(self) -> None:
        """Persist registered frames (lossless mkv), masks and transform params."""
        paths = self._cache_paths()
        for p in paths.values():
            p.parent.mkdir(parents=True, exist_ok=True)

        # The registration input is already lossy (compressed cube.mp4 at cq=18,
        # see scripts/compress.py), so a fast HEVC re-encode is sufficient here.
        cube_to_mp4_fastest(
            self._registered_frames, str(paths["frames"]), cq=18, fps=200
        )
        save_mask(self._registered_masks, paths["masks"])

        tf = self._transform or {}
        dx, dy = tf.get("dx"), tf.get("dy")
        if isinstance(dx, torch.Tensor):
            dx = dx.numpy()
        if isinstance(dy, torch.Tensor):
            dy = dy.numpy()
        np.savez(paths["transform"], dx=dx, dy=dy, **self._cache_meta())

    @property
    def transform(self) -> dict:
        """Applied transform: ``{"dx": (T,), "dy": (T, W)}`` (numpy/torch)."""
        if self._transform is None:
            self.compute_registration()
        return self._transform

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
            if not self.use_encoded_video:
                # load_cube returns (N, W, H); align to the masks' (H, W)
                # orientation so registration sees a consistent frame/mask grid.
                mask_hw = self.raw_masks.shape[1:]
                if self._raw_frames.shape[1:] == mask_hw[::-1]:
                    self._raw_frames = self._raw_frames.transpose(0, 2, 1)
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
        # Reuse a previously cached registration when available.
        if self.cache_dir is not None and self._load_from_cache():
            return

        raw_masks = self.raw_masks
        raw_frames = self.raw_frames
        registered_masks = raw_masks
        registered_frames = raw_frames

        registered_masks, registered_frames, params = register_videos(
            registered_masks,
            registered_frames,
            correct_transversal=self.correct_transversal,
            correct_axial=self.correct_axial,
            flatten_rpe=self.flatten_rpe,
            axial_refinement=self.axial_refinement,
            fovea_correction_enabled=self.fovea_correction_enabled,
            lateral_method=self.lateral_method,
            max_lateral_shift=self.max_lateral_shift,
            smooth_transversal=self.smooth_transversal,
            smooth_transversal_sigma=self.smooth_transversal_sigma,
            max_axial_shift=self.max_axial_shift,
            subpixel=self.subpixel,
            batch_size=self._batch_size,
            device=self._device,
            verbose=self.verbose,
            return_params=True,
            crop_factor=self.crop_factor,
            scale_factor=self.scale_factor,
            transversal_bandpass=self.transversal_bandpass,
            axial_bandpass=self.axial_bandpass,
        )

        self._registered_masks = registered_masks.cpu().numpy() > 0
        self._registered_frames = registered_frames.cpu().numpy()
        self._transform = params
        bm, csi = extract_boundaries_fast(self._registered_masks)
        bm, csi = clean_boundaries(bm, csi)
        self._registered_lines = torch.stack(
            [torch.tensor(bm), torch.tensor(csi)], dim=1
        ).cpu()

        if self.cache_dir is not None:
            self._save_to_cache()

    def plot(self, which="registered", index=None):
        if which == "registered":
            frames = self.registered_frames
            masks = self.registered_masks
        elif which == "raw":
            frames = self.raw_frames
            masks = self.raw_masks
        else:
            raise ValueError(f"Unknown 'which' value: {which}")

        if index is None:
            index = np.random.randint(0, len(frames))

        frame = frames[index]
        mask = masks[index]

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(frame, cmap="gray")
        axes[0].set_title(f"{which} frame {index}")
        axes[0].axis("off")
        axes[1].imshow(frame, cmap="gray")
        # Use contourf to fill the mask area with a semi-transparent color and edge visible
        axes[1].contourf(mask, levels=[0.5, 1], colors=["red"], alpha=0.3)
        axes[1].set_title(f"{which} frame {index} with mask overlay")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()
