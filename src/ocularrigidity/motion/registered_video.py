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


from ocularrigidity.registration.rigid import register_masks_by_displacement
from ocularrigidity.rigidity.features import (
    compute_deltaY_boundaries,
)
from ocularrigidity.segmentation.postprocess.interfaces import (
    clean_boundaries,
    extract_boundaries_fast,
    extract_boundaries_gpu,
)


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
        batch_size: int = 128,
        cache_dir: Path = None,
        lateral_method: Literal["xcorr", "fullframe", "both"] = "xcorr",
        subpixel: bool = True,
        crop_w_x: float = 0.75,
        bp_lo: float = 0.02,
        bp_hi: float = 0.5,
        median_registration: bool = False,
        median_max_vshift: int = 30,
        median_use_shadow: bool = True,
        median_use_log: bool = True,
        median_shadow_n: float = 4.0,
        median_shadow_a: float = 0.8,
        median_log_kernel_size: int = 9,
        median_log_sigma: float = 3.0,
    ):
        self.video = video
        self.root_masks = root_masks
        self.root_data = root_data

        self.skip_first_n_frames = skip_first_n_frames
        self.drop_last_n_frames = drop_last_n_frames
        self.flatten = flatten
        self.horizontal_alignment = horizontal_alignment
        # Lateral (x) shift estimator: "xcorr" (profile cross-correlation) or
        # "fullframe" (2D phase correlation). Part of the cache key below.
        self.lateral_method = lateral_method
        # Apply parabolic sub-pixel refinement to the lateral shift peak. Part of
        # the cache key so integer- and sub-pixel-registered results don't mix.
        self.subpixel = subpixel
        # Central fraction of the WIDTH kept before the FFT in the "fullframe"
        # lateral estimator (height is not cropped). No effect when
        # lateral_method == "xcorr".
        self.crop_w_x = crop_w_x
        # Bornes basse/haute (fraction de Nyquist) du passe-bande spectral de
        # l'estimateur "fullframe". No effect when lateral_method == "xcorr".
        self.bp_lo = bp_lo
        self.bp_hi = bp_hi
        # 2e passe de recalage axial (RPE) sur la mediane du volume. Desactivee
        # par defaut ; ses parametres font partie de la cle de cache (_cache_meta).
        self.median_registration = median_registration
        self.median_max_vshift = median_max_vshift
        self.median_use_shadow = median_use_shadow
        self.median_use_log = median_use_log
        self.median_shadow_n = median_shadow_n
        self.median_shadow_a = median_shadow_a
        self.median_log_kernel_size = median_log_kernel_size
        self.median_log_sigma = median_log_sigma
        self.verbose = verbose
        self.use_encoded_video = use_encoded_video

        # Optional on-disk cache of the registration result. When set, the
        # registered frames/masks and the transform params are persisted under
        # ``cache_dir`` (mirroring the layout of ROOT_COMPRESSED_VIDEO/ROOT_MASKS)
        # and reused on subsequent runs. None (default) disables caching.
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
        """Registration parameters the cache is keyed on (validated on load).

        Includes ``lateral_method`` so an xcorr-registered cache is never served
        for a fullframe request (and vice versa).
        """
        return dict(
            skip_first_n_frames=self.skip_first_n_frames,
            drop_last_n_frames=self.drop_last_n_frames,
            flatten=int(self.flatten),
            horizontal_alignment=int(self.horizontal_alignment),
            lateral_method=self.lateral_method,
            subpixel=int(self.subpixel),
            crop_w_x=float(self.crop_w_x),
            bp_lo=float(self.bp_lo),
            bp_hi=float(self.bp_hi),
            median_registration=int(self.median_registration),
            median_max_vshift=int(self.median_max_vshift),
            median_use_shadow=int(self.median_use_shadow),
            median_use_log=int(self.median_use_log),
            median_shadow_n=float(self.median_shadow_n),
            median_shadow_a=float(self.median_shadow_a),
            median_log_kernel_size=int(self.median_log_kernel_size),
            median_log_sigma=float(self.median_log_sigma),
        )

    def _load_from_cache(self) -> bool:
        """Populate registration results from cache. Returns True on a valid hit."""
        paths = self._cache_paths()
        if not all(p.exists() for p in paths.values()):
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
        if not self.horizontal_alignment:
            registered_masks, registered_frames, params = (
                register_masks_by_displacement(
                    registered_masks,
                    registered_frames,
                    batch_size=self._batch_size,
                    correct_dx=self.horizontal_alignment,
                    flatten_rpe=self.flatten,
                    verbose=self.verbose,
                    return_params=True,
                    lateral_method="xcorr",
                    device=self._device,
                    subpixel=self.subpixel,
                )
            )
        else:
            if self.lateral_method in ("xcorr", "both"):
                registered_masks, registered_frames, params = (
                    register_masks_by_displacement(
                        registered_masks,
                        registered_frames,
                        batch_size=self._batch_size,
                        correct_dx=self.horizontal_alignment,
                        flatten_rpe=self.flatten,
                        verbose=self.verbose,
                        return_params=True,
                        lateral_method="xcorr",
                        device=self._device,
                        subpixel=self.subpixel,
                    )
                )
            if (
                self.lateral_method in ("fullframe", "both")
                and self.horizontal_alignment
            ):
                registered_masks, registered_frames, params = (
                    register_masks_by_displacement(
                        registered_masks,
                        registered_frames,
                        batch_size=self._batch_size,
                        correct_dx=self.horizontal_alignment,
                        flatten_rpe=self.flatten,
                        verbose=self.verbose,
                        return_params=True,
                        lateral_method="fullframe",
                        device=self._device,
                        subpixel=self.subpixel,
                        crop_w_x=self.crop_w_x,
                        bp_lo=self.bp_lo,
                        bp_hi=self.bp_hi,
                    )
                )

        # 2e passe optionnelle : recalage axial de chaque A-scan sur la mediane
        # du volume deja recale (identification de la RPE). Opere sur le volume
        # en memoire ; le deplacement par colonne est stocke dans transform["dy_median"].
        if self.median_registration:
            from ocularrigidity.registration.axial.median_registration import (
                register_ascans_to_median,
            )

            registered_frames, registered_masks, dy_median = (
                register_ascans_to_median(
                    registered_frames,
                    registered_masks,
                    max_vshift=self.median_max_vshift,
                    use_shadow=self.median_use_shadow,
                    use_log=self.median_use_log,
                    shadow_n=self.median_shadow_n,
                    shadow_a=self.median_shadow_a,
                    log_kernel_size=self.median_log_kernel_size,
                    log_sigma=self.median_log_sigma,
                    subpixel=self.subpixel,
                    batch_size=self._batch_size,
                    device=self._device,
                    verbose=self.verbose,
                )
            )
            params = dict(params)
            params["dy_median"] = dy_median

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
