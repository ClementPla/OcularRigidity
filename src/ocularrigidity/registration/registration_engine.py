"""This module is responsible for loading a video and its corresponding masks, performing registration, and providing access to the registered frames, masks, and computed thickness."""

import numpy as np
from pathlib import Path
from typing import Optional
import torch

from ocularrigidity.registration.config import RegistrationConfig

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


def cache_paths_for(cache_dir: Path, video_id: Path) -> dict:
    """Cache file locations for one video's registered frames, masks and transform."""
    cache_dir = Path(cache_dir)
    return {
        "frames": cache_dir / "registered_frames" / video_id / "cube.mp4",
        "masks": cache_dir / "registered_masks" / video_id / "mask.npz",
        "transform": cache_dir / "registered_masks" / video_id / "transform.npz",
    }


def cache_meta_for(config: RegistrationConfig) -> dict:
    """Registration parameters the cache is keyed on (validated on load)."""
    c = config
    return dict(
        skip_first_n_frames=c.skip_first_n_frames,
        drop_last_n_frames=c.drop_last_n_frames,
        correct_transversal=int(c.correct_transversal),
        correct_axial=int(c.correct_axial),
        flatten_rpe=int(c.flatten_rpe),
        fovea_correction_enabled=int(c.fovea_correction_enabled),
        lateral_method=c.lateral_method,
        max_lateral_shift=int(c.max_lateral_shift),
        smooth_transversal=int(c.smooth_transversal),
        smooth_transversal_sigma=float(c.smooth_transversal_sigma),
        axial_refinement=int(c.axial_refinement),
        max_axial_shift=int(c.max_axial_shift),
        subpixel=int(c.subpixel),
        crop_factor=float(c.crop_factor),
        scale_factor=float(c.scale_factor),
        transversal_bandpass=str(c.transversal_bandpass),
        axial_bandpass=str(c.axial_bandpass),
        # Which estimator produced the transform, and (for the learned one)
        # from which weights. Two caches that differ only in this are not
        # interchangeable, so the key has to carry it.
        method=c.method,
        registrator_checkpoint=(
            str(c.registrator_checkpoint) if c.method == "learned" else ""
        ),
        # Both change the transform itself, not just how long it takes to get.
        reference_selection=c.reference_selection,
        probe_frames=int(c.probe_frames),
        # Which pixels went in: the compressed mp4 is lossy against the raw
        # cube.bin (mean |difference| ~11.7 grey levels), so a registration
        # computed from one is not the same artifact as one from the other.
        use_encoded_video=int(c.use_encoded_video),
        filter_bad_columns=int(c.filter_bad_columns),
    )


def cache_is_valid(
    cache_dir: Path, video_id: Path, config: RegistrationConfig
) -> bool:
    """Is there a cached registration for this video, matching ``config``?

    Reads only the small transform.npz, so a batch script can ask this about a
    whole cohort up front — which volumes to re-register is a planning decision,
    and planning must not decode 4.7 GB per video to make it.
    """
    paths = cache_paths_for(cache_dir, video_id)
    if not all(p.exists() for p in paths.values()):
        return False
    try:
        data = np.load(paths["transform"])
    except Exception:
        return False
    # Stale cache if it was produced with different registration params.
    # Compare as strings so int and string keys (e.g. lateral_method) work.
    # A key absent from an older cache is not compared, so caches written
    # before newer keys (crop/scale/bandpass) were added stay valid.
    return all(
        str(data[k]) == str(v)
        for k, v in cache_meta_for(config).items()
        if k in data
    )


def read_cache_payload(
    cache_dir: Path,
    video_id: Path,
    config: RegistrationConfig,
    verbose: bool = False,
) -> Optional[dict]:
    """Decode a cached registration, or return None if there is no valid one.

    Pure file IO and CPU decode — no GPU, no torch autograd, nothing tied to
    this process — so a batch caller can run it in a DataLoader worker ahead of
    time and hand the result to :meth:`VideoRegistrator.prime_from_cache`.
    """
    paths = cache_paths_for(cache_dir, video_id)
    if not cache_is_valid(cache_dir, video_id, config):
        return None
    try:
        data = np.load(paths["transform"])
        frames = read_gray(paths["frames"])
        masks = load_mask(paths["masks"])
    except Exception as e:
        if verbose:
            print(f"Ignoring unreadable registration cache ({e})")
        return None
    if frames.shape[0] != masks.shape[0]:
        return None
    return {
        "frames": frames,
        "masks": masks,
        "transform": {"dx": data["dx"], "dy": data["dy"]},
        "path": paths["masks"].parent,
    }


class VideoRegistrator:
    def __init__(
        self,
        video: Path,
        root_masks: Optional[Path] = None,
        root_data: Optional[Path] = None,
        config: Optional[RegistrationConfig] = None,
        *,
        # --- in-memory inputs (bypass path loading) ---
        frames: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        # --- runtime / cache (per-invocation, not algorithmic) ---
        device: str = "cuda",
        cache_dir: Path = None,
        overwrite_cache: bool = False,
        verbose: bool = True,
        cq_cache: int = 18,
    ):
        """``video`` is the video identifier/name, used for cache and mask paths.

        Frames and masks are normally loaded from ``root_data``/``root_masks``
        using that name. Alternatively, pass ``frames`` and/or ``masks`` as
        numpy arrays to feed them directly, in which case the corresponding
        ``root_*`` argument is not required. ``video`` should still be provided
        as a name so caching and identification keep working.
        """
        self.video = video
        self.root_masks = root_masks
        self.root_data = root_data

        self.config = config if config is not None else RegistrationConfig()

        self.verbose = verbose

        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

        # In-memory inputs supplied directly (bypass path loading). Stored
        # untrimmed; the frame slice is applied lazily like the loaded arrays.
        self._provided_frames = frames
        self._provided_masks = masks

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
        self._overwrite_cache = overwrite_cache
        self._cq_cache = cq_cache
        self._loaded_from_cache = False

    @property
    def skip_first_n_frames(self) -> int:
        return self.config.skip_first_n_frames

    @property
    def drop_last_n_frames(self) -> int:
        return self.config.drop_last_n_frames

    @property
    def flatten_rpe(self) -> bool:
        return self.config.flatten_rpe

    @property
    def correct_transversal(self) -> bool:
        return self.config.correct_transversal

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
        if self.root_data is not None and (self.root_data / self.video).is_file():
            return self.video.parent
        return self.video

    def _cache_paths(self) -> dict:
        """Cache file locations for the registered frames, masks and transform."""
        return cache_paths_for(self.cache_dir, self._video_id)

    def _cache_meta(self) -> dict:
        """Registration parameters the cache is keyed on (validated on load)."""
        return cache_meta_for(self.config)

    def _adopt_cache_payload(self, payload: dict) -> None:
        """Install a decoded cache payload as this registrator's result."""
        frames, masks = payload["frames"], payload["masks"]
        # A payload that travelled through a DataLoader arrives as tensors
        # backed by shared memory; .numpy() views that buffer rather than
        # copying another 4.7 GB, and keeps it alive for as long as we hold it.
        if isinstance(frames, torch.Tensor):
            frames = frames.numpy()
        if isinstance(masks, torch.Tensor):
            masks = masks.numpy()

        self._registered_frames = frames
        self._registered_masks = masks
        self._transform = payload["transform"]
        bm, csi = extract_boundaries_gpu(masks, to_numpy=False)
        self._registered_lines = torch.stack([bm, csi], dim=1).cpu()

    def prime_from_result(
        self,
        registered_frames,
        registered_masks,
        transform: dict,
        raw_masks=None,
    ) -> None:
        """Install a registration computed elsewhere, leaving the cache writable.

        The counterpart of :meth:`prime_from_cache`: that one adopts a payload
        that *came from* the cache and must not be written back, this one
        adopts a freshly computed result that must. It is how the fused
        single-pass stage (:mod:`ocularrigidity.registration.fused`) hands its
        output over — the registrator then behaves as if it had registered the
        video itself, so :meth:`flush_cache` and every downstream consumer are
        unchanged and the old registration path is never entered.
        """
        self._adopt_cache_payload(
            {
                "frames": registered_frames,
                "masks": registered_masks,
                "transform": transform,
            }
        )
        if raw_masks is not None:
            self._raw_masks = raw_masks
        self._loaded_from_cache = False

    def prime_from_cache(self, payload: dict) -> None:
        """Adopt a cache payload read by :func:`read_cache_payload` elsewhere.

        Lets a batch caller move the cache decode (~25 s of CPU per volume) into
        a DataLoader worker and still get a registrator that behaves as if it
        had loaded the cache itself — including not writing it back.
        """
        self._adopt_cache_payload(payload)
        self._loaded_from_cache = True

    def _load_from_cache(self) -> bool:
        """Populate registration results from cache. Returns True on a valid hit."""
        if self._overwrite_cache:
            return False
        payload = read_cache_payload(
            self.cache_dir, self._video_id, self.config, verbose=self.verbose
        )
        if payload is None:
            return False

        self._adopt_cache_payload(payload)
        if self.verbose:
            print(f"Loaded registration from cache: {payload['path']}")
        return True

    def _save_to_cache(self) -> None:
        """Persist registered frames (lossless mkv), masks and transform params."""
        paths = self._cache_paths()
        for p in paths.values():
            p.parent.mkdir(parents=True, exist_ok=True)
        cube_to_mp4_fastest(
            self._registered_frames, str(paths["frames"]), cq=self._cq_cache, fps=60
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
            if self._provided_frames is not None:
                self._raw_frames = np.asarray(self._provided_frames)[self._frame_slice]
                return self._raw_frames
            if self.root_data is None:
                raise ValueError(
                    "No `frames` array and no `root_data` to load frames from."
                )
            if self.config.use_encoded_video:
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
            if not self.config.use_encoded_video:
                # load_cube returns (N, W, H); align to the masks' (H, W)
                # orientation so registration sees a consistent frame/mask grid.
                mask_hw = self.raw_masks.shape[1:]
                if self._raw_frames.shape[1:] == mask_hw[::-1]:
                    self._raw_frames = self._raw_frames.transpose(0, 2, 1)
        return self._raw_frames

    @property
    def raw_masks(self):
        if self._raw_masks is None:
            if self._provided_masks is not None:
                self._raw_masks = np.asarray(self._provided_masks)[self._frame_slice]
                return self._raw_masks
            if self.root_masks is None:
                raise ValueError(
                    "No `masks` array and no `root_masks` to load masks from."
                )
            # If self.video points to a file, take its parent directory as the video id for mask loading
            if self.root_data is not None and (self.root_data / self.video).is_file():
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

    def flush_cache(self) -> None:
        """Persist the computed registration, if there is a cache to write to.

        Split out of :meth:`compute_registration` so a batch caller can run it
        on a background thread. The HEVC encode and the mask compression are
        seconds of CPU and subprocess work per volume, and none of it needs the
        GPU that the next volume is waiting for. A registration that came *from*
        the cache is not written back.
        """
        if self.cache_dir is not None and not self._loaded_from_cache:
            self._save_to_cache()

    def compute_registration(self, save_cache: bool = True):
        """Register the video, reusing a cached result when one is valid.

        ``save_cache=False`` computes without writing; the caller then owns the
        write and must call :meth:`flush_cache` (typically on another thread).
        """
        # Reuse a previously cached registration when available.
        if self.cache_dir is not None and self._load_from_cache():
            self._loaded_from_cache = True
            return

        raw_masks = self.raw_masks
        raw_frames = self.raw_frames
        registered_masks = raw_masks
        registered_frames = raw_frames

        registered_masks, registered_frames, params = register_videos(
            registered_masks,
            registered_frames,
            self.config,
            device=self._device,
            verbose=self.verbose,
            return_params=True,
        )

        self._registered_masks = registered_masks.cpu().numpy() > 0
        self._registered_frames = registered_frames.cpu().numpy()
        self._transform = params
        bm, csi = extract_boundaries_fast(self._registered_masks)
        bm, csi = clean_boundaries(bm, csi)
        self._registered_lines = torch.stack(
            [torch.tensor(bm), torch.tensor(csi)], dim=1
        ).cpu()

        if save_cache:
            self.flush_cache()

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
