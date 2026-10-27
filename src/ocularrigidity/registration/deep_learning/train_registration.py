import argparse
import json
import os
import time
import random
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm

from ocularrigidity.registration.deep_learning.models.losses import (
    UnsupervisedRegistrationLoss,
    cycle_consistency_loss,
    shift_equivariance_loss,
)
from ocularrigidity.registration.deep_learning.correlation_estimator import (
    CorrelationBaseline,
)
from ocularrigidity.registration.deep_learning.models.regressor import (
    RegistrationRegressor,
)

cv2.setNumThreads(0)  # avoid worker/cv2 thread oversubscription


def _load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _video_of(stem: str) -> str:
    """Recover {video} from a '{video}_{idx}' stem by dropping the trailing frame-index token."""
    return stem.rsplit("_", 1)[0]


def _read_uint8(path: Path) -> torch.Tensor:
    """Single-channel PNG -> [1, H, W] uint8."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return torch.from_numpy(img).unsqueeze(0)


def _normalise(img: torch.Tensor) -> torch.Tensor:
    """uint8 -> float32 with the encoder's normalisation ((x/255 - 0.5) / 0.5)."""
    return img.to(torch.float32).mul_(1.0 / 255.0).sub_(0.5).div_(0.5)


def _read_image(path: Path) -> torch.Tensor:
    """Single-channel PNG -> [1, H, W] float32, normalised for the encoder."""
    return _normalise(_read_uint8(path))


class ImageCache:
    """Decoded frames kept in RAM as uint8 (4x smaller than the float input), least recently used evicted past ``max_bytes``. One instance is shared by every dataset of a process, so a frame used by the pair, triplet and shift datasets is stored once."""

    def __init__(self, max_bytes: int):
        self.max_bytes = int(max_bytes)
        self.nbytes = 0
        self._store: "OrderedDict[Path, torch.Tensor]" = OrderedDict()

    def __call__(self, path: Path) -> torch.Tensor:
        img = self._store.get(path)
        if img is not None:
            self._store.move_to_end(path)
            return _normalise(img)
        img = _read_uint8(path)
        size = img.numel()
        if size <= self.max_bytes:
            self._store[path] = img
            self.nbytes += size
            while self.nbytes > self.max_bytes:
                _, old = self._store.popitem(last=False)
                self.nbytes -= old.numel()
        return _normalise(img)


def _load_image(path: Path, cache: Optional[ImageCache]) -> torch.Tensor:
    return cache(path) if cache is not None else _read_image(path)


def _list_frames(root: Path) -> Dict[str, List[Tuple[Path, Optional[Path]]]]:
    """``video -> [(image_path, transform_path | None), ...]``, index 0 is the reference frame (whose transform is the identity by construction"""
    frames: Dict[str, List[Tuple[Path, Optional[Path]]]] = defaultdict(list)
    d_tf = root / "transforms"
    for p in sorted((root / "images" / "fixed").glob("*.png")):
        frames[_video_of(p.stem)].insert(0, (p, None))
    for p in sorted((root / "images" / "moving").glob("*.png")):
        vid = _video_of(p.stem)
        if vid in frames:
            tf = d_tf / f"{p.stem}.pt"
            frames[vid].append((p, tf if tf.exists() else None))
    return dict(frames)


def _crop_dy(dy: torch.Tensor, win, crop) -> torch.Tensor:
    """Take the crop's columns out of a full-width per-column displacement."""
    if crop is None or win is None:
        return dy
    return dy[win[1] : win[1] + crop[1]]


def _key(vid: str, idx: int, win) -> str:
    """FeatureCache key."""
    return f"{vid}#{idx}" if win is None else f"{vid}#{idx}@{win[0]}_{win[1]}"


def _retina_row(paths: Sequence[Path], n: int = 3) -> int:
    """Row to centre a crop on: the brightest band of the B-scan."""
    rows = []
    for q in list(paths)[:n]:
        img = cv2.imread(str(q), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        prof = img.astype(np.float32).mean(axis=1)
        k = 51
        prof = np.convolve(prof, np.ones(k, np.float32) / k, mode="same")
        rows.append(int(prof.argmax()))
    return int(np.median(rows)) if rows else 0


# first-frame path -> (retina row, frame shape): shared by every dataset of a
# process, so the frames are read once rather than once per dataset
_CENTRE_CACHE: Dict[str, Tuple[int, Tuple[int, int]]] = {}


def _video_centres(frames, crop):
    """``video -> retina row``, plus the full frame shape."""
    if crop is None:
        return None, None
    centres = {}
    full = None
    for vid, fl in sorted(frames.items()):
        key = str(fl[0][0])
        if key not in _CENTRE_CACHE:
            img = cv2.imread(key, cv2.IMREAD_GRAYSCALE)
            _CENTRE_CACHE[key] = (_retina_row([q for q, _ in fl]), img.shape[:2])
        centres[vid], shape = _CENTRE_CACHE[key]
        if full is None:
            full = shape
    return centres, full


def load_bm_lines(directory: Path, videos) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """``frame stem -> (rows (W,) float32, present (W,) bool)`` for ``videos``, from
    ``dataset/precompute_bm_lines.py``; every frame of these videos must be there."""
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    missing = [v for v in videos if not (Path(directory) / f"{v}.npz").exists()]
    if missing:
        raise SystemExit(
            f"--bm-lines {directory}: no lines for {len(missing)} video(s), e.g. {missing[0]}; "
            "run dataset/precompute_bm_lines.py"
        )
    for v in videos:
        z = np.load(Path(directory) / f"{v}.npz")
        for stem, rows, present in zip(z["stems"], z["rows"], z["present"]):
            out[str(stem)] = (rows, present)
    return out


def crop_bm_line(rows: np.ndarray, present: np.ndarray, win, crop):
    """The line in the rows/columns of a crop window, exactly as ``_apply_crop``
    cuts the frame: columns ``c0:c0+cw``, rows shifted by ``r0``; a column whose
    line falls outside the window's rows is not present."""
    rows = torch.from_numpy(np.asarray(rows, dtype=np.float32))
    present = torch.from_numpy(np.asarray(present, dtype=bool))
    if crop is None or win is None:
        return rows, present
    (r0, c0), (ch, cw) = win, crop
    rows = rows[c0 : c0 + cw] - r0
    present = present[c0 : c0 + cw] & (rows >= 0) & (rows <= ch - 1)
    return rows, present


def _draw_windows(centres, full, crop, seed: int, jitter: int = 64):
    """``video -> (row0, col0)``, re-drawable per epoch."""
    if crop is None:
        return None
    H, W = full
    ch, cw = crop
    rng = random.Random(seed)
    win = {}
    for vid, r in centres.items():
        r0 = r - ch // 2 + rng.randint(-jitter, jitter)
        win[vid] = (
            int(min(max(r0, 0), max(H - ch, 0))),
            rng.randint(0, max(W - cw, 0)),
        )
    return win


def _apply_crop(img: torch.Tensor, win, crop):
    """Crop ``(1, H, W)`` to ``crop`` at origin ``win``"""
    if crop is None or win is None:
        return img
    ch, cw = crop
    r0, c0 = win
    return img[..., r0 : r0 + ch, c0 : c0 + cw]


# --------------------------------------------------------------------------- #
# Data
class FramePairDataset(Dataset):
    """Ordered (fixed, moving) frame pairs drawn from within a single volume."""

    def __init__(
        self,
        root: Path,
        videos: Optional[Set[str]] = None,
        *,
        pair_mode: str = "within",
        pairs_per_video: int = 200,
        seed: int = 0,
        image_cache: Optional[ImageCache] = None,
        crop: Optional[Tuple[int, int]] = None,
        bm_lines: Optional[Path] = None,
    ):
        assert pair_mode in ("within", "reference")
        root = Path(root)
        self.image_cache = image_cache
        self.crop = crop

        rng = random.Random(seed)
        self.samples: List[Tuple[str, int, int]] = []
        self.frames = _list_frames(root)
        self._seed = seed
        self._centres, self._full = _video_centres(self.frames, crop)
        self.windows = _draw_windows(self._centres, self._full, crop, seed)
        # Dumped transforms are always full-frame width
        _probe = cv2.imread(str(next(iter(self.frames.values()))[0][0]), cv2.IMREAD_GRAYSCALE)
        self.full_w = int(_probe.shape[1])
        for vid, fl in sorted(self.frames.items()):
            if videos is not None and vid not in videos:
                continue
            if len(fl) < 2:
                continue
            if pair_mode == "reference":
                pairs = [(0, j) for j in range(1, len(fl))]
            else:
                pairs = [
                    (i, j) for i in range(len(fl)) for j in range(len(fl)) if i != j
                ]
                rng.shuffle(pairs)
                pairs = pairs[:pairs_per_video]
            self.samples += [(vid, i, j) for i, j in pairs]
        # Precomputed BM lines: the only geometric change to the frames here is
        # the crop, which __getitem__ applies to the lines with the same window
        self.lines = (
            load_bm_lines(bm_lines, sorted({v for v, _, _ in self.samples}))
            if bm_lines is not None
            else None
        )

    def set_epoch(self, epoch: int) -> None:
        """Re-draw this epoch's crop windows (no-op without --crop)."""
        if self.crop is not None:
            self.windows = _draw_windows(
                self._centres, self._full, self.crop, self._seed + 9973 * epoch
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _image(self, path: Path) -> torch.Tensor:
        return _load_image(path, self.image_cache)

    def _transform(self, tf: Optional[Path], width: int):
        """(dx, dy) of a frame w.r.t."""
        if tf is None:
            return torch.zeros(()), torch.zeros(width)
        t = _load(tf)
        return t["x"].float().reshape(()), t["y"].float().reshape(-1)

    def __getitem__(self, k: int):
        vid, i, j = self.samples[k]
        fl = self.frames[vid]
        win = self.windows[vid] if self.windows else None
        fixed = _apply_crop(self._image(fl[i][0]), win, self.crop)
        moving = _apply_crop(self._image(fl[j][0]), win, self.crop)
        W = self.full_w
        dx_i, dy_i = self._transform(fl[i][1], W)
        dx_j, dy_j = self._transform(fl[j][1], W)
        # The diagnostic is only meaningful if *both* frames' transforms are known.
        has_gt = (i == 0 or fl[i][1] is not None) and (j == 0 or fl[j][1] is not None)
        item = (
            fixed,
            moving,
            dx_j - dx_i,
            _crop_dy(dy_j - dy_i, win, self.crop),
            torch.tensor(float(has_gt)),
            _key(vid, i, win),
            _key(vid, j, win),
        )
        if self.lines is None:
            return item
        line_f, present_f = crop_bm_line(*self.lines[fl[i][0].stem], win, self.crop)
        line_m, present_m = crop_bm_line(*self.lines[fl[j][0].stem], win, self.crop)
        return item + (line_f, present_f, line_m, present_m)


class FrameTripletDataset(Dataset):
    """Closed triplets ``(i, j, k)`` of frames from one video, for the cycle- consistency term: a lateral shift is a pure translation, so it composes exactly across any three frames of the same eye"""

    def __init__(
        self,
        root: Path,
        videos: Optional[Set[str]] = None,
        *,
        triplets_per_video: int = 40,
        seed: int = 0,
        image_cache: Optional[ImageCache] = None,
        crop: Optional[Tuple[int, int]] = None,
    ):
        root = Path(root)
        self.image_cache = image_cache
        self.crop = crop
        self.frames = _list_frames(root)
        self._seed = seed
        self._centres, self._full = _video_centres(self.frames, crop)
        self.windows = _draw_windows(self._centres, self._full, crop, seed)

        rng = random.Random(seed)
        self.samples: List[Tuple[str, int, int, int]] = []
        for vid, fl in sorted(self.frames.items()):
            if videos is not None and vid not in videos:
                continue
            n = len(fl)
            if n < 3:
                continue
            for _ in range(triplets_per_video):
                i, j, k = rng.sample(range(n), 3)
                self.samples.append((vid, i, j, k))

    def set_epoch(self, epoch: int) -> None:
        """Re-draw this epoch's crop windows (no-op without --crop)."""
        if self.crop is not None:
            self.windows = _draw_windows(
                self._centres, self._full, self.crop, self._seed + 9973 * epoch
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _image(self, path: Path) -> torch.Tensor:
        return _load_image(path, self.image_cache)

    def __getitem__(self, idx: int):
        vid, i, j, k = self.samples[idx]
        fl = self.frames[vid]
        win = self.windows[vid] if self.windows else None
        c = lambda n: _apply_crop(self._image(fl[n][0]), win, self.crop)
        fi, fj, fk = c(i), c(j), c(k)
        # Row order (i,j), (j,k), (i,k)
        fixed = torch.stack([fi, fj, fi], dim=0)
        moving = torch.stack([fj, fk, fk], dim=0)
        keys_f = [_key(vid, i, win), _key(vid, j, win), _key(vid, i, win)]
        keys_m = [_key(vid, j, win), _key(vid, k, win), _key(vid, k, win)]
        return fixed, moving, keys_f, keys_m


def collate_triplets(batch):
    """Bt items of 3 stacked pairs each -> one (3*Bt,...) batch, triplet-major (item 0's 3 rows, then item 1's, ...) so ``dx.view(Bt, 3)`` recovers ``(dx_ij, dx_jk, dx_ik)`` per triplet after the model runs on the batch."""
    fixed = torch.cat([b[0] for b in batch], dim=0)
    moving = torch.cat([b[1] for b in batch], dim=0)
    keys_f = [key for b in batch for key in b[2]]
    keys_m = [key for b in batch for key in b[3]]
    return fixed, moving, keys_f, keys_m


def _shift_and_mask(img: torch.Tensor, delta: int, margin: int) -> torch.Tensor:
    """Shift ``img`` content right by ``delta`` whole columns and blank ``margin`` columns at both edges."""
    out = torch.zeros_like(img)
    if delta > 0:
        out[..., delta:] = img[..., :-delta]
    elif delta < 0:
        out[..., :delta] = img[..., -delta:]
    else:
        out = img.clone()
    if margin > 0:
        out[..., :margin] = 0
        out[..., -margin:] = 0
    return out


class FrameShiftDataset(Dataset):
    """(fixed, moving) pairs plus a copy of ``moving`` shifted by a known integer ``delta``, for the shift-equivariance term."""

    def __init__(
        self,
        root: Path,
        videos: Optional[Set[str]] = None,
        *,
        shifts_per_video: int = 40,
        shift_max: int = 8,
        seed: int = 0,
        image_cache: Optional[ImageCache] = None,
        crop: Optional[Tuple[int, int]] = None,
    ):
        root = Path(root)
        self.image_cache = image_cache
        self.shift_max = int(shift_max)
        self.crop = crop
        self.frames = _list_frames(root)
        self._seed = seed
        self._centres, self._full = _video_centres(self.frames, crop)
        self.windows = _draw_windows(self._centres, self._full, crop, seed)

        rng = random.Random(seed)
        choices = [d for d in range(-self.shift_max, self.shift_max + 1) if d != 0]
        self.samples: List[Tuple[str, int, int, int]] = []
        for vid, fl in sorted(self.frames.items()):
            if videos is not None and vid not in videos:
                continue
            if len(fl) < 2:
                continue
            for _ in range(shifts_per_video):
                i, j = rng.sample(range(len(fl)), 2)
                self.samples.append((vid, i, j, rng.choice(choices)))

    def set_epoch(self, epoch: int) -> None:
        """Re-draw this epoch's crop windows (no-op without --crop)."""
        if self.crop is not None:
            self.windows = _draw_windows(
                self._centres, self._full, self.crop, self._seed + 9973 * epoch
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _image(self, path: Path) -> torch.Tensor:
        return _load_image(path, self.image_cache)

    def __getitem__(self, idx: int):
        vid, i, j, delta = self.samples[idx]
        fl = self.frames[vid]
        win = self.windows[vid] if self.windows else None
        fixed = _apply_crop(self._image(fl[i][0]), win, self.crop)
        moving = _apply_crop(self._image(fl[j][0]), win, self.crop)
        m = self.shift_max
        return (
            fixed,
            _shift_and_mask(moving, 0, m),
            _shift_and_mask(moving, delta, m),
            torch.tensor(float(delta)),
            _key(vid, i, win),
        )


def _patients(root: Path) -> Dict[str, str]:
    """``video -> patient`` from the ``manifest.csv`` of prepare_unsupervised_data, if any."""
    manifest = Path(root) / "manifest.csv"
    if not manifest.exists():
        return {}
    import csv

    with open(manifest, newline="") as f:
        return {row["name"]: str(row["patient"]) for row in csv.DictReader(f)}


def split_videos(root: Path, val_frac: float, seed: int) -> Tuple[Set[str], Set[str]]:
    """Split by *patient* when ``manifest.csv`` says who each video belongs to (two
    eyes or two visits of one patient are not independent), else by video."""
    vids = sorted(
        {_video_of(p.stem) for p in (Path(root) / "images" / "fixed").glob("*.png")}
    )
    patient_of = _patients(root)
    groups: Dict[str, List[str]] = defaultdict(list)
    for v in vids:
        groups[patient_of.get(v, v)].append(v)
    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_val = max(1, round(len(keys) * val_frac)) if len(keys) > 1 else 0
    val = {v for k in keys[:n_val] for v in groups[k]}
    return set(vids) - val, val


class VideoChunkBatchSampler(Sampler):
    """Batches drawn from a small working set of videos at a time.

    Under DDP (``world_size`` > 1) each rank takes every ``world_size``-th video of
    the epoch's shuffle. ``equalize`` (training) makes every rank yield exactly the
    same number of batches, which DDP needs to keep its gradient all-reduces in
    step: the rank's stream is cut, or re-cycled with another shuffle, at
    ``samples // (batch_size * world_size)`` batches. With one rank the batches
    are exactly those of the single-GPU trainer.
    """

    def __init__(
        self,
        samples: Sequence[Tuple],
        batch_size: int,
        videos_in_flight: int = 4,
        shuffle: bool = True,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        equalize: bool = False,
    ):
        self.groups: Dict[str, List[int]] = defaultdict(list)
        for idx, sample in enumerate(samples):
            self.groups[sample[0]].append(idx)
        self.batch_size = batch_size
        self.k = max(1, videos_in_flight)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.rank = rank
        self.world = max(1, world_size)
        self.equalize = equalize and self.world > 1
        if self.world > 1 and len(self.groups) < self.world:
            raise ValueError(
                f"{len(self.groups)} video(s) for {self.world} DDP ranks: every rank "
                "needs at least one (raise --val-frac if this is the validation set)"
            )
        if self.equalize:
            n_samples = sum(len(g) for g in self.groups.values())
            self._len = max(1, n_samples // (batch_size * self.world))
        else:
            self._len = sum(1 for _ in self._rank_batches(0))

    def _rank_batches(self, epoch: int):
        rng = random.Random(self.seed + epoch)
        vids = list(self.groups)
        if self.shuffle:
            rng.shuffle(vids)
        vids = vids[self.rank :: self.world]
        for s in range(0, len(vids), self.k):
            pool = [i for v in vids[s : s + self.k] for i in self.groups[v]]
            if self.shuffle:
                rng.shuffle(pool)
            for b in range(0, len(pool), self.batch_size):
                yield pool[b : b + self.batch_size]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        if not self.equalize:
            yield from self._rank_batches(self.epoch)
            return
        n, cycle = 0, 0
        while n < self._len:
            yielded = False
            # A different shuffle each time the stream is re-cycled
            for batch in self._rank_batches(self.epoch + 100_003 * cycle):
                if n == self._len:
                    return
                yield batch
                n += 1
                yielded = True
            if not yielded:
                return
            cycle += 1


def encoder_pyramid(encoder, images: torch.Tensor) -> List[torch.Tensor]:
    """The 4 feature levels (strides 4 to 32) the regressor sits on, as at inference
    (``registration.fused``): any smp encoder, identical to ``forward_features`` for MiT."""
    return list(encoder(images)[2:])


class FeatureCache:
    """LRU of frozen-encoder pyramids, keyed by frame."""

    def __init__(self, encoder, capacity: int = 192, amp_dtype=None):
        self.encoder = encoder
        self.capacity = capacity
        self.amp_dtype = amp_dtype
        self._cache: "OrderedDict[str, List[torch.Tensor]]" = OrderedDict()
        self.hits = self.misses = 0

    def _encode(self, images: torch.Tensor) -> List[torch.Tensor]:
        with torch.no_grad():
            if self.amp_dtype is not None:
                with torch.autocast("cuda", dtype=self.amp_dtype):
                    feats = encoder_pyramid(self.encoder, images)
            else:
                feats = encoder_pyramid(self.encoder, images)
        return [f.half() for f in feats]

    def get(self, images: torch.Tensor, keys: Sequence[str]) -> List[torch.Tensor]:
        """Pyramid for each image, encoding only the frames not already held."""
        missing = [i for i, k in enumerate(keys) if k not in self._cache]
        # A frame can repeat inside one batch
        uniq: Dict[str, int] = {}
        for i in missing:
            uniq.setdefault(keys[i], i)
        if uniq:
            idx = list(uniq.values())
            feats = self._encode(images[idx])
            for n, k in enumerate(uniq):
                self._cache[k] = [f[n : n + 1] for f in feats]
            protected = set(keys)
            while len(self._cache) > self.capacity:
                victim = next((k for k in self._cache if k not in protected), None)
                if victim is None:
                    break
                del self._cache[victim]
        self.hits += len(keys) - len(uniq)
        self.misses += len(uniq)
        n_scales = len(next(iter(self._cache.values())))
        out = []
        for s in range(n_scales):
            out.append(torch.cat([self._cache[k][s] for k in keys], 0).float())
        for k in keys:  # mark as recently used
            self._cache.move_to_end(k)
        return out

    def encode(self, images: torch.Tensor) -> List[torch.Tensor]:
        """Encode ``images`` without touching the cache."""
        return [f.float() for f in self._encode(images)]

    @property
    def hit_rate(self) -> float:
        n = self.hits + self.misses
        return self.hits / n if n else 0.0


# --------------------------------------------------------------------------- #
# Frozen networks
def build_segmenter(device: str, checkpoint: Optional[Path] = None):
    """The frozen choroid segmentation module: its encoder feeds the regressor, and (with ``--w-bm``) its decoder provides the BM curve the loss aligns.

    ``checkpoint``: a Lightning ``.ckpt`` from ``scripts/segmentation/train.py``
    (e.g. the sweep winner); default the Hub model pinned by ``SEGMENTATION_REVISION``.
    """
    if checkpoint is None:
        from ocularrigidity.segmentation.utils import get_model

        # get_model loads onto `device` directly
        seg = get_model(device).eval()
    else:
        from ocularrigidity.segmentation.trainer.pl_module import (
            ChoroidSegmentationModule,
        )

        # The checkpoint holds the weights: no ImageNet download
        seg = ChoroidSegmentationModule.load_from_checkpoint(
            str(checkpoint), map_location=device, encoder_weights=None
        ).eval()
        if float(next(seg.parameters()).detach().abs().max()) == 0.0:
            raise RuntimeError(f"segmentation weights in {checkpoint} are all zero")
        hp = seg.hparams
        print(
            f"segmentation model: {checkpoint} "
            f"({hp.get('arch')} / {hp.get('encoder_name')})"
        )
    for p in seg.parameters():
        p.requires_grad_(False)
    return seg


# --------------------------------------------------------------------------- #
# Train / eval
def _forever(loader):
    """Endlessly re-iterate ``loader``, reshuffling as its sampler dictates."""
    while True:
        yield from loader


def run_epoch(
    model,
    loader,
    criterion,
    cache,
    device,
    optimizer=None,
    grad_clip: float = 1.0,
    baseline: bool = False,
    triplet_loader=None,
    w_cycle: float = 0.0,
    shift_loader=None,
    w_shift: float = 0.0,
    shift_beta: float = 8.0,
):
    """One pass."""
    train = optimizer is not None
    model.train(train and not baseline)
    sums = _new_sums()
    triplet_iter = _iter_or_none(triplet_loader)
    shift_iter = _iter_or_none(shift_loader)

    for batch in tqdm(
        loader,
        desc="train" if train else ("baseline" if baseline else "val"),
        leave=False,
    ):
        loss, comp, out = forward_step(
            model,
            batch,
            criterion,
            cache,
            device,
            grad=train and not baseline,
            baseline=baseline,
            triplet_iter=triplet_iter,
            w_cycle=w_cycle,
            shift_iter=shift_iter,
            w_shift=w_shift,
            shift_beta=shift_beta,
        )
        if train and not baseline:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        _accumulate(sums, comp, *out)

    return _finalize(sums)


def _iter_or_none(loader):
    return _forever(loader) if loader is not None and len(loader) > 0 else None


def forward_step(
    model,
    batch,
    criterion,
    cache,
    device,
    *,
    grad: bool,
    baseline: bool = False,
    triplet_iter=None,
    w_cycle: float = 0.0,
    shift_iter=None,
    w_shift: float = 0.0,
    shift_beta: float = 8.0,
):
    """The loss of one pair batch, plus its cycle-triplet and shift-equivariance
    batches drawn from their iterators. Shared by ``run_epoch`` (one GPU) and
    ``RegistrationModule`` (Lightning DDP). Returns ``(loss, components,
    (dx, dy, dx_gt, dy_gt, has_gt, batch size))``."""
    fixed, moving, dx_gt, dy_gt, has_gt, key_f, key_m, *lines = batch
    bm_lines = [t.to(device, non_blocking=True) for t in lines] if lines else None
    fixed = fixed.to(device, non_blocking=True)
    moving = moving.to(device, non_blocking=True)
    dx_gt = dx_gt.to(device)
    dy_gt = dy_gt.to(device)
    bs = fixed.shape[0]

    feats = cache.get(torch.cat([fixed, moving], 0), list(key_f) + list(key_m))
    fixed_feats = [f[:bs] for f in feats]
    moving_feats = [f[bs:] for f in feats]
    shape = tuple(fixed.shape[-2:])

    with torch.set_grad_enabled(grad):
        if baseline:
            dx, dy = dx_gt, dy_gt
        else:
            dx, dy = model(fixed_feats, moving_feats, img_shape=shape)
        loss, comp = criterion(
            dx,
            dy,
            fixed_img=fixed,
            moving_img=moving,
            fixed_feats=fixed_feats,
            moving_feats=moving_feats,
            bm_lines=bm_lines,
        )

        # --- dx auxiliaries. Neither uses the classical estimate: the
        if not baseline and triplet_iter is not None:
            t_fixed, t_moving, t_key_f, t_key_m = next(triplet_iter)
            t_fixed = t_fixed.to(device, non_blocking=True)
            t_moving = t_moving.to(device, non_blocking=True)
            t_bs = t_fixed.shape[0]
            t_feats = cache.get(
                torch.cat([t_fixed, t_moving], 0), list(t_key_f) + list(t_key_m)
            )
            t_fixed_feats = [f[:t_bs] for f in t_feats]
            t_moving_feats = [f[t_bs:] for f in t_feats]
            t_dx, _ = model(
                t_fixed_feats, t_moving_feats,
                img_shape=tuple(t_fixed.shape[-2:]), detach_dy=True,
            )
            dx_ij, dx_jk, dx_ik = t_dx.view(-1, 3).unbind(dim=1)
            l_cycle = cycle_consistency_loss(dx_ij, dx_jk, dx_ik)
            loss = loss + w_cycle * l_cycle
            comp["cycle"] = float(l_cycle.detach())

        if not baseline and shift_iter is not None:
            s_fixed, s_ref, s_shifted, s_delta, s_key = next(shift_iter)
            s_fixed = s_fixed.to(device, non_blocking=True)
            s_ref = s_ref.to(device, non_blocking=True)
            s_shifted = s_shifted.to(device, non_blocking=True)
            s_delta = s_delta.to(device)
            s_bs = s_fixed.shape[0]
            # fixed is a real frame -> cacheable
            s_fixed_feats = cache.get(s_fixed, list(s_key))
            s_var_feats = cache.encode(torch.cat([s_ref, s_shifted], 0))
            s_pair_fixed = [torch.cat([f, f], 0) for f in s_fixed_feats]
            s_dx, _ = model(
                s_pair_fixed, s_var_feats,
                img_shape=tuple(s_fixed.shape[-2:]), detach_dy=True,
            )
            l_shift = shift_equivariance_loss(
                s_dx[:s_bs], s_dx[s_bs:], s_delta, beta=float(shift_beta)
            )
            loss = loss + w_shift * l_shift
            comp["shift"] = float(l_shift.detach())

        if not baseline and (triplet_iter is not None or shift_iter is not None):
            comp["core"] = float(loss.detach()) - (
                w_cycle * comp.get("cycle", 0.0)
                + w_shift * comp.get("shift", 0.0)
            )
            comp["total"] = float(loss.detach())  # keep in sync with backward

    return loss, comp, (dx, dy, dx_gt, dy_gt, has_gt, bs)


def _new_sums() -> Dict[str, float]:
    # Fixed keys, so that the sums of every DDP rank line up
    return defaultdict(float, {"dx_mae": 0.0, "dy_mae": 0.0, "_n": 0.0, "_n_gt": 0.0})


def _accumulate(sums, comp, dx, dy, dx_gt, dy_gt, has_gt, bs) -> None:
    with torch.no_grad():
        for k, v in comp.items():
            sums[k] += v * bs
        # Agreement with the classical estimator, on the pairs it covers.
        w = has_gt.to(dx.device)
        if float(w.sum()) > 0:
            sums["dx_mae"] += float(((dx - dx_gt).abs() * w).sum())
            sums["dy_mae"] += float(((dy - dy_gt).abs().mean(dim=1) * w).sum())
            sums["_n_gt"] += float(w.sum())
    sums["_n"] += bs


def _finalize(sums) -> Dict[str, float]:
    n, n_gt = max(sums["_n"], 1.0), sums["_n_gt"]
    out = {k: v / n for k, v in sums.items() if not k.startswith("_")}
    for k in ("dx_mae", "dy_mae"):
        out[k] = sums[k] / n_gt if n_gt else float("nan")
    return out


# : Everything that defines *which* objective is being optimised.
OBJECTIVE_KEYS = (
    "w_feature",
    "w_photometric",
    "w_lateral",
    "w_bm",
    "w_smooth",
    "w_curvature",
    "w_coverage",
    "w_cycle",
    "w_shift",
    "bm_scale",
)


def _build_criterion(spec: Dict[str, float], n_scales: int, args, segmenter):
    """A loss module for one objective spec."""
    return UnsupervisedRegistrationLoss(
        n_scales=n_scales,
        w_feature=spec["w_feature"],
        w_photometric=spec["w_photometric"],
        w_lateral=spec["w_lateral"],
        w_bm=spec["w_bm"],
        w_smooth=spec["w_smooth"],
        w_curvature=spec["w_curvature"],
        w_coverage=spec["w_coverage"],
        photometric_levels=args.photometric_levels,
        photometric_window=args.photometric_window,
        coarse_to_fine=not args.no_coarse_to_fine,
        segmenter=segmenter if spec["w_bm"] > 0 else None,
        bm_scale=spec["bm_scale"],
        bm_on_features=args.bm_on_features,
    )


def _init_wandb(args, train_obj, eval_obj, n_train_vid, n_val_vid, n_train, n_val):
    """Start a W&B run, or return ``None`` when --wandb-project was not given."""
    if not args.wandb_project:
        return None
    import wandb

    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    cfg.update(
        train_objective=train_obj,
        eval_objective=eval_obj,
        cascade=not args.no_cascade,
        use_correlation=not args.no_correlation,
        n_train_videos=n_train_vid,
        n_val_videos=n_val_vid,
        n_train_pairs=n_train,
        n_val_pairs=n_val,
    )
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        group=args.wandb_group,
        tags=args.wandb_tags,
        mode=args.wandb_mode,
        config=cfg,
    )


def _fmt(tag: str, m: Dict[str, float]) -> str:
    return (
        f"{tag} loss {m['total']:.4f}"
        + (f" (core {m['core']:.4f})" if "core" in m else "")
        + " "
        f"(feat {m.get('feat', float('nan')):.4f} photo {m.get('photo', float('nan')):.4f}"
        + (f" lat {m['lat']:.4f}" if "lat" in m else "")
        + (f" bm {m['bm_px']:.2f}px" if "bm_px" in m else "")
        + (f" cycle {m['cycle']:.3f}px" if "cycle" in m else "")
        + (f" shift {m['shift']:.3f}px" if "shift" in m else "")
        + f") | vs classical: dx {m['dx_mae']:.2f}px dy {m['dy_mae']:.2f}px"
    )


def _finish_run(run, args, best, best_epoch, train_obj, eval_obj) -> None:
    """Final W&B summary (and optional artifact) of a training run."""
    ckpt_path = args.out / "best.pt"
    print(f"done. best val objective: {best:.4f}  ->  {ckpt_path}")
    if run is None:
        return
    run.summary["best/val_total"] = best
    run.summary["best/epoch"] = best_epoch
    run.summary["checkpoint"] = str(ckpt_path.resolve())
    if args.wandb_checkpoint and ckpt_path.exists():
        import wandb

        art = wandb.Artifact(
            f"regressor-{run.name}".replace("/", "-"),
            type="model",
            metadata={
                "best_epoch": best_epoch,
                "val_total": best,
                "train_objective": train_obj,
                "eval_objective": eval_obj,
                "cascade": not args.no_cascade,
                "use_correlation": not args.no_correlation,
            },
        )
        art.add_file(str(ckpt_path))
        run.log_artifact(art)
    run.finish()


# --------------------------------------------------------------------------- #
# Multi-GPU (Lightning DDP)
import pytorch_lightning as pl  # noqa: E402


class RegistrationModule(pl.LightningModule):
    """The regressor's training under Lightning, for ``--devices`` > 1 (DDP).

    Lightning launches one process per GPU, all-reduces the gradients, clips
    them, steps the optimizer and the per-epoch cosine schedule. A step is
    ``forward_step``, the same as the single-GPU loop: a pair batch plus a
    cycle-triplet and a shift batch. Each rank trains on its own share of the
    videos (``VideoChunkBatchSampler``), with its own feature cache; epoch
    metrics are summed over ranks. ``best.pt`` keeps the single-GPU format, and
    ``<out>/lightning/last.ckpt`` (every epoch) resumes a run with ``--resume``.
    """

    def __init__(
        self, model, seg, criterion, eval_criterion, args, train_obj, eval_obj,
        datasets, loader_kw, meta, run=None,
    ):
        super().__init__()
        self.model = model
        self.seg = seg  # frozen; registered so Lightning puts it on the device
        # Plain dict: not registered, so the segmenter they hold is saved once
        self._criteria = {"train": criterion, "eval": eval_criterion}
        self.cfg = args
        self.train_obj, self.eval_obj = train_obj, eval_obj
        self.datasets = datasets
        self.loader_kw = loader_kw
        self.meta = meta
        self.run = run
        self.best, self.best_epoch = float("inf"), -1
        self._va = None
        self._iters: Dict[str, tuple] = {}
        self._sums: Dict[str, Dict[str, float]] = {}

    def train(self, mode: bool = True):
        super().train(mode)
        self.seg.eval()  # frozen network: never in training mode
        return self

    # --- data ---------------------------------------------------------------
    def _loader(self, ds, batch, *, shuffle, collate=None, equalize=False):
        if ds is None or not len(ds):
            return None
        sampler = VideoChunkBatchSampler(
            ds.samples,
            batch,
            self.cfg.videos_in_flight,
            shuffle=shuffle,
            seed=self.cfg.seed,
            rank=self.global_rank,
            world_size=self.trainer.world_size,
            # Training: same number of steps on every rank (gradient all-reduce);
            # validation and the auxiliary streams need not be
            equalize=equalize,
        )
        kw = dict(self.loader_kw)
        if collate is not None:
            kw["collate_fn"] = collate
        return DataLoader(ds, batch_sampler=sampler, **kw)

    def train_dataloader(self):
        return self._loader(
            self.datasets["train"], self.cfg.batch_size, shuffle=True, equalize=True
        )

    def val_dataloader(self):
        loader = self._loader(self.datasets["val"], self.cfg.batch_size, shuffle=False)
        return loader if loader is not None else []

    def on_fit_start(self):
        a = self.cfg
        amp = None if a.encoder_dtype == "fp32" else torch.bfloat16
        self.cache = FeatureCache(self.seg.model.encoder, capacity=a.feature_cache, amp_dtype=amp)
        for c in self._criteria.values():
            c.to(self.device)
        self._aux = {}
        for split in ("train", "val"):
            self._aux[split] = (
                self._loader(
                    self.datasets[f"{split}_triplet"], a.triplet_batch_size,
                    shuffle=split == "train", collate=collate_triplets,
                ),
                self._loader(
                    self.datasets[f"{split}_shift"], a.shift_batch_size,
                    shuffle=split == "train",
                ),
            )

    # --- steps --------------------------------------------------------------
    def _step(self, batch, split: str):
        obj = self.train_obj if split == "train" else self.eval_obj
        loss, comp, out = forward_step(
            self.model,
            batch,
            self._criteria["train" if split == "train" else "eval"],
            self.cache,
            self.device,
            grad=split == "train",
            triplet_iter=self._iters[split][0],
            w_cycle=obj["w_cycle"],
            shift_iter=self._iters[split][1],
            w_shift=obj["w_shift"],
            shift_beta=1.0,
        )
        _accumulate(self._sums[split], comp, *out)
        if split == "train":
            _accumulate(self._window, comp, *out)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def _epoch_float(self) -> float:
        per_epoch = max(1, int(self.trainer.num_training_batches))
        return self.trainer.global_step / per_epoch

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Running training losses of rank 0 every --log-every-steps steps."""
        every = max(1, int(self.cfg.log_every_steps))
        step = self.trainer.global_step
        if step % every or not self.trainer.is_global_zero or self.run is None:
            return
        now = time.time()
        rate = (step - self._win_step) / max(now - self._win_t, 1e-6)
        log = {f"step/train/{k}": v for k, v in _finalize(self._window).items()
               if not k.endswith("_mae")}
        log.update({
            "step/lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            "step/steps_per_s": rate,
            "step/cache_hit_rate": self.cache.hit_rate,
            "epoch": self._epoch_float(),
        })
        self.run.log(log, step=step)
        self._window, self._win_t, self._win_step = _new_sums(), now, step

    def _start(self, split: str):
        """Fresh auxiliary iterators and sums for one split. Train and val keep
        their own, so a mid-epoch validation leaves the training state alone."""
        t, s = self._aux[split]
        self._iters[split] = (_iter_or_none(t), _iter_or_none(s))
        self._sums[split] = _new_sums()

    def on_train_epoch_start(self):
        e = self.current_epoch
        # Coarse-to-fine over the first half of training, all scales after.
        self._criteria["train"].set_progress(min(1.0, 2.0 * (e + 1) / self.cfg.epochs))
        for key in ("train", "train_triplet", "train_shift"):
            if self.datasets[key] is not None:
                self.datasets[key].set_epoch(e)
        for loader in self._aux["train"]:
            if loader is not None:
                loader.batch_sampler.set_epoch(e)
        self._lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self._va = None
        self._start("train")
        self._window, self._win_t, self._win_step = _new_sums(), time.time(), self.trainer.global_step

    def on_validation_epoch_start(self):
        self._start("val")

    def on_validation_epoch_end(self):
        """After every validation (mid-epoch with --val-every-steps, and at the
        end of each epoch): W&B log, and best.pt when the val objective improves."""
        va = self._reduce(self._sums["val"])
        self._va = va
        self._save_if_best(va["total"])
        if not self.trainer.is_global_zero:
            return
        step, epoch = self.trainer.global_step, self._epoch_float()
        print(f"[epoch {epoch:.2f} | step {step}] " + _fmt("val", va), flush=True)
        if self.run is not None:
            log: Dict[str, Any] = {f"val/{k}": v for k, v in va.items()}
            log.update({"epoch": epoch, "best/val_total": self.best})
            self.run.log(log, step=step)

    def _save(self, name: str, score: float) -> None:
        """Weights in the single-GPU ``best.pt`` format (rank 0 only)."""
        if self.trainer.is_global_zero:
            torch.save(
                {
                    **self.meta,
                    "epoch": self.current_epoch,
                    "global_step": self.trainer.global_step,
                    "model": self.model.state_dict(),
                    "val_loss": score,
                },
                self.cfg.out / name,
            )

    def _save_if_best(self, score: float) -> None:
        # last.pt always: the val objective may not track generalisation
        self._save("last.pt", score)
        if score >= self.best:
            return
        self.best, self.best_epoch = score, self.current_epoch
        self._save("best.pt", score)

    def _reduce(self, sums) -> Dict[str, float]:
        """Sum the per-rank sums, then average as the single-GPU loop does."""
        strategy = self.trainer.strategy
        keys = strategy.broadcast(sorted(sums), src=0)
        t = torch.tensor(
            [float(sums.get(k, 0.0)) for k in keys], dtype=torch.float64, device=self.device
        )
        t = strategy.reduce(t, reduce_op="sum")
        return _finalize(defaultdict(float, zip(keys, t.tolist())))

    def on_train_epoch_end(self):
        """Whole-epoch training metrics (all ranks); validation is logged by
        on_validation_epoch_end, which Lightning runs before this hook."""
        epoch = self.current_epoch
        tr = self._reduce(self._sums["train"])
        if self._va is None:  # no validation set: rank on the training loss
            self._save_if_best(tr["total"])
        if not self.trainer.is_global_zero:
            return
        print(f"[{epoch:03d}] " + _fmt("train", tr)
              + f"  [cache {self.cache.hit_rate:.0%} on rank 0]", flush=True)
        if self.run is not None:
            log: Dict[str, Any] = {f"train/{k}": v for k, v in tr.items()}
            log["lr"] = self._lr
            log["cache_hit_rate"] = self.cache.hit_rate
            log["progress"] = self._criteria["train"]._progress
            log["epoch"] = epoch + 1
            self.run.log(log, step=self.trainer.global_step)

    # --- optimisation and resuming -------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg.epochs)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def on_save_checkpoint(self, checkpoint):
        checkpoint["registration_best"] = (self.best, self.best_epoch)

    def on_load_checkpoint(self, checkpoint):
        self.best, self.best_epoch = checkpoint.get("registration_best", (float("inf"), -1))


def fit_lightning(module: RegistrationModule, args) -> None:
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.strategies import DDPStrategy

    trainer = pl.Trainer(
        accelerator="auto",
        devices=args.devices,
        # No buffer broadcast: the only buffers are the frozen segmenter's (the
        # regressor uses GroupNorm), and broadcasting them makes every forward a
        # collective, which hangs when ranks have different validation lengths
        strategy=DDPStrategy(broadcast_buffers=False),
        max_epochs=args.epochs,
        gradient_clip_val=1.0,  # as the single-GPU loop
        gradient_clip_algorithm="norm",
        precision="32-true",
        logger=False,
        # Resumable state: every --val-every-steps steps (else every epoch)
        callbacks=[
            ModelCheckpoint(
                dirpath=args.out / "lightning",
                save_last=True,
                save_top_k=0,
                every_n_train_steps=args.val_every_steps or None,
            )
        ],
        use_distributed_sampler=False,  # VideoChunkBatchSampler shards by rank itself
        num_sanity_val_steps=0,
        enable_model_summary=False,
        val_check_interval=args.val_every_steps if args.val_every_steps > 0 else 1.0,
    )
    trainer.fit(module, ckpt_path=str(args.resume) if args.resume else None)
    if trainer.is_global_zero:
        _finish_run(module.run, args, module.best, module.best_epoch, module.train_obj, module.eval_obj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/home/clement/Documents/data/OcularRigidityAnnotated/dataset/Registration/"
        ),
        help="frames from dataset/prepare_unsupervised_data.py",
    )
    ap.add_argument(
        "--segmentation-checkpoint",
        type=Path,
        default=None,
        help="Lightning .ckpt of the segmentation model whose encoder the "
        "regressor sits on (and whose masks give the BM term); default the Hub "
        "model pinned by SEGMENTATION_REVISION. Any smp encoder works.",
    )
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--embed", type=int, default=128)

    # --- pairing -----------------------------------------------------------
    ap.add_argument(
        "--pair-mode",
        choices=("within", "reference"),
        default="within",
        help="'within': every ordered pair of frames of a volume (no ground "
        "truth needed); 'reference': only (reference, frame) pairs.",
    )
    ap.add_argument("--pairs-per-video", type=int, default=200)
    ap.add_argument(
        "--crop",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help="train on a HxW crop instead of the full frame, e.g. --crop 768 "
        "768. One window per video, centred on that video's retina (a fixed "
        "window will not do: the choroid sits anywhere from row ~250 to ~1080 "
        "across the cohort). The window is shared by every frame of the video "
        "so the FeatureCache keeps working, and is identical for fixed and "
        "moving so no artificial dy is introduced. Inference is unaffected -- "
        "the checkpoint records the full frame shape.",
    )

    # --- cycle consistency ---------------------------------------------------
    ap.add_argument(
        "--w-cycle",
        type=float,
        default=1.0,
        help="weight of the dx cycle-consistency term, (dx_ij + dx_jk) - "
        "dx_ik over closed frame triplets of one video: a lateral shift is a "
        "pure translation, so it composes exactly regardless of content, "
        "which makes this a self-consistency check on the model's own "
        "output rather than a fit to any label. Targets dx specifically -- "
        "it is the quantity direct optimisation leaves weakly constrained "
        "(see --w-lateral) and where the cascade has been seen to lock onto "
        "one of two nearby candidates and hold it for long runs of frames. "
        "0 disables it and skips building the triplet loader.",
    )
    ap.add_argument("--triplets-per-video", type=int, default=40)
    ap.add_argument(
        "--w-shift",
        type=float,
        default=1.0,
        help="weight of the shift-equivariance term: a copy of the moving "
        "frame is displaced by a known whole-column delta and dx must track "
        "it one-for-one (dx_ref - dx_shifted == delta). The only exact signal "
        "for dx that involves no estimate of the truth -- we apply the shift "
        "ourselves. Unlike --w-cycle it has no degenerate solution, so it is "
        "what stops the dx head collapsing to a constant; 0 disables it.",
    )
    ap.add_argument(
        "--shift-max",
        type=int,
        default=8,
        help="range of the synthetic shift, in columns: delta is drawn from "
        "the nonzero integers in [-shift_max, shift_max]. Also the margin "
        "blanked at both edges of every variant so the shifted-in strip "
        "cannot leak delta. This is an augmentation range, not a bound on "
        "what the model may predict.",
    )
    ap.add_argument("--shifts-per-video", type=int, default=40)
    ap.add_argument(
        "--shift-batch-size",
        type=int,
        default=2,
        help="pairs per shift-equivariance step. Each costs two *fresh* "
        "encoder passes (the shifted variants are synthetic and cannot be "
        "cached), so this is the knob that sets the term's cost.",
    )
    ap.add_argument(
        "--triplet-batch-size",
        type=int,
        default=2,
        help="closed triplets per cycle-consistency step (3x this many head "
        "passes; the frozen encoder pass is still shared with the pair batch "
        "through the same FeatureCache).",
    )

    # --- objective ---------------------------------------------------------
    ap.add_argument("--w-feature", type=float, default=1.0)
    ap.add_argument("--w-photometric", type=float, default=1.0)
    ap.add_argument(
        "--w-lateral",
        type=float,
        default=1.0,
        help="weight of the vertical-mean-profile term, which is what "
        "makes dx worth learning; 0 reproduces the run where dx collapsed "
        "to a constant.",
    )
    ap.add_argument(
        "--w-bm",
        type=float,
        default=0.1,
        help="weight of the soft-BM alignment term (the cohort QC metric). "
        "Costs a segmentation decoder pass per step, with grad. It is the "
        "most discriminative of the three terms by two orders of magnitude; "
        "0 disables it.",
    )
    ap.add_argument("--w-smooth", type=float, default=0.05)
    ap.add_argument("--w-curvature", type=float, default=0.02)
    ap.add_argument("--w-coverage", type=float, default=0.5)
    ap.add_argument(
        "--bm-on-features",
        action="store_true",
        help="warp the frozen pyramid and run the decoder only, instead of "
        "warping the frame and re-encoding it. Skips an encoder pass, but "
        "measured to give a 3-7x weaker dy gradient with no growth away from "
        "the optimum, so dy converges worse. Off by default; useful only when "
        "bm_px is wanted as a cheap measurement rather than as a driver.",
    )
    ap.add_argument(
        "--bm-lines",
        type=Path,
        default=None,
        help="directory of precomputed BM lines (dataset/precompute_bm_lines.py, "
        "same --segmentation-checkpoint): the BM term compares them directly, "
        "with no segmentation pass per step; validation also reports the "
        "re-segmented error as bm_px_reseg",
    )
    ap.add_argument(
        "--bm-scale",
        type=float,
        default=1.0,
        help="segment a downscaled pair for the BM term (0.5 cuts its memory "
        "fourfold); bm_px is still reported in full-resolution pixels",
    )
    ap.add_argument("--photometric-levels", type=int, default=3)
    ap.add_argument("--photometric-window", type=int, default=9)
    ap.add_argument("--no-coarse-to-fine", action="store_true")

    # --- model -------------------------------------------------------------
    ap.add_argument(
        "--no-cascade",
        action="store_true",
        help="single-shot fusion instead of coarse-to-fine refinement",
    )
    ap.add_argument(
        "--no-correlation",
        action="store_true",
        help="drop the per-scale correlation volume (concat-only fusion); "
        "only meaningful with --no-cascade",
    )
    ap.add_argument("--corr-dy-radius", type=int, default=6)
    ap.add_argument(
        "--dx-step",
        type=float,
        default=4.0,
        help="pixels per unit of the cascade's lateral output. Sets how far "
        "one step of the dx head moves the answer, so it is the conditioning "
        "knob for dx's gain: raise it if --w-shift reports the head tracking "
        "only a fraction of the applied displacement.",
    )
    ap.add_argument("--corr-dx-radius", type=int, default=10)
    ap.add_argument("--scale-dx", type=float, default=16.0)
    ap.add_argument("--scale-dy-bulk", type=float, default=128.0)
    ap.add_argument("--scale-dy-residual", type=float, default=8.0)

    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument(
        "--image-cache-gb",
        type=float,
        default=16.0,
        help="RAM for decoded frames (uint8, shared by all datasets, least "
        "recently used evicted). Only with --num-workers 0; 0 disables it. A "
        "1536x1024 frame takes 1.5 MB.",
    )
    ap.add_argument(
        "--videos-in-flight",
        type=int,
        default=4,
        help="how many videos a batch may be drawn from. Larger keeps batches "
        "more heterogeneous, smaller makes the feature cache hit more often -- "
        "and that is the dominant cost: 4 gives 69%% hits at 778 ms/step, "
        "2 gives 87%% at 607 ms/step, against 1195 ms with no cache.",
    )
    ap.add_argument(
        "--feature-cache",
        type=int,
        default=192,
        help="frames of encoder pyramid to keep (float16, ~24 MB each)",
    )
    ap.add_argument(
        "--encoder-dtype",
        choices=("bf16", "fp32"),
        default="bf16",
        help="precision of the frozen encoder forward. Measured worth only "
        "~5%% once the feature cache is on (815 -> 778 ms/step): the cache "
        "already removes most encoder calls, so there is little left to speed up.",
    )
    ap.add_argument(
        "--eval-correlation",
        action="store_true",
        help="score the training-free correlation-peak estimator on the "
        "validation set, under the same objective, and exit. The zero-parameter "
        "null the ablation arms cannot provide: they all share this backbone, "
        "so they only compare learned against learned.",
    )
    ap.add_argument(
        "--eval-correlation-scales",
        type=int,
        nargs="*",
        default=None,
        help="pyramid levels for --eval-correlation (default: all, coarse-to-"
        "fine). A single level, e.g. 3, is the same estimator without the "
        "coarse-to-fine refinement.",
    )
    ap.add_argument(
        "--eval-baseline",
        action="store_true",
        help="also score the classical transform under the same objective",
    )
    ap.add_argument("--out", type=Path, default=Path("checkpoints"))
    ap.add_argument(
        "--devices",
        type=int,
        default=1,
        help="GPUs. 1: the single-GPU loop; >1: Lightning DDP, each GPU taking "
        "--batch-size pairs per step (effective batch = devices x batch-size)",
    )
    ap.add_argument(
        "--val-every-steps",
        type=int,
        default=0,
        help="--devices > 1 only: also validate every N training steps (logged "
        "to W&B, best.pt saved on improvement); 0 = at the end of each epoch only",
    )
    ap.add_argument(
        "--log-every-steps",
        type=int,
        default=50,
        help="--devices > 1 only: running training losses to W&B every N steps",
    )
    ap.add_argument(
        "--val-pairs-per-video",
        type=int,
        default=None,
        help="pairs per validation video (default: --pairs-per-video); fewer "
        "makes frequent validation affordable",
    )
    ap.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="--devices > 1 only: a <out>/lightning/last.ckpt to continue from",
    )
    ap.add_argument("--dry-run", action="store_true")

    # --- comparable evaluation ---------------------------------------------
    ap.add_argument(
        "--eval-objective",
        type=str,
        default=None,
        metavar="JSON",
        help="score validation under a *different* objective than the one "
        "being trained, given as a JSON object of "
        f"{{{', '.join(OBJECTIVE_KEYS)}}} overrides. This is what makes an "
        "ablation readable: dropping a term changes the training loss, so "
        "`val/total` of an ablated arm is not on the same scale as the "
        "reference's and ranking arms by it compares nothing. Pass the "
        "reference objective here in every arm and every run is scored on "
        "one fixed ruler -- including checkpoint selection, which otherwise "
        "means a different thing in each arm. Validation-side triplet/shift "
        "loaders follow this spec, the training-side ones follow the "
        "training weights, so an ablated term still costs nothing to train.",
    )

    # --- logging ------------------------------------------------------------
    ap.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="log to this Weights & Biases project; omitted = no logging",
    )
    ap.add_argument("--wandb-entity", type=str, default=None)
    ap.add_argument("--wandb-name", type=str, default=None, help="run name")
    ap.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="group runs of one sweep together (the ablation runner sets this)",
    )
    ap.add_argument("--wandb-tags", type=str, nargs="*", default=None)
    ap.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    ap.add_argument(
        "--wandb-checkpoint",
        action="store_true",
        help="upload the best checkpoint as a W&B artifact at the end of the "
        "run. It is written to --out regardless.",
    )
    args = ap.parse_args()

    distributed = args.devices > 1
    if distributed and (args.eval_baseline or args.eval_correlation):
        raise SystemExit("--eval-baseline / --eval-correlation run on one GPU: drop --devices")
    if args.resume and not distributed:
        raise SystemExit("--resume needs --devices > 1")
    if args.bm_lines is not None:
        meta_path = args.bm_lines / "meta.json"
        if not meta_path.exists():
            raise SystemExit(f"--bm-lines {args.bm_lines}: no meta.json")
        made_with = json.loads(meta_path.read_text())["segmentation_checkpoint"]
        expected = (
            str(args.segmentation_checkpoint.resolve()) if args.segmentation_checkpoint else "hub"
        )
        if made_with != expected:
            raise SystemExit(
                f"--bm-lines were made with {made_with}, not --segmentation-checkpoint {expected}"
            )
        if args.bm_on_features:
            raise SystemExit("--bm-lines and --bm-on-features are two BM modes: pick one")
        if args.bm_scale != 1.0:
            print("--bm-lines: the lines are full resolution, --bm-scale is ignored")
    # Lightning re-runs this script in one process per GPU (LOCAL_RANK set in
    # the children): only the first one prints and logs to W&B
    is_rank0 = os.environ.get("LOCAL_RANK", "0") == "0"
    log = print if is_rank0 else (lambda *a, **k: None)

    torch.manual_seed(args.seed)
    # Under DDP the networks are built on the CPU; Lightning moves them to each GPU
    device = "cuda" if torch.cuda.is_available() and not distributed else "cpu"

    # What we optimise, and what we score.
    train_obj = {k: float(getattr(args, k)) for k in OBJECTIVE_KEYS}
    eval_obj = dict(train_obj)
    if args.eval_objective:
        override = json.loads(args.eval_objective)
        unknown = set(override) - set(OBJECTIVE_KEYS)
        if unknown:
            raise SystemExit(
                f"--eval-objective: unknown key(s) {sorted(unknown)}; "
                f"expected a subset of {list(OBJECTIVE_KEYS)}"
            )
        eval_obj.update({k: float(v) for k, v in override.items()})
        log("train objective:", json.dumps(train_obj))
        log("eval  objective:", json.dumps(eval_obj), "(common ruler)")

    seg = build_segmenter(device, args.segmentation_checkpoint)
    encoder = seg.model.encoder

    crop = tuple(args.crop) if args.crop else None
    # Worker processes would each hold their own copy, so the cache only lives
    # in the main process (--num-workers 0).
    image_cache = (
        ImageCache(int(args.image_cache_gb * 2**30))
        if args.num_workers == 0 and args.image_cache_gb > 0
        else None
    )
    train_vids, val_vids = split_videos(args.root, args.val_frac, args.seed)
    ds_kw = dict(
        bm_lines=args.bm_lines,
        pair_mode=args.pair_mode,
        pairs_per_video=args.pairs_per_video,
        seed=args.seed,
        image_cache=image_cache,
        crop=crop,
    )
    train_ds = FramePairDataset(args.root, train_vids, **ds_kw)
    val_ds = FramePairDataset(
        args.root,
        val_vids,
        **{**ds_kw, "pairs_per_video": args.val_pairs_per_video or args.pairs_per_video},
    )

    triplet_kw = dict(
        triplets_per_video=args.triplets_per_video,
        seed=args.seed,
        image_cache=image_cache,
        crop=crop,
    )
    train_triplet_ds = (
        FrameTripletDataset(args.root, train_vids, **triplet_kw)
        if train_obj["w_cycle"] > 0
        else None
    )
    val_triplet_ds = (
        FrameTripletDataset(args.root, val_vids, **triplet_kw)
        if eval_obj["w_cycle"] > 0
        else None
    )

    shift_kw = dict(
        shifts_per_video=args.shifts_per_video,
        shift_max=args.shift_max,
        seed=args.seed,
        image_cache=image_cache,
        crop=crop,
    )
    train_shift_ds = (
        FrameShiftDataset(args.root, train_vids, **shift_kw)
        if train_obj["w_shift"] > 0
        else None
    )
    val_shift_ds = (
        FrameShiftDataset(args.root, val_vids, **shift_kw)
        if eval_obj["w_shift"] > 0
        else None
    )

    sample = train_ds[0]
    H, W = sample[0].shape[-2:]          # what the model is fed while training
    _probe = cv2.imread(str(next(iter(train_ds.frames.values()))[0][0]), cv2.IMREAD_GRAYSCALE)
    FULL_H, FULL_W = _probe.shape[:2]
    with torch.no_grad():
        in_channels = [
            f.shape[1]
            for f in encoder_pyramid(encoder, sample[0].unsqueeze(0).to(device))
        ]

    if crop:
        log(f"crop: {H}x{W} per video, retina-centred (full frame {FULL_H}x{FULL_W})")
    log(f"images {H}x{W} | encoder channels (fine->coarse): {in_channels}")
    log(f"videos: {len(train_vids)} train / {len(val_vids)} val")
    log(f"pairs:  {len(train_ds)} train / {len(val_ds)} val  [{args.pair_mode}]")
    def _n(ds) -> int:
        return len(ds) if ds is not None else 0

    if train_triplet_ds is not None or val_triplet_ds is not None:
        log(
            f"triplets: {_n(train_triplet_ds)} train / {_n(val_triplet_ds)} val "
            f"[cycle consistency, w_cycle={train_obj['w_cycle']} "
            f"train / {eval_obj['w_cycle']} eval]"
        )
    if train_shift_ds is not None or val_shift_ds is not None:
        log(
            f"shifts:   {_n(train_shift_ds)} train / {_n(val_shift_ds)} val "
            f"[equivariance, w_shift={train_obj['w_shift']} train / "
            f"{eval_obj['w_shift']} eval, delta=+-1..{args.shift_max}px]"
        )

    if args.dry_run:
        for k in range(min(3, len(train_ds))):
            item = train_ds[k]
            dx, dy, has_gt = item[2], item[3], item[4]
            log(
                f"  pair {k}: dx={dx.item():+.2f}px  "
                f"dy[min/mean/max]={dy.min():+.1f}/{dy.mean():+.1f}/{dy.max():+.1f}px  "
                f"gt={bool(has_gt)}"
            )
        return

    common = dict(
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0 and crop is None,
    )
    train_sampler = VideoChunkBatchSampler(
        train_ds.samples,
        args.batch_size,
        args.videos_in_flight,
        shuffle=True,
        seed=args.seed,
    )
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, **common)
    val_loader = (
        DataLoader(
            val_ds,
            batch_sampler=VideoChunkBatchSampler(
                val_ds.samples, args.batch_size, args.videos_in_flight, shuffle=False
            ),
            **common,
        )
        if len(val_ds)
        else None
    )

    def _aux_loader(ds, batch, *, shuffle, collate=None):
        """One auxiliary loader, plus its sampler so the epoch can be advanced."""
        if ds is None or not len(ds):
            return None, None
        sampler = VideoChunkBatchSampler(
            ds.samples,
            batch,
            args.videos_in_flight,
            shuffle=shuffle,
            seed=args.seed,
        )
        kw = dict(common)
        if collate is not None:
            kw["collate_fn"] = collate
        return DataLoader(ds, batch_sampler=sampler, **kw), sampler

    train_triplet_loader, train_triplet_sampler = _aux_loader(
        train_triplet_ds, args.triplet_batch_size, shuffle=True,
        collate=collate_triplets,
    )
    val_triplet_loader, _ = _aux_loader(
        val_triplet_ds, args.triplet_batch_size, shuffle=False,
        collate=collate_triplets,
    )
    train_shift_loader, train_shift_sampler = _aux_loader(
        train_shift_ds, args.shift_batch_size, shuffle=True
    )
    val_shift_loader, _ = _aux_loader(
        val_shift_ds, args.shift_batch_size, shuffle=False
    )

    amp_dtype = None if args.encoder_dtype == "fp32" else torch.bfloat16
    cache = FeatureCache(encoder, capacity=args.feature_cache, amp_dtype=amp_dtype)

    scales = (args.scale_dx, args.scale_dy_bulk, args.scale_dy_residual)
    model = RegistrationRegressor(
        in_channels=in_channels,
        embed=args.embed,
        img_shape=(H, W),
        cascade=not args.no_cascade,
        scales=scales,
        use_correlation=not args.no_correlation,
        corr_dy_radius=args.corr_dy_radius,
        corr_dx_radius=args.corr_dx_radius,
        dx_step=args.dx_step,
    ).to(device)

    # With precomputed lines, training needs no segmentation pass at all;
    # validation keeps the segmenter for the bm_px_reseg check
    criterion = _build_criterion(
        train_obj, len(in_channels), args, None if args.bm_lines else seg
    )
    eval_criterion = _build_criterion(eval_obj, len(in_channels), args, seg)
    eval_criterion.set_progress(1.0)

    run = (
        _init_wandb(args, train_obj, eval_obj, len(train_vids), len(val_vids),
                    len(train_ds), len(val_ds))
        if is_rank0
        else None
    )
    # What best.pt holds besides the weights (same format on one or many GPUs)
    meta = {
        "in_channels": in_channels,
        "segmentation_checkpoint": (
            str(args.segmentation_checkpoint) if args.segmentation_checkpoint else "hub"
        ),
        "out_width": W,
        "embed": args.embed,
        "img_shape": (FULL_H, FULL_W),
        "train_shape": (H, W),
        "cascade": not args.no_cascade,
        "scales": scales,
        "use_correlation": not args.no_correlation,
        "train_objective": train_obj,
        "eval_objective": eval_obj,
        "args": vars(args),
    }

    if distributed:
        args.out.mkdir(parents=True, exist_ok=True)
        module = RegistrationModule(
            model, seg, criterion, eval_criterion, args, train_obj, eval_obj,
            datasets={
                "train": train_ds,
                "val": val_ds,
                "train_triplet": train_triplet_ds,
                "val_triplet": val_triplet_ds,
                "train_shift": train_shift_ds,
                "val_shift": val_shift_ds,
            },
            loader_kw=common,
            meta=meta,
            run=run,
        )
        fit_lightning(module, args)
        return

    baseline_metrics = None
    if args.eval_baseline and val_loader is not None:
        baseline_metrics = run_epoch(
            model,
            val_loader,
            eval_criterion,
            cache,
            device,
            baseline=True,
        )
        log(_fmt("[baseline: classical transform] val", baseline_metrics))
        if run is not None:
            for k, v in baseline_metrics.items():
                run.summary[f"baseline/{k}"] = v

    if args.eval_correlation and val_loader is not None:
        # Scored exactly as an arm is: same val loader, same eval_criterion,
        # same cycle/shift batches
        corr_model = CorrelationBaseline(
            img_shape=(H, W), scales=args.eval_correlation_scales
        ).to(device)
        corr_metrics = run_epoch(
            corr_model,
            val_loader,
            eval_criterion,
            cache,
            device,
            triplet_loader=val_triplet_loader,
            w_cycle=eval_obj["w_cycle"],
            shift_loader=val_shift_loader,
            w_shift=eval_obj["w_shift"],
            shift_beta=1.0,
        )
        tag = (
            "all scales"
            if args.eval_correlation_scales is None
            else f"scales {args.eval_correlation_scales}"
        )
        log(_fmt(f"[correlation peak, no training: {tag}] val", corr_metrics))
        if run is not None:
            for k, v in corr_metrics.items():
                run.summary[f"correlation/{k}"] = v
            run.finish()
        return

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    args.out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    best_epoch = -1

    for epoch in tqdm(range(args.epochs), desc="epochs"):
        # Coarse-to-fine over the first half of training, all scales after.
        criterion.set_progress(min(1.0, 2.0 * (epoch + 1) / args.epochs))
        train_sampler.set_epoch(epoch)
        for _ds in (train_ds, train_triplet_ds, train_shift_ds):
            if _ds is not None:
                _ds.set_epoch(epoch)
        if train_triplet_sampler is not None:
            train_triplet_sampler.set_epoch(epoch)
        if train_shift_sampler is not None:
            train_shift_sampler.set_epoch(epoch)
        tr = run_epoch(
            model,
            train_loader,
            criterion,
            cache,
            device,
            optimizer=optimizer,
            triplet_loader=train_triplet_loader,
            w_cycle=train_obj["w_cycle"],
            shift_loader=train_shift_loader,
            w_shift=train_obj["w_shift"],
            shift_beta=1.0,
        )
        lr_now = scheduler.get_last_lr()[0]
        scheduler.step()
        msg = f"[{epoch:03d}] " + _fmt("train", tr)

        score = tr["total"]
        va = None
        if val_loader is not None:
            va = run_epoch(
                model,
                val_loader,
                eval_criterion,
                cache,
                device,
                triplet_loader=val_triplet_loader,
                w_cycle=eval_obj["w_cycle"],
                shift_loader=val_shift_loader,
                w_shift=eval_obj["w_shift"],
                shift_beta=1.0,
            )
            msg += "  ||  " + _fmt("val", va)
            score = va["total"]
        msg += f"  [cache {cache.hit_rate:.0%}]"
        tqdm.write(msg)

        if run is not None:
            log: Dict[str, Any] = {f"train/{k}": v for k, v in tr.items()}
            if va is not None:
                log.update({f"val/{k}": v for k, v in va.items()})
            log["lr"] = lr_now
            log["cache_hit_rate"] = cache.hit_rate
            log["progress"] = criterion._progress
            run.log(log, step=epoch)

        state = {**meta, "epoch": epoch, "model": model.state_dict(), "val_loss": score}
        # last.pt always: the val objective may not track generalisation
        torch.save(state, args.out / "last.pt")
        if score < best:
            best, best_epoch = score, epoch
            torch.save(state, args.out / "best.pt")

    _finish_run(run, args, best, best_epoch, train_obj, eval_obj)


if __name__ == "__main__":
    main()
