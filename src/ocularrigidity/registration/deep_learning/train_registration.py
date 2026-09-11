import argparse
import json
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
    """Recover {video} from a '{video}_{idx}' stem by dropping the trailing
    frame-index token. Correct even if {video} itself ends in '_<int>'."""
    return stem.rsplit("_", 1)[0]


def _read_image(path: Path) -> torch.Tensor:
    """Single-channel PNG -> [1, H, W] float32, with the encoder's normalisation
    ((x/255 - 0.5) / 0.5) — the same one ``prepare_data.py`` used."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    t = torch.from_numpy(img).to(torch.float32).unsqueeze(0)
    t.mul_(1.0 / 255.0).sub_(0.5).div_(0.5)
    return t


def _list_frames(root: Path) -> Dict[str, List[Tuple[Path, Optional[Path]]]]:
    """``video -> [(image_path, transform_path | None), ...]``, index 0 is the
    reference frame (whose transform is the identity by construction --
    ``prepare_data`` picks it as the frame with ``dx == 0``). Shared by every
    frame-sampling dataset so they agree on frame order and indexing."""
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
    """FeatureCache key. The crop window is part of the identity: the same
    frame under a different window is a different pyramid, and serving a stale
    one would be silent and wrong."""
    return f"{vid}#{idx}" if win is None else f"{vid}#{idx}@{win[0]}_{win[1]}"


def _retina_row(paths: Sequence[Path], n: int = 3) -> int:
    """Row to centre a crop on: the brightest band of the B-scan.

    A *fixed* window is not an option. Measured over the cohort the choroid
    centre sits anywhere from row ~250 to ~1080 depending on the acquisition,
    so one shared window holds the full mask for only about half the frames.
    Per video it is much tighter -- the worst video needed 720 rows to cover
    every frame -- which is what makes a 768-row crop safe once it is centred.
    """
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


def _video_centres(frames, crop):
    """``video -> retina row``, plus the full frame shape. Computed once: it
    reads a few frames per video, which is far too slow to redo every epoch."""
    if crop is None:
        return None, None
    centres = {}
    full = None
    for vid, fl in sorted(frames.items()):
        img = cv2.imread(str(fl[0][0]), cv2.IMREAD_GRAYSCALE)
        if full is None:
            full = img.shape[:2]
        centres[vid] = _retina_row([q for q, _ in fl])
    return centres, full


def _draw_windows(centres, full, crop, seed: int, jitter: int = 64):
    """``video -> (row0, col0)``, re-drawable per epoch.

    One window per video rather than per sample, because the FeatureCache is
    keyed by frame: a frame's pyramid is only reusable while its crop holds
    still, so a fresh window per pair would turn every hit into a miss. Redrawn
    each epoch instead, which gives the crop variety of random cropping at no
    cost to the cache -- over a run each video is seen through as many windows
    as there are epochs.

    Rows are jittered around the video's retina rather than drawn uniformly:
    the choroid sits anywhere from row ~250 to ~1080 across the cohort, so a
    uniform row offset would spend much of its time on empty vitreous.
    """
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
    """Crop ``(1, H, W)`` to ``crop`` at origin ``win``; identity if no crop."""
    if crop is None or win is None:
        return img
    ch, cw = crop
    r0, c0 = win
    return img[..., r0 : r0 + ch, c0 : c0 + cw]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class FramePairDataset(Dataset):
    """Ordered (fixed, moving) frame pairs drawn from within a single volume.

    With no target to match, the reference frame loses its special status: any
    frame can play fixed. ``pair_mode="within"`` therefore samples ordered pairs
    from all of a video's dumped frames, which turns N frames into N(N-1) samples
    instead of N-1. ``pair_mode="reference"`` keeps the old pairing (every frame
    against the volume's reference), which is what the diagnostic ground truth
    was computed for.

    Pairs are drawn once, from a seeded RNG, so an epoch is reproducible and the
    validation set is fixed.
    """

    def __init__(
        self,
        root: Path,
        videos: Optional[Set[str]] = None,
        *,
        pair_mode: str = "within",
        pairs_per_video: int = 200,
        seed: int = 0,
        cache_images: bool = True,
        crop: Optional[Tuple[int, int]] = None,
    ):
        assert pair_mode in ("within", "reference")
        root = Path(root)
        self.cache_images = cache_images
        self._cache: Dict[Path, torch.Tensor] = {}
        self.crop = crop

        rng = random.Random(seed)
        self.samples: List[Tuple[str, int, int]] = []
        self.frames = _list_frames(root)
        self._seed = seed
        self._centres, self._full = _video_centres(self.frames, crop)
        self.windows = _draw_windows(self._centres, self._full, crop, seed)
        # Dumped transforms are always full-frame width; a crop must be taken
        # from them *after* the difference, never baked into their length.
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

    def set_epoch(self, epoch: int) -> None:
        """Re-draw this epoch's crop windows (no-op without --crop)."""
        if self.crop is not None:
            self.windows = _draw_windows(
                self._centres, self._full, self.crop, self._seed + 9973 * epoch
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _image(self, path: Path) -> torch.Tensor:
        if self.cache_images and path in self._cache:
            return self._cache[path]
        img = _read_image(path)
        if self.cache_images:
            self._cache[path] = img
        return img

    def _transform(self, tf: Optional[Path], width: int):
        """(dx, dy) of a frame w.r.t. its volume's reference; identity for the
        reference itself and for a frame whose transform was not dumped."""
        if tf is None:
            return torch.zeros(()), torch.zeros(width)
        t = _load(tf)
        return t["x"].float().reshape(()), t["y"].float().reshape(-1)

    def __getitem__(self, k: int):
        vid, i, j = self.samples[k]
        fl = self.frames[vid]
        win = self.windows[vid] if self.windows else None
        # The same window for both frames: cropping them at different offsets
        # would inject a dy of exactly that difference into the pair.
        fixed = _apply_crop(self._image(fl[i][0]), win, self.crop)
        moving = _apply_crop(self._image(fl[j][0]), win, self.crop)
        W = self.full_w
        # Both transforms are relative to the same reference, so the pair's
        # transform is their difference (exact for dx; for dy exact up to the
        # column re-indexing induced by dx, which is a few pixels).
        dx_i, dy_i = self._transform(fl[i][1], W)
        dx_j, dy_j = self._transform(fl[j][1], W)
        # The diagnostic is only meaningful if *both* frames' transforms are
        # known. Index 0 is the reference, whose transform is the identity; any
        # other frame without a dumped transform makes the pair unscorable.
        has_gt = (i == 0 or fl[i][1] is not None) and (j == 0 or fl[j][1] is not None)
        # The frame identities travel with the pair so run_epoch can cache the
        # encoder pyramid per frame instead of per pair — see FeatureCache.
        return (
            fixed,
            moving,
            dx_j - dx_i,
            _crop_dy(dy_j - dy_i, win, self.crop),
            torch.tensor(float(has_gt)),
            _key(vid, i, win),
            _key(vid, j, win),
        )


class FrameTripletDataset(Dataset):
    """Closed triplets ``(i, j, k)`` of frames from one video, for the cycle-
    consistency term: a lateral shift is a pure translation, so it composes
    exactly across any three frames of the same eye --
    ``dx(i,k) == dx(i,j) + dx(j,k)`` -- regardless of what the model predicts
    for any single pair. Checking that composition needs no ground truth, only
    three ordinary forward passes on frames the pair loader already visits.

    One item yields the three *pairs* a closed triplet needs -- ``(i,j)``,
    ``(j,k)``, ``(i,k)`` -- stacked so a caller can push them through the
    model exactly like a pair batch and then read off ``dx.view(-1, 3)``.
    """

    def __init__(
        self,
        root: Path,
        videos: Optional[Set[str]] = None,
        *,
        triplets_per_video: int = 40,
        seed: int = 0,
        cache_images: bool = True,
        crop: Optional[Tuple[int, int]] = None,
    ):
        root = Path(root)
        self.cache_images = cache_images
        self._cache: Dict[Path, torch.Tensor] = {}
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
        if self.cache_images and path in self._cache:
            return self._cache[path]
        img = _read_image(path)
        if self.cache_images:
            self._cache[path] = img
        return img

    def __getitem__(self, idx: int):
        vid, i, j, k = self.samples[idx]
        fl = self.frames[vid]
        win = self.windows[vid] if self.windows else None
        c = lambda n: _apply_crop(self._image(fl[n][0]), win, self.crop)
        fi, fj, fk = c(i), c(j), c(k)
        # Row order (i,j), (j,k), (i,k) -- collate_triplets keeps it, so a
        # batch of Bt items reshapes to (Bt, 3) in exactly this column order.
        fixed = torch.stack([fi, fj, fi], dim=0)
        moving = torch.stack([fj, fk, fk], dim=0)
        keys_f = [_key(vid, i, win), _key(vid, j, win), _key(vid, i, win)]
        keys_m = [_key(vid, j, win), _key(vid, k, win), _key(vid, k, win)]
        return fixed, moving, keys_f, keys_m


def collate_triplets(batch):
    """Bt items of 3 stacked pairs each -> one (3*Bt,...) batch, triplet-major
    (item 0's 3 rows, then item 1's, ...) so ``dx.view(Bt, 3)`` recovers
    ``(dx_ij, dx_jk, dx_ik)`` per triplet after the model runs on the batch."""
    fixed = torch.cat([b[0] for b in batch], dim=0)
    moving = torch.cat([b[1] for b in batch], dim=0)
    keys_f = [key for b in batch for key in b[2]]
    keys_m = [key for b in batch for key in b[3]]
    return fixed, moving, keys_f, keys_m


def _shift_and_mask(img: torch.Tensor, delta: int, margin: int) -> torch.Tensor:
    """Shift ``img`` content right by ``delta`` whole columns and blank
    ``margin`` columns at both edges.

    Whole columns, so the shift is an exact index copy -- no interpolation, and
    therefore none of the sub-pixel resampling bias that makes the photometric
    terms prefer integer offsets.

    The blanking is what keeps the task honest. Shifting alone leaves a
    ``|delta|``-wide empty strip on one side, whose width and side give
    ``delta`` away; a model could read the strip instead of registering
    anything. Masking a fixed ``margin >= max|delta|`` on *both* edges of
    *every* variant makes their border geometry identical, so the strip
    carries no information and only the content offset differs.
    """
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
    """(fixed, moving) pairs plus a copy of ``moving`` shifted by a known
    integer ``delta``, for the shift-equivariance term.

    The only exactly-known quantity available for ``dx``: we apply the shift
    ourselves, so ``dx_ref - dx_shifted == delta`` holds whatever the true
    underlying displacement is, and the classical estimator is not involved at
    any point. See :func:`shift_equivariance_loss`.

    ``fixed`` is returned raw and keyed, so it comes from the FeatureCache like
    any other frame. The two ``moving`` variants are transformed images, which
    no per-frame cache can serve, so they are encoded fresh every step -- that
    is what this term costs.
    """

    def __init__(
        self,
        root: Path,
        videos: Optional[Set[str]] = None,
        *,
        shifts_per_video: int = 40,
        shift_max: int = 8,
        seed: int = 0,
        cache_images: bool = True,
        crop: Optional[Tuple[int, int]] = None,
    ):
        root = Path(root)
        self.cache_images = cache_images
        self.shift_max = int(shift_max)
        self._cache: Dict[Path, torch.Tensor] = {}
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
        if self.cache_images and path in self._cache:
            return self._cache[path]
        img = _read_image(path)
        if self.cache_images:
            self._cache[path] = img
        return img

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


def split_videos(root: Path, val_frac: float, seed: int) -> Tuple[Set[str], Set[str]]:
    """Split by *video*: two frames of one volume are far from independent."""
    vids = sorted(
        {_video_of(p.stem) for p in (Path(root) / "images" / "fixed").glob("*.png")}
    )
    rng = random.Random(seed)
    rng.shuffle(vids)
    n_val = max(1, round(len(vids) * val_frac)) if len(vids) > 1 else 0
    return set(vids[n_val:]), set(vids[:n_val])


class VideoChunkBatchSampler(Sampler):
    """Batches drawn from a small working set of videos at a time.

    Encoding is 57% of a training step (a MiT forward over 1536x1024 frames),
    and with ``pair_mode="within"`` the same ~40 frames per video are re-encoded
    for every pair they appear in — about 240 passes per video where 40 would
    do. Caching the pyramid per frame fixes that, but only if consecutive
    batches keep hitting the same frames.

    The obvious way to get that locality — one video per batch — would make
    every gradient step see a single volume, which is a real change to the
    optimisation and not one we want. So instead a *working set* of
    ``videos_in_flight`` volumes is held at once and batches are drawn at random
    from the union of their pairs: batches stay heterogeneous, while the cache
    only ever needs to hold a few volumes' frames.
    """

    def __init__(
        self,
        samples: Sequence[Tuple],
        batch_size: int,
        videos_in_flight: int = 4,
        shuffle: bool = True,
        seed: int = 0,
    ):
        # Only the leading video id is read, so this works unchanged on
        # FramePairDataset's (vid, i, j) and FrameTripletDataset's (vid, i, j, k).
        self.groups: Dict[str, List[int]] = defaultdict(list)
        for idx, sample in enumerate(samples):
            self.groups[sample[0]].append(idx)
        self.batch_size = batch_size
        self.k = max(1, videos_in_flight)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        vids = list(self.groups)
        self._len = sum(
            (sum(len(self.groups[v]) for v in vids[s : s + self.k]) + batch_size - 1)
            // batch_size
            for s in range(0, len(vids), self.k)
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        vids = list(self.groups)
        if self.shuffle:
            rng.shuffle(vids)
        for s in range(0, len(vids), self.k):
            pool = [i for v in vids[s : s + self.k] for i in self.groups[v]]
            if self.shuffle:
                rng.shuffle(pool)
            for b in range(0, len(pool), self.batch_size):
                yield pool[b : b + self.batch_size]


class FeatureCache:
    """LRU of frozen-encoder pyramids, keyed by frame.

    Stored as float16 (~24 MB a frame against 49 in float32) and cast back on
    use: these features are consumed by a cosine similarity and a correlation
    volume, both of which normalise, so the storage precision is far finer than
    anything the objective resolves.
    """

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
                    feats = self.encoder.forward_features(images)
            else:
                feats = self.encoder.forward_features(images)
        return [f.half() for f in feats]

    def get(self, images: torch.Tensor, keys: Sequence[str]) -> List[torch.Tensor]:
        """Pyramid for each image, encoding only the frames not already held."""
        missing = [i for i, k in enumerate(keys) if k not in self._cache]
        # A frame can repeat inside one batch; encode it once.
        uniq: Dict[str, int] = {}
        for i in missing:
            uniq.setdefault(keys[i], i)
        if uniq:
            idx = list(uniq.values())
            feats = self._encode(images[idx])
            for n, k in enumerate(uniq):
                self._cache[k] = [f[n : n + 1] for f in feats]
            # Evict only after the whole batch is in, and never evict a frame
            # this batch still has to read back: a capacity below 2*batch_size
            # would otherwise drop entries between insertion and use.
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
        """Encode ``images`` without touching the cache.

        For synthetically transformed frames (see FrameShiftDataset): they are
        not any frame's pyramid, so caching them by frame id would poison the
        cache for the real thing.
        """
        return [f.float() for f in self._encode(images)]

    @property
    def hit_rate(self) -> float:
        n = self.hits + self.misses
        return self.hits / n if n else 0.0


# --------------------------------------------------------------------------- #
# Frozen networks
# --------------------------------------------------------------------------- #
def build_segmenter(device: str):
    """The frozen choroid segmentation module: its encoder feeds the regressor,
    and (with ``--w-bm``) its decoder provides the BM curve the loss aligns."""
    from ocularrigidity.scripts.cohort_analysis.segment_n_cycles import get_model

    # get_model loads onto `device` directly -- see its docstring on why
    # an already-loaded model must never be moved across GPUs.
    seg = get_model(device).eval()
    for p in seg.parameters():
        p.requires_grad_(False)
    return seg


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
def _forever(loader):
    """Endlessly re-iterate ``loader``, reshuffling as its sampler dictates.

    Deliberately not ``itertools.cycle``: that saves a copy of every element it
    yields so it can replay them, which for image batches is ~75 MB a batch and
    tens of GB of pinned host RAM over one epoch, and it then replays the
    *identical* tensors rather than drawing freshly shuffled triplets.
    """
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
    """One pass. ``optimizer=None`` evaluates; ``baseline=True`` scores the
    classical (dx, dy) instead of the model's, under the identical objective.

    ``triplet_loader`` (optional) yields closed frame triplets for the cycle-
    consistency term (see ``cycle_consistency_loss``); it is cycled
    independently of ``loader`` since the two have different lengths. Skipped
    entirely under ``baseline``: the classical labels compose by construction
    (``dx_gt(i,j) = dx_j - dx_i`` sums telescopically), so the term would be
    tautologically zero there and measure nothing.
    """
    train = optimizer is not None
    model.train(train and not baseline)
    sums: Dict[str, float] = defaultdict(float)
    n = 0.0
    n_gt = 0.0
    triplet_iter = (
        _forever(triplet_loader)
        if triplet_loader is not None and len(triplet_loader) > 0
        else None
    )
    shift_iter = (
        _forever(shift_loader)
        if shift_loader is not None and len(shift_loader) > 0
        else None
    )

    for fixed, moving, dx_gt, dy_gt, has_gt, key_f, key_m in tqdm(
        loader,
        desc="train" if train else ("baseline" if baseline else "val"),
        leave=False,
    ):
        fixed = fixed.to(device, non_blocking=True)
        moving = moving.to(device, non_blocking=True)
        dx_gt = dx_gt.to(device)
        dy_gt = dy_gt.to(device)
        bs = fixed.shape[0]

        # Frozen pyramids, encoded once per *frame* rather than once per pair:
        # the same frames recur across the pairs of a video, and the encoder is
        # the single most expensive part of a step.
        feats = cache.get(torch.cat([fixed, moving], 0), list(key_f) + list(key_m))
        fixed_feats = [f[:bs] for f in feats]
        moving_feats = [f[bs:] for f in feats]
        # Taken from the batch, so crops and full frames both work: the cascade
        # reads its strides off this and would silently mis-scale otherwise.
        shape = tuple(fixed.shape[-2:])

        with torch.set_grad_enabled(train and not baseline):
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
            )

            # --- dx auxiliaries. Neither uses the classical estimate: the
            # cycle term checks the model against itself, the shift term
            # against a displacement we applied ourselves. They are
            # complementary -- shift fixes dx's gain but is blind to a constant
            # offset, cycle removes the offset but is minimised by collapse.
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
                # fixed is a real frame -> cacheable; the two shifted variants
                # are synthetic, so they are encoded fresh (see cache.encode).
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
                # `core` is the part of the objective the classical baseline is
                # also scored on: the aux dx terms are skipped in baseline mode,
                # so `total` alone cannot be compared against that row.
                comp["core"] = float(loss.detach()) - (
                    w_cycle * comp.get("cycle", 0.0)
                    + w_shift * comp.get("shift", 0.0)
                )
                comp["total"] = float(loss.detach())  # keep in sync with backward

        if train and not baseline:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        with torch.no_grad():
            for k, v in comp.items():
                sums[k] += v * bs
            # Agreement with the classical estimator, on the pairs it covers.
            w = has_gt.to(device)
            if float(w.sum()) > 0:
                sums["dx_mae"] += float(((dx - dx_gt).abs() * w).sum())
                sums["dy_mae"] += float(((dy - dy_gt).abs().mean(dim=1) * w).sum())
                n_gt += float(w.sum())
        n += bs

    out = {k: v / max(n, 1.0) for k, v in sums.items()}
    for k in ("dx_mae", "dy_mae"):
        out[k] = sums[k] / max(n_gt, 1.0) if n_gt else float("nan")
    return out


#: Everything that defines *which* objective is being optimised. An ablation
#: varies these and nothing else, so they are also exactly the knobs that have
#: to be pinned to a common setting for two arms to be comparable.
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
    """A loss module for one objective spec.

    The segmenter is only attached when the BM term is actually on: with
    ``w_bm == 0`` the term is skipped, and passing ``None`` makes that
    structural rather than a weight that happens to be zero.
    """
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
    """Start a W&B run, or return ``None`` when --wandb-project was not given.

    The config carries both objectives explicitly, so a sweep's runs can be
    grouped and diffed by which term was dropped without re-deriving it from
    the flat weight list.
    """
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/home/clement/Documents/data/OcularRigidity/Registration/Dataset/"
        ),
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
    # Output scales, in pixels, for --no-cascade only: the heads there are
    # linear, so these are a normalisation, not a bound, and they set how fast
    # the answer can be reached (Adam moves a head's output by about `lr` per
    # step). The cascade derives its own scales from the pyramid strides.
    ap.add_argument("--scale-dx", type=float, default=16.0)
    ap.add_argument("--scale-dy-bulk", type=float, default=128.0)
    ap.add_argument("--scale-dy-residual", type=float, default=8.0)

    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
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
        "--eval-baseline",
        action="store_true",
        help="also score the classical transform under the same objective",
    )
    ap.add_argument("--out", type=Path, default=Path("checkpoints"))
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

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # What we optimise, and what we score. Identical unless --eval-objective
    # says otherwise, in which case the latter is the common ruler every arm
    # of an ablation is measured against.
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
        print("train objective:", json.dumps(train_obj))
        print("eval  objective:", json.dumps(eval_obj), "(common ruler)")

    seg = build_segmenter(device)
    encoder = seg.model.encoder

    crop = tuple(args.crop) if args.crop else None
    train_vids, val_vids = split_videos(args.root, args.val_frac, args.seed)
    ds_kw = dict(
        pair_mode=args.pair_mode,
        pairs_per_video=args.pairs_per_video,
        seed=args.seed,
        cache_images=args.num_workers == 0,
        crop=crop,
    )
    train_ds = FramePairDataset(args.root, train_vids, **ds_kw)
    val_ds = FramePairDataset(args.root, val_vids, **ds_kw)

    # Each side is built only if *its own* objective uses the term: ablating a
    # term out of training costs nothing to train, while the eval side keeps
    # measuring it so the arm stays comparable.
    triplet_kw = dict(
        triplets_per_video=args.triplets_per_video,
        seed=args.seed,
        cache_images=args.num_workers == 0,
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
        cache_images=args.num_workers == 0,
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
    # The checkpoint must record the *full* frame, not the crop: inference
    # rebuilds the model from it and would otherwise mis-scale every stride.
    _probe = cv2.imread(str(next(iter(train_ds.frames.values()))[0][0]), cv2.IMREAD_GRAYSCALE)
    FULL_H, FULL_W = _probe.shape[:2]
    with torch.no_grad():
        in_channels = [
            f.shape[1]
            for f in encoder.forward_features(sample[0].unsqueeze(0).to(device))
        ]

    if crop:
        print(f"crop: {H}x{W} per video, retina-centred (full frame {FULL_H}x{FULL_W})")
    print(f"images {H}x{W} | encoder channels (fine->coarse): {in_channels}")
    print(f"videos: {len(train_vids)} train / {len(val_vids)} val")
    print(f"pairs:  {len(train_ds)} train / {len(val_ds)} val  [{args.pair_mode}]")
    def _n(ds) -> int:
        return len(ds) if ds is not None else 0

    if train_triplet_ds is not None or val_triplet_ds is not None:
        print(
            f"triplets: {_n(train_triplet_ds)} train / {_n(val_triplet_ds)} val "
            f"[cycle consistency, w_cycle={train_obj['w_cycle']} "
            f"train / {eval_obj['w_cycle']} eval]"
        )
    if train_shift_ds is not None or val_shift_ds is not None:
        print(
            f"shifts:   {_n(train_shift_ds)} train / {_n(val_shift_ds)} val "
            f"[equivariance, w_shift={train_obj['w_shift']} train / "
            f"{eval_obj['w_shift']} eval, delta=+-1..{args.shift_max}px]"
        )

    if args.dry_run:
        for k in range(min(3, len(train_ds))):
            _, _, dx, dy, has_gt, _, _ = train_ds[k]
            print(
                f"  pair {k}: dx={dx.item():+.2f}px  "
                f"dy[min/mean/max]={dy.min():+.1f}/{dy.mean():+.1f}/{dy.max():+.1f}px  "
                f"gt={bool(has_gt)}"
            )
        return

    common = dict(
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        # With --crop the windows are redrawn each epoch in the parent, so
        # workers have to be re-forked to see them; persistent workers would
        # keep serving the first epoch's crops forever.
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
        """One auxiliary loader, plus its sampler so the epoch can be advanced.

        Returns ``(loader, sampler)``, both ``None`` when the term is off on
        this side. Without advancing the sampler the aux stream would replay
        one fixed order every epoch.
        """
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

    criterion = _build_criterion(train_obj, len(in_channels), args, seg)
    # A separate module rather than the same one re-weighted: it is always held
    # at full progress, so validation is not measured under the coarse-to-fine
    # ramp's objective-of-the-epoch and stays comparable with itself.
    eval_criterion = _build_criterion(eval_obj, len(in_channels), args, seg)
    eval_criterion.set_progress(1.0)

    run = _init_wandb(args, train_obj, eval_obj, len(train_vids), len(val_vids),
                      len(train_ds), len(val_ds))

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
        print(_fmt("[baseline: classical transform] val", baseline_metrics))
        if run is not None:
            for k, v in baseline_metrics.items():
                run.summary[f"baseline/{k}"] = v

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
            # eval_criterion is permanently at full progress: with the
            # coarse-to-fine ramp on, the training loss is measured under a
            # different objective every epoch and cannot be compared with
            # itself -- nor, under an ablation, with another arm.
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

        if score < best:
            best, best_epoch = score, epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "in_channels": in_channels,
                    "out_width": W,
                    "embed": args.embed,
                    "img_shape": (FULL_H, FULL_W),
                    "train_shape": (H, W),
                    "cascade": not args.no_cascade,
                    "scales": scales,
                    "use_correlation": not args.no_correlation,
                    "val_loss": score,
                    "train_objective": train_obj,
                    "eval_objective": eval_obj,
                    "args": vars(args),
                },
                args.out / "best.pt",
            )

    ckpt_path = args.out / "best.pt"
    print(f"done. best val objective: {best:.4f}  ->  {ckpt_path}")

    if run is not None:
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


if __name__ == "__main__":
    main()
