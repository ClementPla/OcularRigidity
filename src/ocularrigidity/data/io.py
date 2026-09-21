import imageio.v3 as iio
import os
from pathlib import Path
from urllib.parse import urlparse
from functools import lru_cache
import zstandard as zstd
import numpy as np
import smbclient
from dotenv import load_dotenv
from tqdm.auto import tqdm


def load_cube_mp4(path: str, collapse_rgb: bool = True) -> np.ndarray:
    """
    Decode an MP4 video to a (n_frames, H, W) uint8 grayscale numpy array.

    Assumes the MP4 was encoded from a grayscale source (channels should be
    identical or near-identical after codec round-trip).
    """
    video = iio.imread(path)  # shape (n, H, W, 3) or (n, H, W)

    if video.ndim == 4 and collapse_rgb:
        return video[..., 0]
    elif video.ndim == 4 and not collapse_rgb:
        return video
    elif video.ndim == 3:
        return video
    else:
        raise ValueError(f"Unexpected video shape: {video.shape}")


def load_mj2_video(video_path):
    """Load a video in MJ2 format and return it as a numpy array."""
    video = iio.imread(video_path)
    return video


load_dotenv()


@lru_cache(maxsize=1)
def _register_smb(server: str) -> None:
    smbclient.register_session(
        server,
        username=os.environ["SMB_USERNAME"],
        password=os.environ["SMB_PASSWORD"],
    )


def _open(path: str, mode: str):
    """Open a local or smb:// path uniformly."""
    if path.startswith("smb://"):
        parsed = urlparse(path)
        _register_smb(parsed.netloc)
        unc = f"//{parsed.netloc}{parsed.path}"
        return smbclient.open_file(unc, mode=mode)
    return open(path, mode)


def _exists(path: str) -> bool:
    if path.startswith("smb://"):
        parsed = urlparse(path)
        _register_smb(parsed.netloc)
        unc = f"//{parsed.netloc}{parsed.path}"
        try:
            smbclient.stat(unc)
            return True
        except (OSError, FileNotFoundError):
            return False
    return Path(path).exists()


def _read_all(path: str, chunk_size: int = 64 * 1024 * 1024) -> bytes:
    """Read a file fully into memory with a progress bar."""
    if path.startswith("smb://"):
        parsed = urlparse(path)
        _register_smb(parsed.netloc)
        unc = f"//{parsed.netloc}{parsed.path}"
        total = smbclient.stat(unc).st_size
        opener = lambda: smbclient.open_file(unc, mode="rb")
        desc = f"Downloading {Path(parsed.path).name}"
    else:
        total = os.path.getsize(path)
        opener = lambda: open(path, "rb")
        desc = f"Loading {Path(path).name}"

    chunks = []
    with (
        opener() as f,
        tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=desc,
            leave=False,
        ) as pbar,
    ):
        while chunk := f.read(chunk_size):
            chunks.append(chunk)
            pbar.update(len(chunk))

    return b"".join(chunks)


DEPTH = 1536  # A-scan length, in samples


def load_octa(root: Path, depth: int = DEPTH):
    """Memory-map amplitude/phase volumes as (n_bscans, n_ascans, depth) uint8."""
    ts = np.loadtxt(root / "timestamp.txt", dtype=np.int64)
    n_bscans = ts.size
    nbytes = (root / "amplitude.bin").stat().st_size
    n_ascans, rem = divmod(nbytes, n_bscans * depth)
    if rem:
        raise ValueError(
            f"{nbytes} bytes not divisible by {n_bscans} B-scans x {depth} depth"
        )

    shape = (n_bscans, n_ascans, depth)
    amp = np.memmap(root / "amplitude.bin", dtype=np.uint8, mode="r").reshape(shape)
    phase = np.memmap(root / "phase.bin", dtype=np.uint8, mode="r").reshape(shape)
    return amp, phase, ts




def load_cube(
    folder: str,
    H: int = 1024,
    W: int = 1536,
    dtype=np.uint8,
    reorder: bool = True,
) -> np.ndarray:
    """
    Load a cube.bin file fully into memory as a (n_frames, H, W) array.

    Accepts:
      - local folder:  /path/to/folder
      - SMB folder:    smb://server/share/path/to/folder

    If `reorder` and `timestamp.txt` exists alongside `cube.bin`,
    frames are sorted by timestamp.
    """
    folder = str(folder)
    folder = folder.rstrip("/")
    cube_path = f"{folder}/cube.bin"
    ts_path = f"{folder}/timestamp.txt"

    # Read raw bytes and interpret as array
    buf = _read_all(cube_path)

    itemsize = np.dtype(dtype).itemsize
    n_frames = len(buf) // (H * W * itemsize)
    data = np.frombuffer(buf, dtype=dtype, count=n_frames * H * W).reshape(
        n_frames, H, W
    )
    # frombuffer returns a read-only view; copy so we own the memory
    data = data.copy()

    # Reorder if timestamps available
    if reorder and _exists(ts_path):
        with _open(ts_path, "r") as f:
            timestamps = np.array([float(line) for line in f if line.strip()])
        order = np.argsort(timestamps)
        if not np.array_equal(order, np.arange(n_frames)):
            data = data[order]  # in-RAM fancy indexing, no laziness needed
    # Swap axes to N, W, H
    data = data.transpose(0, 2, 1)
    return data


def save_mask(mask: np.ndarray, path):
    """Save a boolean mask, packed + zstd compressed."""
    packed = np.packbits(mask.reshape(-1))  # flatten and pack to uint8
    compressed = zstd.ZstdCompressor(level=10).compress(packed.tobytes())
    np.savez(
        path,
        compressed=np.frombuffer(compressed, dtype=np.uint8),
        shape=np.array(mask.shape, dtype=np.int64),
    )


def load_mask(path) -> np.ndarray:
    data = np.load(path)
    decompressed = zstd.ZstdDecompressor().decompress(data["compressed"].tobytes())
    packed = np.frombuffer(decompressed, dtype=np.uint8)
    shape = tuple(data["shape"])
    flat = np.unpackbits(packed)[: np.prod(shape)]
    return flat.reshape(shape).astype(bool)


def load_mask_frames(path, indices) -> np.ndarray:
    """Decode only ``indices`` frames of a packed mask, as ``(len(indices), H, W)``.

    zstd has to inflate the whole payload (a few ms -- these files are tens of
    KB), but ``np.unpackbits`` and the bool conversion are what actually cost:
    they expand every frame to a byte per pixel. Callers that need a handful of
    reference frames out of a folded cycle stack pay ~30x for frames they throw
    away, so unpack per frame instead. Equivalent to ``load_mask(path)[indices]``.
    """
    data = np.load(path)
    shape = tuple(int(v) for v in data["shape"])
    T, H, W = shape
    n_px = H * W
    packed = np.frombuffer(
        zstd.ZstdDecompressor().decompress(data["compressed"].tobytes()),
        dtype=np.uint8,
    )
    out = np.empty((len(indices), H, W), dtype=bool)
    for k, i in enumerate(indices):
        i = int(i) % T
        lo, hi = i * n_px, (i + 1) * n_px  # bit range of this frame
        b0, b1 = lo // 8, -(-hi // 8)  # byte range covering it
        bits = np.unpackbits(packed[b0:b1])
        # view, not astype: unpackbits yields 0/1 uint8, same width as bool.
        out[k] = bits[lo - b0 * 8 : lo - b0 * 8 + n_px].reshape(H, W).view(bool)
    return out
