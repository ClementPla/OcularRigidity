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


def load_cube_mp4(path: str) -> np.ndarray:
    """
    Decode an MP4 video to a (n_frames, H, W) uint8 grayscale numpy array.

    Assumes the MP4 was encoded from a grayscale source (channels should be
    identical or near-identical after codec round-trip).
    """
    video = iio.imread(path)  # shape (n, H, W, 3) or (n, H, W)

    if video.ndim == 4:
        # RGB — collapse to grayscale. Taking channel 0 is fine if the source
        # was grayscale replicated to 3 channels. Use luminance formula if
        # you want to be safe against color drift from chroma subsampling.
        return video[..., 0]
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
