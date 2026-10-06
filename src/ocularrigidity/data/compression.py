from pathlib import Path
import numpy as np
import os
from tqdm.auto import tqdm


import subprocess
import av

from decord import VideoReader, cpu

os.environ["IMAGEIO_FFMPEG_EXE"] = "/home/clement/miniforge-pypy3/envs/dl/bin/ffmpeg"
import imageio


def cube_to_mp4(
    cube: np.ndarray, out_path: str, crf: int = 23, fps: int = 30, verbose: bool = False
):
    """Compress (n, H, W) grayscale or (n, H, W, 3) RGB uint8 cube to H.265 MP4."""
    is_color = cube.ndim == 4
    writer = imageio.get_writer(
        out_path,
        fps=fps,
        codec="libx265",
        quality=None,
        macro_block_size=2,
        ffmpeg_log_level="info" if verbose else "error",
        ffmpeg_params=[
            "-crf",
            str(crf),
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv444p" if is_color else "yuv420p",
            # x265 prints its own banner/stats independently of ffmpeg's loglevel.
            "-x265-params",
            "log-level=none",
        ],
    )
    for frame in cube:
        writer.append_data(frame)
    writer.close()


def cube_to_mp4_fast(cube: np.ndarray, out_path: str, fps: int = 30, verbose=False):
    """Ultra-fast grayscale compression."""

    writer = imageio.get_writer(
        out_path,
        fps=fps,
        codec="hevc_nvenc",
        macro_block_size=2,
        ffmpeg_log_level="warning",
        input_params=["-pix_fmt", "gray"],
        output_params=[
            "-preset",
            "p4",
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            "20",
            "-b:v",
            "0",
            "-pix_fmt",
            "yuv420p",
        ],
    )

    for frame in tqdm(
        cube, desc="Compressing frames", leave=False, disable=not verbose
    ):  # frame is HxW uint8
        writer.append_data(frame)
    writer.close()


def cube_to_mp4_fastest(
    cube,
    out_path,
    fps=30,
    cq=15,
    ffmpeg="/home/clement/miniforge-pypy3/envs/dl/bin/ffmpeg",
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cube = np.ascontiguousarray(cube, dtype=np.uint8)
    is_color = cube.ndim == 4 and cube.shape[3] == 3
    if is_color:
        T, H, W, C = cube.shape
    else:
        T, H, W = cube.shape
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray" if not is_color else "rgb24",
        "-s",
        f"{W}x{H}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "hevc_nvenc",
        "-preset",
        "p4",
        "-rc",
        "constqp",
        "-qp",
        str(cq),
        "-bf",
        "0",
        "-rc-lookahead",
        "0",
        "-multipass",
        "0",
        "-g",
        "30",
        "-pix_fmt",
        "yuv420p",
        "-loglevel",
        "warning",
        out_path,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    # memoryview avoids the .tobytes() copy
    p.stdin.write(memoryview(cube))
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg failed")


def cube_to_mkv_lossless(
    cube, out_path, fps=30, ffmpeg="/home/clement/miniforge-pypy3/envs/dl/bin/ffmpeg"
):
    cube = np.ascontiguousarray(cube, dtype=np.uint8)
    T, H, W = cube.shape
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-s",
        f"{W}x{H}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-coder",
        "1",  # range coder, smaller files
        "-context",
        "1",
        "-g",
        "1",  # all-intra, instant seek
        "-slices",
        "16",  # parallelism
        "-slicecrc",
        "1",  # per-slice CRC, small overhead
        "-threads",
        "0",  # use all cores
        "-pix_fmt",
        "gray",  # FFV1 supports gray natively
        "-loglevel",
        "warning",
        out_path,  # use .mkv
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    p.stdin.write(memoryview(cube))
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg failed")


def mkv_to_cube(
    path,
    T=None,
    H=None,
    W=None,
    ffmpeg="/home/clement/miniforge-pypy3/envs/dl/bin/ffmpeg",
):
    """Decode an FFV1/MKV lossless cube to a TxHxW uint8 numpy cube."""
    if T is None or H is None or W is None:
        T, H, W = _probe(path, ffmpeg)

    cmd = [
        ffmpeg,
        "-loglevel",
        "warning",
        "-threads",
        "0",
        "-i",
        path,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-an",
        "pipe:1",
    ]
    return _read_raw_cube(cmd, T, H, W)


def read_gray(path, indices=None):
    path = str(path)
    vr = VideoReader(path, ctx=cpu(0))
    if indices is None:
        indices = list(range(len(vr)))
    return vr.get_batch(indices).asnumpy()[..., 0]  # T,H,W uint8


def read_luma(path, indices):
    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    fps = float(stream.average_rate)
    indices = sorted(indices)
    out = {}
    for idx in indices:
        ts = int(idx / fps / stream.time_base)
        container.seek(ts, stream=stream, backward=True)
        for frame in container.decode(stream):
            cur = int(round(float(frame.pts * stream.time_base) * fps))
            if cur >= idx:
                out[idx] = frame.to_ndarray(format="gray")  # Y plane, no conversion
                break
    container.close()
    return np.stack([out[i] for i in indices])


def mp4_to_cube(path: str | Path, verbose: bool = True) -> np.ndarray:
    """Fastest PyAV extraction: raw Y-plane memory copy into a pre-allocated array."""
    container = av.open(str(path))
    stream = container.streams.video[0]

    # Maximize C-level multi-threading
    stream.thread_type = "AUTO"
    stream.thread_count = 0  # 0 allows FFmpeg to autodetect optimal CPU core threads

    num_frames = stream.frames
    height = stream.codec_context.height
    width = stream.codec_context.width

    # Fast path: Pre-allocated contiguous uint8 buffer
    if num_frames > 0:
        cube = np.empty((num_frames, height, width), dtype=np.uint8)
        idx = 0
        for frame in tqdm(
            container.decode(stream),
            desc=f"Decoding {Path(path).name}",
            total=num_frames,
            disable=not verbose,
            leave=False,
        ):
            # Direct memory view of the Y-plane (grayscale) without swscale conversion
            cube[idx] = np.frombuffer(frame.planes[0], dtype=np.uint8).reshape(
                height, width
            )
            idx += 1
        container.close()
        return cube[:idx]  # Slice in case stream frame count was off by a frame

    # Fallback if container doesn't report total frame count in header
    frames = []
    for frame in tqdm(
        container.decode(stream),
        desc=f"Decoding {Path(path).name}",
        disable=not verbose,
        leave=False,
    ):
        frames.append(
            np.frombuffer(frame.planes[0], dtype=np.uint8).reshape(height, width)
        )
    container.close()
    return np.array(frames, dtype=np.uint8)


def mp4_to_cube_(
    path,
    T=None,
    H=None,
    W=None,
    ffmpeg="/home/clement/miniforge-pypy3/envs/dl/bin/ffmpeg",
    use_gpu=True,
):
    """Decode HEVC mp4 to a TxHxW uint8 numpy cube."""
    if T is None or H is None or W is None:
        T, H, W = _probe(path, ffmpeg)

    cmd = [ffmpeg, "-loglevel", "warning"]
    if use_gpu:
        cmd += ["-hwaccel", "cuda"]
    cmd += [
        "-i",
        path,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-an",
        "pipe:1",
    ]
    return _read_raw_cube(cmd, T, H, W)


def _probe(path, ffmpeg):
    """``(T, H, W)`` from the container header."""
    ffprobe = ffmpeg.replace("ffmpeg", "ffprobe")
    out = (
        subprocess.check_output(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_frames",
                "-of",
                "csv=p=0",
                path,
            ]
        )
        .decode()
        .strip()
        .split(",")
    )
    W, H = int(out[0]), int(out[1])
    T = int(out[2]) if len(out) > 2 and out[2].strip().lstrip("-").isdigit() else None
    return T, H, W


def _read_raw_cube(cmd, T, H, W):
    """Run ``cmd`` (raw gray video on stdout) and collect a ``(T, H, W)`` cube."""
    frame_bytes = H * W
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)

    if T is not None:
        cube = np.empty((T, H, W), dtype=np.uint8)
        buf = memoryview(cube).cast("B")
        n, total = 0, buf.nbytes
        while n < total:
            got = p.stdout.readinto(buf[n:])
            if not got:
                break
            n += got
        p.stdout.close()
        rc = p.wait()
        if rc != 0 or n != total:
            raise RuntimeError(f"ffmpeg failed: rc={rc}, read {n}/{total} bytes")
        return cube

    chunks = []
    while True:
        block = p.stdout.read(frame_bytes * 32)
        if not block:
            break
        chunks.append(block)
    p.stdout.close()
    rc = p.wait()
    data = bytearray().join(chunks)
    if rc != 0 or not data or len(data) % frame_bytes:
        raise RuntimeError(
            f"ffmpeg failed: rc={rc}, read {len(data)} bytes = "
            f"{len(data) / frame_bytes:.3f} frames of {H}x{W}"
        )
    return np.frombuffer(data, dtype=np.uint8).reshape(-1, H, W)

