import numpy as np
import os
from tqdm.auto import tqdm


import subprocess
import av

import matplotlib.pyplot as plt
from decord import VideoReader, cpu

os.environ["IMAGEIO_FFMPEG_EXE"] = "/home/clement/miniforge-pypy3/envs/dl/bin/ffmpeg"
import imageio


def cube_to_mp4(cube: np.ndarray, out_path: str, crf: int = 23, fps: int = 30):
    """
    Compress (n, H, W) uint8 cube to H.265 MP4.

    crf: 0 (lossless) to 51 (terrible). 18-23 is visually near-lossless.
         For OCT display, 23-28 is usually fine.
    """
    writer = imageio.get_writer(
        out_path,
        fps=fps,
        codec="libx265",
        quality=None,
        ffmpeg_params=["-crf", str(crf), "-preset", "medium", "-pix_fmt", "yuv420p"],
    )
    for frame in cube:
        writer.append_data(frame)
    writer.close()


def cube_to_mp4_fast(cube: np.ndarray, out_path: str, fps: int = 30, verbose=False):
    """
    Ultra-fast grayscale compression.
    Target: Speed and Numpy-readability.
    """

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


def mp4_to_cube(
    path,
    T=None,
    H=None,
    W=None,
    ffmpeg="/home/clement/miniforge-pypy3/envs/dl/bin/ffmpeg",
    use_gpu=True,
):
    """Decode HEVC mp4 to a TxHxW uint8 numpy cube.

    If T/H/W are None, probes the file with ffprobe first.
    """
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

    cube = np.empty((T, H, W), dtype=np.uint8)
    buf = memoryview(cube).cast("B")

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)
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


def _probe(path, ffmpeg):
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
    W, H, T = int(out[0]), int(out[1]), int(out[2])
    return T, H, W


def compare_compression(lossless_path, lossy_path, n_summary_frames=200):
    # global metrics on a random subsample (full read on 3000 frames is fine too)
    rng = np.random.default_rng(0)
    vr_lossless = VideoReader(lossless_path, ctx=cpu(0))
    T = len(vr_lossless)
    summary_idx = rng.choice(T, size=min(n_summary_frames, T), replace=False)
    summary_idx = np.sort(summary_idx).tolist()

    a = read_luma(lossless_path, summary_idx).astype(np.int16)
    b = read_luma(lossy_path, summary_idx).astype(np.int16)
    diff = b - a  # signed; reveals whether lossy is biased high/low

    abs_diff = np.abs(diff)
    mae = abs_diff.mean(axis=(1, 2))
    maxe = abs_diff.max(axis=(1, 2))
    psnr = 10 * np.log10(255.0**2 / np.maximum((diff**2).mean(axis=(1, 2)), 1e-9))

    print(f"frames analyzed:         {len(summary_idx)}")
    print(f"mean abs error (global): {abs_diff.mean():.3f} / 255")
    print(f"max  abs error (global): {abs_diff.max()}")
    print(f"PSNR  median / min:      {np.median(psnr):.2f} / {psnr.min():.2f} dB")
    print(f"% pixels exact:          {(abs_diff == 0).mean() * 100:.2f}")
    print(f"% pixels |err|<=1:       {(abs_diff <= 1).mean() * 100:.2f}")
    print(f"% pixels |err|<=3:       {(abs_diff <= 3).mean() * 100:.2f}")

    # spatial heatmap: average |error| per pixel across sampled frames
    spatial_err = abs_diff.mean(axis=0)

    # error histogram (signed, to see bias)
    hist_bins = np.arange(-30, 31)
    hist, _ = np.histogram(diff, bins=hist_bins)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(summary_idx, mae, lw=0.8, label="MAE")
    axes[0, 0].plot(summary_idx, maxe, lw=0.8, alpha=0.5, label="max |err|")
    axes[0, 0].set_xlabel("frame")
    axes[0, 0].set_ylabel("error (gray levels)")
    axes[0, 0].set_title("Per-frame error")
    axes[0, 0].legend()

    axes[0, 1].plot(summary_idx, psnr, lw=0.8, color="k")
    axes[0, 1].set_xlabel("frame")
    axes[0, 1].set_ylabel("PSNR (dB)")
    axes[0, 1].set_title("Per-frame PSNR (higher = better)")

    im = axes[1, 0].imshow(spatial_err, cmap="magma")
    axes[1, 0].set_title("Mean |error| per pixel")
    axes[1, 0].axis("off")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    axes[1, 1].bar(hist_bins[:-1], hist, width=1.0, color="steelblue")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("signed error (lossy − lossless)")
    axes[1, 1].set_ylabel("count (log)")
    axes[1, 1].set_title("Error distribution")

    plt.tight_layout()
    return fig, {"mae": mae, "psnr": psnr, "spatial_err": spatial_err}


def inspect_frame(lossless_path, lossy_path, idx, amp=10):
    a = read_gray(lossless_path, [idx])[0].astype(np.int16)
    b = read_gray(lossy_path, [idx])[0].astype(np.int16)
    d = b - a

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    axes[0].imshow(a, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("lossless")
    axes[1].imshow(b, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("lossy")
    im2 = axes[2].imshow(np.abs(d) * amp, cmap="magma", vmin=0, vmax=255)
    axes[2].set_title(f"|error| ×{amp}")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    im3 = axes[3].imshow(d, cmap="seismic", vmin=-15, vmax=15)
    axes[3].set_title("signed error")
    plt.colorbar(im3, ax=axes[3], fraction=0.046)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    return fig
