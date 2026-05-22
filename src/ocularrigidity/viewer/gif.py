from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from scipy.signal import savgol_filter
from matplotlib import cm
from ocularrigidity.registration.sparse_demons import track_points_with_demons


def render_membrane_trace(
    frames: np.ndarray,
    masks: np.ndarray,
    output_path: str | Path,
    side_by_side: bool = False,
    *,
    fps: int = 10,
    decay: float = 0.9,
    current_color_rgb: tuple[int, int, int] = (0, 255, 0),  # green: just stamped
    old_color_rgb: tuple[int, int, int] = (255, 0, 0),  # red: about to vanish
    border_kernel_size: int = 2,
) -> None:
    """Render a looping animation where each mask boundary leaves a fading trace.

    For every frame the mask boundary is extracted via morphological gradient
    and stamped into a trace buffer at full intensity. The buffer decays
    multiplicatively between frames, so older positions fade out. The trace is
    composited over the grayscale frame with a color that interpolates from
    ``current_color_rgb`` (fresh) to ``old_color_rgb`` (faded), giving the
    green -> yellow -> red -> gone trail.

    Parameters
    ----------
    frames : (T, H, W) uint8
        Grayscale source frames.
    masks : (T, H, W)
        Binary segmentation masks (bool or 0/1 integer).
    output_path : str or Path
        Destination path; the extension determines the format (.gif, .mp4, ...).
    side_by_side : bool, default False
        If True, concatenate the original frame and the overlay horizontally.
    fps : int
        Output frame rate.
    decay : float
        Per-frame multiplicative decay in (0, 1) applied to the trace buffer.
    current_color_rgb, old_color_rgb : (R, G, B), each component in [0, 255]
        Endpoints of the color ramp along the trace.
    border_kernel_size : int
        Side length of the square structuring element used for the boundary
        extraction (morphological gradient).
    """
    if frames.ndim != 3:
        raise ValueError(f"`frames` must have shape (T, H, W); got {frames.shape}.")
    if frames.shape != masks.shape:
        raise ValueError(
            f"`frames` and `masks` must share shape; got {frames.shape} vs {masks.shape}."
        )

    T, H, W = frames.shape
    kernel = np.ones((border_kernel_size, border_kernel_size), dtype=np.uint8)
    current_color = np.asarray(current_color_rgb, dtype=np.float32)
    old_color = np.asarray(old_color_rgb, dtype=np.float32)

    trace = np.zeros((H, W), dtype=np.float32)
    overlay: list[np.ndarray] = []

    for i, (frame, mask) in enumerate(zip(frames, masks)):
        # Decay the running trace, then refresh the current border to full intensity.
        border = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
        trace *= decay
        trace[border] = 1.0

        # Composite: blend the gray frame with an age-dependent color.
        gray = frame[..., None].astype(np.float32)  # (H, W, 1)
        alpha = trace[..., None]  # (H, W, 1)
        color = old_color + (current_color - old_color) * alpha  # (H, W, 3)
        out = gray * (1.0 - alpha) + color * alpha  # (H, W, 3)

        overlay.append(np.clip(out, 0, 255).astype(np.uint8))
        # Write the frame number on the top-left corner for debugging.
        cv2.putText(
            overlay[-1],
            f"{i:04d} / {T}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

    if side_by_side:
        overlay = [
            np.concatenate(
                [cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR), overlay[i]], axis=1
            )
            for i, frame in enumerate(frames)
        ]

    imageio.mimwrite(str(output_path), overlay, fps=fps, loop=0)


def render_mask_quiver(
    frames: np.ndarray,
    masks: np.ndarray,
    output_path: str | Path,
    side_by_side: bool = False,
    *,
    fps: int = 10,
    stride: int = 8,
    reference: int = 0,
    arrow_scale: float = 20.0,
    min_magnitude: float = 0.05,  # pixels, pre-scale
    arrow_cmap: str = "viridis",
    arrow_thickness: int = 1,
    tip_length: float = 0.35,
    border_kernel_size: int = 2,
    lk_window: int = 21,
    lk_levels: int = 3,
    annotate_scale: bool = True,
    smooth_window: int = 4,
    cyclic: bool = True,
) -> None:
    if frames.ndim != 3:
        raise ValueError(f"`frames` must have shape (T, H, W); got {frames.shape}.")
    if frames.shape != masks.shape:
        raise ValueError(
            f"`frames` and `masks` must share shape; got {frames.shape} vs {masks.shape}."
        )
    T, H, W = frames.shape
    if not 0 <= reference < T:
        raise ValueError(f"`reference` must be in [0, {T}); got {reference}.")

    kernel = np.ones((border_kernel_size, border_kernel_size), dtype=np.uint8)
    ref_border = (
        cv2.morphologyEx(masks[reference].astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
        > 0
    )
    ys, xs = np.nonzero(ref_border)
    if ys.size == 0:
        raise ValueError(f"Reference frame {reference} has an empty boundary.")
    ys, xs = ys[::stride], xs[::stride]
    p0 = np.stack([xs, ys], axis=1).astype(np.float32).reshape(-1, 1, 2)
    N = p0.shape[0]

    lk_params = dict(
        winSize=(lk_window, lk_window),
        maxLevel=lk_levels,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        minEigThreshold=1e-4,
    )

    # Track every frame against the reference. NaN marks lost tracks.
    ref_frame = frames[reference]
    positions = np.full((T, N, 2), np.nan, dtype=np.float32)
    positions[reference] = p0[:, 0, :]
    for t in range(T):
        if t == reference:
            continue
        p1, status, _ = cv2.calcOpticalFlowPyrLK(
            ref_frame, frames[t], p0, None, **lk_params
        )
        # p1, status = track_points_with_demons(ref_frame, frames[t], p0, std_dev=3.0)
        ok = status[:, 0].astype(bool)
        positions[t, ok] = p1[ok, 0, :]

    if smooth_window > 0:
        # Per-anchor linear interpolation through NaN gaps along time.
        t_axis = np.arange(T)
        for n in range(N):
            valid = np.isfinite(positions[:, n, 0])
            nv = int(valid.sum())
            if nv == T:
                continue
            if nv < 2:
                positions[:, n, :] = p0[n]
                continue
            idx = np.where(valid)[0]
            for c in (0, 1):
                positions[:, n, c] = np.interp(
                    t_axis,
                    idx,
                    positions[idx, n, c],
                    period=T if cyclic else None,
                )
        positions = savgol_filter(
            positions,
            window_length=smooth_window,
            polyorder=3,
            axis=0,
            mode="wrap" if cyclic else "interp",
        )
    p0_xy = p0[:, 0, :]  # (N, 2) in (x, y)

    overlay: list[np.ndarray] = []
    # Find min and max magnitudes for color scaling.
    disp = positions - p0_xy[None, :, :]  # (T, N, 2)
    mags = np.hypot(disp[..., 0], disp[..., 1])  # (T, N)
    valid = np.isfinite(mags) & (mags >= min_magnitude)
    if valid.any():
        vmin, vmax = mags[valid].min(), mags[valid].max()
    else:
        vmin, vmax = 0.0, 1.0
    cmap = cm.get_cmap(arrow_cmap)
    for i in range(T):
        out = cv2.cvtColor(frames[i], cv2.COLOR_GRAY2RGB).copy()

        disp = positions[i] - p0_xy  # (N, 2)
        mags = np.hypot(disp[:, 0], disp[:, 1])
        valid = np.isfinite(mags) & (mags >= min_magnitude)

        for (x0, y0), (dx, dy), mag in zip(p0_xy[valid], disp[valid], mags[valid]):
            ex = x0 + dx * arrow_scale
            ey = y0 + dy * arrow_scale
            norm = 0.0 if vmax == vmin else (mag - vmin) / (vmax - vmin)
            arrow_color_rgb = tuple(int(c * 255) for c in cmap(norm)[:3])
            cv2.arrowedLine(
                out,
                (int(round(x0)), int(round(y0))),
                (int(round(ex)), int(round(ey))),
                arrow_color_rgb,
                arrow_thickness,
                line_type=cv2.LINE_AA,
                tipLength=tip_length,
            )

        cv2.putText(
            out,
            f"{i:04d} / {T}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        if annotate_scale:
            cv2.putText(
                out,
                f"x{arrow_scale:g}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
        overlay.append(out)

    if side_by_side:
        overlay = [
            np.concatenate(
                [cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB), overlay[i]], axis=1
            )
            for i, frame in enumerate(frames)
        ]
    imageio.mimwrite(str(output_path), overlay, fps=fps, loop=0)
