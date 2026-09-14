from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import savgol_filter
from ocularrigidity.viewer.quiver import (
    QuiverStyle,
    boundary_anchors,
    draw_quiver,
    track_anchors,
)
from tqdm.auto import tqdm


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
    only_y: bool = False,
    only_orthogonal_to_border: bool = False,
    border_normal_sigma: float = 2.0,
    show_csi_summary: bool = True,
    show_only_csi_anchors: bool = False,
) -> None:
    """Track the mask boundary with optical flow and animate it as a quiver.

    Anchors are sampled on the reference frame's mask boundary (every
    ``stride``-th border pixel, so the flow only tracks what gets drawn) and
    followed across the sequence; the drawing itself — and every option below —
    is :func:`ocularrigidity.viewer.quiver.draw_quiver`, shared with the
    stored-displacement renderer used by the Streamlit viewer.
    """
    if frames.ndim != 3:
        raise ValueError(f"`frames` must have shape (T, H, W); got {frames.shape}.")
    if frames.shape != masks.shape:
        raise ValueError(
            f"`frames` and `masks` must share shape; got {frames.shape} vs {masks.shape}."
        )
    if not 0 <= reference < frames.shape[0]:
        raise ValueError(
            f"`reference` must be in [0, {frames.shape[0]}); got {reference}."
        )

    ref_xy = boundary_anchors(
        masks[reference], stride=stride, border_kernel_size=border_kernel_size
    )
    disp = track_anchors(
        frames,
        ref_xy,
        reference=reference,
        lk_window=lk_window,
        lk_levels=lk_levels,
    )
    style = QuiverStyle(
        stride=1,  # already applied when sampling the anchors
        arrow_scale=arrow_scale,
        min_magnitude=min_magnitude,
        arrow_cmap=arrow_cmap,
        arrow_thickness=arrow_thickness,
        tip_length=tip_length,
        smooth_window=smooth_window,
        cyclic=cyclic,
        only_y=only_y,
        only_orthogonal_to_border=only_orthogonal_to_border,
        border_normal_sigma=border_normal_sigma,
        show_csi_summary=show_csi_summary,
        show_only_csi_anchors=show_only_csi_anchors,
        annotate_scale=annotate_scale,
        side_by_side=side_by_side,
    )
    overlay = draw_quiver(
        frames, ref_xy, disp, masks=masks, reference=reference, style=style
    )
    imageio.mimwrite(str(output_path), list(overlay), fps=fps, loop=0)


def render_tissue_quiver_map(
    frames: np.ndarray,
    output_path: str | Path,
    masks: np.ndarray | None = None,
    *,
    fps: int = 10,
    win_size: int = 31,
    window_len: int = 11,
    poly_order: int = 3,
    grid_step: int = 16,
    arrow_scale: float = 3.0,
    min_magnitude: float = 0.5,
    cmap: str = "turbo",  # 'turbo', 'jet', or 'magma' are great for vector fields
) -> None:
    T, H, W = frames.shape
    raw_flows = np.zeros((T, H, W, 2), dtype=np.float32)
    flow = None

    # 1. Extract raw dense optical flow
    for i in tqdm(range(1, T), desc="Analyzing tissue movement vectors"):
        prev = frames[i - 1]
        current = frames[i]

        flow = cv2.calcOpticalFlowFarneback(
            prev,
            current,
            flow,
            pyr_scale=0.5,
            levels=2,
            winsize=win_size,
            iterations=5,
            poly_n=7,
            poly_sigma=5.0,
            flags=0 if flow is None else cv2.OPTFLOW_USE_INITIAL_FLOW,
        )

        if masks is not None:
            raw_flows[i, ..., 0] = flow[..., 0]

            raw_flows[i, ..., 1] = flow[..., 1]

        else:
            raw_flows[i] = flow

    raw_flows[0] = raw_flows[1]

    # 2. Temporal Filtering on X and Y independently to kill noise
    print("Applying Savitzky-Golay filter to vector field...")
    smoothed_flows = savgol_filter(
        raw_flows,
        window_length=window_len,
        polyorder=poly_order,
        axis=0,
        mode="wrap",
    )

    all_mags = np.sqrt(smoothed_flows[..., 0] ** 2 + smoothed_flows[..., 1] ** 2)
    global_max = np.max(all_mags)

    colormap = plt.get_cmap(cmap)

    # 3. Draw the Quiver Map
    print("Rendering quiver overlay...")
    overlay = np.zeros((T, H, W, 3), dtype=np.uint8)

    for i in tqdm(range(T), desc="Drawing frames"):
        bg = cv2.cvtColor(frames[i], cv2.COLOR_GRAY2BGR)
        flow_field = smoothed_flows[i]

        for y in range(0, H, grid_step):
            for x in range(0, W, grid_step):
                fx = flow_field[y, x, 0]
                fy = flow_field[y, x, 1]

                mag = np.sqrt(fx**2 + fy**2)

                if mag > min_magnitude:
                    # Normalize magnitude to [0.0, 1.0] relative to the global max
                    norm_mag = np.clip(mag / global_max, 0.0, 1.0)

                    # Get RGBA from Matplotlib, convert to BGR for OpenCV
                    rgba = colormap(norm_mag)
                    color_bgr = (
                        int(rgba[2] * 255),  # Blue
                        int(rgba[1] * 255),  # Green
                        int(rgba[0] * 255),  # Red
                    )

                    start_pt = (x, y)
                    end_pt = (int(x + fx * arrow_scale), int(y + fy * arrow_scale))

                    cv2.arrowedLine(
                        bg,
                        start_pt,
                        end_pt,
                        color_bgr,
                        thickness=1,
                        tipLength=0.3,
                        line_type=cv2.LINE_AA,
                    )

        overlay[i] = bg

    imageio.mimwrite(str(output_path), overlay, fps=fps, loop=0)
