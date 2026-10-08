from pathlib import Path

import numpy as np
from ocularrigidity.registration.sparse_demons import track_points_with_demons
from ocularrigidity.segmentation.postprocess.interfaces import (
    clean_boundaries,
    extract_boundaries_fast,
)
from scipy.signal import savgol_filter

from ocularrigidity.data.compression import read_gray
from ocularrigidity.data.io import load_mask
from ocularrigidity.pipeline_config import DELTA_A, N_CYCLES
from ocularrigidity.segmentation.closing_structures import trim_choroid

import SimpleITK as sitk
import cv2


def reference_boundary_points(mask: np.ndarray) -> np.ndarray:
    """The points to track, ``(N, 2)`` float32 in ``(x, y)``: one on the RPE and
    one on the CSI in every column the mask covers.

    They are read off the per-column boundary curves the thickness measurement
    uses, so every column weighs the same in a mean over the points. A mask
    outline does not have that property: it has as many pixels per column as
    the interface is steep there, and none along a flat run once simplified.

    Ordered as a closed polygon (RPE left to right, then CSI right to left),
    which is what the area computed from them relies on.
    """
    rpe, csi = extract_boundaries_fast(np.asarray(mask, dtype=bool)[None])
    rpe, csi = clean_boundaries(rpe, csi)
    rpe, csi = rpe[0], csi[0]
    cols = np.flatnonzero(np.isfinite(rpe) & np.isfinite(csi))
    if cols.size == 0:
        raise ValueError("Reference mask has no column with both interfaces.")
    return np.concatenate(
        [
            np.stack([cols, rpe[cols]], axis=1),
            np.stack([cols[::-1], csi[cols[::-1]]], axis=1),
        ]
    ).astype(np.float32)


def extract_displacement_at_boundaries(
    frames,
    masks,
    reference_frame_idx=0,
    smooth_window=11,
    max_displacement=40,  # Estimated max pixels a boundary moves over the video,
    method="optical_flow",
    lk_window=35,
    lk_levels: int = 3,
    lk_initial_flow: bool = DELTA_A.lk_initial_flow,
):
    """Compute pixel displacement optimized via dynamic ROI cropping.

    Every frame is tracked from the reference frame. With ``lk_initial_flow``
    the Lucas-Kanade search of each point starts from where the point was last
    found (in the neighbouring frame, walking away from the reference) rather
    than from zero displacement.
    """
    ref_mask = masks[reference_frame_idx]
    ref_contours = reference_boundary_points(ref_mask)
    ref_frame = frames[reference_frame_idx]

    T = len(frames)
    N = len(ref_contours)

    p0 = ref_contours.astype(np.float32).reshape(-1, 1, 2)  # (N, 1, 2) in (x, y)
    positions = np.full((T, N, 2), np.nan, dtype=np.float32)
    positions[reference_frame_idx] = p0[:, 0, :]

    sitk_ref = sitk.GetImageFromArray(ref_frame.astype(np.float32))

    # Walk away from the reference on both sides, so that the frame visited
    # before `t` is always its neighbour on the reference's side.
    order = [
        *range(reference_frame_idx + 1, T),
        *range(reference_frame_idx - 1, -1, -1),
    ]
    guess = p0.copy()  # where each point was last found

    # 3. Process video loop
    for t in order:
        if t == reference_frame_idx - 1:
            guess = p0.copy()  # second side: back at the reference

        match method:
            case "demons":
                # Pass the pre-cropped or bounded regions to the demons tracker
                p1, status = track_points_with_demons(
                    ref_frame=ref_frame,
                    current_frame=frames[t],
                    p0=p0,
                    std_dev=3.0,
                    roi_margin=max_displacement,
                    fixed_image=sitk_ref,
                )

            case "optical_flow":
                lk_params = dict(
                    winSize=(lk_window, lk_window),
                    maxLevel=lk_levels,
                    criteria=(
                        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        30,
                        0.01,
                    ),
                    minEigThreshold=1e-4,
                )
                if lk_initial_flow:
                    # `guess` is the starting point and is overwritten with the
                    # result, hence the copy.
                    p1, status, _ = cv2.calcOpticalFlowPyrLK(
                        ref_frame,
                        frames[t],
                        p0,
                        guess.copy(),
                        flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
                        **lk_params,
                    )
                else:
                    p1, status, _ = cv2.calcOpticalFlowPyrLK(
                        ref_frame, frames[t], p0, None, **lk_params
                    )

        ok = status[:, 0].astype(bool)
        positions[t, ok] = p1[ok, 0, :]
        # A point that was lost keeps its last known position as its guess.
        guess[ok] = p1[ok]

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
                    period=T,
                )
        positions = savgol_filter(
            positions, window_length=smooth_window, polyorder=3, axis=0, mode="wrap"
        )

    p0_xy = p0[:, 0, :]
    disp = positions - p0_xy[None, :, :]
    return disp, p0[:, 0, :]


def shoelace_area(coords):
    # Extract row (y) and col (x) coordinates
    y = coords[:, 0]
    x = coords[:, 1]
    # Shift and cross-multiply
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(np.roll(x, 1), y))


def compute_delta_A_from_displacements(reference_border, displacements):
    """Computes the change in choroidal area (delta A) across frames"""
    T, N, _ = displacements.shape
    delta_A = np.zeros(T, dtype=np.float64)

    A_0 = shoelace_area(reference_border)

    for t in range(T):
        # Apply the tracking displacement to the reference skeleton
        deformed_border = reference_border + displacements[t]
        # Drop any points that became NaN due to lost tracking
        deformed_border = deformed_border[~np.isnan(deformed_border).any(axis=1)]
        if len(deformed_border) < 3:
            delta_A[t] = np.nan
            continue

        # Calculate the new spatial area
        A_t = shoelace_area(deformed_border)

        # 3. Extract the delta scalar value
        delta_A[t] = A_t - A_0

    return delta_A


def compute_minimal_A(reference_border, displacements):
    """Computes the minimal area enclosed by the deformed boundary across frames."""
    T, N, _ = displacements.shape
    min_A = np.inf

    for t in range(T):
        deformed_border = reference_border + displacements[t]
        deformed_border = deformed_border[~np.isnan(deformed_border).any(axis=1)]
        if len(deformed_border) < 3:
            continue
        A_t = shoelace_area(deformed_border)
        if A_t < min_A:
            min_A = A_t

    return min_A


def compute_delta_A_differential(reference_border, displacements):
    """Computes delta A directly using the differential boundary integral"""
    T, N, _ = displacements.shape
    delta_A = np.zeros(T, dtype=np.float64)

    # Extract baseline positions (y=row, x=col)
    y = reference_border[:, 0]
    x = reference_border[:, 1]

    # Compute segment differentials (vectors joining vertex i to i+1)
    dx = np.roll(x, -1) - x
    dy = np.roll(y, -1) - y

    for t in range(T):
        dy_disp = displacements[t, :, 0]
        dx_disp = displacements[t, :, 1]
        # Drop coordinates with NaN displacements (lost tracking)
        valid = ~np.isnan(dy_disp) & ~np.isnan(dx_disp)
        if valid.sum() < 3:
            delta_A[t] = np.nan
            continue
        dy_disp = dy_disp[valid]
        dx_disp = dx_disp[valid]
        dx_valid = dx[valid]
        dy_valid = dy[valid]

        # Calculate average displacement for each segment face
        avg_dy = 0.5 * (dy_disp + np.roll(dy_disp, -1))
        avg_dx = 0.5 * (dx_disp + np.roll(dx_disp, -1))

        # Core cross-product integration: (dx * delta_y) - (dy * delta_x)
        segment_area_changes = (dx_valid * avg_dy) - (dy_valid * avg_dx)

        # Sum along the entire closed boundary contour
        delta_A[t] = np.sum(segment_area_changes)

    # Adjust sign if reference loop direction is inverted relative to standard coordinate axes
    reference_orientation = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    if reference_orientation < 0:
        delta_A = -delta_A

    return delta_A


def extract_displacement(
    video_path: Path = None,
    mask_path: Path = None,
    video: np.ndarray = None,
    mask: np.ndarray = None,
    N_cycles: int = N_CYCLES,
    method=DELTA_A.method,
    smooth_window: int = DELTA_A.smooth_window,
    lk_window: int = DELTA_A.lk_window,
    trim: int = DELTA_A.trim,
    lk_initial_flow: bool = DELTA_A.lk_initial_flow,
):
    if video is None:
        video = read_gray(video_path)
    if mask is None:
        mask = load_mask(mask_path)

    trimmed_masks = trim_choroid(
        mask,
        trim,
    )

    frame_per_cycle = video.shape[0] // N_cycles
    deltaA_per_cycle = []
    minA_per_cycle = []
    displacement_per_cycle = []
    reference_coordinates_per_cycle = []
    for i in range(N_cycles):
        start_frame = i * frame_per_cycle
        end_frame = (i + 1) * frame_per_cycle
        video_cycle = video[start_frame:end_frame]
        mask_cycle = trimmed_masks[start_frame:end_frame]
        displacement, reference_border_coordinates = extract_displacement_at_boundaries(
            video_cycle,
            mask_cycle,
            smooth_window=smooth_window,
            lk_window=lk_window,
            method=method,
            lk_initial_flow=lk_initial_flow,
        )
        delta_a_differential = compute_delta_A_from_displacements(
            reference_border_coordinates, displacement
        )
        deltaA_per_cycle.append(delta_a_differential)
        minA = compute_minimal_A(reference_border_coordinates, displacement)
        minA_per_cycle.append(minA)
        displacement_per_cycle.append(displacement)
        reference_coordinates_per_cycle.append(reference_border_coordinates)
    return (
        deltaA_per_cycle,
        minA_per_cycle,
        displacement_per_cycle,
        reference_coordinates_per_cycle,
    )
