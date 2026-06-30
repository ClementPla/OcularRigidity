import numpy as np
from ocularrigidity.registration.sparse_demons import track_points_with_demons
from ocularrigidity.segmentation.postprocess.interfaces import (
    get_masks_contours,
)
from scipy.signal import savgol_filter

import numpy as np
import SimpleITK as sitk
import cv2


def extract_displacement_at_boundaries(
    frames,
    masks,
    reference_frame_idx=0,
    smooth_window=11,
    max_displacement=20,  # Estimated max pixels a boundary moves over the video,
    method="demons",
    lk_window=35,
    lk_levels: int = 3,
):
    """Compute pixel displacement optimized via dynamic ROI cropping."""
    ref_mask = masks[reference_frame_idx]
    ref_contours = get_masks_contours(ref_mask)
    ref_frame = frames[reference_frame_idx]

    T = len(frames)
    N = len(ref_contours)

    p0 = ref_contours.astype(np.float32).reshape(-1, 1, 2)  # (N, 1, 2) in (x, y)
    positions = np.full((T, N, 2), np.nan, dtype=np.float32)
    positions[reference_frame_idx] = p0[:, 0, :]

    sitk_ref = sitk.GetImageFromArray(ref_frame.astype(np.float32))

    # 3. Process video loop
    for t in range(T):
        if t == reference_frame_idx:
            continue

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
                p1, status, _ = cv2.calcOpticalFlowPyrLK(
                    ref_frame, frames[t], p0, None, **lk_params
                )

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
    """Computes the change in choroidal area (delta A) across frames

    using boundary displacement vectors.

    Parameters:
    -----------
    reference_border : np.ndarray
        The baseline polygon boundary coordinates (N, 2) where columns are [row, col].
    displacements : np.ndarray
        The structural displacement array of shape (T, N, 2) tracking [d_row, d_col].

    Returns:
    --------
    delta_A : np.ndarray
        Array of shape (T,) containing the change in area for each frame.
    """
    T, N, _ = displacements.shape
    delta_A = np.zeros(T, dtype=np.float64)

    A_0 = shoelace_area(reference_border)

    for t in range(T):
        # Apply the tracking displacement to the reference skeleton
        deformed_border = reference_border + displacements[t]
        # Drop any points that became NaN due to lost tracking
        deformed_border = deformed_border[~np.isnan(deformed_border).any(axis=1)]
        if len(deformed_border) < 3:
            # Not enough points to form a polygon, skip this frame
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
    """Computes delta A directly using the differential boundary integral

    (u * n) ds along the ordered perimeter. This should give the same result as the shoelace method (up to sign).

    Parameters:
    -----------
    reference_border : np.ndarray
        Ordered baseline coordinates (N, 2) mapped as [row, col] -> [y, x].
    displacements : np.ndarray
        Displacement vectors (T, N, 2) mapped as [d_row, d_col] -> [dy, dx].

    Returns:
    --------
    delta_A : np.ndarray
        Array of shape (T,) containing the structural area variance.
    """
    T, N, _ = displacements.shape
    delta_A = np.zeros(T, dtype=np.float64)

    # Extract baseline positions (y=row, x=col)
    y = reference_border[:, 0]
    x = reference_border[:, 1]

    # Compute segment differentials (vectors joining vertex i to i+1)
    # np.roll(..., -1) gets the index i+1
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
        # The sign automatically adjusts for expansion vs contraction
        segment_area_changes = (dx_valid * avg_dy) - (dy_valid * avg_dx)

        # Sum along the entire closed boundary contour
        # Take absolute value of the sum to stay invariant to clockwise/counter-clockwise ordering
        delta_A[t] = np.sum(segment_area_changes)

    # Adjust sign if reference loop direction is inverted relative to standard coordinate axes
    # We can verify the orientation by checking a simple shoelace sign
    reference_orientation = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    if reference_orientation < 0:
        delta_A = -delta_A

    return delta_A
