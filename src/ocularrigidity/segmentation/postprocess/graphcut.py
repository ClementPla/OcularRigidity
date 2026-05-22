import numpy as np
from scipy.ndimage import sobel


def _build_cost(probs: np.ndarray, prob_threshold: float = 0.3) -> np.ndarray:
    """
    Build cost image: low cost where CSI (bottom boundary) likely is.

    CSI is where probs transition from high (above, interior) to low (below, sclera),
    i.e. strong negative vertical gradient.
    """
    grad_y = sobel(probs, axis=0)
    transition = np.maximum(-grad_y, 0)
    transition_norm = transition / (transition.max() + 1e-6)
    cost = 1.0 - transition_norm

    # Forbid pixels that are definitely outside the choroid
    cost[probs < prob_threshold] = 1e3

    return cost


def _dp_shortest_path(
    cost: np.ndarray, max_step: int = 2, lambda_smooth: float = 0.0
) -> np.ndarray:
    """
    Dynamic-programming shortest path left→right through cost image.
    Returns y-coordinate per column.
    """
    H, W = cost.shape
    dp = np.full((H, W), np.inf, dtype=np.float32)
    trace = np.zeros((H, W), dtype=np.int8)

    dp[:, 0] = cost[:, 0]

    for x in range(1, W):
        best = np.full(H, np.inf, dtype=np.float32)
        best_dy = np.zeros(H, dtype=np.int8)

        for dy in range(-max_step, max_step + 1):
            shifted = np.roll(dp[:, x - 1], dy)
            if dy > 0:
                shifted[:dy] = np.inf
            elif dy < 0:
                shifted[dy:] = np.inf

            candidate = shifted + cost[:, x] + lambda_smooth * abs(dy)
            better = candidate < best
            best = np.where(better, candidate, best)
            best_dy = np.where(better, dy, best_dy)

        dp[:, x] = best
        trace[:, x] = best_dy

    # Backtrack from min of last column
    y_per_col = np.zeros(W, dtype=np.int32)
    y_per_col[-1] = int(np.argmin(dp[:, -1]))
    for x in range(W - 1, 0, -1):
        dy = trace[y_per_col[x], x]
        y_per_col[x - 1] = y_per_col[x] - dy

    return y_per_col


def graphcut_mask_from_probs(
    probs: np.ndarray,
    max_step: int = 2,
    lambda_smooth: float = 0.5,
    prob_threshold: float = 0.3,
    bm_threshold: float = 0.5,
) -> np.ndarray:
    """
    Given a per-pixel sigmoid probability map of the choroid, find BM and CSI
    via shortest-path, and rebuild a clean binary mask.
    """
    H, W = probs.shape

    # BM (top): first row per column where prob > bm_threshold
    above = probs > bm_threshold
    bm = np.argmax(above, axis=0)
    has_mask = above.any(axis=0)
    bm[~has_mask] = H  # columns without mask → BM at bottom (will give empty col)

    # CSI (bottom): shortest-path through cost image
    cost = _build_cost(probs, prob_threshold=prob_threshold)
    # Force CSI to be below BM by raising cost above each column's BM
    for x in range(W):
        cost[: max(0, bm[x] + 2), x] = 1e3

    csi = _dp_shortest_path(cost, max_step=max_step, lambda_smooth=lambda_smooth)

    # Enforce CSI > BM at each column
    csi = np.maximum(csi, bm + 2)

    # Rebuild the mask
    mask = np.zeros((H, W), dtype=bool)
    y_grid = np.arange(H)[:, None]
    bm_row = bm[None, :]
    csi_row = csi[None, :]
    mask = (y_grid >= bm_row) & (y_grid <= csi_row)
    mask[:, ~has_mask] = False

    return mask
