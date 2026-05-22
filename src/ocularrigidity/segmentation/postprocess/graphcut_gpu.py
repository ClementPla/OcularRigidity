import torch
import torch.nn.functional as F


def _sobel_y(x: torch.Tensor) -> torch.Tensor:
    """Sobel along H axis. x: (T, H, W), returns (T, H, W)."""
    kernel = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)
    x_pad = F.pad(x.unsqueeze(1), (1, 1, 1, 1), mode="replicate")
    return F.conv2d(x_pad, kernel).squeeze(1)


def _build_cost_batch_torch(
    probs: torch.Tensor, prob_threshold: float = 0.3
) -> torch.Tensor:
    """probs: (T, H, W) → cost: (T, H, W)."""
    grad_y = _sobel_y(probs)
    transition = torch.clamp(-grad_y, min=0)
    max_per_frame = transition.flatten(1).amax(dim=1).clamp(min=1e-6)
    transition = transition / max_per_frame[:, None, None]
    cost = 1.0 - transition
    cost = torch.where(probs < prob_threshold, torch.full_like(cost, 1e3), cost)
    return cost


def _shift_h(arr: torch.Tensor, dy: int) -> torch.Tensor:
    """Shift (T, H) along H by dy, filling shifted-in rows with +inf."""
    out = torch.roll(arr, dy, dims=1)
    if dy > 0:
        out[:, :dy] = float("inf")
    elif dy < 0:
        out[:, dy:] = float("inf")
    return out


def _dp_shortest_path_batch_torch(
    cost: torch.Tensor,
    max_step: int = 2,
    lambda_smooth: float = 0.5,
) -> torch.Tensor:
    """
    cost: (T, H, W) float tensor on device
    returns: (T, W) int tensor of y per column per frame
    """
    T, H, W = cost.shape
    device, dtype = cost.device, cost.dtype

    dp = torch.full((T, H, W), float("inf"), dtype=dtype, device=device)
    trace = torch.zeros((T, H, W), dtype=torch.int8, device=device)

    dp[:, :, 0] = cost[:, :, 0]

    steps = list(range(-max_step, max_step + 1))
    steps_tensor = torch.tensor(steps, dtype=torch.int8, device=device)

    for x in range(1, W):
        # Stack all shifted previous-column DPs: (K, T, H)
        shifted = torch.stack(
            [_shift_h(dp[:, :, x - 1], dy) + lambda_smooth * abs(dy) for dy in steps],
            dim=0,
        )
        candidates = shifted + cost[:, :, x].unsqueeze(0)  # (K, T, H)

        best_val, best_dy_idx = candidates.min(dim=0)  # (T, H)
        dp[:, :, x] = best_val
        trace[:, :, x] = steps_tensor[best_dy_idx]

    # Backtrack, vectorized across T
    y = torch.zeros((T, W), dtype=torch.long, device=device)
    y[:, -1] = dp[:, :, -1].argmin(dim=1)

    # Gather trace along y at each x step, walking backward
    for x in range(W - 1, 0, -1):
        dy = trace[torch.arange(T, device=device), y[:, x], x].to(torch.long)
        y[:, x - 1] = y[:, x] - dy

    return y


def _gaussian_kernel_1d(sigma: float, device, dtype) -> torch.Tensor:
    radius = max(1, int(round(3 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x**2) / (2 * sigma**2))
    return k / k.sum()


def _temporal_smooth_y(y: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian-smooth a (T, W) integer path along T, per column. Returns float (T, W)."""
    if sigma <= 0:
        return y.float()
    kernel = _gaussian_kernel_1d(sigma, y.device, torch.float32).view(1, 1, -1)
    pad = kernel.shape[-1] // 2
    # Treat W as batch, T as the spatial axis
    yf = y.float().T.unsqueeze(1)  # (W, 1, T)
    yf = F.pad(yf, (pad, pad), mode="replicate")
    yf = F.conv1d(yf, kernel)  # (W, 1, T)
    return yf.squeeze(1).T  # (T, W)


def _dp_with_temporal_anchor(
    cost: torch.Tensor,
    anchor: torch.Tensor,
    shadow_col: torch.Tensor,
    mu_temporal: float,
    max_step: int,
    lambda_smooth: float,
) -> torch.Tensor:
    """
    Re-solve the CSI DP with a temporal anchor term.

    cost:        (T, H, W) original cost volume (with 1e3 walls intact).
    anchor:      (T, W) float, target y per (t, x), typically a temporally
                 smoothed previous solution.
    shadow_col:  (T, W) bool. On True columns, the data term is replaced by
                 the anchor term so the 1e3 wall does not drown out the prior.
                 On False columns, the anchor term is added to the data term.
    """
    T, H, W = cost.shape
    y_grid = torch.arange(H, device=cost.device, dtype=cost.dtype).view(1, H, 1)
    anchor_b = anchor.to(cost.dtype).unsqueeze(1)  # (T, 1, W)
    temporal = mu_temporal * (y_grid - anchor_b).abs()  # (T, H, W)

    augmented = torch.where(
        shadow_col.unsqueeze(1),
        temporal,  # shadow: anchor only
        cost + temporal,  # otherwise: data + anchor
    )
    return _dp_shortest_path_batch_torch(
        augmented, max_step=max_step, lambda_smooth=lambda_smooth
    )


def graphcut_masks_from_probs_batch_torch(
    probs: torch.Tensor,
    max_step: int = 2,
    lambda_smooth: float = 0.5,
    prob_threshold: float = 0.3,
    bm_threshold: float = 0.5,
    *,
    temporal_smooth: bool = False,
    temporal_sigma: float = 2.0,
    temporal_mu: float = 0.3,
    temporal_iterations: int = 1,
) -> torch.Tensor:
    """
    probs: (T, H, W) float tensor on GPU (sigmoid output)
    returns: (T, H, W) bool tensor on GPU

    Temporal smoothing (off by default) refines only the CSI by re-solving
    the DP with an anchor term derived from a temporally smoothed first pass.
    On columns where the entire A-line is below `prob_threshold` (treated as
    shadowed), the data term is dropped so the temporal prior can drive the
    path. BM is unchanged.

    Temporal kwargs:
        temporal_smooth:     master on/off switch.
        temporal_sigma:      Gaussian sigma (in frames) for smoothing the
                             anchor along T. Keep well below the cardiac
                             period to preserve pulsation.
        temporal_mu:         weight of the |y - anchor| anchor term, in the
                             same units as the [0, 1] data term.
        temporal_iterations: number of anchor-refinement passes.
    """
    T, H, W = probs.shape
    device = probs.device

    # BM: first row above threshold per column
    above = probs > bm_threshold
    bm = above.to(torch.int32).argmax(dim=1)  # (T, W)
    has_mask = above.any(dim=1)  # (T, W)
    bm = torch.where(has_mask, bm, torch.full_like(bm, H))

    # Cost volume
    cost = _build_cost_batch_torch(probs, prob_threshold=prob_threshold)

    # Forbid positions above BM + 2
    y_grid = torch.arange(H, device=device).view(1, H, 1)
    bm_b = bm.unsqueeze(1)
    above_bm = y_grid < (bm_b + 2)
    cost = torch.where(above_bm, torch.full_like(cost, 1e3), cost)

    # CSI shortest path: per-frame baseline
    csi = _dp_shortest_path_batch_torch(
        cost, max_step=max_step, lambda_smooth=lambda_smooth
    )

    # Optional temporal refinement
    if temporal_smooth and T > 1 and temporal_iterations > 0:
        shadow_col = (probs < prob_threshold).all(dim=1)  # (T, W)
        for _ in range(temporal_iterations):
            anchor = _temporal_smooth_y(csi, temporal_sigma)
            csi = _dp_with_temporal_anchor(
                cost,
                anchor=anchor,
                shadow_col=shadow_col,
                mu_temporal=temporal_mu,
                max_step=max_step,
                lambda_smooth=lambda_smooth,
            )

    csi = torch.maximum(csi, bm + 2)

    # Rebuild mask
    csi_b = csi.unsqueeze(1)
    mask = (y_grid >= bm_b) & (y_grid <= csi_b)
    mask = mask & has_mask.unsqueeze(1)
    return mask
