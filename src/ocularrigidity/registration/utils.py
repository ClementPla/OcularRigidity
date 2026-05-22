from typing import Literal, Tuple

import torch


def _interp1d_at(ref: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Linear interp of a 1D signal at fractional positions.

    NaN where pos is out-of-bounds or where either neighbour is NaN.

    Args:
        ref: (W,) signal, may contain NaN.
        pos: any shape, fractional positions in pixel units.
    """
    W = ref.shape[0]
    in_bounds = (pos >= 0) & (pos <= W - 1)
    p = torch.nan_to_num(pos.clamp(0.0, float(W - 1)), nan=0.0)
    i0 = torch.floor(p).long()
    i1 = (i0 + 1).clamp(max=W - 1)
    frac = p - i0.to(p.dtype)
    v0 = ref[i0]
    v1 = ref[i1]
    out = v0 * (1.0 - frac) + v1 * frac
    bad = (~in_bounds) | torch.isnan(v0) | torch.isnan(v1)
    return torch.where(bad, out.new_full((), float("nan")), out)


def _interp1d_batch_at(ref: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Linear interp of per-batch 1D curves at fractional positions.

    Args:
        ref: (B, W) batched 1D signals.
        pos: (B, ...) fractional positions in pixel units. Any rank ≥ 1 after
            the batch dim is allowed (e.g. (B, W), (B, H, W), (B, N, H, W)).

    Returns:
        Tensor with the same shape as ``pos``.
    """
    if ref.ndim != 2:
        raise ValueError(f"ref must be 2D (B, W); got shape {tuple(ref.shape)}")
    if pos.ndim < 2 or pos.shape[0] != ref.shape[0]:
        raise ValueError(
            f"pos must start with B={ref.shape[0]}; got shape {tuple(pos.shape)}"
        )

    B, W = ref.shape
    p = pos.clamp(0.0, float(W - 1))
    i0 = torch.floor(p).long()
    i1 = (i0 + 1).clamp(max=W - 1)
    frac = p - i0.to(p.dtype)

    # Flatten trailing dims to a single sample axis, gather from (B, W), reshape back.
    i0f = i0.reshape(B, -1)
    i1f = i1.reshape(B, -1)
    v0 = torch.gather(ref, 1, i0f).reshape(pos.shape)
    v1 = torch.gather(ref, 1, i1f).reshape(pos.shape)
    return v0 * (1.0 - frac) + v1 * frac


def _solve_y_linear_batched(
    y_src: torch.Tensor,  # (T, W)
    x_grid: torch.Tensor,  # (W,)
    y_dst: torch.Tensor,  # (T, W)
    valid: torch.Tensor,  # (T, W) bool
    *,
    min_pts: int,
    fit_a: bool,
    fit_b: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-frame LS fit of  y_dst ≈ a·y_src + b·x + c  via normal equations.

    Fixed parameters are moved to the LHS:
        fit_a=False  =>  target = y_dst - y_src       (a is fixed at 1)
        fit_b=False  =>  drop x column                (b is fixed at 0)
    """
    T, W = y_src.shape
    dtype, device = y_src.dtype, y_src.device

    # LS target with fixed coefficients moved to the LHS.
    target = y_dst if fit_a else (y_dst - y_src)

    # Design columns whose coefficient is being fit.
    cols = []
    if fit_a:
        cols.append(y_src)
    if fit_b:
        cols.append(x_grid.expand(T, W))
    cols.append(torch.ones(T, W, dtype=dtype, device=device))
    A = torch.stack(cols, dim=-1)  # (T, W, K)
    K = A.shape[-1]

    # Mask invalid rows (zero-contribution to the normal equations).
    mask = valid.unsqueeze(-1).to(dtype)
    A_m = A * mask
    y_m = torch.where(valid, target, target.new_zeros(())).unsqueeze(-1)  # (T, W, 1)

    # Normal equations: AᵀA · θ = Aᵀy. Tiny ridge for singular frames.
    AtA = A_m.transpose(-1, -2) @ A_m  # (T, K, K)
    Aty = A_m.transpose(-1, -2) @ y_m  # (T, K, 1)
    eye = torch.eye(K, dtype=dtype, device=device).expand(T, K, K)
    AtA_reg = AtA + 1e-8 * eye

    sol_full, info = torch.linalg.solve_ex(AtA_reg, Aty)
    sol = sol_full.squeeze(-1)  # (T, K)
    solver_ok = info == 0

    # Unpack at fixed positions; missing terms get identity values.
    a = torch.ones(T, dtype=dtype, device=device)
    b = torch.zeros(T, dtype=dtype, device=device)
    idx = 0
    if fit_a:
        a = sol[:, idx]
        idx += 1
    if fit_b:
        b = sol[:, idx]
        idx += 1
    c = sol[:, idx]

    n = valid.sum(dim=-1)
    enough = n >= min_pts
    finite = torch.isfinite(a) & torch.isfinite(b) & torch.isfinite(c)
    ok = enough & finite & solver_ok

    a = torch.where(ok, a, a.new_ones(()))
    b = torch.where(ok, b, b.new_zeros(()))
    c = torch.where(ok, c, c.new_zeros(()))
    return a, b, c, ok
