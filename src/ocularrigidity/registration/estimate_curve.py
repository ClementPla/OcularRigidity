from warnings import warn

from typing import Optional

import numpy as np
from tqdm.auto import tqdm


from typing import Literal, Tuple, Union

import torch
import torch.nn.functional as F
from ocularrigidity.registration.utils import (
    _interp1d_at,
    _solve_y_linear_batched,
)

warn(
    "The registration.estimate_curve module is deprecated and will likely be removed in a future release. ",
    category=DeprecationWarning,
)


def bbox_fast(mask):
    z_any = mask.any(axis=(1, 2))
    y_any = mask.any(axis=(0, 2))
    x_any = mask.any(axis=(0, 1))
    z_min, z_max = np.where(z_any)[0][[0, -1]]
    y_min, y_max = np.where(y_any)[0][[0, -1]]
    x_min, x_max = np.where(x_any)[0][[0, -1]]
    return z_min, z_max + 1, y_min, y_max + 1, x_min, x_max + 1


def extract_bm_lines_batch(masks):
    """Extract BM lines for all frames at once. Returns a list of (N_t, 2)."""
    y_top = np.argmax(masks, axis=1).astype(np.float32)
    has_mask = masks.any(axis=1)
    x = np.arange(masks.shape[2], dtype=np.float32)
    lines = []
    for t in range(masks.shape[0]):
        hm = has_mask[t]
        lines.append(np.stack([x[hm], y_top[t][hm]], axis=1))
    return lines


def _masked_ncc(src: torch.Tensor, ref: torch.Tensor, min_overlap: int) -> torch.Tensor:
    """NaN-aware NCC along the last dim. -1e30 for degenerate slices."""
    valid = ~(torch.isnan(src) | torch.isnan(ref))
    src_z = torch.where(valid, src, src.new_zeros(()))
    ref_z = torch.where(valid, ref, ref.new_zeros(()))
    n = valid.sum(dim=-1).to(src.dtype)
    n_safe = n.clamp(min=1.0)
    m_src = src_z.sum(dim=-1) / n_safe
    m_ref = ref_z.sum(dim=-1) / n_safe
    ds = torch.where(valid, src - m_src.unsqueeze(-1), src.new_zeros(()))
    dr = torch.where(valid, ref - m_ref.unsqueeze(-1), ref.new_zeros(()))
    num = (ds * dr).sum(dim=-1)
    d_s = (ds * ds).sum(dim=-1)
    d_r = (dr * dr).sum(dim=-1)
    denom = torch.sqrt(d_s.clamp(min=0) * d_r.clamp(min=0))
    bad = (n < min_overlap) | (denom < 1e-12)
    safe_denom = denom.clamp(min=1e-30)
    ncc = num / safe_denom
    return torch.where(bad, num.new_full((), -1e30), ncc)


def _initial_dx_search(
    src: torch.Tensor,
    ref: torch.Tensor,
    max_shift: int,
    min_overlap: int,
) -> torch.Tensor:
    """Per-frame dx (T,) with sub-pixel precision via parabolic peak refinement.

    Convention: x_ref = x + dx, i.e. src[x] ≈ ref[x + dx]. dx > 0 means src
    has features at smaller x than ref.
    """
    T, W = src.shape
    device, dtype = src.device, src.dtype
    x = torch.arange(W, device=device, dtype=dtype)
    S = 2 * max_shift + 1
    scores = src.new_full((T, S), -1e30)
    for k in range(S):
        dx_test = float(k - max_shift)
        ref_s = _interp1d_at(ref, x + dx_test)
        ref_b = ref_s.unsqueeze(0).expand(T, W)
        scores[:, k] = _masked_ncc(src, ref_b, min_overlap)

    best_k = scores.argmax(dim=1)
    dx_int = best_k.to(dtype) - max_shift

    has_room = (best_k > 0) & (best_k < S - 1)
    bk = best_k.clamp(1, S - 2)
    y_m = scores.gather(1, (bk - 1).unsqueeze(1)).squeeze(1)
    y_0 = scores.gather(1, bk.unsqueeze(1)).squeeze(1)
    y_p = scores.gather(1, (bk + 1).unsqueeze(1)).squeeze(1)
    nbrs_ok = (y_m > -1e20) & (y_p > -1e20)
    denom = y_m - 2.0 * y_0 + y_p
    safe = denom.abs() > 1e-12
    delta = 0.5 * (y_m - y_p) / torch.where(safe, denom, torch.ones_like(denom))
    delta = delta.clamp(-1.0, 1.0)
    apply = has_room & nbrs_ok & safe
    dx = torch.where(apply, dx_int + delta, dx_int)

    no_signal = scores.max(dim=1).values <= -1e29
    return torch.where(no_signal, dx.new_zeros(()), dx)


def _grid_search_dx_sx(
    src_corr: torch.Tensor,
    ref: torch.Tensor,
    dx: torch.Tensor,
    sx: torch.Tensor,
    cx: float,
    sx_grid: torch.Tensor,
    dx_offsets: torch.Tensor,
    min_pts: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Loop sx, vectorise over dx_offsets. O(D·T·W) peak memory."""
    T, W = src_corr.shape
    device, dtype = src_corr.device, src_corr.dtype
    x = torch.arange(W, device=device, dtype=dtype)
    S = sx_grid.numel()
    D = dx_offsets.numel()

    best_score = src_corr.new_full((T,), -1e30)
    best_dx = dx.clone()
    best_sx = sx.clone()

    for s_i in range(S):
        sx_t = sx_grid[s_i]
        base = sx_t * (x - cx) + cx + dx.unsqueeze(1)
        x_ref = base.unsqueeze(0) + dx_offsets.view(D, 1, 1)
        ref_s = _interp1d_at(ref, x_ref)
        scores = _masked_ncc(src_corr.unsqueeze(0).expand(D, T, W), ref_s, min_pts)
        s_max, d_argmax = scores.max(dim=0)
        better = s_max > best_score
        best_score = torch.where(better, s_max, best_score)
        best_dx = torch.where(better, dx + dx_offsets[d_argmax], best_dx)
        best_sx = torch.where(better, sx_t.expand_as(best_sx), best_sx)
    return best_dx, best_sx


def _refine_dx_subpixel(
    src_corr: torch.Tensor,
    ref: torch.Tensor,
    dx: torch.Tensor,
    sx: torch.Tensor,
    cx: float,
    min_pts: int,
) -> torch.Tensor:
    """Parabolic sub-pixel refinement of dx at the current (dx, sx)."""
    T, W = src_corr.shape
    device, dtype = src_corr.device, src_corr.dtype
    x = torch.arange(W, device=device, dtype=dtype)
    base = sx.unsqueeze(1) * (x - cx) + cx + dx.unsqueeze(1)  # (T, W)
    offsets = torch.tensor([-1.0, 0.0, 1.0], device=device, dtype=dtype)
    x_ref = base.unsqueeze(0) + offsets.view(3, 1, 1)  # (3, T, W)
    ref_s = _interp1d_at(ref, x_ref)
    scores = _masked_ncc(src_corr.unsqueeze(0).expand(3, T, W), ref_s, min_pts)
    y_m, y_0, y_p = scores[0], scores[1], scores[2]
    denom = y_m - 2.0 * y_0 + y_p
    is_peak = (denom < -1e-12) & (y_m > -1e20) & (y_p > -1e20)
    safe_denom = torch.where(is_peak, denom, denom.new_ones(()))
    delta = (0.5 * (y_m - y_p) / safe_denom).clamp(-1.0, 1.0)
    return dx + torch.where(is_peak, delta, delta.new_zeros(()))


def register_curves_torch(
    bm,
    ref_idx: int,
    transform: Literal["euclidean", "tilt", "similarity", "affine"] = "euclidean",
    *,
    min_pts: int = 10,
    max_shift: int = 50,
    max_sx_stretch: float = 0.05,
    sx_steps: int = 11,
    dx_search_radius: int = 5,
    refine_iters: int = 2,
    device: Union[torch.device, str, None] = None,
    dtype: torch.dtype = torch.float32,
    verbose: bool = False,
    horizontal_alignment: bool = True,
    horizontal_scaling: bool = False,
    log_diagnostics: bool = False,
    log_frames: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """GPU-batched 1D-curve registration.

    Model fitted per frame:
        ref(x_ref) ≈ a·src(x) + b·(x - cx) + c
        x_ref      = sx·(x - cx) + cx + dx

    Output convention (uncentered for downstream warps):
        ref(x_ref) ≈ a·src(x) + b_out·x + c_out
    where b_out = b and c_out = c - b·cx.

    Args:
        bm: (T, W) tensor or array of y-coordinates per frame; NaN allowed.
        ref_idx: index of the reference frame (gets identity params).
        transform: which subset of {dx, sx, a, b, c} is free.
        min_pts: minimum valid overlap to attempt a fit / score an NCC.
        max_shift: half-width of the initial integer-shift NCC scan.
        max_sx_stretch, sx_steps: sx grid over [1 - s, 1 + s], sx_steps points.
        dx_search_radius: per-iter grid search over current dx ± this (int px).
        refine_iters: extra (LS → MAD → re-LS → grid) cycles after the first.
        horizontal_alignment, horizontal_scaling: post-grid overrides; force
            dx=0 or sx=1 regardless of transform.
        log_diagnostics: print per-iter (dx, sx, b) stats.

    Returns:
        params: (T, 5) [(dx, sx, a, b, c)]. Reference frame is identity.
    """
    if not isinstance(bm, torch.Tensor):
        bm = torch.as_tensor(bm)
    if device is None:
        device = (
            bm.device
            if bm.is_cuda
            else (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        )
    bm = bm.to(device=device, dtype=dtype)
    if bm.dim() != 2:
        raise ValueError(f"bm must be (T, W); got {tuple(bm.shape)}")
    T, W = bm.shape
    if not (0 <= ref_idx < T):
        raise IndexError(f"ref_idx {ref_idx} out of range for T={T}")
    cx = (W - 1) * 0.5

    ref = bm[ref_idx]
    src = bm
    x = torch.arange(W, device=device, dtype=dtype)
    x_c = x - cx  # centered design column (decorrelates b and c)

    # 1. Initial dx
    dx = (
        _initial_dx_search(src, ref, max_shift, min_pts)
        if horizontal_alignment
        else torch.zeros(T, device=device, dtype=dtype)
    )
    sx = torch.ones(T, device=device, dtype=dtype)
    a = torch.ones(T, device=device, dtype=dtype)
    b = torch.zeros(T, device=device, dtype=dtype)  # slope about cx
    c = torch.zeros(T, device=device, dtype=dtype)  # offset at cx

    fit_kwargs = {
        "affine": dict(fit_a=True, fit_b=True),
        "tilt": dict(fit_a=False, fit_b=True),
        "similarity": dict(fit_a=False, fit_b=False),
        "euclidean": dict(fit_a=False, fit_b=False),
    }[transform]

    do_grid = transform != "euclidean"
    if do_grid:
        if sx_steps <= 1:
            sx_grid = torch.ones(1, device=device, dtype=dtype)
        else:
            sx_grid = torch.linspace(
                1.0 - max_sx_stretch,
                1.0 + max_sx_stretch,
                sx_steps,
                device=device,
                dtype=dtype,
            )
        dx_offsets = torch.arange(
            -dx_search_radius,
            dx_search_radius + 1,
            device=device,
            dtype=dtype,
        )

    if log_diagnostics:
        if log_frames is None:
            log_frames = tuple(sorted({0, T // 2, T - 1}))
        log_frames = tuple(i for i in log_frames if 0 <= i < T)
        print(
            f"[reg] T={T} W={W} transform={transform} "
            f"ref_idx={ref_idx} dx_radius={dx_search_radius} "
            f"horiz_align={horizontal_alignment} horiz_scale={horizontal_scaling}"
        )
        _log_iter("init", dx, sx, b, log_frames)

    for it in tqdm(
        range(refine_iters + 1),
        disable=not verbose,
        leave=False,
        desc="Refining registration",
    ):
        # --- LS fit (centered: ref_s ≈ a·src + b·x_c + c) ---
        x_ref_cur = sx.unsqueeze(1) * x_c + cx + dx.unsqueeze(1)
        ref_s = _interp1d_at(ref, x_ref_cur)
        valid = (~torch.isnan(src)) & (~torch.isnan(ref_s))

        a_n, b_n, c_n, ok_fit = _solve_y_linear_batched(
            src, x_c, ref_s, valid, min_pts=min_pts, **fit_kwargs
        )
        a = torch.where(ok_fit, a_n, a)
        b = torch.where(ok_fit, b_n, b)
        c = torch.where(ok_fit, c_n, c)

        # --- MAD-based outlier rejection on residuals ---
        x_b = x_c.unsqueeze(0).expand_as(src)
        residuals = ref_s - (
            a.unsqueeze(1) * src + b.unsqueeze(1) * x_b + c.unsqueeze(1)
        )
        res_m = torch.where(valid, residuals, residuals.new_full((), float("nan")))
        med = torch.nanmedian(res_m, dim=-1).values
        med_safe = torch.where(torch.isnan(med), med.new_zeros(()), med)
        abs_dev = (residuals - med_safe.unsqueeze(1)).abs()
        abs_dev_m = torch.where(valid, abs_dev, abs_dev.new_full((), float("nan")))
        mad = torch.nanmedian(abs_dev_m, dim=-1).values
        mad_safe = torch.where(torch.isnan(mad), mad.new_zeros(()), mad)
        thresh = (3.0 * 1.4826 * mad_safe).clamp(min=1e-6)
        valid_inlier = valid & (abs_dev <= thresh.unsqueeze(1))

        a_r, b_r, c_r, ok_re = _solve_y_linear_batched(
            src, x_c, ref_s, valid_inlier, min_pts=min_pts, **fit_kwargs
        )
        a = torch.where(ok_re, a_r, a)
        b = torch.where(ok_re, b_r, b)
        c = torch.where(ok_re, c_r, c)

        # --- Grid search dx, sx (skipped on final iter and for euclidean) ---
        # Model: ref_s ≈ a·src + b·x_c + c. NCC ignores a and c, so under NCC
        # we have ref_s ≈ src + b·x_c. Therefore src_corr := src + b·x_c is
        # what should match the warped ref. This is what breaks the (dx, b)
        # degeneracy without falsely steering dx when real tilt is present:
        # b·x_c is accounted for explicitly, so the remaining shift signal in
        # NCC is uncontaminated by the modeled tilt.
        if (
            it < refine_iters
            and do_grid
            and (horizontal_alignment or horizontal_scaling)
        ):
            if fit_kwargs["fit_b"]:  # tilt or affine
                src_corr = src + b.unsqueeze(1) * x_b
            else:
                src_corr = src
            dx, sx = _grid_search_dx_sx(
                src_corr, ref, dx, sx, cx, sx_grid, dx_offsets, min_pts
            )
            if not horizontal_alignment:
                dx = _refine_dx_subpixel(src_corr, ref, dx, sx, cx, min_pts)
            if not horizontal_scaling:
                sx = torch.ones_like(sx)

        if log_diagnostics:
            _log_iter(f"it{it}", dx, sx, b, log_frames)

    # --- Convert centered (b, c) → uncentered (b_out·x + c_out) for downstream
    # ref(x_ref) ≈ a·src + b·(x - cx) + c  =  a·src + b·x + (c - b·cx)
    c_out = c - b * cx
    b_out = b

    params = torch.stack([dx, sx, a, b_out, c_out], dim=-1)
    params[ref_idx] = params.new_tensor([0.0, 1.0, 1.0, 0.0, 0.0])

    if not fit_kwargs["fit_a"]:
        params[:, 2] = 1.0
    if not fit_kwargs["fit_b"]:
        params[:, 3] = 0.0
    if transform == "euclidean":
        params[:, 1] = 1.0

    if log_diagnostics:
        print(
            f"[reg] done. dx: mean={dx.mean():+.3f} std={dx.std():.3f} "
            f"[{dx.min():+.2f},{dx.max():+.2f}]  "
            f"|b|: mean={b.abs().mean():.4f} max={b.abs().max():.4f}"
        )
    return smooth_params_temporally(params, ref_idx)


def smooth_params_temporally(
    params: torch.Tensor,
    ref_idx: int,
    *,
    dx_sigma: float = 3.0,  # frames; tune to your sampling rate / HR
    median_kernel: int = 5,
    smooth_sx: bool = False,
) -> torch.Tensor:
    """Median-then-Gaussian smooth dx (and optionally sx) across time.

    Median removes per-frame outliers from bad NCC peaks; Gaussian enforces
    the smooth motion prior.
    """
    out = params.clone()
    dx = params[:, 0].detach().cpu().numpy()
    # 1) median filter to kill outliers
    from scipy.signal import medfilt

    dx = medfilt(dx, kernel_size=median_kernel)
    # 2) gaussian filter to enforce smoothness
    from scipy.ndimage import gaussian_filter1d

    dx = gaussian_filter1d(dx, sigma=dx_sigma)
    out[:, 0] = torch.as_tensor(dx, dtype=params.dtype, device=params.device)
    out[ref_idx, 0] = 0.0  # keep ref identity
    if smooth_sx:
        sx = medfilt(params[:, 1].detach().cpu().numpy(), kernel_size=median_kernel)
        sx = gaussian_filter1d(sx, sigma=dx_sigma)
        out[:, 1] = torch.as_tensor(sx, dtype=params.dtype, device=params.device)
        out[ref_idx, 1] = 1.0
    return out


def estimate_dx_from_images(stack, ref_idx, max_shift=50, axis=0):
    """Cross-correlate vertically-collapsed B-scan profiles for dx.
    stack: (T, H, W) float images. Returns (T,) dx in pixels.
    """
    # Collapse y so each frame becomes a 1D x-profile with rich texture
    prof = stack.mean(axis=axis + 1) if stack.ndim == 4 else stack.mean(axis=1)
    prof = torch.as_tensor(prof, dtype=torch.float32)
    # High-pass to kill DC and broad vignetting bias
    prof = prof - prof.mean(dim=-1, keepdim=True)
    return _initial_dx_search(prof, prof[ref_idx], max_shift, min_overlap=10)


def _log_iter(
    tag: str,
    dx: torch.Tensor,
    sx: torch.Tensor,
    b: torch.Tensor,
    frames: Tuple[int, ...],
) -> None:
    dx_c = dx.detach().float().cpu()
    sx_c = sx.detach().float().cpu()
    b_c = b.detach().float().cpu()
    print(
        f"[reg:{tag}] "
        f"dx[mean±std]={dx_c.mean():+.3f}±{dx_c.std():.3f} "
        f"[{dx_c.min():+.2f},{dx_c.max():+.2f}]  "
        f"sx[mean±std]={sx_c.mean():.4f}±{sx_c.std():.4f}  "
        f"|b|[mean,max]={b_c.abs().mean():.4f},{b_c.abs().max():.4f}"
    )
    if frames:
        parts = [
            f"f{i}:(dx={dx_c[i]:+.3f},sx={sx_c[i]:.4f},b={b_c[i]:+.4f})" for i in frames
        ]
        print(f"[reg:{tag}]   " + "  ".join(parts))
