"""
Log-Demons image registration in PyTorch.

Supports 2D and 3D, additive and BCH (diffeomorphic) updates, fluid + elastic
regularization, multi-resolution pyramid. Designed to stay on GPU end-to-end:
no Python-level loops over pixels/voxels, only over iterations and pyramid
levels. Written to be torch.compile-friendly.

Conventions
-----------
- Images:     (B, 1, *S) where *S is (H, W) or (D, H, W).
- Velocities: (B, ndim, *S), in VOXEL units (pixels). This is the natural unit
  for warping: phi(x) = x + u(x) where u is also in voxels.
- grid_sample wants normalized coords in [-1, 1], so we convert at the edges
  only. This keeps the math in voxel space where it is interpretable and
  avoids repeatedly re-normalizing.
- Coordinate ordering inside velocity tensors is (x, y) for 2D and (x, y, z)
  for 3D — i.e., the channel dim indexes spatial axes in the order that
  grid_sample expects for the last axis of its `grid` argument.
  That means: for 2D, v[:, 0] is the column (W) displacement and v[:, 1] is
  the row (H) displacement. For 3D, v[:, 0] = W, v[:, 1] = H, v[:, 2] = D.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Grid utilities
# ---------------------------------------------------------------------------

def _identity_grid(shape: tuple[int, ...], device, dtype) -> Tensor:
    """Identity sampling grid in voxel units, shape (ndim, *S)."""
    axes = [torch.arange(s, device=device, dtype=dtype) for s in shape]
    # meshgrid with 'ij' gives coords in (D,H,W) or (H,W) order; we want the
    # channel axis to be (W, H[, D]) so grid_sample sees (x, y[, z]).
    grids = torch.meshgrid(*axes, indexing="ij")      # each (*S,)
    grid = torch.stack(grids[::-1], dim=0)            # (ndim, *S), reversed to (x,y[,z])
    return grid


def _voxel_to_normalized(grid_vox: Tensor, shape: tuple[int, ...]) -> Tensor:
    """
    Convert (B, ndim, *S) voxel-unit sampling grid to grid_sample's expected
    (B, *S, ndim) normalized grid. shape is the *spatial* shape of the image
    being sampled (D,H,W) or (H,W).

    Normalization uses align_corners=True convention: voxel i in [0, N-1] maps
    to normalized 2*i/(N-1) - 1.
    """
    ndim = len(shape)
    # Sizes in (x, y[, z]) order to match channel order of grid_vox.
    sizes = list(shape[::-1])
    scales = torch.tensor(
        [2.0 / max(s - 1, 1) for s in sizes],
        device=grid_vox.device, dtype=grid_vox.dtype,
    ).view(1, ndim, *([1] * ndim))
    normed = grid_vox * scales - 1.0
    # Move channel axis to the end: (B, ndim, *S) -> (B, *S, ndim)
    perm = (0, *range(2, 2 + ndim), 1)
    return normed.permute(perm).contiguous()


def warp(image: Tensor, disp: Tensor, mode: str = "bilinear") -> Tensor:
    """
    Warp `image` by displacement field `disp` (voxel units).
    phi(x) = x + disp(x); sampled at phi(x).
    """
    B = image.shape[0]
    spatial = image.shape[2:]
    ndim = len(spatial)
    assert disp.shape == (B, ndim, *spatial), \
        f"disp {tuple(disp.shape)} vs image {tuple(image.shape)}"

    idgrid = _identity_grid(spatial, image.device, disp.dtype)    # (ndim, *S)
    phi_vox = idgrid.unsqueeze(0) + disp                          # (B, ndim, *S)
    grid = _voxel_to_normalized(phi_vox, spatial)
    return F.grid_sample(image, grid, mode=mode,
                         padding_mode="border", align_corners=True)


def compose(u: Tensor, v: Tensor) -> Tensor:
    """
    Compose two displacement fields: (u ∘ v)(x) = u(x + v(x)) + v(x).
    Both in voxel units.
    """
    B, ndim = v.shape[0], v.shape[1]
    spatial = v.shape[2:]
    idgrid = _identity_grid(spatial, v.device, v.dtype)
    phi = idgrid.unsqueeze(0) + v
    grid = _voxel_to_normalized(phi, spatial)
    u_at_phi = F.grid_sample(u, grid, mode="bilinear",
                             padding_mode="border", align_corners=True)
    return u_at_phi + v


# ---------------------------------------------------------------------------
# Scaling and squaring: phi = exp(v)
# ---------------------------------------------------------------------------

def exp_velocity(v: Tensor, n_steps: int = 6) -> Tensor:
    """
    Compute the displacement of the diffeomorphism exp(v) from the velocity v,
    via scaling and squaring. Returns a displacement field (same shape as v).

    n_steps of squaring: phi = (id + v/2^n) squared n times.
    """
    disp = v / (2.0 ** n_steps)
    for _ in range(n_steps):
        disp = compose(disp, disp)
    return disp


# ---------------------------------------------------------------------------
# Gradients and Demons forces
# ---------------------------------------------------------------------------

def _central_gradient(field: Tensor) -> Tensor:
    """
    Central-difference spatial gradient, computed via replicate-padding so
    that boundary samples get a one-sided (half-magnitude) estimate. Works
    for any channel count C.

    Input : (B, C, *S). Output : (B, C * ndim, *S) stacked as
            [grad_c0_axis_x, grad_c0_axis_y, ..., grad_cN_axis_z].

    For a single-channel image (C=1), the output has shape (B, ndim, *S) with
    channel order (x, y[, z]) — matching the velocity field convention.
    """
    B, C = field.shape[:2]
    ndim = field.ndim - 2
    out_channels = []
    # F.pad order: last dim first. pad[2*i], pad[2*i+1] pad dim (ndim-1-i).
    for c in range(ndim):                        # c = 0 -> x, 1 -> y, 2 -> z
        axis_from_end = c                        # x is last axis, y second-to-last, ...
        pad = [0] * (2 * ndim)
        pad[2 * axis_from_end]     = 1
        pad[2 * axis_from_end + 1] = 1
        padded = F.pad(field, pad, mode="replicate")
        # Build slicer for "shifted +1" and "shifted -1" along the axis.
        axis = field.ndim - 1 - c                # actual tensor axis
        slicer_fwd = [slice(None)] * padded.ndim
        slicer_bwd = [slice(None)] * padded.ndim
        slicer_fwd[axis] = slice(2, None)        # drop 2 from the front
        slicer_bwd[axis] = slice(None, -2)       # drop 2 from the back
        g = 0.5 * (padded[tuple(slicer_fwd)] - padded[tuple(slicer_bwd)])
        out_channels.append(g)
    # Interleave so that channel order is (c0_x, c0_y, ..., c1_x, c1_y, ...)
    # which for C=1 collapses to (x, y[, z]).
    stacked = torch.stack(out_channels, dim=2)   # (B, C, ndim, *S)
    return stacked.view(B, C * ndim, *field.shape[2:])


def demons_force(fixed: Tensor,
                 moving_warped: Tensor,
                 grad_fixed: Tensor,
                 use_esm: bool = True,
                 alpha: float = 0.5,
                 eps: float = 1e-6) -> Tensor:
    """
    Compute the Demons update field u (voxel units).

    ESM (symmetric, Vercauteren et al.):
        u = -(I_m - I_f) * (∇I_m + ∇I_f) / (|∇I_m + ∇I_f|^2 + alpha^2 (I_m - I_f)^2 + eps)

    Additive (Thirion):
        u = -(I_m - I_f) * ∇I_f / (|∇I_f|^2 + alpha^2 (I_m - I_f)^2 + eps)

    `alpha` controls the max step length; ~0.5 means |u| ≲ 1/alpha = 2 vox.
    """
    diff = moving_warped - fixed                      # (B,1,*S)
    if use_esm:
        grad_moving = _central_gradient(moving_warped)
        grad = grad_fixed + grad_moving
    else:
        grad = grad_fixed
    grad_sq = grad.pow(2).sum(dim=1, keepdim=True)    # (B,1,*S)
    denom = grad_sq + (alpha ** 2) * diff.pow(2) + eps
    u = -(diff * grad) / denom                        # (B,ndim,*S)
    return u


# ---------------------------------------------------------------------------
# Gaussian smoothing (separable)
# ---------------------------------------------------------------------------

def _gaussian_kernel_1d(sigma: float, device, dtype, truncate: float = 4.0) -> Tensor:
    if sigma <= 0:
        return torch.ones(1, device=device, dtype=dtype)
    radius = max(1, int(math.ceil(truncate * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def gaussian_smooth(field: Tensor, sigma: float) -> Tensor:
    """
    Separable Gaussian smoothing along all spatial axes.
    `field` is (B, C, *S). Applies the same sigma per axis. Reflect padding.
    """
    if sigma is None or sigma <= 0:
        return field
    B, C = field.shape[:2]
    spatial = field.shape[2:]
    ndim = len(spatial)
    kernel = _gaussian_kernel_1d(sigma, field.device, field.dtype)
    ksize = kernel.numel()
    pad = ksize // 2

    conv = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[ndim]

    out = field
    for axis in range(ndim):
        # Build a kernel that is length ksize along `axis` and 1 elsewhere,
        # with groups=C (depthwise).
        shape = [1, 1] + [1] * ndim
        shape[2 + axis] = ksize
        k = kernel.view(*shape).expand(C, 1, *shape[2:]).contiguous()

        pad_list = [0] * (2 * ndim)
        # F.pad pads the last dim first; invert axis order.
        pad_list[2 * (ndim - 1 - axis)]     = pad
        pad_list[2 * (ndim - 1 - axis) + 1] = pad
        out = F.pad(out, pad_list, mode="replicate")
        out = conv(out, k, groups=C)
    return out


# ---------------------------------------------------------------------------
# Log-Demons core
# ---------------------------------------------------------------------------

@dataclass
class LogDemonsConfig:
    iterations: tuple[int, ...] = (100, 50, 25)      # per pyramid level, coarse -> fine
    sigma_fluid: float = 1.0                         # smoothing on update u
    sigma_elastic: float = 0.5                       # smoothing on total velocity v
    alpha: float = 0.5                               # Demons step control
    use_esm: bool = True                             # symmetric (ESM) forces
    diffeomorphic: bool = True                       # BCH update vs additive
    bch_order: int = 1                               # 1: v+u, 2: v+u+0.5[v,u]
    n_scaling_squaring: int = 6                      # steps in exp(v)
    pyramid_levels: int = 3                          # number of pyramid levels
    stop_rel_tol: float = 1e-4                       # relative change to stop early
    verbose: bool = False


def _jacobian_vector(field: Tensor, vec: Tensor) -> Tensor:
    """
    (J_field @ vec) at each voxel.
    field, vec: (B, ndim, *S). Returns (B, ndim, *S).
    J_field is an (ndim x ndim) Jacobian of the vector field `field`; we
    contract it with the ndim-vector `vec` at each voxel.
    """
    ndim = field.shape[1]
    # Gradient of each component of `field` along each spatial axis.
    # partial[c, a] = d field[c] / d x_a    where x_0=x, x_1=y, x_2=z
    out = torch.zeros_like(field)
    for c in range(ndim):
        comp = field[:, c:c + 1]                     # (B,1,*S)
        grad_c = _central_gradient(comp)             # (B,ndim,*S), ordered (x,y[,z])
        out[:, c:c + 1] = (grad_c * vec).sum(dim=1, keepdim=True)
    return out


def bch_update(v: Tensor, u: Tensor, order: int = 1) -> Tensor:
    """
    Baker-Campbell-Hausdorff update:
        order 1:  v <- v + u
        order 2:  v <- v + u + 0.5 [v, u] = v + u + 0.5 (J_v u - J_u v)
    """
    if order <= 1:
        return v + u
    lie = _jacobian_vector(v, u) - _jacobian_vector(u, v)
    return v + u + 0.5 * lie


def _downsample(img: Tensor) -> Tensor:
    ndim = img.ndim - 2
    pool = {2: F.avg_pool2d, 3: F.avg_pool3d}[ndim]
    return pool(img, kernel_size=2, stride=2, ceil_mode=True)


def _upsample_velocity(v: Tensor, size: tuple[int, ...]) -> Tensor:
    """
    Upsample a velocity field to target spatial `size` and rescale magnitudes
    to match the new voxel spacing (velocities are in voxel units).
    """
    ndim = v.ndim - 2
    mode = {2: "bilinear", 3: "trilinear"}[ndim]
    old = v.shape[2:]
    up = F.interpolate(v, size=size, mode=mode, align_corners=True)
    # Rescale each component by new_size/old_size along its corresponding axis.
    # Channel c corresponds to axis (ndim-1-c) (because channel 0 = x = last).
    scales = [size[ndim - 1 - c] / max(old[ndim - 1 - c], 1) for c in range(ndim)]
    s = torch.tensor(scales, device=v.device, dtype=v.dtype).view(1, ndim, *([1] * ndim))
    return up * s


def log_demons(fixed: Tensor,
               moving: Tensor,
               cfg: LogDemonsConfig | None = None,
               v_init: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
    """
    Register `moving` to `fixed` using Log-Demons.

    Args
    ----
    fixed, moving : (B, 1, *S) tensors, same shape, on the same device.
    cfg : LogDemonsConfig (optional).
    v_init : optional (B, ndim, *S) velocity field at full resolution to warm-
             start from. Downsampled to the coarsest pyramid level internally.
             Hugely useful for sequential video registration where motion
             between adjacent frame-pairs is similar.

    Returns
    -------
    warped  : (B, 1, *S) — moving warped into fixed space.
    disp    : (B, ndim, *S) — total displacement field (voxel units), s.t.
              warped(x) = moving(x + disp(x)).
    v       : (B, ndim, *S) — stationary velocity field. exp(v) = disp when
              diffeomorphic=True; otherwise v == disp.
    """
    if cfg is None:
        cfg = LogDemonsConfig()

    assert fixed.shape == moving.shape, "fixed and moving must match"
    ndim = fixed.ndim - 2
    assert ndim in (2, 3), "only 2D/3D supported"

    # Build pyramid (coarse first). Level 0 is coarsest.
    L = cfg.pyramid_levels
    fixed_pyr = [fixed]
    moving_pyr = [moving]
    for _ in range(L - 1):
        fixed_pyr.append(_downsample(fixed_pyr[-1]))
        moving_pyr.append(_downsample(moving_pyr[-1]))
    fixed_pyr.reverse()
    moving_pyr.reverse()

    # Velocity initialized at coarsest level.
    spatial0 = fixed_pyr[0].shape[2:]
    if v_init is None:
        v = torch.zeros(fixed.shape[0], ndim, *spatial0,
                        device=fixed.device, dtype=fixed.dtype)
    else:
        # Resize v_init from full res to coarsest level, rescaling magnitudes.
        # _upsample_velocity uses F.interpolate which handles downsample too.
        v = _upsample_velocity(v_init, spatial0)

    iters = cfg.iterations
    # Broadcast iteration counts if fewer entries than levels.
    if len(iters) < L:
        iters = iters + (iters[-1],) * (L - len(iters))

    for lvl in range(L):
        F_l = fixed_pyr[lvl]
        M_l = moving_pyr[lvl]
        grad_F = _central_gradient(F_l)
        prev_energy = None

        for it in range(iters[lvl]):
            # 1) current diffeomorphism displacement
            if cfg.diffeomorphic:
                disp = exp_velocity(v, n_steps=cfg.n_scaling_squaring)
            else:
                disp = v

            # 2) warp moving and compute force
            warped = warp(M_l, disp)
            u = demons_force(F_l, warped, grad_F,
                             use_esm=cfg.use_esm, alpha=cfg.alpha)

            # 3) fluid-like smoothing on the update
            u = gaussian_smooth(u, cfg.sigma_fluid)

            # 4) compose in log-domain
            if cfg.diffeomorphic:
                v = bch_update(v, u, order=cfg.bch_order)
            else:
                v = v + u

            # 5) elastic-like smoothing on the total velocity
            v = gaussian_smooth(v, cfg.sigma_elastic)

            # 6) monitor (SSD on current level) for optional early stop
            if cfg.stop_rel_tol > 0 or cfg.verbose:
                with torch.no_grad():
                    energy = (warped - F_l).pow(2).mean().item()
                if cfg.verbose:
                    print(f"[lvl {lvl} it {it:4d}] SSD={energy:.6e}")
                if prev_energy is not None and prev_energy > 0:
                    rel = abs(prev_energy - energy) / prev_energy
                    if rel < cfg.stop_rel_tol:
                        break
                prev_energy = energy

        # Upsample velocity to next level
        if lvl < L - 1:
            v = _upsample_velocity(v, fixed_pyr[lvl + 1].shape[2:])

    # Final outputs at full resolution
    disp = exp_velocity(v, n_steps=cfg.n_scaling_squaring) if cfg.diffeomorphic else v
    warped = warp(moving, disp)
    return warped, disp, v


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------

def _demo():
    """Synthetic 2D test: warp a disc image, recover the warp."""
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H = W = 128

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )
    # Moving: a disc + a small bump (gives the algorithm something to lock onto)
    moving = ((xx ** 2 + yy ** 2) < 0.4 ** 2).float()
    moving += 0.5 * torch.exp(-((xx - 0.3) ** 2 + (yy + 0.2) ** 2) / 0.02)
    moving = moving.clamp(0, 1).view(1, 1, H, W)

    # Create a known smooth displacement and apply it to make `fixed`
    true_disp = torch.zeros(1, 2, H, W, device=device)
    true_disp[0, 0] = 5 * torch.exp(-((xx) ** 2 + (yy) ** 2) / 0.3)      # x shift
    true_disp[0, 1] = 3 * torch.sin(3 * xx) * torch.exp(-yy ** 2 / 0.3)  # y shift
    fixed = warp(moving, true_disp)

    cfg = LogDemonsConfig(
        iterations=(80, 60, 40),
        sigma_fluid=1.0,
        sigma_elastic=0.5,
        alpha=0.5,
        use_esm=True,
        diffeomorphic=True,
        bch_order=1,
        verbose=False,
    )
    warped, disp, v = log_demons(fixed, moving, cfg)

    ssd_before = (moving - fixed).pow(2).mean().item()
    ssd_after  = (warped - fixed).pow(2).mean().item()
    print(f"device      : {device}")
    print(f"SSD before  : {ssd_before:.4e}")
    print(f"SSD after   : {ssd_after:.4e}")
    print(f"reduction   : {100 * (1 - ssd_after / ssd_before):.1f}%")
    print(f"|v| max     : {v.norm(dim=1).max().item():.3f} vox")
    print(f"|disp| max  : {disp.norm(dim=1).max().item():.3f} vox")


if __name__ == "__main__":
    _demo()