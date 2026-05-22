


import argparse
import os
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm

from ocularrigidity.registration.demons import (
    LogDemonsConfig,
    _central_gradient,
    compose,
    log_demons,
    warp,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class VideoRegistrationConfig:
    # Registration per frame-pair (used with warm-start, so keep iterations modest)
    demons: LogDemonsConfig = field(default_factory=lambda: LogDemonsConfig(
        iterations=(60, 40, 20),
        sigma_fluid=1.0,
        sigma_elastic=0.5,
        alpha=0.5,
        use_esm=True,
        diffeomorphic=True,
        bch_order=1,
        pyramid_levels=3,
        stop_rel_tol=1e-4,
        verbose=False,
    ))
    # Video-level
    cyclic: Literal["yes", "no", "diagnostic"] = "yes"
    # "yes"        : apply linear closure correction across the composed fields.
    # "no"         : skip closure diagnostics entirely.
    # "diagnostic" : compute drift map and RMS but do NOT correct the fields.
    primary_rigidity: Literal["strain", "affine", "motion"] = "strain"
    affine_patch: int = 9          # patch radius (pixels) for local affine fit
    verbose: bool = True


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class VideoRegistrationResult:
    # Per-pair fields, in voxel units, shape (N-1, 2, H, W):
    #   phi_pair[t] = displacement from frame t to frame t+1
    #   velocities_pair[t] = stationary velocity whose exp equals phi_pair[t]
    phi_pair: np.ndarray
    velocities_pair: np.ndarray
    # Composed fields (N, 2, H, W): phi_0_to_t[t] maps frame 0 -> frame t.
    # Index 0 is the zero field (frame 0 to itself).
    displacement_0_to_t: np.ndarray
    # Rigidity maps, each (H, W)
    rigidity_maps: dict
    # Closure diagnostics
    drift_map: np.ndarray           # (H, W) per-pixel drift magnitude
    closure_residual_rms: float     # scalar, pixels
    # Timing
    seconds_total: float
    seconds_per_pair: float


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_tensor(video: np.ndarray, device: str, dtype: torch.dtype) -> Tensor:
    """(N, H, W) np.ndarray -> (N, 1, H, W) tensor, normalized to [0, 1]."""
    if video.ndim != 3:
        raise ValueError(f"expected (N, H, W), got {video.shape}")
    arr = video.astype(np.float32, copy=False)
    # Normalize by global max to [0,1]. If already in [0,1] this is a no-op-ish.
    m = arr.max()
    if m > 1.5:   # looks like uint / raw intensity
        arr = arr / (m + 1e-12)
    t = torch.from_numpy(arr).to(device=device, dtype=dtype)
    return t.unsqueeze(1)    # (N, 1, H, W)


# ---------------------------------------------------------------------------
# Sequential registration with warm-start
# ---------------------------------------------------------------------------

def sequential_register(frames: Tensor,
                        cfg: VideoRegistrationConfig) -> tuple[Tensor, Tensor]:
    """
    Register frame t to frame t+1 for t = 0 .. N-2, warm-starting each
    registration from the previous frame pair's velocity.

    frames : (N, 1, H, W) tensor
    Returns
    -------
    phi_pair       : (N-1, 2, H, W) displacement fields (voxel units)
    velocities_pair: (N-1, 2, H, W) stationary velocities
    """
    N = frames.shape[0]
    H, W = frames.shape[2], frames.shape[3]
    device, dtype = frames.device, frames.dtype

    phi_pair = torch.empty(N - 1, 2, H, W, device=device, dtype=dtype)
    v_pair = torch.empty(N - 1, 2, H, W, device=device, dtype=dtype)

    # We register fixed=frame[t], moving=frame[t+1]. Result `disp` satisfies
    #     frame[t](x) ≈ frame[t+1](x + disp(x)),
    # i.e. "for point x in frame t, its location in frame t+1 is x + disp(x)".
    # That is the FORWARD trajectory direction we want.
    v_prev = None
    for t in tqdm(range(N - 1), desc="Registering frames"):
        fixed  = frames[t:t + 1]          # (1,1,H,W)
        moving = frames[t + 1:t + 2]
        _, disp, v = log_demons(fixed, moving, cfg.demons, v_init=v_prev)
        phi_pair[t] = disp[0]
        v_pair[t] = v[0]
        v_prev = v        # warm-start next pair
        if cfg.verbose and (t % max(1, (N - 1) // 20) == 0 or t == N - 2):
            max_disp = disp.norm(dim=1).max().item()
            print(f"  pair {t:4d} -> {t+1:<4d}  |disp|_max = {max_disp:5.2f} vox")
    return phi_pair, v_pair


# ---------------------------------------------------------------------------
# Composition of sequential fields
# ---------------------------------------------------------------------------

def compose_sequence(phi_pair: Tensor) -> Tensor:
    """
    Given per-pair displacement fields phi_pair[t] = phi_{t -> t+1}, return
    phi_0_to_t[t] = phi_{0 -> t} for t = 0 .. N-1, where:
        phi_{0 -> 0} = 0
        phi_{0 -> t+1} = phi_{t -> t+1} ∘ phi_{0 -> t}

    Semantics: a point x in frame 0 is at position x + phi_{0->t}(x) in frame t.

    phi_pair : (N-1, 2, H, W)
    Returns  : (N, 2, H, W)
    """
    N_minus_1 = phi_pair.shape[0]
    H, W = phi_pair.shape[2], phi_pair.shape[3]
    device, dtype = phi_pair.device, phi_pair.dtype

    out = torch.zeros(N_minus_1 + 1, 2, H, W, device=device, dtype=dtype)
    cur = out[0:1]                        # (1,2,H,W), zero field
    for t in range(N_minus_1):
        step = phi_pair[t:t + 1]          # phi_{t -> t+1}
        cur = compose(step, cur)          # u∘v: apply 'step' outside 'cur'
        out[t + 1] = cur[0]
    return out


# ---------------------------------------------------------------------------
# Cycle closure
# ---------------------------------------------------------------------------

def apply_closure_correction(phi_0_to_t: Tensor) -> tuple[Tensor, Tensor, float]:
    """
    Redistribute the cycle-closure residual linearly over time so that
    phi_{0 -> N-1} becomes the identity (assumes the video spans exactly one
    cycle or an integer number of cycles).

    Returns corrected field, drift map (before correction), and RMS drift.
    """
    N = phi_0_to_t.shape[0]
    drift = phi_0_to_t[-1]                           # (2, H, W)
    drift_mag = drift.norm(dim=0)                    # (H, W)
    rms = drift_mag.pow(2).mean().sqrt().item()

    # Linear redistribution: subtract (t / (N-1)) * drift from each frame.
    t = torch.arange(N, device=phi_0_to_t.device, dtype=phi_0_to_t.dtype)
    weights = (t / max(N - 1, 1)).view(N, 1, 1, 1)
    corrected = phi_0_to_t - weights * drift.unsqueeze(0)
    return corrected, drift_mag, rms


def closure_diagnostic(phi_0_to_t: Tensor) -> tuple[Tensor, float]:
    drift = phi_0_to_t[-1]
    drift_mag = drift.norm(dim=0)
    rms = drift_mag.pow(2).mean().sqrt().item()
    return drift_mag, rms


# ---------------------------------------------------------------------------
# Rigidity maps
# ---------------------------------------------------------------------------

def strain_map(phi_0_to_t: Tensor) -> Tensor:
    """
    Time-averaged Frobenius norm of the Green-Lagrange strain tensor,
    E = 0.5 (F^T F - I), with F = I + grad(u). u = phi_0_to_t.

    phi_0_to_t : (N, 2, H, W) displacement fields (voxel units)
    Returns    : (H, W) rigidity map. Low value = locally rigid.

    We skip t=0 (zero field gives zero strain trivially) to avoid biasing
    the mean toward zero.
    """
    N = phi_0_to_t.shape[0]
    acc = torch.zeros(phi_0_to_t.shape[2:], device=phi_0_to_t.device,
                      dtype=phi_0_to_t.dtype)
    count = 0
    for t in range(1, N):
        u = phi_0_to_t[t:t + 1]                        # (1, 2, H, W)
        # _central_gradient on a 2-channel field gives (1, 4, H, W):
        #   [du0/dx, du0/dy, du1/dx, du1/dy]
        # Here channel 0 of u is x-displacement, channel 1 is y-displacement.
        g = _central_gradient(u)[0]                    # (4, H, W)
        dux_dx, dux_dy, duy_dx, duy_dy = g[0], g[1], g[2], g[3]
        # F = I + grad(u):
        F00 = 1.0 + dux_dx
        F01 =       dux_dy
        F10 =       duy_dx
        F11 = 1.0 + duy_dy
        # C = F^T F
        C00 = F00 * F00 + F10 * F10
        C01 = F00 * F01 + F10 * F11
        C11 = F01 * F01 + F11 * F11
        # E = 0.5 (C - I)
        E00 = 0.5 * (C00 - 1.0)
        E01 = 0.5 * C01
        E11 = 0.5 * (C11 - 1.0)
        # ||E||_F  (symmetric, so include off-diagonal twice)
        E_frob = torch.sqrt(E00 * E00 + 2.0 * E01 * E01 + E11 * E11 + 1e-12)
        acc = acc + E_frob
        count += 1
    return acc / max(count, 1)


def motion_map(phi_0_to_t: Tensor) -> Tensor:
    """Std over time of |phi_{0->t}|. Low = stationary (not the same as rigid)."""
    mag = phi_0_to_t.norm(dim=1)          # (N, H, W)
    return mag.std(dim=0)


def affine_residual_map(phi_0_to_t: Tensor, patch_radius: int = 9) -> Tensor:
    """
    For each pixel x, fit an affine model of the trajectory
        phi_{0->t}(y) ≈ A_t (y - x) + b_t        for y in a patch around x
    and report the time-averaged fitting residual at the center pixel.

    Low residual = the patch moves as an affine body (rigid + uniform strain).

    Implementation: for each pixel the affine model has 6 parameters per
    dimension-per-frame. The closed-form least-squares solution at each pixel
    reduces to local moments that can be computed with box filters. We use
    box filters (avg_pool) for speed — O(N*H*W) overall.

    phi_0_to_t : (N, 2, H, W)
    Returns    : (H, W)
    """
    N, C, H, W = phi_0_to_t.shape
    assert C == 2
    device, dtype = phi_0_to_t.device, phi_0_to_t.dtype
    r = patch_radius
    k = 2 * r + 1

    # Local coordinates centered at each pixel.
    # For patch around pixel (i, j), we fit:
    #   u(i+di, j+dj, t) ≈ a_t * di + b_t * dj + c_t    (one eq per component)
    # We solve the 3-parameter linear LS per (pixel, component, time)
    # using local sums over the patch:
    #   S_xx = sum di^2, S_yy = sum dj^2, S_xy = sum di*dj
    # Because the patch is symmetric around zero, S_x = S_y = S_xy = 0, so the
    # normal equations decouple into:
    #   a_t = sum(di * u) / S_xx
    #   b_t = sum(dj * u) / S_yy
    #   c_t = sum(u) / n
    # and the residual per patch is sum(u^2) - a*sum(di*u) - b*sum(dj*u) - c*sum(u).

    # Precompute sums that only depend on the patch geometry.
    di = torch.arange(-r, r + 1, device=device, dtype=dtype)
    dj = di.clone()
    S_xx = (di.pow(2).sum() * k).item()   # sum di^2 over patch = k * sum_i di^2
    S_yy = (dj.pow(2).sum() * k).item()
    n_patch = float(k * k)

    # Kernels for box sums weighted by di or dj.
    ii, jj = torch.meshgrid(di, dj, indexing="ij")    # (k, k)
    kernel_1   = torch.ones(1, 1, k, k, device=device, dtype=dtype)
    kernel_di  = ii.view(1, 1, k, k).contiguous()
    kernel_dj  = jj.view(1, 1, k, k).contiguous()

    def boxsum(x, kernel):
        # x: (N, 1, H, W) -> same shape, conv2d with reflect padding
        x_pad = F.pad(x, (r, r, r, r), mode="replicate")
        return F.conv2d(x_pad, kernel)

    total_residual = torch.zeros(H, W, device=device, dtype=dtype)
    count = 0
    # Process each time frame (skip t=0 which is zero).
    for t in range(1, N):
        u = phi_0_to_t[t]                              # (2, H, W)
        for c in range(2):
            x = u[c:c + 1].unsqueeze(0)                # (1, 1, H, W)
            Su   = boxsum(x, kernel_1)                 # sum u
            Sxu  = boxsum(x, kernel_di)                # sum di * u
            Syu  = boxsum(x, kernel_dj)                # sum dj * u
            Su2  = boxsum(x * x, kernel_1)             # sum u^2

            a_coef = Sxu / (S_xx + 1e-12)
            b_coef = Syu / (S_yy + 1e-12)
            c_coef = Su  / n_patch
            # Residual sum of squares over patch:
            rss = Su2 - a_coef * Sxu - b_coef * Syu - c_coef * Su
            total_residual = total_residual + rss[0, 0].clamp(min=0)
            count += 1
    # Per-pixel RMS residual, averaged over time and both components.
    return torch.sqrt(total_residual / (max(count, 1) * n_patch) + 1e-12)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def analyze_video(video: np.ndarray,
                  cfg: VideoRegistrationConfig | None = None,
                  device: str = "cuda",
                  dtype: torch.dtype = torch.float32) -> VideoRegistrationResult:
    """
    Full pipeline: sequential registration + composition + rigidity maps.

    video : (N, H, W) numpy array.
    """
    if cfg is None:
        cfg = VideoRegistrationConfig()
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU (will be slow).")
        device = "cpu"

    t0 = time.time()
    frames = _to_tensor(video, device, dtype)
    N = frames.shape[0]
    if N < 2:
        raise ValueError("need at least 2 frames")

    if cfg.verbose:
        print(f"Video: N={N}, H×W={frames.shape[2]}×{frames.shape[3]}, "
              f"device={device}, dtype={dtype}")
        print("Sequential registration with warm-start:")

    phi_pair, v_pair = sequential_register(frames, cfg)
    t_reg = time.time()

    if cfg.verbose:
        print(f"Composing {N-1} fields into frame-0-to-t trajectories...")
    phi_0_to_t = compose_sequence(phi_pair)              # (N, 2, H, W)

    # Closure handling
    if cfg.cyclic == "yes":
        phi_0_to_t, drift_map, rms = apply_closure_correction(phi_0_to_t)
        if cfg.verbose:
            print(f"Cycle-closure correction applied. Pre-correction RMS drift "
                  f"= {rms:.3f} vox (max = {drift_map.max().item():.3f}). "
                  f"High values indicate regions where sequential composition "
                  f"was least reliable — often the least rigid regions.")
    elif cfg.cyclic == "diagnostic":
        drift_map, rms = closure_diagnostic(phi_0_to_t)
        if cfg.verbose:
            print(f"Closure diagnostic: RMS drift = {rms:.3f} voxels "
                  f"(max = {drift_map.max().item():.3f})")
    else:
        drift_map = torch.zeros(frames.shape[2], frames.shape[3],
                                device=device, dtype=dtype)
        rms = float("nan")

    if cfg.verbose:
        print("Computing rigidity maps...")
    rig_strain = strain_map(phi_0_to_t)
    rig_motion = motion_map(phi_0_to_t)
    rig_affine = affine_residual_map(phi_0_to_t, patch_radius=cfg.affine_patch)

    t_end = time.time()
    seconds_total = t_end - t0
    seconds_per_pair = (t_reg - t0) / max(N - 1, 1)
    if cfg.verbose:
        print(f"Done in {seconds_total:.1f} s "
              f"({seconds_per_pair*1000:.1f} ms / frame-pair).")

    return VideoRegistrationResult(
        phi_pair=phi_pair.detach().cpu().numpy(),
        velocities_pair=v_pair.detach().cpu().numpy(),
        displacement_0_to_t=phi_0_to_t.detach().cpu().numpy(),
        rigidity_maps={
            "strain": rig_strain.detach().cpu().numpy(),
            "motion": rig_motion.detach().cpu().numpy(),
            "affine": rig_affine.detach().cpu().numpy(),
        },
        drift_map=drift_map.detach().cpu().numpy(),
        closure_residual_rms=rms,
        seconds_total=seconds_total,
        seconds_per_pair=seconds_per_pair,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _save_outputs(result: VideoRegistrationResult, out_dir: str,
                  primary: str = "strain"):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "phi_pair.npy"), result.phi_pair)
    np.save(os.path.join(out_dir, "displacement_0_to_t.npy"),
            result.displacement_0_to_t)
    np.save(os.path.join(out_dir, "drift_map.npy"), result.drift_map)
    for name, m in result.rigidity_maps.items():
        np.save(os.path.join(out_dir, f"rigidity_{name}.npy"), m)
    # Symlink-like copy of the primary map under a stable name.
    np.save(os.path.join(out_dir, "rigidity_primary.npy"),
            result.rigidity_maps[primary])

    # Optional: PNG previews if matplotlib is available.
    try:
        import matplotlib.pyplot as plt
        for name, m in result.rigidity_maps.items():
            fig, ax = plt.subplots(figsize=(5, 5))
            im = ax.imshow(m, cmap="magma")
            title = f"Rigidity map ({name})\nlower = more rigid (by this metric)"
            if name == primary:
                title += "  [PRIMARY]"
            ax.set_title(title)
            plt.colorbar(im, ax=ax, fraction=0.046)
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, f"rigidity_{name}.png"), dpi=120)
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(result.drift_map, cmap="viridis")
        ax.set_title(f"Cycle-closure drift (RMS={result.closure_residual_rms:.2f} vox)")
        plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "drift_map.png"), dpi=120)
        plt.close(fig)
    except ImportError:
        pass


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="Path to .npy file, shape (N, H, W).")
    p.add_argument("--out", default="rigidity_out", help="Output directory.")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--cyclic", default="yes",
                   choices=["yes", "no", "diagnostic"],
                   help="Default 'yes' assumes the video spans an integer "
                        "number of cycles and applies linear closure correction.")
    p.add_argument("--primary", default="strain",
                   choices=["strain", "affine", "motion"],
                   help="Which rigidity map to treat as the primary output.")
    p.add_argument("--affine-patch", type=int, default=9)
    p.add_argument("--iters", type=int, nargs=3, default=[60, 40, 20],
                   help="Iterations per pyramid level (coarse to fine).")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    args = p.parse_args()

    video = np.load(args.input)
    cfg = VideoRegistrationConfig(
        demons=LogDemonsConfig(iterations=tuple(args.iters)),
        cyclic=args.cyclic,
        primary_rigidity=args.primary,
        affine_patch=args.affine_patch,
    )
    dtype = {"float32": torch.float32, "float16": torch.float16}[args.dtype]
    result = analyze_video(video, cfg=cfg, device=args.device, dtype=dtype)
    _save_outputs(result, args.out, primary=args.primary)

    prim = result.rigidity_maps[args.primary]
    print(f"\nOutputs written to {args.out}/")
    print(f"Primary rigidity map: {args.primary}")
    print(f"  shape  : {prim.shape}")
    print(f"  range  : [{prim.min():.4g}, {prim.max():.4g}]")
    print(f"  median : {np.median(prim):.4g}   (lower = more rigid)")


if __name__ == "__main__":
    _cli()


if __name__ == "__main__":
    _cli()