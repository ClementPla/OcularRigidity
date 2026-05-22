"""
Predict choroid mask evolution from observed vessel masks.
==========================================================

Given:
    mask_choroid  : (H, W) bool, choroid region at t = 0 (rest).
    mask_vessels  : (T, H, W) bool, vessel masks at each timestamp.
                    mask_vessels[0] is the rest configuration.

Output:
    masks_simulated : (T, H, W) bool, predicted choroid masks per frame.

Method
------
Vessel deformation is *observed* (segmented), not parameterized. We therefore
impose the observed per-vessel motion as a Dirichlet BC inside the rest
vessel footprint and let linear elasticity propagate the response through
the surrounding stroma.

Per frame t:
  1. Label connected components in mask_vessels[0] and mask_vessels[t].
  2. Match each rest component to its closest counterpart (centroid distance).
  3. Fit an affine A, c per pair by matching first + second pixel moments
     (mean and covariance). Affine is enough to capture translation, scale
     and shear — the dominant modes for a vessel cross-section in a B-scan.
  4. For every FEM node inside a rest vessel, prescribe
        u(x) = A x + c - x.
  5. BM nodes (top edge of mask_choroid per column) -> u = 0.
     Outside-choroid nodes -> u = 0 (these sit beyond our domain).
  6. Free everywhere else. Solve linear elasticity.
  7. Backward-warp mask_choroid by the resulting displacement field.

Linearity speedup: the SET of constrained DOFs is fixed across frames
(because the rest vessel mask doesn't change), only their VALUES change.
We slice K into K_ff, K_fd once, factor K_ff once, and reuse for every
frame — turning a per-frame O(n^1.5) sparse LU into a single solve plus
cheap back-substitutions.

Image-coordinate convention assumed throughout
----------------------------------------------
* Axes: (row=y, col=x), origin top-left, y increases downward.
* BM is at the top of mask_choroid (smallest y per column).
* CSI is at the bottom of mask_choroid (largest y per column).
* If your data has BM at the bottom instead, flip the masks vertically
  before calling, or set bm_at_top=False.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
from scipy.ndimage import label, map_coordinates
from scipy.interpolate import RegularGridInterpolator

from skfem import MeshTri, Basis, ElementVector, ElementTriP1, BilinearForm, asm
from skfem.helpers import sym_grad, trace, ddot


# ====================================================================
# Helpers
# ====================================================================
def _extract_bm_pixels(mask_choroid, bm_at_top=True):
    """Topmost (or bottommost) True pixel per column."""
    H, W = mask_choroid.shape
    bm = np.zeros_like(mask_choroid)
    has = mask_choroid.any(axis=0)
    cols = np.where(has)[0]
    src = mask_choroid if bm_at_top else mask_choroid[::-1]
    rows = np.argmax(src[:, cols], axis=0)
    if not bm_at_top:
        rows = H - 1 - rows
    bm[rows, cols] = True
    return bm


def _sqrt_psd_2x2(M):
    """Stable principal square root of a 2x2 (near-)PSD matrix."""
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 0.0, None)
    return V @ np.diag(np.sqrt(w)) @ V.T


def _fit_affine_from_components(labels_rest, labels_t, max_track_dist=80.0):
    """
    For each component in `labels_rest`, find the closest component in
    `labels_t` (centroid distance, greedy nearest, no reuse). Fit an affine
    A, c such that  x_t = A @ x_rest + c  matches the first and second
    pixel moments of each pair.

    Why moments? For two point clouds with means mu_r, mu_t and covariances
    Cov_r, Cov_t, the affine that maps the first onto the second by
    matching both is:
            A = sqrt(Cov_t) @ sqrt(Cov_r)^{-1}
            c = mu_t - A @ mu_r.
    No iteration, no correspondences needed -- it's the unique linear-plus-
    translation map that pushes the whitened cloud onto the target cloud.

    Returns
    -------
    transforms : dict {rest_label: (A, c)}
    """
    transforms = {}
    n_rest, n_t = labels_rest.max(), labels_t.max()
    if n_rest == 0 or n_t == 0:
        return transforms

    # Per-component pixel coordinate stacks ((x, y) order)
    rest_pts = {
        L: np.argwhere(labels_rest == L)[:, ::-1].astype(float)
        for L in range(1, n_rest + 1)
    }
    t_pts = {
        L: np.argwhere(labels_t == L)[:, ::-1].astype(float) for L in range(1, n_t + 1)
    }
    rest_mu = {L: P.mean(axis=0) for L, P in rest_pts.items()}
    t_mu = {L: P.mean(axis=0) for L, P in t_pts.items()}

    used = set()
    for Lr, mu_r in rest_mu.items():
        best_Lt, best_d = None, max_track_dist
        for Lt, mu_t in t_mu.items():
            if Lt in used:
                continue
            d = np.linalg.norm(mu_t - mu_r)
            if d < best_d:
                best_d, best_Lt = d, Lt
        if best_Lt is None:
            continue
        used.add(best_Lt)

        Pr, Pt = rest_pts[Lr], t_pts[best_Lt]
        Pcr = Pr - mu_r
        Pct = Pt - t_mu[best_Lt]
        # Tiny ridge to keep covariances strictly PD even for tiny vessels.
        Cov_r = (Pcr.T @ Pcr) / max(len(Pcr), 1) + 1e-3 * np.eye(2)
        Cov_t = (Pct.T @ Pct) / max(len(Pct), 1) + 1e-3 * np.eye(2)

        S_r = _sqrt_psd_2x2(Cov_r)
        S_t = _sqrt_psd_2x2(Cov_t)
        A = S_t @ np.linalg.inv(S_r)
        c = t_mu[best_Lt] - A @ mu_r
        transforms[Lr] = (A, c)
    return transforms


def _backward_warp_mask(mask, ux_pix, uy_pix):
    """
    Warp `mask` from rest to deformed configuration using displacement
    fields (ux_pix, uy_pix) sampled on the rest pixel grid.

    Backward warp: for each output pixel x_def, sample mask at
        x_rest ≈ x_def - u(x_def).
    For small displacements (a few pixels), approximating u(x_def) by
    u(x_rest) at the same pixel location is fine. For larger u, a few
    fixed-point iterations would be needed -- not done here.
    """
    H, W = mask.shape
    yy, xx = np.indices((H, W), dtype=float)
    src_y = yy - uy_pix
    src_x = xx - ux_pix
    out = map_coordinates(
        mask.astype(float), [src_y, src_x], order=1, mode="constant", cval=0.0
    )
    return out > 0.5


# ====================================================================
# Main: forward elasticity simulation
# ====================================================================
def simulate_choroid_pulsation(
    mask_choroid,
    mask_vessels,
    E=5.0e3,
    nu=0.45,
    mesh_step=2,
    bm_at_top=True,
    verbose=False,
):
    """
    See module docstring.

    Parameters
    ----------
    mask_choroid : (H, W) bool
    mask_vessels : (T, H, W) bool
    E, nu        : linear-elastic material parameters
    mesh_step    : pixels between FEM nodes (1 = node-per-pixel, slow but
                   accurate; 2-4 is a good speed/accuracy tradeoff)
    bm_at_top    : True if BM is at the smallest y per column
    """
    T, H, W = mask_vessels.shape

    # --- Lame parameters (plane strain) ---------------------------
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    # --- Mesh: tensor grid covering the image ---------------------
    xs = np.arange(0, W, mesh_step, dtype=float)
    ys = np.arange(0, H, mesh_step, dtype=float)
    if xs[-1] < W - 1:
        xs = np.append(xs, float(W - 1))
    if ys[-1] < H - 1:
        ys = np.append(ys, float(H - 1))
    nx, ny = len(xs), len(ys)

    mesh = MeshTri.init_tensor(xs, ys)
    basis = Basis(mesh, ElementVector(ElementTriP1()))
    n_nodes = mesh.p.shape[1]
    n_dofs = 2 * n_nodes

    # Fast pixel <-> node lookup (each node falls on a pixel).
    j_node = np.clip(np.round(mesh.p[0]).astype(int), 0, W - 1)  # column = x
    i_node = np.clip(np.round(mesh.p[1]).astype(int), 0, H - 1)  # row    = y

    # --- Stiffness ------------------------------------------------
    @BilinearForm
    def stiffness(u, v, w):
        return 2.0 * mu * ddot(sym_grad(u), sym_grad(v)) + lam * trace(
            sym_grad(u)
        ) * trace(sym_grad(v))

    K = asm(stiffness, basis).tocsr()

    # --- Constrained DOF set (constant across frames!) ------------
    inside = mask_choroid[i_node, j_node]
    bm_pixel = _extract_bm_pixels(mask_choroid, bm_at_top=bm_at_top)
    bm_node_mask = bm_pixel[i_node, j_node]
    vrest_node_mask = mask_vessels[0][i_node, j_node]

    dir_node_mask = (~inside) | bm_node_mask | vrest_node_mask
    dir_nodes = np.where(dir_node_mask)[0]
    # Each node has 2 DOFs (ux, uy), interleaved as [ux_0, uy_0, ux_1, ...]
    dir_dofs = np.sort(np.concatenate([2 * dir_nodes, 2 * dir_nodes + 1]))
    free_dofs = np.setdiff1d(np.arange(n_dofs), dir_dofs)

    # --- Slice K into free/constrained blocks and factor once -----
    K_ff = K[free_dofs, :][:, free_dofs].tocsc()
    K_fd = K[free_dofs, :][:, dir_dofs].tocsr()

    if verbose:
        print(f"DOFs total/free/constrained: {n_dofs}/{len(free_dofs)}/{len(dir_dofs)}")
        print("Factorizing K_ff once...")
    K_ff_solve = spla.factorized(K_ff)

    # --- Vessel labels at rest, per-node label lookup -------------
    labels_rest, _ = label(mask_vessels[0])
    vessel_node_labels = labels_rest[i_node, j_node]  # 0 outside vessels

    # Cache mesh node coordinates for quick affine application.
    node_xy = mesh.p.T  # (n_nodes, 2), columns = (x, y)

    # --- Time loop ------------------------------------------------
    masks_simulated = np.zeros_like(mask_vessels)
    masks_simulated[0] = mask_choroid

    yy_pix, xx_pix = np.indices((H, W), dtype=float)
    pix_yx = np.stack([yy_pix, xx_pix], axis=-1)  # (H, W, 2) for interpolator

    for t in range(1, T):
        # Track vessels and fit affines.
        labels_t, _ = label(mask_vessels[t])
        transforms = _fit_affine_from_components(labels_rest, labels_t)

        # Build the prescribed-displacement vector at all DOFs.
        # Outside choroid and BM rows stay zero; vessel rows get u = Ax + c - x.
        u_full = np.zeros(n_dofs)
        for L, (A, c) in transforms.items():
            sel = vessel_node_labels == L
            if not sel.any():
                continue
            xy = node_xy[sel]  # (n_sel, 2)
            disp = xy @ A.T + c - xy  # (n_sel, 2)
            ks = np.where(sel)[0]
            u_full[2 * ks] = disp[:, 0]
            u_full[2 * ks + 1] = disp[:, 1]

        # Solve K_ff u_f = -K_fd u_d
        u_d = u_full[dir_dofs]
        rhs = -K_fd @ u_d
        u_f = K_ff_solve(rhs)
        u_full[free_dofs] = u_f
        # u_full[dir_dofs] stays at u_d (its prescribed value)

        # Per-node displacement and reshape to mesh-aligned grids.
        # init_tensor orders nodes with x as inner index (y outer), so
        # reshape (n_nodes,) -> (ny, nx).
        u_node = u_full.reshape(-1, 2)
        grid_ux = u_node[:, 0].reshape(ny, nx)
        grid_uy = u_node[:, 1].reshape(ny, nx)

        # Bilinearly resample u onto every pixel.
        ux_interp = RegularGridInterpolator(
            (ys, xs), grid_ux, bounds_error=False, fill_value=0.0
        )
        uy_interp = RegularGridInterpolator(
            (ys, xs), grid_uy, bounds_error=False, fill_value=0.0
        )
        ux_pix = ux_interp(pix_yx)
        uy_pix = uy_interp(pix_yx)

        # Warp the rest choroid mask.
        masks_simulated[t] = _backward_warp_mask(mask_choroid, ux_pix, uy_pix)

        if verbose and (t % max(T // 10, 1) == 0):
            print(f"  frame {t}/{T - 1}")

    return masks_simulated
