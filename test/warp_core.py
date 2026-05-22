"""
Shared warp engine for choroid pulsation visualization.
=======================================================

Provides:
    compute_displacement_fields(mask_choroid, mask_vessels, nu, ...)
        -> (T, H, W, 2) displacement field per pixel per frame.
    warp_image(image, ux, uy)
        -> backward-warped image at a single frame.
    warp_mask(mask, ux, uy)
        -> backward-warped boolean mask at a single frame.
    extract_csi(mask) -> per-column y of the CSI for contour overlays.

Strategy
--------
The expensive part of the FEM is K's factorization. For pre-rendering at
multiple nu values we re-factor K each time, but each factorization is
amortized over T frames -- still very tractable.
"""

import numpy as np
import scipy.sparse.linalg as spla
from scipy.ndimage import label, map_coordinates
from scipy.interpolate import RegularGridInterpolator
from skfem import (MeshTri, Basis, ElementVector, ElementTriP1,
                   BilinearForm, asm)
from skfem.helpers import sym_grad, trace, ddot


# --------------------------------------------------------------------
# Affine fitting (copied from earlier; same moment-matching method)
# --------------------------------------------------------------------
def _sqrt_psd_2x2(M):
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 0.0, None)
    return V @ np.diag(np.sqrt(w)) @ V.T


def _fit_affine_from_components(labels_rest, labels_t, max_track_dist=80.0):
    transforms = {}
    n_rest, n_t = labels_rest.max(), labels_t.max()
    if n_rest == 0 or n_t == 0:
        return transforms
    rest_pts = {L: np.argwhere(labels_rest == L)[:, ::-1].astype(float)
                for L in range(1, n_rest + 1)}
    t_pts = {L: np.argwhere(labels_t == L)[:, ::-1].astype(float)
             for L in range(1, n_t + 1)}
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
        Cov_r = (Pcr.T @ Pcr) / max(len(Pcr), 1) + 1e-3 * np.eye(2)
        Cov_t = (Pct.T @ Pct) / max(len(Pct), 1) + 1e-3 * np.eye(2)
        S_r = _sqrt_psd_2x2(Cov_r)
        S_t = _sqrt_psd_2x2(Cov_t)
        A = S_t @ np.linalg.inv(S_r)
        c = t_mu[best_Lt] - A @ mu_r
        transforms[Lr] = (A, c)
    return transforms


def _extract_bm_pixels(mask_choroid, bm_at_top=True):
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


# --------------------------------------------------------------------
# Main displacement-field computation
# --------------------------------------------------------------------
def compute_displacement_fields(mask_choroid, mask_vessels,
                                nu=0.45, mesh_step=2,
                                bm_at_top=True, verbose=False):
    """
    Run the FEM forward simulation and return per-pixel displacement
    fields for every frame.

    Returns
    -------
    ux : (T, H, W) horizontal pixel displacement
    uy : (T, H, W) vertical pixel displacement
    """
    T, H, W = mask_vessels.shape

    # E cancels in pure-Dirichlet problems; just pick something.
    E = 1.0
    mu_l = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

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
    j_node = np.clip(np.round(mesh.p[0]).astype(int), 0, W - 1)
    i_node = np.clip(np.round(mesh.p[1]).astype(int), 0, H - 1)

    @BilinearForm
    def stiffness(u, v, w):
        return (2.0 * mu_l * ddot(sym_grad(u), sym_grad(v))
                + lam * trace(sym_grad(u)) * trace(sym_grad(v)))

    K = asm(stiffness, basis).tocsr()

    inside = mask_choroid[i_node, j_node]
    bm_pix = _extract_bm_pixels(mask_choroid, bm_at_top=bm_at_top)
    bm_node = bm_pix[i_node, j_node]
    vrest_node = mask_vessels[0][i_node, j_node]

    dir_node = (~inside) | bm_node | vrest_node
    dir_nodes = np.where(dir_node)[0]
    dir_dofs = np.sort(np.concatenate([2 * dir_nodes, 2 * dir_nodes + 1]))
    free_dofs = np.setdiff1d(np.arange(n_dofs), dir_dofs)

    K_ff = K[free_dofs, :][:, free_dofs].tocsc()
    K_fd = K[free_dofs, :][:, dir_dofs].tocsr()
    if verbose:
        print(f"  [nu={nu:.3f}] factorizing K_ff ({K_ff.shape[0]} dofs)...")
    K_ff_solve = spla.factorized(K_ff)

    labels_rest, _ = label(mask_vessels[0])
    vessel_node_labels = labels_rest[i_node, j_node]
    node_xy = mesh.p.T

    ux_out = np.zeros((T, H, W), dtype=np.float32)
    uy_out = np.zeros((T, H, W), dtype=np.float32)

    yy_pix, xx_pix = np.indices((H, W), dtype=float)
    pix_yx = np.stack([yy_pix, xx_pix], axis=-1)

    for t in range(T):
        if t == 0:
            continue   # rest frame: zero displacement
        labels_t, _ = label(mask_vessels[t])
        transforms = _fit_affine_from_components(labels_rest, labels_t)

        u_full = np.zeros(n_dofs)
        for L, (A, c) in transforms.items():
            sel = (vessel_node_labels == L)
            if not sel.any():
                continue
            xy = node_xy[sel]
            disp = xy @ A.T + c - xy
            ks = np.where(sel)[0]
            u_full[2 * ks] = disp[:, 0]
            u_full[2 * ks + 1] = disp[:, 1]

        u_d = u_full[dir_dofs]
        rhs = -K_fd @ u_d
        u_f = K_ff_solve(rhs)
        u_full[free_dofs] = u_f

        u_node = u_full.reshape(-1, 2)
        grid_ux = u_node[:, 0].reshape(ny, nx)
        grid_uy = u_node[:, 1].reshape(ny, nx)
        ux_out[t] = RegularGridInterpolator(
            (ys, xs), grid_ux, bounds_error=False,
            fill_value=0.0)(pix_yx).astype(np.float32)
        uy_out[t] = RegularGridInterpolator(
            (ys, xs), grid_uy, bounds_error=False,
            fill_value=0.0)(pix_yx).astype(np.float32)

        if verbose and (t % max(T // 10, 1) == 0):
            print(f"    frame {t}/{T-1}")

    return ux_out, uy_out


# --------------------------------------------------------------------
# Per-frame warps
# --------------------------------------------------------------------
def warp_image(image, ux, uy):
    """Backward-warp a grayscale or RGB image with order=1 interpolation."""
    H, W = image.shape[:2]
    yy, xx = np.indices((H, W), dtype=float)
    src_y = yy - uy
    src_x = xx - ux
    if image.ndim == 2:
        return map_coordinates(image, [src_y, src_x],
                               order=1, mode='nearest')
    out = np.empty_like(image)
    for c in range(image.shape[2]):
        out[..., c] = map_coordinates(image[..., c], [src_y, src_x],
                                      order=1, mode='nearest')
    return out


def warp_mask(mask, ux, uy):
    """Backward-warp a boolean mask, returning bool."""
    H, W = mask.shape
    yy, xx = np.indices((H, W), dtype=float)
    src_y = yy - uy
    src_x = xx - ux
    out = map_coordinates(mask.astype(float), [src_y, src_x],
                          order=1, mode='constant', cval=0.0)
    return out > 0.5


def extract_csi(mask, bm_at_top=True):
    """Per-column y of the CSI (boundary opposite BM). NaN where empty."""
    H, W = mask.shape
    out = np.full(W, np.nan)
    has = mask.any(axis=0)
    cols = np.where(has)[0]
    if cols.size == 0:
        return out
    if bm_at_top:
        rows = H - 1 - np.argmax(mask[::-1][:, cols], axis=0)
    else:
        rows = np.argmax(mask[:, cols], axis=0)
    out[cols] = rows
    return out


def extract_bm(mask, bm_at_top=True):
    """Per-column y of the BM (the FIXED boundary)."""
    H, W = mask.shape
    out = np.full(W, np.nan)
    has = mask.any(axis=0)
    cols = np.where(has)[0]
    if cols.size == 0:
        return out
    if bm_at_top:
        rows = np.argmax(mask[:, cols], axis=0)
    else:
        rows = H - 1 - np.argmax(mask[::-1][:, cols], axis=0)
    out[cols] = rows
    return out