

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from choroid_pulse import simulate_choroid_pulsation


# ====================================================================
# CSI extraction
# ====================================================================
def extract_csi(mask, bm_at_top=True):
    """
    Per-column y-coordinate of the CSI (the boundary OPPOSITE to BM).

    Returns
    -------
    csi  : (W,) float, NaN where the column is empty.
    """
    H, W = mask.shape
    out = np.full(W, np.nan)
    has = mask.any(axis=0)
    cols = np.where(has)[0]
    if cols.size == 0:
        return out
    if bm_at_top:
        # CSI is the bottommost True pixel per column.
        rows = H - 1 - np.argmax(mask[::-1][:, cols], axis=0)
    else:
        rows = np.argmax(mask[:, cols], axis=0)
    out[cols] = rows
    return out


def csi_displacement_curve(masks, bm_at_top=True):
    """
    Per-frame CSI displacement (relative to frame 0) at every column.

    Returns
    -------
    disp : (T, W) float, NaN where either rest or current column is empty.
    """
    T = masks.shape[0]
    csi0 = extract_csi(masks[0], bm_at_top=bm_at_top)
    out  = np.full((T, masks.shape[2]), np.nan)
    for t in range(T):
        csi_t = extract_csi(masks[t], bm_at_top=bm_at_top)
        out[t] = csi_t - csi0
    return out


# ====================================================================
# Residual analysis
# ====================================================================
def run_diagnostic(mask_choroid, mask_vessels, mask_choroid_t,
                   E=5.0e3, nu=0.45, mesh_step=2,
                   bm_at_top=True, save_path=None):
    """
    Run the forward model and produce a 5-panel diagnostic figure.

    Parameters
    ----------
    mask_choroid    : (H, W) bool, rest choroid.
    mask_vessels    : (T, H, W) bool, vessels per frame.
    mask_choroid_t  : (T, H, W) bool, observed choroid per frame.
    E, nu           : default elastic parameters for the FIRST PASS.
                      Don't tune these yet; we want to see the unfit error.
    mesh_step       : pixels between FEM nodes.
    bm_at_top       : convention for which choroid edge is fixed.
    save_path       : where to save the figure.

    Returns
    -------
    info : dict with predicted masks, CSI curves, residuals, and the SVD
           of the residual matrix. Use this for follow-up analysis.
    """
    T, H, W = mask_vessels.shape
    assert mask_choroid_t.shape == (T, H, W), \
        "Observed and vessel masks must have the same TxHxW shape."

    # 1. Forward simulation with default parameters.
    print("Running forward simulation...")
    masks_pred = simulate_choroid_pulsation(
        mask_choroid, mask_vessels,
        E=E, nu=nu, mesh_step=mesh_step,
        bm_at_top=bm_at_top, verbose=True)

    # 2. CSI displacement curves.
    disp_pred = csi_displacement_curve(masks_pred, bm_at_top=bm_at_top)
    disp_obs  = csi_displacement_curve(mask_choroid_t, bm_at_top=bm_at_top)

    # 3. Residual (observed minus predicted). Mask NaN columns.
    residual = disp_obs - disp_pred
    valid_cols = np.isfinite(residual).all(axis=0)
    residual_clean = residual[:, valid_cols]
    xs_valid = np.where(valid_cols)[0]

    # 4. SVD of the residual matrix (T x W_valid).
    # Each rank-1 term  s_k * u_k(t) * v_k(x)  is a separable temporal x
    # spatial pattern. We center per-frame mean to remove rigid offset.
    R = residual_clean - np.nanmean(residual_clean, axis=1, keepdims=True)
    R[~np.isfinite(R)] = 0.0
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    var_explained = S ** 2 / max((S ** 2).sum(), 1e-12)

    # 5. Per-frame peak-to-peak amplitudes (scatter plot).
    ptp_obs  = np.nanmax(disp_obs,  axis=1) - np.nanmin(disp_obs,  axis=1)
    ptp_pred = np.nanmax(disp_pred, axis=1) - np.nanmin(disp_pred, axis=1)

    # 6. Pick the most-deformed frame (largest observed PTP) for plot 1.
    t_peak = int(np.nanargmax(ptp_obs))

    # ---------------- Plot ----------------
    fig = plt.figure(figsize=(13, 14), constrained_layout=True)
    gs  = fig.add_gridspec(5, 2)

    # Plot 1: CSI overlays at peak frame
    ax = fig.add_subplot(gs[0, :])
    ax.plot(disp_obs[t_peak],  label='observed CSI displacement', lw=1.6)
    ax.plot(disp_pred[t_peak], label='predicted CSI displacement', lw=1.6)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('x [pixel]'); ax.set_ylabel('CSI shift [pixel]')
    ax.set_title(f'Plot 1 — CSI displacement at peak frame (t={t_peak})')
    ax.legend(); ax.grid(alpha=0.3)

    # Plot 2: residual heatmap
    ax = fig.add_subplot(gs[1, :])
    vmax = max(abs(np.nanmin(residual)), abs(np.nanmax(residual)), 1e-6)
    im = ax.imshow(residual, aspect='auto', origin='lower',
                   cmap='RdBu_r',
                   norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
    ax.set_xlabel('x [pixel]'); ax.set_ylabel('frame t')
    ax.set_title('Plot 2 — residual = observed - predicted '
                 '(pixel; red=under-predicted)')
    fig.colorbar(im, ax=ax, label='pixel')

    # Plot 3a: leading spatial mode
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(xs_valid, Vt[0])
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_title(f'Plot 3a — residual spatial mode 1 '
                 f'({var_explained[0]*100:.0f}% variance)')
    ax.set_xlabel('x [pixel]'); ax.set_ylabel('mode amplitude')
    ax.grid(alpha=0.3)

    # Plot 3b: leading temporal mode
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(np.arange(T), U[:, 0] * S[0], '-o', ms=4)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_title('Plot 3b — residual temporal mode 1')
    ax.set_xlabel('frame t'); ax.set_ylabel('mode amplitude')
    ax.grid(alpha=0.3)

    # Plot 4: amplitude scatter
    ax = fig.add_subplot(gs[3, 0])
    ax.scatter(ptp_obs, ptp_pred, s=22)
    lo = 0
    hi = max(np.nanmax(ptp_obs), np.nanmax(ptp_pred), 1e-6) * 1.1
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, label='y = x')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('observed PTP [pixel]'); ax.set_ylabel('predicted PTP [pixel]')
    ax.set_title('Plot 4 — per-frame amplitude scatter')
    ax.legend(); ax.grid(alpha=0.3)

    # Plot 4b: singular value spectrum
    ax = fig.add_subplot(gs[3, 1])
    k = min(8, len(S))
    ax.bar(np.arange(k), var_explained[:k] * 100)
    ax.set_xlabel('mode index')
    ax.set_ylabel('% variance explained')
    ax.set_title('Plot 4b — residual SVD spectrum')
    ax.grid(alpha=0.3, axis='y')

    # Plot 5: waveforms at three columns
    ax = fig.add_subplot(gs[4, :])
    cols_pick = np.linspace(W * 0.25, W * 0.75, 3).astype(int)
    cmap = plt.get_cmap('tab10')
    for i, c in enumerate(cols_pick):
        ax.plot(disp_obs[:, c],  '-',  color=cmap(i), label=f'obs  x={c}')
        ax.plot(disp_pred[:, c], '--', color=cmap(i), label=f'pred x={c}')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('frame t'); ax.set_ylabel('CSI shift [pixel]')
    ax.set_title('Plot 5 — waveforms at three columns (solid=obs, dashed=pred)')
    ax.legend(ncol=3, fontsize=8); ax.grid(alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=130)
    plt.show()

    return dict(
        masks_pred=masks_pred,
        disp_pred=disp_pred,
        disp_obs=disp_obs,
        residual=residual,
        svd=(U, S, Vt),
        ptp_obs=ptp_obs,
        ptp_pred=ptp_pred,
    )


# ====================================================================
# Synthetic demo
# ====================================================================
def _demo():
    """
    Quick synthetic test: generate vessels + 'observed' choroid that
    follows a SLIGHTLY DIFFERENT physics than our model (e.g., a higher
    effective stiffness contrast). Run the diagnostic.
    """
    from choroid_pulse_image import _make_synthetic
    mask_choroid, mask_vessels = _make_synthetic()

    # Pretend observation: simulate with a different nu so there IS
    # a residual to look at. In real life this comes from segmentation.
    print("Generating synthetic 'observed' masks with nu=0.30 ...")
    mask_choroid_t = simulate_choroid_pulsation(
        mask_choroid, mask_vessels, E=5.0e3, nu=0.30, mesh_step=2,
        verbose=False)

    info = run_diagnostic(
        mask_choroid, mask_vessels, mask_choroid_t,
        E=5.0e3, nu=0.45, mesh_step=2,
        save_path='/home/claude/diagnostic_output.png')

    print("\nSummary statistics")
    print("-" * 50)
    print(f"Residual RMS over all frames:     "
          f"{np.sqrt(np.nanmean(info['residual']**2)):.3f} pixel")
    print(f"Residual mode 1 explains:         "
          f"{(info['svd'][1][0]**2/(info['svd'][1]**2).sum())*100:.1f}% var")
    print(f"Mean predicted/observed PTP ratio: "
          f"{np.nanmean(info['ptp_pred']/np.maximum(info['ptp_obs'],1e-6)):.2f}")


if __name__ == "__main__":
    _demo()