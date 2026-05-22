"""
make_animation.py
=================

Generate an MP4 (or GIF) animation of:
  - The original B-scan video on the left
  - The warped-from-rest B-scan on the right
  - Vessel masks overlaid in red on both
  - Simulated CSI contour overlaid in cyan on the warped panel
  - Observed CSI contour overlaid in yellow on the original panel
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from warp_core import (compute_displacement_fields, warp_image,
                       warp_mask, extract_csi)


def _normalize(img):
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)


def _overlay_mask(rgb, mask, color, alpha=0.45):
    out = rgb.copy()
    for c in range(3):
        out[..., c] = np.where(mask, (1 - alpha) * out[..., c]
                                       + alpha * color[c], out[..., c])
    return out


def make_animation(bscans, mask_choroid, mask_vessels,
                   mask_choroid_t=None,
                   nu=0.45, mesh_step=2,
                   bm_at_top=True,
                   fps=10, output_path='choroid_animation.mp4',
                   verbose=True):
    """
    Render a side-by-side animation. Pass mask_choroid_t to overlay the
    OBSERVED CSI on the left panel; otherwise only the rest contour shows.
    """
    T, H, W = bscans.shape
    if verbose:
        print("Computing displacement fields...")
    ux, uy = compute_displacement_fields(
        mask_choroid, mask_vessels,
        nu=nu, mesh_step=mesh_step,
        bm_at_top=bm_at_top, verbose=verbose)

    bscan0 = _normalize(bscans[0])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5),
                                   constrained_layout=True)
    axL.set_title('Observed B-scan')
    axR.set_title(f'Simulated warp (nu = {nu:.2f})')
    for ax in (axL, axR):
        ax.set_xticks([]); ax.set_yticks([])

    # Initial frames
    rgb_L = np.stack([bscan0] * 3, axis=-1)
    rgb_L = _overlay_mask(rgb_L, mask_vessels[0], (1.0, 0.2, 0.2))
    rgb_R = rgb_L.copy()

    im_L = axL.imshow(rgb_L)
    im_R = axR.imshow(rgb_R)

    csi0 = extract_csi(mask_choroid, bm_at_top=bm_at_top)
    xs_csi = np.arange(W)

    (line_obs,)  = axL.plot(xs_csi, csi0, color='yellow', lw=1.4,
                            label='observed CSI')
    (line_sim,)  = axR.plot(xs_csi, csi0, color='cyan',   lw=1.4,
                            label='simulated CSI')
    axL.legend(loc='lower left', fontsize=8)
    axR.legend(loc='lower left', fontsize=8)
    txt = fig.suptitle(f'frame 0 / {T-1}', fontsize=11)

    def update(t):
        # Left: original B-scan + vessel mask + (if available) observed CSI
        bscan_t = _normalize(bscans[t])
        rgb_L = np.stack([bscan_t] * 3, axis=-1)
        rgb_L = _overlay_mask(rgb_L, mask_vessels[t], (1.0, 0.2, 0.2))
        im_L.set_data(rgb_L)
        if mask_choroid_t is not None:
            line_obs.set_ydata(extract_csi(mask_choroid_t[t],
                                           bm_at_top=bm_at_top))

        # Right: warp the REST B-scan with current displacement
        warped = warp_image(bscans[0], ux[t], uy[t])
        warped = _normalize(warped)
        warped_vessels = warp_mask(mask_vessels[0], ux[t], uy[t])
        rgb_R = np.stack([warped] * 3, axis=-1)
        rgb_R = _overlay_mask(rgb_R, warped_vessels, (1.0, 0.2, 0.2))
        im_R.set_data(rgb_R)
        # Simulated CSI: warp the rest choroid mask, extract CSI from it
        warped_choroid = warp_mask(mask_choroid, ux[t], uy[t])
        line_sim.set_ydata(extract_csi(warped_choroid, bm_at_top=bm_at_top))

        txt.set_text(f'frame {t} / {T-1}')
        return im_L, im_R, line_obs, line_sim, txt

    if verbose:
        print(f"Rendering {T} frames -> {output_path}")
    ani = anim.FuncAnimation(fig, update, frames=T, blit=False,
                             interval=1000 // fps)

    if output_path.endswith('.gif'):
        ani.save(output_path, writer='pillow', fps=fps, dpi=110)
    else:
        # Try ffmpeg; fall back to gif if not present.
        try:
            ani.save(output_path, writer='ffmpeg', fps=fps, dpi=110,
                     bitrate=2400)
        except Exception as e:
            fallback = output_path.rsplit('.', 1)[0] + '.gif'
            print(f"ffmpeg unavailable ({e}); writing GIF: {fallback}")
            ani.save(fallback, writer='pillow', fps=fps, dpi=110)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    # Synthetic demo
    from choroid_pulse_image import _make_synthetic
    mask_choroid, mask_vessels = _make_synthetic(T=24)
    bscans = np.tile(mask_choroid.astype(np.float32) * 0.6
                     + np.random.rand(*mask_choroid.shape) * 0.1,
                     (24, 1, 1)).astype(np.float32)
    make_animation(bscans, mask_choroid, mask_vessels,
                   output_path='/home/claude/choroid_animation.gif',
                   fps=8, verbose=True)