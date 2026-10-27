"""Animated 3-D rendering of the choroidal vessels deformed by a pulsatile field."""

from __future__ import annotations

import sys

import imageio.v2 as imageio
import numpy as np
import pyvista as pv
from scipy.ndimage import map_coordinates
from skimage.measure import marching_cubes

pv.OFF_SCREEN = True


def _vessel_mesh(mask, px_um, dz_um, depth_scale):
    verts, faces, *_ = marching_cubes(
        np.pad(mask, 1).astype(np.float32), 0.5, spacing=(px_um[0], dz_um, px_um[1])
    )
    verts -= np.array([px_um[0], dz_um, px_um[1]])  # undo the padding
    mesh = pv.PolyData(
        # x = col, y = -row (fundus up), z = -depth (BM on top)
        np.c_[verts[:, 2], -verts[:, 0], -verts[:, 1] * depth_scale],
        np.c_[np.full(len(faces), 3), faces].ravel(),
    )
    return mesh.smooth_taubin(n_iter=30, pass_band=0.1).decimate(0.5)


def _sample(img, x, y, px_um):
    return map_coordinates(img, np.stack([-y / px_um[0], x / px_um[1]]), order=1)


def render(
    scene: dict,
    out_path: str,
    amplify: float = 25.0,
    depth_scale: float = 2.0,
    loops: int = 2,
    fps: int = 15,
    size=(1200, 900),
    elevation: float = 40.0,
    azimuth: float = -60.0,
    csi_step: int = 4,
    retina_color: str = "#d4a017",
    retina_opacity: float = 0.9,
):
    """Write the animation to ``out_path`` (.mp4 or .gif)."""
    mask, dct, ct = scene["mask"], scene["dct"], np.nan_to_num(scene["ct"])
    px_um, dz_um = tuple(scene["px_um"]), float(scene["dz_um"])
    inside = np.isfinite(dct[0])
    dct0 = np.nan_to_num(dct)

    mesh = _vessel_mesh(mask & inside[:, None, :], px_um, dz_um, depth_scale)
    bounds = mesh.bounds  # camera framed on the choroid, retina or not
    P0 = mesh.points.copy()
    ct_pts = _sample(ct, P0[:, 0], P0[:, 1], px_um)
    frac = np.clip(-P0[:, 2] / depth_scale / np.maximum(ct_pts, 1), 0, 1)

    H, W = ct.shape
    s = csi_step
    rr, cc = np.mgrid[0:H:s, 0:W:s]
    X, Y = cc * px_um[1], -rr * px_um[0]
    keep = inside[::s, ::s].ravel(order="F")

    def csi_surface(t):
        d = dct0[t, ::s, ::s]
        g = pv.StructuredGrid(X, Y, -(ct[::s, ::s] + amplify * d) * depth_scale)
        g["d"] = d.ravel(order="F")
        return g.extract_points(keep, adjacent_cells=False)

    T = dct.shape[0]
    lim = float(np.nanpercentile(np.abs(dct), 99))
    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.set_background("white")
    mesh["displacement (um)"] = np.zeros(mesh.n_points)
    pl.add_mesh(
        mesh,
        scalars="displacement (um)",
        cmap="RdBu_r",
        clim=(-lim, lim),
        smooth_shading=True,
        scalar_bar_args=dict(color="black", title="displacement at the vessel (um)"),
    )
    csi_actor = pl.add_mesh(
        csi_surface(0), scalars="d", cmap="RdBu_r", clim=(-lim, lim), opacity=0.3, show_scalar_bar=False
    )
    if "retina" in scene:
        # The retina rides on the BM, which is the fixed reference here: static layer.
        verts, faces, *_ = marching_cubes(
            np.pad(scene["retina"] & inside[:, None, :], 1).astype(np.float32),
            0.5,
            spacing=(px_um[0], dz_um, px_um[1]),
        )
        verts -= np.array([px_um[0], dz_um, px_um[1]])
        z = float(scene["retina_z0_um"]) + verts[:, 1]
        retina = pv.PolyData(
            np.c_[verts[:, 2], -verts[:, 0], -z * depth_scale],
            np.c_[np.full(len(faces), 3), faces].ravel(),
        ).smooth_taubin(n_iter=30, pass_band=0.1)
        pl.add_mesh(retina, color=retina_color, smooth_shading=True, opacity=retina_opacity)
    text = pl.add_text("", position="upper_left", font_size=12, color="black")

    # Oblique view from above, centred on the field
    b = np.array(bounds).reshape(3, 2).mean(axis=1)
    radius = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
    el, az = np.deg2rad(elevation), np.deg2rad(azimuth)
    direction = np.array([np.cos(el) * np.sin(az), -np.cos(el) * np.cos(az), np.sin(el)])
    pl.camera.focal_point = b
    pl.camera.position = b + 2.1 * radius * direction
    pl.camera.up = (0, 0, 1)

    writer = imageio.get_writer(out_path, fps=fps, **({"quality": 8, "macro_block_size": 1} if out_path.endswith(".mp4") else {}))
    for k in range(T * loops):
        t = k % T
        d = _sample(dct0[t], P0[:, 0], P0[:, 1], px_um) * frac
        mesh.points = P0 - np.c_[np.zeros_like(d), np.zeros_like(d), amplify * d * depth_scale]
        mesh["displacement (um)"] = d
        csi_actor.mapper.SetInputData(csi_surface(t))
        text.SetText(
            2,
            f"{scene['title']}  |  {t / T:.0%} of the cardiac cycle  |  "
            f"displacement x{amplify:g}, depth x{depth_scale:g}",
        )
        pl.render()
        writer.append_data(pl.screenshot(return_img=True))
    writer.close()
    pl.close()
    return out_path


if __name__ == "__main__":
    import json

    scene_path, out = sys.argv[1], sys.argv[2]
    kwargs = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    with np.load(scene_path) as z:
        scene = {k: z[k] for k in z.files}
    scene["title"] = str(scene.get("title", ""))
    render(scene, out, **kwargs)
    print(out)
