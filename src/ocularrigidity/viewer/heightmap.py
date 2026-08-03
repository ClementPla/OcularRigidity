import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap


def visualize_csi_surface(
    csi: np.ndarray,
    timestamps: np.ndarray = None,
    z_scale: float = 1.0,
    gap_threshold_factor: float = 1.5,
    screenshot: str | None = None,
    off_screen: bool = False,
    jupyter_backend: str | None = "static",
):
    """Render a (T, W) depth-map as a surface in 3D with gaps highlighted.

    Parameters
    ----------
    csi : (T, W) array
        Depth / intensity values. First axis is time, second is column.
    timestamps : (T,) array, optional
        Acquisition times in seconds. Defaults to np.arange(T).
    z_scale : float
        Multiplicative scaling for the depth axis.
    gap_threshold_factor : float
        An inter-sample interval exceeding `gap_threshold_factor * median(dt)`
        is flagged as a temporal gap.
    screenshot : str, optional
        If given, save the render to this path (implies off_screen=True).
    off_screen : bool
        Render without opening a GUI window (useful on headless servers).
    jupyter_backend : str or None
        PyVista Jupyter backend used by ``plotter.show()`` in a notebook.
        Defaults to ``"static"`` (a server-side render embedded as an image),
        which is robust in VS Code / remote kernels. The default ``"trame"``
        backend uses an interactive widget/websocket that frequently crashes the
        kernel there; use ``"trame"``/``"client"`` only for smaller (decimated)
        surfaces where interactivity is needed. Ignored outside a notebook.
    """
    T, W = csi.shape
    t = np.asarray(timestamps) if timestamps is not None else np.arange(T, dtype=float)
    x = np.arange(W, dtype=float)

    # Normalize axes to a comparable visual scale. Subtract the min so an
    # offset (e.g. t starting at 100 s) doesn't compress the spacing.
    def _norm(v, target=1000.0):
        v0 = v - v.min()
        span = v0.max() if v0.max() > 0 else 1.0
        return v0 / span * target

    t_norm = _norm(t)
    x_norm = _norm(x)

    # ---- 1. Gap detection on timestamps --------------------------------
    dt = np.diff(t)
    median_dt = np.median(dt)
    gap_mask = dt > (median_dt * gap_threshold_factor)  # (T-1,) bool

    # ---- 2. Build the structured grid ----------------------------------
    # meshgrid with indexing="ij" -> arrays of shape (T, W).
    # PyVista sets dimensions=[T, W, 1] and internally ravels with F-order.
    #
    # VTK segfaults (crashing the kernel) on non-finite *point coordinates*, and
    # the boundary signal legitimately has NaN gaps. Keep NaN/Inf out of the
    # geometry by pinning those points to a finite floor value; every cell that
    # touches such a point is hidden below, so the floor value is never shown.
    finite = np.isfinite(csi)
    if not finite.any():
        raise ValueError("`csi` has no finite values to display.")
    z_floor_val = float(csi[finite].min())
    Z_grid = np.where(finite, csi, z_floor_val).astype(float) * z_scale

    T_grid, X_grid = np.meshgrid(t_norm, x_norm, indexing="ij")
    grid = pv.StructuredGrid(T_grid, X_grid, Z_grid)

    # ---- 3. Attach data in F-order to match PyVista's internal layout --
    # Point data: (T, W) -> ravel("F"): point_id = i + T*j, i fastest. NaN
    # scalars are fine (rendered with the colormap's nan color); only NaN
    # coordinates crash, and those were handled above.
    grid.point_data["depth"] = csi.ravel("F")

    # Cells to hide: temporal gaps OR any cell with a non-finite corner. A cell
    # (i, j) spans corners (i, j), (i+1, j), (i, j+1), (i+1, j+1); it is kept
    # only if all four are finite. Cell ids are F-order: cell_id = i + (T-1)*j.
    is_gap_cells = np.tile(gap_mask, W - 1)  # (T-1)*(W-1) bool
    cell_finite = (
        finite[:-1, :-1] & finite[1:, :-1] & finite[:-1, 1:] & finite[1:, 1:]
    )  # (T-1, W-1)
    hidden_cells = is_gap_cells | (~cell_finite).ravel("F")
    grid.cell_data["is_gap"] = is_gap_cells.astype(np.uint8)

    # ---- 4. Hide gap / NaN cells (preserves StructuredGrid) ------------
    # hide_cells keeps the grid structured; extract_cells would convert to
    # an UnstructuredGrid, which renders fine but loses structure info.
    grid.hide_cells(hidden_cells, inplace=True)

    # ---- 5. Floor plane that flags gap columns in red ------------------
    floor_z = float(Z_grid.min() - 0.1 * np.ptp(Z_grid))
    floor_grid = pv.StructuredGrid(T_grid, X_grid, np.full_like(Z_grid, floor_z))
    # Turn a per-cell gap mask (T-1,) into a per-point mark (T,) by padding
    # with False at the last timestamp (no outgoing interval there).
    gap_per_t = np.append(gap_mask, False)  # (T,)
    floor_scalars = np.broadcast_to(gap_per_t[:, None], (T, W)).ravel("F")
    floor_grid.point_data["is_gap"] = floor_scalars.astype(np.uint8)

    # ---- 6. Plot --------------------------------------------------------
    plotter = pv.Plotter(off_screen=off_screen or (screenshot is not None))
    plotter.add_mesh(
        grid,
        scalars="depth",
        cmap="viridis",
        smooth_shading=True,
        label="Acquired data",
    )
    gap_cmap = ListedColormap(["#333333", "#ff4444"])  # 0 = ok, 1 = gap
    plotter.add_mesh(
        floor_grid,
        scalars="is_gap",
        cmap=gap_cmap,
        clim=(0, 1),
        opacity=0.5,
        show_scalar_bar=False,
        label="Sampling quality (red = gap)",
    )
    plotter.add_axes()
    plotter.show_bounds(
        xtitle="time (s)",
        ytitle="column (px)",
        ztitle="depth (px)",
    )
    plotter.add_legend()

    if screenshot:
        plotter.show(screenshot=screenshot)
    else:
        plotter.show(jupyter_backend=jupyter_backend)
    return plotter
