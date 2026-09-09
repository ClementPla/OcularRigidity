"""Overlay HEYEX layer annotations on a single B-scan.

Works with anything exposing ``shape``, ``data[i]`` and ``layers[name].data`` --
an eyepy ``EyeVolume`` or, via :func:`plot_acquisition`, an
:class:`~ocularrigidity.data.heyex.dicom.Acquisition`.
"""

from __future__ import annotations

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from eyepy.config import layer_colors

__all__ = ["plot_acquisition", "plot_bscan_layers"]

# Fixed order, assigned by position and never cycled. Used when the caller opts
# out of eyepy's canonical colours, which alias PR1/EZ and PR2/IZ and put ILM
# and BM uncomfortably close together.
_FALLBACK_RAMP = (
    "#3BA3EC", "#E68332", "#34AF84", "#E866F4",
    "#97A431", "#A48CF4", "#36ADA4", "#F77189",
)


def plot_bscan_layers(
    volume,
    bscan_index: int = 0,
    layers=None,
    use_eyepy_colors: bool = True,
    show_coverage: bool = True,
    crop: bool = True,
    crop_pad: float = 0.25,
    figsize=None,
    ax=None,
):
    """Plot the layers present on one B-scan.

    Args:
        volume: an eyepy ``EyeVolume`` or any object with ``shape``, ``data``
            and ``layers`` (see module docstring).
        bscan_index: index along the slow axis; negative indices are allowed.
        layers: restrict to these layer names. Default: all present.
        use_eyepy_colors: eyepy's canonical per-layer colours, so the figure
            matches ``EyeVolume.plot()``. False picks a colourblind-safer ramp.
        show_coverage: add a layer x B-scan availability panel. Worth keeping
            for ONH data, where it shows at a glance that RNFL and BM exist
            only on the circle scans.
        crop: zoom the depth axis onto the annotated band. The retina occupies
            a small part of the A-scan, so without this most of the panel is
            empty and the labels collide.
        crop_pad: margin around that band, as a fraction of its thickness.

    Returns:
        ``(fig, info)``; ``info`` carries the resolved index and per-layer
        coverage for this B-scan.
    """
    n_bscans, height, width = volume.shape
    if not -n_bscans <= bscan_index < n_bscans:
        raise IndexError(f"bscan_index {bscan_index} out of range for {n_bscans} B-scans")
    bscan_index %= n_bscans

    names = list(volume.layers.keys()) if layers is None else list(layers)
    missing = [n for n in names if n not in volume.layers]
    if missing:
        raise KeyError(f"layers not in volume: {missing}. Available: {list(volume.layers)}")

    heights = {
        n: np.asarray(volume.layers[n].data[bscan_index], dtype=float) for n in names
    }
    coverage = {n: float(np.isfinite(h).mean()) for n, h in heights.items()}
    present = [n for n in names if coverage[n] > 0]

    # Order anatomically, top of the B-scan first, comparing only on columns
    # where every layer is defined: a layer annotated over a wider span (ILM
    # often runs past the others, onto a steep slope) would otherwise get a
    # skewed mean and sort out of order.
    if present:
        common = np.all([np.isfinite(heights[n]) for n in present], axis=0)
        if common.sum() >= 8:
            present.sort(key=lambda n: heights[n][common].mean())
        else:
            present.sort(key=lambda n: np.nanmedian(heights[n]))

    if use_eyepy_colors:
        colors = {n: "#" + layer_colors[n] for n in present}
    else:
        colors = {
            n: _FALLBACK_RAMP[i] if i < len(_FALLBACK_RAMP) else "#8A8A8A"
            for i, n in enumerate(present)
        }

    if ax is not None:
        fig, axes, show_coverage = ax.figure, [ax], False
    elif show_coverage:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=figsize or (13, 9),
            gridspec_kw={"height_ratios": [3, 1.5], "hspace": 0.55},
        )
        axes = list(axes)
    else:
        fig, single = plt.subplots(figsize=figsize or (13, 6))
        axes = [single]

    # --- B-scan with layer curves -----------------------------------------
    ax0 = axes[0]
    ax0.imshow(volume.data[bscan_index], cmap="gray", aspect="auto", interpolation="nearest")

    x = np.arange(width)
    label_targets = []
    for name in present:
        h = heights[name]
        ax0.plot(
            x, h, color=colors[name], linewidth=2, solid_capstyle="round",
            label=f"{name}  ({coverage[name]:.0%})",
        )
        finite = np.flatnonzero(np.isfinite(h))
        if finite.size:
            label_targets.append([h[finite[-1]], name])

    if crop and present:
        stack = np.concatenate([heights[n][np.isfinite(heights[n])] for n in present])
        lo, hi = float(stack.min()), float(stack.max())
        pad = max((hi - lo) * crop_pad, 5.0)
        top, bottom = max(0.0, lo - pad), min(height - 1.0, hi + pad)
    else:
        top, bottom = 0.0, height - 1.0

    # Direct labels at the right edge, nudged apart so identity is never
    # carried by colour alone.
    label_targets.sort(key=lambda t: t[0])
    min_gap = (bottom - top) * 0.045
    for i in range(1, len(label_targets)):
        if label_targets[i][0] - label_targets[i - 1][0] < min_gap:
            label_targets[i][0] = label_targets[i - 1][0] + min_gap
    for y, name in label_targets:
        ax0.annotate(
            name, xy=(width - 1, y), xytext=(6, 0), textcoords="offset points",
            va="center", ha="left", fontsize=8, color=colors[name], clip_on=False,
        )

    ax0.set_xlim(0, width - 1)
    ax0.set_ylim(bottom, top)
    ax0.set_xlabel("A-scan (x)")
    ax0.set_ylabel("depth (row)")
    ax0.set_title(
        f"B-scan {bscan_index} / {n_bscans - 1}  —  "
        f"{len(present)} of {len(names)} layers annotated",
        loc="left",
    )
    # Legend below the image so it never occludes the curves.
    ax0.legend(
        loc="upper left", bbox_to_anchor=(0, -0.13), fontsize=8,
        ncol=min(6, max(1, len(present))), frameon=False,
        handlelength=1.4, columnspacing=1.4,
    )

    absent = [n for n in names if coverage[n] == 0]
    if absent:
        ax0.annotate(
            "no annotation on this B-scan: " + ", ".join(absent),
            xy=(0.0, -0.34), xycoords="axes fraction", fontsize=8, color="0.35",
        )

    # --- Availability across the stack ------------------------------------
    if show_coverage:
        ax1 = axes[1]

        def depth(name):  # all-NaN layers sort last instead of warning
            d = np.asarray(volume.layers[name].data, dtype=float)
            return d[np.isfinite(d)].mean() if np.any(np.isfinite(d)) else np.inf

        order = sorted(names, key=depth)
        matrix = np.vstack(
            [
                np.isfinite(np.asarray(volume.layers[n].data, dtype=float)).mean(axis=1)
                for n in order
            ]
        )
        im = ax1.imshow(
            matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest"
        )
        ax1.set_yticks(range(len(order)), order, fontsize=8)
        ax1.set_xlabel("B-scan index")
        ax1.set_title("Fraction of A-scans annotated, per layer per B-scan", loc="left")
        ax1.axvline(bscan_index, color="#E4572E", linewidth=2)
        # Inside the heatmap, so low indices cannot collide with the title.
        ax1.annotate(
            f"B-scan {bscan_index}", xy=(bscan_index, 0.0), xytext=(4, 4),
            textcoords="offset points", ha="left", va="top", fontsize=8,
            color="#E4572E",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
        )
        fig.colorbar(im, ax=ax1, pad=0.01, fraction=0.025).set_label("coverage", fontsize=8)

    info = {
        "bscan_index": bscan_index,
        "coverage": coverage,
        "present": present,
        "absent": absent,
    }
    return fig, info


class _LayerShim:
    __slots__ = ("data",)

    def __init__(self, data):
        self.data = data


class _VolumeShim:
    """Adapts an Acquisition to the attributes :func:`plot_bscan_layers` uses."""

    def __init__(self, acq):
        self.data = acq.bscans
        self.shape = acq.bscans.shape
        self.layers = {k: _LayerShim(np.asarray(v, float)) for k, v in acq.layers.items()}


def plot_acquisition(acq, bscan_index: int = 0, **kwargs):
    """:func:`plot_bscan_layers` for an :class:`Acquisition`, with its identity
    in the title and the circle radius on the x-label where relevant.

    Layers only line up when ``acq`` was loaded with ``bscan_source="e2e"``
    (the default); a "dicom" acquisition is flagged in the title.
    """
    fig, info = plot_bscan_layers(_VolumeShim(acq), bscan_index=bscan_index, **kwargs)

    circles = set(acq.circle_frames)
    index = info["bscan_index"]
    kind = "circle scan" if index in circles else "line B-scan"
    warn = "" if acq.bscan_source == "e2e" else f"  [{acq.bscan_source} pixels: layers offset]"

    ax0 = fig.axes[0]
    ax0.set_title(
        f"{acq.patient_id}  {acq.study_date}  {acq.laterality}  —  {acq.label}{warn}\n"
        f"frame {index} / {acq.n_bscans - 1} ({kind})  —  "
        f"{len(info['present'])} of {len(acq.layers)} layers annotated",
        loc="left",
        fontsize=10,
    )

    if index in circles:
        radius = acq.details["circle_radii_px"][sorted(circles).index(index)]
        ax0.set_xlabel(f"position along circle (A-scan)  —  radius {radius:.1f} px")

    # Manually placed disc-margin markers, when this B-scan carries them.
    markers = getattr(acq, "markers", None)
    if markers is not None and np.isfinite(markers[index]).all():
        pts = markers[index]
        ax0.plot(
            pts[:, 0], pts[:, 1], "o", ms=9, mfc="none", mec="#00E5FF", mew=2,
            zorder=5, label="marker",
        )
        for x, y in pts:
            ax0.annotate(
                f"({x:.0f}, {y:.0f})", xy=(x, y), xytext=(0, -12),
                textcoords="offset points", ha="center", va="top",
                fontsize=7, color="#00E5FF",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")],
            )

    info["is_circle"] = index in circles
    info["markers"] = None if markers is None else markers[index]
    return fig, info
