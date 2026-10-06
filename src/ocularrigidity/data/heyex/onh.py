"""Minimum rim width at the optic nerve head, from a HEYEX ONH acquisition."""

from __future__ import annotations

import numpy as np

__all__ = [
    "QUADRANTS",
    "GARWAY_HEATH",
    "find_fovea",
    "fobmo_angle",
    "SECTOR_COUNTS",
    "SECTOR_NAMES",
    "marker_ring",
    "minimum_rim_width",
    "register_localizers",
    "rim_width_by_marker_index",
    "rim_width_by_sector",
    "temporal_reference",
]

# : Quadrant bounds in degrees from the temporal meridian, counter-clockwise : through superior.
QUADRANTS = {
    "temporal": (-45.0, 45.0),
    "superior": (45.0, 135.0),
    "nasal": (135.0, 225.0),
    "inferior": (225.0, 315.0),
}


GARWAY_HEATH = {
    "temporal": (316.0, 40.0),
    "superotemporal": (40.0, 85.0),
    "superonasal": (86.0, 125.0),
    "nasal": (126.0, 235.0),
    "inferonasal": (236.0, 275.0),
    "inferotemporal": (276.0, 315.0),
}


def _disc_centre(acq):
    """ONH centre on the localizer, in localizer pixels."""
    centres = acq.details.get("circle_centres")
    if centres is not None and len(centres):
        return np.asarray(centres[0], float)
    lines = acq.details.get("line_frames", [])
    mids = np.array([acq.geometry[i]["coords"].mean(0) for i in lines])
    return mids.mean(0)


def _point_angle(acq, frame, column, centre):
    """Angle of a B-scan column about the disc centre, in degrees."""
    p1, p2 = acq.geometry[frame]["coords"][0], acq.geometry[frame]["coords"][-1]
    width = acq.layers[next(iter(acq.layers))].shape[1]
    pos = p1 + (column / max(width - 1, 1)) * (p2 - p1)

    dx, dy = pos[0] - centre[0], pos[1] - centre[1]
    temporal_sign = -1.0 if str(acq.laterality).upper().startswith("R") else 1.0
    # +x towards temporal, +y towards superior
    return np.degrees(np.arctan2(-dy, temporal_sign * dx)) % 360.0


def minimum_rim_width(acq, layer="ILM", search_halfwidth=None):
    """Minimum distance from each disc-margin marker to ``layer``."""
    if getattr(acq, "markers", None) is None:
        raise ValueError("acquisition carries no marker points")
    if layer not in acq.layers:
        raise KeyError(f"no {layer!r} layer; available: {list(acq.layers)}")

    heights = np.asarray(acq.layers[layer], float)
    n_bscans, width = heights.shape
    columns = np.arange(width)
    centre = _disc_centre(acq)

    values = np.full((n_bscans, 2), np.nan)
    angles = np.full((n_bscans, 2), np.nan)

    for frame in range(n_bscans):
        pts = acq.markers[frame]
        if not np.isfinite(pts).all():
            continue
        row_mm, col_mm = acq.spacing_for(frame)
        h = heights[frame]
        for k, (px, py) in enumerate(pts):
            valid = np.isfinite(h)
            if search_halfwidth is not None:
                valid &= np.abs(columns - px) <= search_halfwidth
            if not valid.any():
                continue
            dist = np.hypot((columns[valid] - px) * col_mm, (h[valid] - py) * row_mm)
            values[frame, k] = dist.min() * 1000.0  # mm -> um
            angles[frame, k] = _point_angle(acq, frame, px, centre)

    return {
        "value_um": values,
        "angle_deg": angles,
        "disc_centre_px": centre,
        "layer": layer,
    }


def rim_width_by_sector(acq, scheme=QUADRANTS, layer="ILM", rotation_deg=0.0, **kwargs):
    """Average :func:`minimum_rim_width` per sector."""
    raw = minimum_rim_width(acq, layer=layer, **kwargs)
    values = raw["value_um"].ravel()
    angles = raw["angle_deg"].ravel()
    ok = np.isfinite(values) & np.isfinite(angles)
    values, angles = values[ok], (angles[ok] + rotation_deg) % 360.0

    sectors, counts = {}, {}
    for name, (start, end) in scheme.items():
        lo, hi = start % 360.0, end % 360.0
        inside = (
            (angles >= lo) & (angles < hi)
            if lo < hi
            else (angles >= lo) | (angles < hi)
        )
        sectors[name] = (
            float(np.median(values[inside])) if inside.any() else float("nan")
        )
        counts[name] = int(inside.sum())

    return {
        "global_um": float(values.mean()) if values.size else float("nan"),
        "sectors": sectors,
        "counts": counts,
        "raw": raw,
    }


# ---------------------------------------------------------------------------
# Fovea-to-BMO axis
def find_fovea(macular_acq):
    """Locate the fovea in a macular raster: ``(frame, column)``."""
    from scipy.ndimage import gaussian_filter

    thickness = np.asarray(macular_acq.layers["BM"], float) - np.asarray(
        macular_acq.layers["ILM"], float
    )
    ok = np.isfinite(thickness)
    if not ok.any():
        raise ValueError("no ILM/BM segmentation to locate the fovea with")
    n_frames, width = thickness.shape

    _, col_mm = macular_acq.pixel_spacing
    slice_mm = (
        macular_acq.slice_thickness
        if np.isfinite(macular_acq.slice_thickness)
        else col_mm * 10.0
    )
    filled = np.where(ok, thickness, np.nanmedian(thickness[ok]))

    def blur(sigma_mm):
        return gaussian_filter(
            filled, (max(sigma_mm / slice_mm, 0.5), max(sigma_mm / col_mm, 0.5))
        )

    dog = blur(0.9) - blur(0.15)  # positive where the centre is thinner
    inside = np.zeros_like(dog, bool)
    inside[
        int(0.2 * n_frames) : int(0.8 * n_frames), int(0.2 * width) : int(0.8 * width)
    ] = True
    inside &= gaussian_filter(ok.astype(float), (1, 8)) > 0.99
    return tuple(
        int(v)
        for v in np.unravel_index(np.argmax(np.where(inside, dog, -np.inf)), dog.shape)
    )


def register_localizers(onh_localizer, macular_localizer):
    """Translation mapping macular-localizer pixels onto the ONH localizer."""
    from scipy.ndimage import gaussian_filter
    from skimage.registration import phase_cross_correlation
    from skimage.transform import resize

    ref = np.asarray(onh_localizer, float)
    mov = resize(
        np.asarray(macular_localizer, float), ref.shape, order=1, preserve_range=True
    )

    def bandpass(img):
        valid = gaussian_filter((img > 8).astype(float), 6) > 0.99
        return (gaussian_filter(img, 2) - gaussian_filter(img, 25)) * valid, valid

    a, mask_a = bandpass(ref)
    b, mask_b = bandpass(mov)
    shift, _, _ = phase_cross_correlation(
        a, b, reference_mask=mask_a, moving_mask=mask_b, overlap_ratio=0.2
    )
    return float(shift[0]), float(shift[1])


def fobmo_angle(
    onh_acq, macular_acq, onh_localizer, macular_localizer, localizer_pixel_mm=None
):
    """Fovea-to-BMO-centre axis, in degrees, for one examination."""
    ref = np.asarray(onh_localizer, float)
    scale = ref.shape[1] / np.asarray(macular_localizer).shape[1]
    d_row, d_col = register_localizers(ref, macular_localizer)

    frame, column = find_fovea(macular_acq)
    width = np.asarray(macular_acq.layers["ILM"]).shape[1]
    p1 = macular_acq.geometry[frame]["coords"][0]
    p2 = macular_acq.geometry[frame]["coords"][-1]
    pos = p1 + (column / max(width - 1, 1)) * (p2 - p1)  # macular localizer (x, y)
    fovea = np.array([pos[0] * scale + d_col, pos[1] * scale + d_row])

    disc = _disc_centre(onh_acq)
    delta = fovea - disc
    if localizer_pixel_mm is None:
        localizer_pixel_mm = float("nan")
    temporal_sign = -1.0 if str(onh_acq.laterality).upper().startswith("R") else 1.0

    return {
        "angle_deg": float(np.degrees(np.arctan2(-delta[1], temporal_sign * delta[0]))),
        "disc_fovea_mm": float(np.hypot(*delta) * localizer_pixel_mm),
        "fovea_px": fovea,
        "disc_px": disc,
        "shift_px": (d_row, d_col),
        "fovea_bscan": (frame, column),
    }


# ---------------------------------------------------------------------------
SECTOR_COUNTS = (12, 6, 6, 12, 6, 6)

SECTOR_NAMES = (
    "temporal",
    "superotemporal",
    "superonasal",
    "nasal",
    "inferonasal",
    "inferotemporal",
)


def marker_ring(acq, layer="ILM"):
    """The markers ordered around the disc."""
    m = minimum_rim_width(acq, layer=layer)
    values, angles = m["value_um"].ravel(), m["angle_deg"].ravel()
    ids = [(i // 2, i % 2) for i in range(values.size)]
    ok = np.isfinite(values) & np.isfinite(angles)
    values, angles = values[ok], angles[ok]
    ids = [ids[i] for i in np.flatnonzero(ok)]
    order = np.argsort(angles)
    return values[order], angles[order], [ids[i] for i in order]


def temporal_reference(acq, layer="ILM"):
    """The marker that the scan pattern places on the temporal meridian."""
    _, angles, ids = marker_ring(acq, layer=layer)
    lines = acq.details.get("line_frames") or []
    middle = lines[len(lines) // 2] if lines else 0
    candidates = [i for i, (f, _) in enumerate(ids) if f == middle]
    if not candidates:
        candidates = range(len(ids))
    return min(candidates, key=lambda i: abs(((angles[i] + 180.0) % 360.0) - 180.0))


def rim_width_by_marker_index(
    acq, reference="scan", counts=SECTOR_COUNTS, names=SECTOR_NAMES, layer="ILM"
):
    """Sector means built by counting markers around the disc, not by angle."""
    values, angles, ids = marker_ring(acq, layer=layer)
    n = len(values)
    if n == 0:
        raise ValueError("no markers to build sectors from")
    if sum(counts) > n:
        raise ValueError(f"counts sum to {sum(counts)} but only {n} markers exist")

    if reference == "scan":
        ref = temporal_reference(acq, layer=layer)
    elif reference == "temporal":
        ref = int(np.argmin(np.abs(((angles + 180.0) % 360.0) - 180.0)))
    elif isinstance(reference, tuple):
        ref = ids.index(reference)
    else:
        ref = int(reference)

    # Centre the first sector on the reference, as 5-before / 5-after for 11.
    pos = ref - (counts[0] - 1) // 2
    sectors, per, spans = {}, {}, {}
    for name, count in zip(names, counts):
        idx = [(pos + j) % n for j in range(count)]
        sectors[name] = float(values[idx].mean())
        per[name] = count
        spans[name] = (
            round(float(angles[idx[0]]), 1),
            round(float(angles[idx[-1]]), 1),
        )
        pos += count

    return {
        "sectors": sectors,
        "counts": per,
        "angle_spans": spans,
        "global_um": float(values.mean()),
        "reference": ids[ref],
        "reference_angle": round(float(angles[ref]), 1),
        "unassigned": n - sum(counts),
    }
