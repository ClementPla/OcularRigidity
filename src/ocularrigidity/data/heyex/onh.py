"""Minimum rim width at the optic nerve head, from a HEYEX ONH acquisition.

The distance from each disc-margin marker to the ILM is the BMO-MRW measurement
HEYEX reports per sector. **That derived value is not in the DICOM export** --
see :func:`minimum_rim_width` for what we checked -- so it is recomputed here
from the two things the export does carry: the marker points
(:func:`~ocularrigidity.data.heyex.dicom.read_marker_points`) and the ILM
(:func:`~ocularrigidity.data.heyex.dicom.read_layers`).

    from ocularrigidity.data.heyex import load_acquisition
    from ocularrigidity.data.heyex.onh import minimum_rim_width, rim_width_by_sector

    acq = load_acquisition(onh_path)
    mrw = minimum_rim_width(acq)          # per marker point, in micrometres
    rim_width_by_sector(acq)              # averaged per quadrant
"""

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

#: Quadrant bounds in degrees from the temporal meridian, counter-clockwise
#: through superior. Name -> (start, end).
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
    # No circle scans: the radials pivot about the disc, so their shared
    # midpoint is the centre.
    lines = acq.details.get("line_frames", [])
    mids = np.array([acq.geometry[i]["coords"].mean(0) for i in lines])
    return mids.mean(0)


def _point_angle(acq, frame, column, centre):
    """Angle of a B-scan column about the disc centre, in degrees.

    Measured from the temporal meridian, increasing towards superior, so the
    result is directly comparable between eyes. Localizer pixel y grows
    downwards and temporal is -x for a right eye, +x for a left eye.
    """
    p1, p2 = acq.geometry[frame]["coords"][0], acq.geometry[frame]["coords"][-1]
    width = acq.layers[next(iter(acq.layers))].shape[1]
    pos = p1 + (column / max(width - 1, 1)) * (p2 - p1)

    dx, dy = pos[0] - centre[0], pos[1] - centre[1]
    temporal_sign = -1.0 if str(acq.laterality).upper().startswith("R") else 1.0
    # +x towards temporal, +y towards superior
    return np.degrees(np.arctan2(-dy, temporal_sign * dx)) % 360.0


def minimum_rim_width(acq, layer="ILM", search_halfwidth=None):
    """Minimum distance from each disc-margin marker to ``layer``.

    This is BMO-MRW when the markers are the Bruch's-membrane-opening points and
    ``layer`` is the ILM: the shortest distance from the marker to the inner
    retinal surface, measured in the B-scan plane.

    Distances are computed in physical units, which matters here -- an ONH
    B-scan pixel is about 1.5x wider than it is tall, and the lateral spacing
    differs from radial to radial, so the per-frame ``PixelSpacing`` is used
    (see :meth:`Acquisition.spacing_for`).

    .. note::
       HEYEX's own MRW numbers, and its sector averages, are **not present in
       the DICOM export**. We looked: the encapsulated PDFs are macular
       Thickness Map reports with no BMO/MRW/RNFL content, and a byte-level
       sweep of every folder in both private E2E blobs (float32/float64/int
       at every offset) found nothing matching either the values or the shape
       of a sector summary. Exporting the Glaucoma Module report instead of the
       Thickness Map report would be the way to get Heidelberg's own figures.

    Args:
        acq: an ONH :class:`Acquisition` carrying ``markers``.
        layer: layer name to measure to.
        search_halfwidth: if given, restrict the search to this many columns
            either side of the marker. Left as None the whole B-scan is used,
            which is normally right because the nearest ILM point is the one on
            the marker's own side of the cup.

    Returns:
        dict with ``value_um`` ``(n_bscans, 2)``, ``angle_deg`` ``(n_bscans, 2)``
        and ``sector``-ready angles; NaN on B-scans without markers.
    """
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
    """Average :func:`minimum_rim_width` per sector.

    Each marker is assigned by its own angle about the disc, not by the angle of
    the B-scan it sits on: one radial crosses the disc, so its two markers fall
    in opposite sectors.

    Args:
        scheme: name -> (start_deg, end_deg), e.g. :data:`QUADRANTS` or
            :data:`GARWAY_HEATH`. Ranges may wrap past 360.
        rotation_deg: rotate the sector grid by this much before assigning.
            The Spectralis Glaucoma Module does not put sector boundaries on the
            horizontal -- it aligns them to each eye's **fovea-to-BMO-centre
            (FoBMO) axis**, which differs by several degrees between patients.
            The axis is not stored in the export, but :func:`fobmo_angle`
            reconstructs it; pass ``-fobmo_angle(...)["angle_deg"]`` to rotate
            the grid onto it. Left at 0 the sectors sit on the horizontal.

    Returns:
        dict with ``global_um``, ``sectors`` (name -> mean), ``counts`` and the
        raw :func:`minimum_rim_width` output under ``raw``.
    """
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
# ---------------------------------------------------------------------------
def find_fovea(macular_acq):
    """Locate the fovea in a macular raster: ``(frame, column)``.

    Detected as a *dip* in the ILM-BM thickness map rather than its global
    minimum -- peripheral retina is genuinely thinner than the fovea, so a plain
    ``argmin`` lands at the edge of the scan. A difference-of-Gaussians responds
    to the local depression instead, and the two blurs are specified in
    millimetres so the ~10:1 anisotropy between B-scan spacing and A-scan
    spacing is handled correctly.
    """
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
    """Translation mapping macular-localizer pixels onto the ONH localizer.

    Both are 30-degree IR images of the same eye differing only in resolution,
    so the mapping is a pure translation once the smaller is upsampled. Uses
    *masked* normalised cross-correlation: the two views only partly overlap,
    and plain phase correlation picks a wrong peak on roughly a third of visits.

    Returns ``(d_row, d_col)`` in ONH-localizer pixels, to be added to macular
    coordinates already scaled to the ONH grid.
    """
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
    """Fovea-to-BMO-centre axis, in degrees, for one examination.

    This is the axis the Spectralis Glaucoma Module aligns its sectors to. It is
    not stored anywhere in the export, so it is reconstructed: find the fovea in
    the macular raster, register that scan's localizer onto the ONH localizer,
    and take the angle from the BMO centre to the fovea.

    Sign matches :func:`minimum_rim_width`'s angles (from temporal, increasing
    towards superior), so feed the *negated* value to
    ``rim_width_by_sector(rotation_deg=...)`` to rotate the grid onto the axis.

    Returns a dict with ``angle_deg``, ``disc_fovea_mm``, ``fovea_px``,
    ``disc_px`` and ``shift_px``. ``disc_fovea_mm`` is the sanity check worth
    asserting on: anything outside roughly 3.5-6 mm means the registration
    picked the wrong peak.
    """
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
# Index-based sectors (using the scan pattern's own alignment)
# ---------------------------------------------------------------------------
#: Marker counts per sector, walking from temporal towards superior. The 48
#: markers of a 24-radial ONH scan sit ~7.5 degrees apart, so these counts are
#: sector widths of 90/45/45/90/45/45 degrees.
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
    """The markers ordered around the disc.

    Returns ``(values_um, angles_deg, ids)`` sorted by angle, where ``ids`` are
    ``(frame, marker)`` pairs. On a 24-radial ONH scan this is 48 points spaced
    about 7.5 degrees apart.
    """
    m = minimum_rim_width(acq, layer=layer)
    values, angles = m["value_um"].ravel(), m["angle_deg"].ravel()
    ids = [(i // 2, i % 2) for i in range(values.size)]
    ok = np.isfinite(values) & np.isfinite(angles)
    values, angles = values[ok], angles[ok]
    ids = [ids[i] for i in np.flatnonzero(ok)]
    order = np.argsort(angles)
    return values[order], angles[order], [ids[i] for i in order]


def temporal_reference(acq, layer="ILM"):
    """The marker that the scan pattern places on the temporal meridian.

    HEYEX rotates the ONH radial pattern onto the eye's fovea-to-BMO axis at
    acquisition time, so the *middle* B-scan of the star already lies along that
    axis -- measured against :func:`fobmo_angle` on a five-visit export, the two
    agree to about 1 degree (mean |diff| 0.8, SD 0.9). That makes the scan index
    itself a FoBMO-aligned reference, with no fovea detection or localizer
    registration needed.

    Which of that B-scan's two markers is the temporal one flips with laterality
    (they are 180 degrees apart), so the nearer to the temporal meridian is
    taken -- unambiguous, since the axis only tilts by a few degrees.

    Returns the marker's rank in :func:`marker_ring`.
    """
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
    """Sector means built by counting markers around the disc, not by angle.

    An alternative to :func:`rim_width_by_sector`. Instead of cutting the ring
    at fixed angles, it walks a fixed number of markers per sector starting from
    a temporal reference. Because the scan pattern is already FoBMO-aligned (see
    :func:`temporal_reference`), this inherits that alignment for free, and it
    also guarantees equal marker counts per sector -- angle cuts can move a
    point between neighbouring sectors and shift a mean by several micrometres.

    Args:
        reference: ``"scan"`` to use :func:`temporal_reference`; ``"temporal"``
            to use whichever marker is closest to the horizontal temporal
            meridian; a ``(frame, marker)`` tuple; or an integer ring rank.
        counts: markers per sector, in ``names`` order, walking towards
            superior. Must not exceed the number of markers.

    Returns:
        dict with ``sectors``, ``counts``, ``angle_spans``, ``global_um``,
        ``reference`` and ``unassigned`` (markers left over if the counts do not
        sum to the ring length).
    """
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
