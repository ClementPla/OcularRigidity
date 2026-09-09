"""Read Heidelberg HEYEX DICOM exports (a DICOMDIR file-set).

Complements :mod:`ocularrigidity.data.spectralis`, which reads the XML export of
the same device. The DICOM export is the richer of the two: it keeps the
per-frame B-scan geometry on the localizer, which is what lets us tell a macular
raster from an ONH radial-plus-circles acquisition without guessing.

Hierarchy recovered from the file-set::

    patient -> study (one per visit) -> eye (R/L) -> acquisition

with, per eye and visit, one macular raster, one ONH acquisition, their two
localizers and an encapsulated PDF report.

**Where the segmentation lives.** No public DICOM tag carries it. HEYEX embeds
two complete E2E files in a private block (creator "Ashvins Private Object
Information", group 0x0051):

    0x13 -> E2E blob holding the B-scan IMAGE folders
    0x23 -> E2E blob holding bscanmeta + LAYER_ANNOTATION folders

**Why we default to the blob images rather than PixelData.** The two are the
same B-scans, but HEYEX vertically realigns each frame when it writes
``PixelData``, while the layer heights stay in the *unaligned* E2E frame. Layers
then land on the wrong rows for most frames — tens of pixels out, occasionally
over a hundred — and the per-frame offset is often zero, which makes the bug
easy to miss on a lucky frame. So ``load_acquisition`` reads images from the
blob by default; ask for ``bscan_source="dicom"`` only if you need the DICOM
greyscale, and the layers are then mapped into that frame for you (see
``estimate_frame_affine``).
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from eyepy.core.utils import from_e2e_intensity
from eyepy.io.he import HeE2eReader
from eyepy.io.he.vol_reader import SEG_MAPPING

__all__ = [
    "Acquisition",
    "build_tree",
    "classify_scan",
    "apply_frame_affine",
    "estimate_frame_affine",
    "estimate_frame_offsets",
    "extract_pdfs",
    "frame_geometry",
    "index_export",
    "load_acquisition",
    "read_bscans_from_blob",
    "read_frame_affine",
    "read_layers",
    "read_marker_points",
]

#: HEYEX layer ids -> eyepy layer names (0 = ILM, 1 = BM, 2 = RNFL, ...).
LAYER_ID_TO_NAME = {v: k for k, v in SEG_MAPPING.items()}

PRIVATE_CREATOR = "Ashvins Private Object Information"
PRIVATE_GROUP = 0x0051
BLOCK_META = 0x23
BLOCK_IMAGE = 0x13

SOP_OPT = "1.2.840.10008.5.1.4.1.1.77.1.5.4"  # Ophthalmic Tomography Image
SOP_OP8 = "1.2.840.10008.5.1.4.1.1.77.1.5.1"  # Ophthalmic Photography 8 Bit
SOP_PDF = "1.2.840.10008.5.1.4.1.1.104.1"  # Encapsulated PDF

_KIND_BY_SOP = {SOP_OPT: "OCT", SOP_OP8: "LOCALIZER", SOP_PDF: "PDF"}


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
def index_export(root: str | Path) -> list[dict[str, Any]]:
    """One row per instance in the export.

    Reads the files rather than the ``DICOMDIR`` records: HEYEX populates the
    directory records sparsely (study rows carry no laterality, series rows no
    frame count), so walking the tree is both simpler and more informative.
    Feed the result to :class:`pandas.DataFrame` if you want to pivot it.
    """
    root = Path(root)
    files = [p for p in (root / "DICOM").rglob("*") if p.is_file()]
    if not files:
        files = [
            p
            for p in root.rglob("*")
            if p.is_file() and p.name != "DICOMDIR" and "IHE_PDI" not in p.parts
        ]

    rows = []
    for path in sorted(files):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except Exception:
            continue  # thumbnails, index files, anything not a Part-10 object
        sop = str(ds.get("SOPClassUID", ""))
        rows.append(
            {
                "path": str(path),
                "patient_id": str(ds.get("PatientID", "")),
                "patient_name": str(ds.get("PatientName", "")),
                "study_date": str(ds.get("StudyDate", "")),
                "study_uid": str(ds.get("StudyInstanceUID", "")),
                "series_uid": str(ds.get("SeriesInstanceUID", "")),
                "sop_uid": str(ds.get("SOPInstanceUID", "")),
                "modality": str(ds.get("Modality", "")),
                "sop_class": sop,
                "kind": _KIND_BY_SOP.get(sop, "OTHER"),
                "laterality": str(
                    ds.get("ImageLaterality", ds.get("Laterality", "")) or ""
                ),
                "series_desc": str(ds.get("SeriesDescription", "")),
                "n_frames": int(ds.get("NumberOfFrames", 1) or 1),
                "rows": ds.get("Rows"),
                "cols": ds.get("Columns"),
                "acq_datetime": str(ds.get("AcquisitionDateTime", "")),
            }
        )
    return rows


def build_tree(root: str | Path) -> dict:
    """``patient -> study_date -> laterality -> [acquisition summary]``."""
    tree: dict = {}
    for row in index_export(root):
        if row["kind"] != "OCT":
            continue
        ds = pydicom.dcmread(row["path"], stop_before_pixels=True)
        label, _ = classify_scan(frame_geometry(ds))
        (
            tree.setdefault(row["patient_id"], {})
            .setdefault(row["study_date"], {})
            .setdefault(row["laterality"], [])
            .append(
                {
                    "label": label,
                    "n_frames": row["n_frames"],
                    "series_desc": row["series_desc"],
                    "path": row["path"],
                }
            )
        )
    return tree


# ---------------------------------------------------------------------------
# Private E2E blobs
# ---------------------------------------------------------------------------
def _as_dataset(ds_or_path) -> pydicom.Dataset:
    if isinstance(ds_or_path, pydicom.Dataset):
        return ds_or_path
    return pydicom.dcmread(ds_or_path, stop_before_pixels=True)


def _private_blob(ds: pydicom.Dataset, block_offset: int) -> bytes | None:
    """Raw bytes of one HEYEX private E2E blob, or None if absent."""
    try:
        return bytes(ds.private_block(PRIVATE_GROUP, PRIVATE_CREATOR)[block_offset].value)
    except Exception:
        # Creator string missing or renamed: address the element directly. The
        # block is always 0x10 in the exports we have seen.
        element = ds.get((PRIVATE_GROUP, 0x1000 + block_offset))
        return bytes(element.value) if element is not None and element.value else None


def _pick_series(reader: HeE2eReader, expect_bscans: int | None):
    """The one real series in a blob (each blob holds a single acquisition)."""
    candidates = [s for s in reader.series if s.n_bscans > 1]
    if expect_bscans is not None:
        candidates = [s for s in candidates if s.n_bscans == expect_bscans] or candidates
    return max(candidates, key=lambda s: s.n_bscans) if candidates else None


def _open_blob(ds: pydicom.Dataset, block: int):
    """Context-manager-free helper: HeE2eReader needs a real path on disk."""
    blob = _private_blob(ds, block)
    if blob is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".e2e", delete=False) as fh:
        fh.write(blob)
        return fh.name


def read_layers(ds_or_path, expect_bscans: int | None = None):
    """Layer segmentation and B-scan metadata from the private E2E blob.

    Returns ``(layers, bscan_meta)`` where ``layers`` maps a layer name to an
    ``(n_bscans, width)`` array of row positions (NaN where unset), in DICOM
    frame order, and ``bscan_meta`` is a list of eyepy ``EyeBscanMeta``.
    Both are empty when the instance carries no blob.

    Note the coverage is informative: on an ONH acquisition ILM is annotated on
    every frame while BM and RNFL exist only on the three circle scans, which is
    the peripapillary RNFL analysis.
    """
    ds = _as_dataset(ds_or_path)
    tmp = _open_blob(ds, BLOCK_META)
    if tmp is None:
        return {}, None
    try:
        with HeE2eReader(tmp) as reader:
            series = _pick_series(reader, expect_bscans)
            if series is None:
                return {}, None
            layers = {
                LAYER_ID_TO_NAME.get(k, f"id{k}"): np.asarray(v, float)
                for k, v in sorted(series.get_layers().items())
            }
            try:
                meta = series.get_bscan_meta()
            except KeyError:
                meta = None  # metadata folder absent in some blobs
        return layers, meta
    finally:
        os.unlink(tmp)


def read_bscans_from_blob(ds_or_path, expect_bscans: int | None = None):
    """B-scan images from the private E2E *image* blob, as uint8.

    These are the images the layers were drawn on -- see the module docstring
    for why they, not ``PixelData``, are the ones the annotations index.
    Returns None when the blob is absent.
    """
    ds = _as_dataset(ds_or_path)
    tmp = _open_blob(ds, BLOCK_IMAGE)
    if tmp is None:
        return None
    try:
        with HeE2eReader(tmp) as reader:
            series = _pick_series(reader, expect_bscans)
            if series is None:
                return None
            # Not series.get_bscans(): that needs the bscanmeta folder, which
            # lives in the *other* blob.
            stack = np.stack([series.slices[k].get_bscan() for k in sorted(series.slices)])
    finally:
        os.unlink(tmp)

    # eyepy's E2E transform: >1.99 marks empty regions, the rest is log-scaled.
    return from_e2e_intensity(np.asarray(stack, np.float32).copy())


#: Series-level E2E folder holding manually placed point pairs (see
#: :func:`read_marker_points`). eyepy leaves it as raw bytes.
TYPE_MARKERS = 10038

#: Per-slice E2E folder holding the export-time affine (eyepy leaves it as bytes).
TYPE_ALIGNMENT = 10012

#: Byte offsets of the y-row of the 2x3 affine inside that record.
_ALIGN_SHEAR_OFFSET = 28
_ALIGN_SHIFT_OFFSET = 36


def read_frame_affine(ds_or_path, expect_bscans=None, width=None):
    """Exact depth mapping from the E2E frame to ``PixelData``, read from file.

    HEYEX stores the export-time realignment per B-scan, in the private
    metadata blob, as a 2x3 affine in folder type 10012. eyepy does not parse
    that type, so it arrives as raw bytes; the y-row sits at byte offsets 28
    (shear) and 36 (shift), and the transform pivots about the image centre::

        dicom_row(x) = e2e_row(x) + a + b * x
        b = -shear
        a = -shift + shear * width / 2

    This is the same quantity :func:`estimate_frame_affine` measures from the
    pixels, but read rather than fitted -- prefer it when the record is present.

    Returns an ``(n, 2)`` array of ``(a, b)``, or None if the record is absent.
    """
    import struct

    ds = _as_dataset(ds_or_path)
    tmp = _open_blob(ds, BLOCK_META)
    if tmp is None:
        return None
    try:
        with HeE2eReader(tmp) as reader:
            series = _pick_series(reader, expect_bscans)
            if series is None:
                return None
            if width is None:
                width = int(series.get_bscan_meta()[0]["size_x"])
            records = []
            for key in sorted(series.slices):
                folders = series.slices[key].folders.get(TYPE_ALIGNMENT, [])
                if not folders:
                    return None
                data = folders[0].data
                if not isinstance(data, (bytes, bytearray)) or len(data) < 40:
                    return None
                records.append(bytes(data))
    finally:
        os.unlink(tmp)

    shear = np.array([struct.unpack_from("<f", r, _ALIGN_SHEAR_OFFSET)[0] for r in records])
    shift = np.array([struct.unpack_from("<f", r, _ALIGN_SHIFT_OFFSET)[0] for r in records])
    return np.c_[-shift + shear * (width / 2.0), -shear]


def read_marker_points(ds_or_path, expect_bscans=None, line_frames=None):
    """Manually placed point pairs -- the disc-margin / RNFL-endpoint markers.

    These are the points you drop in the HEYEX interface on the ONH radial
    B-scans, two per scan (one either side of the disc). They are *not* layers,
    so they never show up in :func:`read_layers`; they live in their own
    series-level record, folder type 10038, which eyepy leaves as raw bytes.

    Layout, little-endian::

        int32  kind        (102 in our exports)
        int32  n_records   (equals the number of *line* B-scans)
        n_records x 18 bytes:
            int32 x1, int32 y1, int32 x2, int32 y2, 2 bytes padding

    The 18-byte stride is the thing to watch: the record is not 4-byte aligned,
    so reading it as a flat int32 array silently garbles everything after the
    first entry.

    Coordinates are pixel positions in the *E2E* B-scan frame, the same frame as
    :func:`read_layers`, so the same affine applies when moving to ``PixelData``.

    Args:
        line_frames: indices of the B-scans the records belong to, in order.
            Records are written for the line B-scans only -- an ONH acquisition
            stores 24, one per radial, and nothing for the 3 circle scans.
            Defaults to the first ``n_records`` frames.

    Returns:
        ``(n_bscans, 2, 2)`` array of ``[[x1, y1], [x2, y2]]`` per B-scan, NaN
        where no point was placed, or None if the record is absent.
    """
    import struct

    ds = _as_dataset(ds_or_path)
    tmp = _open_blob(ds, BLOCK_META)
    if tmp is None:
        return None
    try:
        with HeE2eReader(tmp) as reader:
            series = _pick_series(reader, expect_bscans)
            if series is None:
                return None
            folders = series.folders.get(TYPE_MARKERS, [])
            if not folders:
                return None
            data = bytes(folders[0].data)
            n_bscans = series.n_bscans
    finally:
        os.unlink(tmp)

    if len(data) < 8:
        return None
    _kind, n_records = struct.unpack_from("<ii", data, 0)
    if len(data) < 8 + 18 * n_records:
        return None

    out = np.full((n_bscans, 2, 2), np.nan)
    frames = list(line_frames) if line_frames is not None else list(range(n_records))
    for i in range(min(n_records, len(frames))):
        x1, y1, x2, y2 = struct.unpack_from("<iiii", data, 8 + 18 * i)
        out[frames[i]] = [[x1, y1], [x2, y2]]
    return out


def apply_frame_affine_points(points, affine):
    """Move marker points from the E2E frame into the ``PixelData`` frame."""
    points = np.asarray(points, float).copy()
    affine = np.asarray(affine, float)
    # y' = y + a + b * x, with x the point's own column.
    points[..., 1] += affine[:, 0:1] + affine[:, 1:2] * points[..., 0]
    return points


def _column_shifts(dicom_frame, e2e_frame, search=170, min_std=1e-3):
    """Per-A-scan vertical shift between two versions of the same B-scan.

    Returns ``(dy, quality)``, both length ``width``; ``dy`` is NaN where the
    column carries no signal. Correlating each column separately (rather than
    the frame as a whole) is what exposes the tilt.
    """
    d = np.asarray(dicom_frame, float)
    e = np.asarray(e2e_frame, float)
    height, width = d.shape

    d = d - d.mean(axis=0)
    e = e - e.mean(axis=0)
    nd = np.linalg.norm(d, axis=0)
    ne = np.linalg.norm(e, axis=0)

    dy = np.full(width, np.nan)
    quality = np.zeros(width)
    lo, hi = height - 1 - search, height - 1 + search
    for x in range(width):
        if nd[x] < min_std or ne[x] < min_std:
            continue
        cc = np.correlate(d[:, x], e[:, x], mode="full") / (nd[x] * ne[x])
        k = int(np.argmax(cc[lo : hi + 1])) + lo
        dy[x] = k - (height - 1)
        quality[x] = cc[k]
    return dy, quality


def _robust_line(x, y, iters=3, min_points=20):
    """Least-squares line with iterative 3-MAD trimming. Returns (a, b)."""
    if x.size < min_points:
        return np.nan, np.nan
    keep = np.ones(x.size, bool)
    coef = (np.nan, np.nan)
    for _ in range(iters):
        if keep.sum() < min_points:
            break
        design = np.c_[np.ones(keep.sum()), x[keep]]
        coef, *_ = np.linalg.lstsq(design, y[keep], rcond=None)
        resid = y - (coef[0] + coef[1] * x)
        mad = np.median(np.abs(resid[keep] - np.median(resid[keep]))) or 1.0
        keep = np.abs(resid) <= 3.0 * 1.4826 * mad
    return float(coef[0]), float(coef[1])


def estimate_frame_affine(dicom_bscans, e2e_bscans, min_quality=0.5, step=4):
    """Per-frame depth mapping from the E2E frame to ``PixelData``.

    HEYEX does not merely translate each B-scan when it writes ``PixelData`` --
    it also tilts it, so the correction is affine *along the A-scan axis*::

        dicom_row(x) = e2e_row(x) + a + b * x

    Measured on a five-visit export this model is exact to well under a pixel
    (median residual 0.25 px, p90 0.5 px), whereas a per-frame constant leaves
    up to ~95 px of tilt on a single frame.

    Args:
        dicom_bscans: ``(n, height, width)`` stack from ``PixelData``.
        e2e_bscans: the matching stack from the private image blob.
        min_quality: minimum per-column correlation to trust a shift estimate.
        step: sample every ``step``-th column (the tilt is smooth; 4 is plenty).

    Returns:
        ``(n, 2)`` array of ``(a, b)``. Frames with too little signal to fit are
        filled by linear interpolation over the frame index, so the result is
        always finite and usable.
    """
    dicom = np.asarray(dicom_bscans, float)
    e2e = np.asarray(e2e_bscans, float)
    n, _, width = dicom.shape
    xs = np.arange(0, width, step)

    out = np.full((n, 2), np.nan)
    for i in range(n):
        dy, quality = _column_shifts(dicom[i][:, xs], e2e[i][:, xs])
        good = np.isfinite(dy) & (quality > min_quality)
        out[i] = _robust_line(xs[good].astype(float), dy[good])

    # Fill unfittable frames from their neighbours rather than returning NaN.
    idx = np.arange(n)
    for col in (0, 1):
        ok = np.isfinite(out[:, col])
        if ok.all():
            continue
        out[:, col] = np.interp(idx, idx[ok], out[ok, col]) if ok.any() else 0.0
    return out


def apply_frame_affine(layers, affine, width=None):
    """Move layer heights from the E2E frame into the ``PixelData`` frame.

    Args:
        layers: ``{name: (n_bscans, width)}`` as returned by :func:`read_layers`.
        affine: ``(n_bscans, 2)`` array of ``(a, b)`` from
            :func:`estimate_frame_affine`.
        width: A-scan count; taken from the arrays when omitted.

    Returns:
        A new dict with the same keys and shapes. NaNs are preserved.
    """
    affine = np.asarray(affine, float)
    out = {}
    for name, values in layers.items():
        values = np.asarray(values, float)
        w = width or values.shape[1]
        x = np.arange(w)[None, :]
        out[name] = values + affine[:, 0:1] + affine[:, 1:2] * x
    return out


def estimate_frame_offsets(dicom_bscans, e2e_bscans) -> np.ndarray:
    """Per-frame constant offset only. Kept for the simple case.

    Prefer :func:`estimate_frame_affine`: a constant ignores the tilt and can
    be ~50 px out at the edges of a frame.
    """
    return estimate_frame_affine(dicom_bscans, e2e_bscans)[:, 0]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def frame_geometry(ds: pydicom.Dataset) -> list[dict[str, Any]]:
    """Per-frame B-scan location on the localizer.

    From ``OphthalmicFrameLocationSequence`` (0022,0031). ``orientation`` is
    ``LINEAR`` for a line B-scan, whose ``coords`` are the two endpoints, or
    ``NONLINEAR`` for a circle scan, whose ``coords`` are the full traced
    polyline (768 points) -- so a circle's centre and radius come straight out
    of the mean, with no convention to remember.

    ``coords`` are returned as **(x, y)** localizer pixels, ready to pass to
    ``plot`` or ``imshow`` axes. The stored pairs are (row, column), i.e.
    (y, x), and are swapped here: taken literally they put the ONH scan pattern
    off the disc entirely, and only the swap centres the radial star on it.
    Distances and the concentricity test are symmetric, so classification is
    unaffected either way -- but anything directional (sector assignment,
    disc-to-fovea axis) is wrong without the swap.
    """
    out = []
    for fg in ds.PerFrameFunctionalGroupsSequence:
        ofl = fg.OphthalmicFrameLocationSequence[0]
        coords = np.array(list(ofl.ReferenceCoordinates), float).reshape(-1, 2)
        out.append(
            {
                "orientation": str(ofl.get("OphthalmicImageOrientation", "")),
                "coords": coords[:, ::-1],  # (row, col) -> (x, y)
                "ref_uid": str(ofl.get("ReferencedSOPInstanceUID", "")),
            }
        )
    return out


def classify_scan(geom: list[dict], tol: float = 1.0) -> tuple[str, dict]:
    """Label an acquisition from its frame geometry.

    Circles are flagged by ``NONLINEAR`` orientation; radial and raster are told
    apart by whether the line B-scans share one midpoint (a star pattern pivots
    about the ONH, a raster marches across the retina).
    """
    circles = [i for i, g in enumerate(geom) if g["orientation"] == "NONLINEAR"]
    lines = [i for i, g in enumerate(geom) if g["orientation"] != "NONLINEAR"]
    details: dict[str, Any] = {
        "n_frames": len(geom),
        "circle_frames": circles,
        "line_frames": lines,
    }

    if circles:
        centres, radii = [], []
        for i in circles:
            coords = geom[i]["coords"]
            centre = coords.mean(0)
            centres.append(centre)
            radii.append(float(np.hypot(*(coords - centre).T).mean()))
        details["circle_centres"] = np.array(centres)
        details["circle_radii_px"] = np.array(radii)
        details["concentric"] = bool(np.ptp(np.array(centres), axis=0).max() < tol)

    radial = False
    if lines:
        mids = np.array([geom[i]["coords"].mean(0) for i in lines])
        details["line_midpoint_spread_px"] = np.ptp(mids, axis=0)
        radial = bool(np.ptp(mids, axis=0).max() < tol)
        details["radial"] = radial

    if circles and radial:
        label = f"ONH radial + {len(circles)} circles"
    elif circles and not lines:
        label = f"{len(circles)} circle scan(s)"
    elif circles:
        label = f"mixed: {len(lines)} line + {len(circles)} circle"
    elif radial:
        label = "radial / star"
    else:
        label = "raster volume"
    return label, details


# ---------------------------------------------------------------------------
# One acquisition
# ---------------------------------------------------------------------------
@dataclass
class Acquisition:
    """One multi-frame OPT instance: pixels, geometry and layer annotations."""

    path: str
    patient_id: str
    patient_name: str
    study_date: str
    laterality: str
    series_desc: str
    label: str
    bscans: np.ndarray  # (n, height, width) uint8
    layers: dict[str, np.ndarray]  # name -> (n, width), NaN where unset
    geometry: list = field(repr=False, default_factory=list)
    details: dict = field(repr=False, default_factory=dict)
    bscan_meta: Any = field(repr=False, default=None)
    bscan_source: str = "e2e"
    frame_affine: Any = field(repr=False, default=None)  # (n, 2) of (a, b)
    markers: Any = field(repr=False, default=None)  # (n, 2, 2), NaN where unset
    pixel_spacing: tuple = ()  # (row_mm, col_mm) within a B-scan
    frame_pixel_spacing: Any = field(repr=False, default=None)  # (n, 2), ONH only
    slice_thickness: float = float("nan")
    localizer_uid: str = ""

    @property
    def n_bscans(self) -> int:
        return self.bscans.shape[0]

    @property
    def circle_frames(self) -> list[int]:
        return list(self.details.get("circle_frames", []))

    def spacing_for(self, index: int) -> tuple[float, float]:
        """``(row_mm, col_mm)`` for one B-scan, per-frame value when present."""
        if self.frame_pixel_spacing is not None:
            row, col = self.frame_pixel_spacing[index]
            return float(row), float(col)
        return self.pixel_spacing

    def layer_coverage(self) -> dict[str, float]:
        """Fraction of A-scans annotated, per layer, over the whole stack."""
        return {k: float(np.isfinite(v).mean()) for k, v in self.layers.items()}

    def circle_radii_mm(self, localizer_pixel_spacing_mm: float) -> np.ndarray:
        """Circle radii in mm, given the localizer's pixel spacing."""
        return self.details.get("circle_radii_px", np.empty(0)) * localizer_pixel_spacing_mm


def load_acquisition(path, with_pixels: bool = True, bscan_source: str = "e2e") -> Acquisition:
    """Load one OPT instance with its geometry and layer annotations.

    ``bscan_source="e2e"`` (default) takes the images from the private image
    blob, which is the coordinate frame the layers are expressed in.
    ``bscan_source="dicom"`` takes ``PixelData`` instead and fills
    :attr:`Acquisition.frame_affine` with the fitted ``(a, b)`` per frame. In
    both cases ``layers`` is returned already aligned to ``bscans``.
    """
    if bscan_source not in ("e2e", "dicom"):
        raise ValueError(f"bscan_source must be 'e2e' or 'dicom', got {bscan_source!r}")

    ds = pydicom.dcmread(path, stop_before_pixels=not with_pixels)
    if str(ds.get("SOPClassUID", "")) != SOP_OPT:
        raise ValueError(f"not an Ophthalmic Tomography instance: {path}")

    geom = frame_geometry(ds)
    label, details = classify_scan(geom)
    n_frames = int(ds.get("NumberOfFrames", 1) or 1)
    layers, bscan_meta = read_layers(ds, expect_bscans=n_frames)
    # Marker records are written for the line B-scans only, in their order.
    markers = read_marker_points(
        ds, expect_bscans=n_frames, line_frames=details.get("line_frames")
    )

    pixels: np.ndarray = np.empty((0,))
    offsets = None
    if with_pixels:
        dicom_px = ds.pixel_array
        if dicom_px.ndim == 2:
            dicom_px = dicom_px[None]
        if bscan_source == "e2e":
            pixels = read_bscans_from_blob(ds, expect_bscans=n_frames)
            if pixels is None:  # no blob: fall back, and say so
                pixels, bscan_source = dicom_px, "dicom (fallback)"
        else:
            pixels = dicom_px
            # Read the transform HEYEX stored; only measure it if absent.
            offsets = read_frame_affine(
                ds, expect_bscans=n_frames, width=dicom_px.shape[2]
            )
            if offsets is None:
                blob_px = read_bscans_from_blob(ds, expect_bscans=n_frames)
                if blob_px is not None:
                    offsets = estimate_frame_affine(dicom_px, blob_px)
            if offsets is not None:
                # Map annotations into PixelData's frame, so they always index
                # `bscans` whichever source the caller picked.
                layers = apply_frame_affine(layers, offsets)
                if markers is not None:
                    markers = apply_frame_affine_points(markers, offsets)

    shared = ds.SharedFunctionalGroupsSequence[0]
    measures = (
        shared.PixelMeasuresSequence[0] if "PixelMeasuresSequence" in shared else None
    )
    # ONH acquisitions put PixelMeasures per frame instead of shared: the lateral
    # spacing differs from radial to radial, and again for the circle scans
    # (a circle's arc is sampled over a different physical length). Distances in
    # such a B-scan are wrong unless the per-frame value is used.
    frame_spacing = None
    if "PixelMeasuresSequence" in ds.PerFrameFunctionalGroupsSequence[0]:
        frame_spacing = np.array(
            [
                [float(v) for v in fg.PixelMeasuresSequence[0].PixelSpacing]
                for fg in ds.PerFrameFunctionalGroupsSequence
            ]
        )
        if measures is None:
            measures = ds.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0]

    return Acquisition(
        path=str(path),
        patient_id=str(ds.get("PatientID", "")),
        patient_name=str(ds.get("PatientName", "")),
        study_date=str(ds.get("StudyDate", "")),
        laterality=str(ds.get("ImageLaterality", ds.get("Laterality", "")) or ""),
        series_desc=str(ds.get("SeriesDescription", "")),
        label=label,
        bscans=pixels,
        layers=layers,
        geometry=geom,
        details=details,
        bscan_meta=bscan_meta,
        markers=markers,
        bscan_source=bscan_source,
        frame_affine=offsets,
        pixel_spacing=tuple(float(x) for x in measures.PixelSpacing) if measures else (),
        frame_pixel_spacing=frame_spacing,
        slice_thickness=(
            float(measures.SliceThickness)
            if measures is not None and "SliceThickness" in measures
            else float("nan")
        ),
        localizer_uid=geom[0]["ref_uid"] if geom else "",
    )


def extract_pdfs(root: str | Path, outdir: str | Path) -> list[Path]:
    """Write every encapsulated PDF report to ``outdir``."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for row in index_export(root):
        if row["kind"] != "PDF":
            continue
        ds = pydicom.dcmread(row["path"])
        name = (
            f"{row['patient_id']}_{row['study_date']}"
            f"_{row['laterality'] or 'NA'}_{row['sop_uid'][-8:]}.pdf"
        )
        out = outdir / name
        out.write_bytes(bytes(ds.EncapsulatedDocument))
        written.append(out)
    return written
