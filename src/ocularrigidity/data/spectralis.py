"""Read Heidelberg Spectralis (HEYEX) OCT XML exports."""

from __future__ import annotations

import ntpath
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Optional
import pandas as pd
import numpy as np
import cv2


# --------------------------------------------------------------------------- #
# Low-level reading helpers
def _open_xml(path):
    """Open a local or ``smb://`` path for reading bytes."""
    path = str(path)
    if path.startswith("smb://"):
        from ocularrigidity.data.io import _open  # lazy: pulls smbclient etc.

        return _open(path, "rb")
    return open(path, "rb")


def _strip_namespaces(root: ET.Element) -> ET.Element:
    """Drop any ``{namespace}`` prefix from every tag in the tree."""
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.rsplit("}", 1)[1]
    return root


def read_root(source) -> ET.Element:
    """Parse an XML file (or accept an already-parsed element) -> root element."""
    if isinstance(source, ET.Element):
        return _strip_namespaces(source)
    try:
        with _open_xml(source) as f:
            raw = f.read()
        try:
            root = ET.fromstring(raw)
        except ET.ParseError:
            root = ET.fromstring(raw.decode("latin-1"))
    except (OSError, ET.ParseError) as exc:
        raise OSError(f"Failed to read XML file {source!r}: {exc}") from exc
    return _strip_namespaces(root)


def _text(node: Optional[ET.Element], tag: str) -> Optional[str]:
    """Stripped text of the first ``tag`` child, or ``None``."""
    if node is None:
        return None
    child = node.find(tag)
    if child is None or child.text is None:
        return None
    txt = child.text.strip()
    return txt or None


def _int(node: Optional[ET.Element], tag: str) -> Optional[int]:
    txt = _text(node, tag)
    return int(round(float(txt))) if txt is not None else None


def _float(node: Optional[ET.Element], tag: str) -> Optional[float]:
    txt = _text(node, tag)
    return float(txt) if txt is not None else None


def _first(node: Optional[ET.Element], *tags: str) -> Optional[ET.Element]:
    """First child matching any of ``tags`` (tolerant to naming variants)."""
    if node is None:
        return None
    for tag in tags:
        child = node.find(tag)
        if child is not None:
            return child
    return None


def _leaf_dict(node: Optional[ET.Element]) -> dict[str, str]:
    """Map every simple leaf child (tag -> text) of ``node``."""
    out: dict[str, str] = {}
    if node is None:
        return out
    for child in node:
        if len(child) == 0 and child.text and child.text.strip():
            out[child.tag] = child.text.strip()
    return out


def _coord(node: Optional[ET.Element]) -> Optional[tuple[float, float]]:
    """Parse a ``<Start>/<End>`` node -> (x, y) in millimetres."""
    coord = _first(node, "Coord")
    if coord is None:
        return None
    x, y = _float(coord, "X"), _float(coord, "Y")
    if x is None and y is None:
        return None
    return (x, y)


# --------------------------------------------------------------------------- #
# Data model
@dataclass(frozen=True)
class AcquisitionTime:
    """Wall-clock time an image was acquired (Spectralis stores no date here)."""

    hour: int
    minute: int
    second: float
    utc_bias: Optional[int] = None  # offset from UTC, in minutes

    @property
    def seconds_of_day(self) -> float:
        """Seconds since midnight"""
        return self.hour * 3600 + self.minute * 60 + self.second

    def to_time(self) -> time:
        whole = int(self.second)
        micro = int(round((self.second - whole) * 1_000_000))
        # guard against rounding 0.9999..
        if micro >= 1_000_000:
            whole, micro = whole + 1, 0
        return time(self.hour, self.minute, whole, micro)

    def __str__(self) -> str:
        return self.to_time().isoformat()


@dataclass(frozen=True)
class Image:
    """One image inside a Series (the IR/fundus localizer or an OCT B-scan)."""

    kind: str  # "oct", "fundus", or "unknown"
    width: Optional[int] = None  # pixels
    height: Optional[int] = None  # pixels
    scale_x: Optional[float] = None  # lateral pixel spacing, mm / pixel
    scale_y: Optional[float] = None  # axial pixel spacing,   mm / pixel
    num_average: Optional[int] = None
    quality: Optional[float] = None
    start_xy: Optional[tuple[float, float]] = None  # B-scan start, mm
    end_xy: Optional[tuple[float, float]] = None  # B-scan end, mm
    acquisition_time: Optional[AcquisitionTime] = None  # each image is timed
    file_path: Optional[str] = None  # ExamURL, as stored (e.g. file:///C:\...)
    context: dict[str, str] = field(default_factory=dict)  # every other leaf

    @property
    def file_name(self) -> Optional[str]:
        # ExamURL looks like "file:///C:\XMLDATA\90F83740.tif"
        return ntpath.basename(self.file_path) if self.file_path else None

    @property
    def lateral_resolution(self) -> Optional[float]:
        """Lateral (en-face) resolution in mm / pixel == ScaleX."""
        return self.scale_x

    @property
    def axial_resolution(self) -> Optional[float]:
        """Axial (depth) resolution in mm / pixel == ScaleY (OCT only)."""
        return self.scale_y


@dataclass(frozen=True)
class Series:
    """A Spectralis acquisition series: one fundus localizer + one OCT scan."""

    series_id: Optional[int] = None
    laterality: Optional[str] = None  # "OD" / "OS" / "R" / "L" if present
    acquisition_time: Optional[AcquisitionTime] = None
    oct: Optional[Image] = None
    fundus: Optional[Image] = None
    context: dict[str, str] = field(default_factory=dict)  # series-level leaves

    # ---- convenience pass-throughs to the OCT image -------------------------
    @property
    def axial_resolution(self) -> Optional[float]:
        return self.oct.axial_resolution if self.oct else None

    @property
    def lateral_resolution(self) -> Optional[float]:
        return self.oct.lateral_resolution if self.oct else None

    @property
    def oct_file_name(self) -> Optional[str]:
        return self.oct.file_name if self.oct else None

    @property
    def fundus_file_name(self) -> Optional[str]:
        return self.fundus.file_name if self.fundus else None


@dataclass(frozen=True)
class Patient:
    last_name: Optional[str] = None
    first_name: Optional[str] = None
    sex: Optional[str] = None
    patient_id: Optional[str] = None
    birth_date: Optional[date] = None
    context: dict[str, str] = field(default_factory=dict)  # every other leaf

    @property
    def full_name(self) -> str:
        return " ".join(p for p in (self.first_name, self.last_name) if p)


@dataclass(frozen=True)
class SpectralisStudy:
    """Top-level handle returned by :meth:`from_file` / :func:`analyze`."""

    patient: Patient
    series: list[Series]
    study_date: Optional[date] = None
    source: Optional[str] = None  # path it was read from, for traceability

    # ---- constructors -------------------------------------------------------
    @classmethod
    def from_file(cls, path) -> "SpectralisStudy":
        return _parse_study(read_root(path), source=str(path))

    @classmethod
    def from_root(cls, root: ET.Element) -> "SpectralisStudy":
        return _parse_study(read_root(root), source=None)

    # ---- timestamps (the headline use case) ---------------------------------
    @property
    def timestamps(self) -> list[Optional[AcquisitionTime]]:
        return [s.acquisition_time for s in self.series]

    def datetimes(self) -> list[Optional[datetime]]:
        """Full ``datetime`` per series, combining ``study_date`` + B-scan time."""
        out: list[Optional[datetime]] = []
        for s in self.series:
            if self.study_date is not None and s.acquisition_time is not None:
                out.append(
                    datetime.combine(self.study_date, s.acquisition_time.to_time())
                )
            else:
                out.append(None)
        return out

    # ---- optional flat view (only when you really want a table) -------------
    def to_dataframe(self):
        """One row per series."""
        import pandas as pd

        rows = []
        for s in self.series:
            t = s.acquisition_time
            rows.append(
                {
                    "series_id": s.series_id,
                    "laterality": s.laterality,
                    "time": t.to_time().isoformat() if t else None,
                    "seconds_of_day": t.seconds_of_day if t else None,
                    "utc_bias": t.utc_bias if t else None,
                    "lateral_resolution": s.lateral_resolution,
                    "axial_resolution": s.axial_resolution,
                    "oct_width": s.oct.width if s.oct else None,
                    "oct_height": s.oct.height if s.oct else None,
                    "num_average": s.oct.num_average if s.oct else None,
                    "quality": s.oct.quality if s.oct else None,
                    "oct_file": s.oct_file_name,
                    "fundus_file": s.fundus_file_name,
                }
            )
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Parsing  (port of analyzeSpectralisXML.m, navigating by tag name)
def _classify_image(img_el: ET.Element) -> str:
    """Best-effort fundus/OCT classification from ``ImageType/Type``."""
    type_txt = (_text(img_el, "ImageType/Type") or "").upper()
    if "OCT" in type_txt:
        return "oct"
    if "LOCALIZER" in type_txt or "FUNDUS" in type_txt or "IR" in type_txt:
        return "fundus"
    return "unknown"


def _parse_image(img_el: ET.Element, kind: str) -> Image:
    ctx = img_el.find("OphthalmicAcquisitionContext")
    return Image(
        kind=kind,
        width=_int(ctx, "Width"),
        height=_int(ctx, "Height"),
        scale_x=_float(ctx, "ScaleX"),
        scale_y=_float(ctx, "ScaleY"),
        # tag name varies across HEYEX versions -> try the common spellings
        num_average=_int(ctx, "NumAve")
        or _int(ctx, "NumAverage")
        or _int(ctx, "NumImages"),
        quality=_float(ctx, "ImageQuality") or _float(ctx, "Quality"),
        start_xy=_coord(_first(ctx, "Start")),
        end_xy=_coord(_first(ctx, "End")),
        acquisition_time=_parse_time(img_el),
        file_path=_text(img_el, "ImageData/ExamURL"),
        context=_leaf_dict(ctx),
    )


def _parse_time(img_el: ET.Element) -> Optional[AcquisitionTime]:
    t = img_el.find("AcquisitionTime/Time")
    if t is None:
        return None
    hour, minute, second = _int(t, "Hour"), _int(t, "Minute"), _float(t, "Second")
    if hour is None and minute is None and second is None:
        return None
    return AcquisitionTime(
        hour=hour or 0,
        minute=minute or 0,
        second=second if second is not None else 0.0,
        utc_bias=_int(t, "UTCBias"),
    )


def _parse_series(series_el: ET.Element) -> list[Series]:
    """One :class:`Series` per OCT B-scan."""
    images = series_el.findall("Image")
    classified = [(img, _classify_image(img)) for img in images]

    oct_els = [img for img, k in classified if k == "oct"]
    fundus_el = next((img for img, k in classified if k == "fundus"), None)

    if not oct_els and len(images) >= 2:
        oct_els = [images[1]]
    if fundus_el is None and len(images) >= 1:
        fundus_el = images[0] if images[0] not in oct_els else None

    fundus_img = _parse_image(fundus_el, "fundus") if fundus_el is not None else None
    series_id = _int(series_el, "ID")
    laterality = _text(series_el, "Laterality") or _text(series_el, "Eye")
    context = _leaf_dict(series_el)

    out = []
    for oct_el in oct_els or [None]:
        oct_img = _parse_image(oct_el, "oct") if oct_el is not None else None

        # The OCT B-scan time is the one that matters for a time series
        acq_time = oct_img.acquisition_time if oct_img else None
        if acq_time is None and fundus_img is not None:
            acq_time = fundus_img.acquisition_time

        out.append(
            Series(
                series_id=series_id,
                laterality=laterality,
                acquisition_time=acq_time,
                oct=oct_img,
                fundus=fundus_img,
                context=context,
            )
        )
    return out


def _parse_patient(patient_el: ET.Element) -> Patient:
    birth = patient_el.find("Birthdate/Date") or patient_el.find("BirthDate/Date")
    birth_date = None
    if birth is not None:
        y, m, d = _int(birth, "Year"), _int(birth, "Month"), _int(birth, "Day")
        if y and m and d:
            birth_date = date(y, m, d)
    return Patient(
        last_name=_text(patient_el, "LastName"),
        # HEYEX exports use the plural "FirstNames"
        first_name=_text(patient_el, "FirstNames") or _text(patient_el, "FirstName"),
        sex=_text(patient_el, "Sex") or _text(patient_el, "Gender"),
        patient_id=_text(patient_el, "PatientID") or _text(patient_el, "ID"),
        birth_date=birth_date,
        context=_leaf_dict(patient_el),
    )


def _parse_study_date(study_el: ET.Element) -> Optional[date]:
    node = _first(study_el, "StudyDate", "ExamDate")
    d = _first(node, "Date") if node is not None else None
    if d is None:
        return None
    y, m, dd = _int(d, "Year"), _int(d, "Month"), _int(d, "Day")
    return date(y, m, dd) if (y and m and dd) else None


def _parse_study(root: ET.Element, source: Optional[str]) -> SpectralisStudy:
    patient_el = root.find("BODY/Patient")
    if patient_el is None:
        raise ValueError("No BODY/Patient node found -- not a Spectralis XML export?")

    study_el = patient_el.find("Study")
    if study_el is None:
        raise ValueError("No Study node found under Patient.")

    series = [e for s in study_el.findall("Series") for e in _parse_series(s)]
    return SpectralisStudy(
        patient=_parse_patient(patient_el),
        series=series,
        study_date=_parse_study_date(study_el),
        source=source,
    )


# --------------------------------------------------------------------------- #
# Functional alias mirroring analyzeSpectralisXML.m
def analyze(path) -> SpectralisStudy:
    """Drop-in replacement for ``analyzeSpectralisXML`` returning one object."""
    return SpectralisStudy.from_file(path)


def load_video(path, return_fundus=False):
    """Returns a np.ndarray and the timestamps associated."""
    xml_file = list(path.glob("*.xml"))
    if not xml_file:
        raise ValueError(f"No XML file found in {path}")
    xml_file = xml_file[0]
    study = analyze(xml_file)
    df = study.to_dataframe()
    oct_width = df["oct_width"].iloc[0]
    oct_height = df["oct_height"].iloc[0]
    N = len(df)
    oct_files = df["oct_file"].to_list()
    video = np.zeros((N, oct_height, oct_width), dtype=np.uint8)
    for i, f in enumerate(oct_files):
        img_path = path / f
        if not img_path.exists():
            raise ValueError(f"File {img_path} does not exist")
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image {img_path}")
        video[i] = img

    timestamps = df["time"]

    if return_fundus:
        fundus_files = df["fundus_file"].to_list()
        fundus_video = np.zeros((N, 768, 768), dtype=np.uint8)
        for i, f in enumerate(fundus_files):
            img_path = path / f
            if not img_path.exists():
                raise ValueError(f"File {img_path} does not exist")
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to read image {img_path}")
            fundus_video[i] = img
        return video, fundus_video, timestamps, df
    else:
        return video, timestamps, df
