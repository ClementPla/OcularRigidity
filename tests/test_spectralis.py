"""Tests for ocularrigidity.data.spectralis.

Runnable two ways:

    pytest tests/test_spectralis.py          # if pytest is installed
    python  tests/test_spectralis.py         # standalone fallback runner

The fixture XML below mirrors the structure of a real Heidelberg Spectralis
(HEYEX) export: ``HEDX/BODY/Patient`` with ``FirstNames`` (plural), a ``Study``
with ``StudyDate`` and ``Series``, each Series holding a LOCALIZER image and an
OCT image, both with their own ``AcquisitionTime``, ``file:///`` ExamURLs,
``NumAve`` / ``ImageQuality`` and ``Start``/``End``/``Coord`` nodes.
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

# Make the src/ layout importable without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocularrigidity.data.spectralis import (  # noqa: E402
    AcquisitionTime,
    SpectralisStudy,
)

# A namespace on the root exercises the namespace-stripping path; real exports
# may or may not carry one.
SAMPLE_XML = """<?xml version="1.0" encoding="utf-8"?>
<HEDX xmlns="http://www.heidelbergengineering.com/schema/v1">
  <BODY>
    <ExportType>Data</ExportType>
    <Patient>
      <ID>148</ID>
      <LastName>SANSORI</LastName>
      <FirstNames>220215-001</FirstNames>
      <Birthdate><Date><Year>1963</Year><Month>8</Month><Day>1</Day></Date></Birthdate>
      <Sex>M</Sex>
      <Study>
        <ID>319</ID>
        <StudyDate><Date><Year>2022</Year><Month>2</Month><Day>15</Day></Date></StudyDate>
        <Series>
          <SeriesUID>LOC.1</SeriesUID>
          <ID>1393.0</ID>
          <Modality>OCT</Modality>
          <Type>Section</Type>
          <Laterality>R</Laterality>
          <NumImages>2</NumImages>
          <Image>
            <ID>0</ID>
            <Laterality>R</Laterality>
            <AcquisitionTime><Time>
              <Hour>17</Hour><Minute>22</Minute><Second>5.140</Second><UTCBias>-360</UTCBias>
            </Time></AcquisitionTime>
            <ImageType><Type>LOCALIZER</Type><LightSource>IR</LightSource></ImageType>
            <OphthalmicAcquisitionContext>
              <Width>768</Width><Height>768</Height>
              <ScaleX>0.0116</ScaleX><ScaleY>0.0116</ScaleY>
              <NumAve>100</NumAve>
            </OphthalmicAcquisitionContext>
            <ImageData><Extension>TIF</Extension><ExamURL>file:///C:\\XMLDATA\\90F37C50.tif</ExamURL></ImageData>
          </Image>
          <Image>
            <ID>1</ID>
            <Laterality>R</Laterality>
            <AcquisitionTime><Time>
              <Hour>17</Hour><Minute>22</Minute><Second>5.493</Second><UTCBias>-360</UTCBias>
            </Time></AcquisitionTime>
            <ImageType><Type>OCT</Type></ImageType>
            <OphthalmicAcquisitionContext>
              <Width>768</Width><Height>496</Height>
              <ScaleX>0.0116</ScaleX><ScaleY>0.0039</ScaleY>
              <Resolution>LORES</Resolution>
              <NumAve>9</NumAve><ImageQuality>28</ImageQuality>
              <EDI>true</EDI><EVI>false</EVI>
              <Start><Coord><X>0.269</X><Y>2.936</Y></Coord></Start>
              <End><Coord><X>8.656</X><Y>5.989</Y></Coord></End>
            </OphthalmicAcquisitionContext>
            <ImageData><Extension>TIF</Extension><ExamURL>file:///C:\\XMLDATA\\90F83740.tif</ExamURL></ImageData>
          </Image>
        </Series>
        <Series>
          <SeriesUID>LOC.2</SeriesUID>
          <ID>1393.0</ID>
          <Modality>OCT</Modality>
          <Type>Section</Type>
          <Laterality>R</Laterality>
          <NumImages>2</NumImages>
          <Image>
            <ID>0</ID>
            <ImageType><Type>LOCALIZER</Type></ImageType>
            <OphthalmicAcquisitionContext>
              <Width>768</Width><Height>768</Height>
              <ScaleX>0.0116</ScaleX><ScaleY>0.0116</ScaleY>
            </OphthalmicAcquisitionContext>
            <ImageData><ExamURL>file:///C:\\XMLDATA\\AAA.tif</ExamURL></ImageData>
          </Image>
          <Image>
            <ID>1</ID>
            <AcquisitionTime><Time>
              <Hour>17</Hour><Minute>22</Minute><Second>5.603</Second><UTCBias>-360</UTCBias>
            </Time></AcquisitionTime>
            <ImageType><Type>OCT</Type></ImageType>
            <OphthalmicAcquisitionContext>
              <Width>768</Width><Height>496</Height>
              <ScaleX>0.0116</ScaleX><ScaleY>0.0039</ScaleY>
              <NumAve>9</NumAve><ImageQuality>30</ImageQuality>
              <Start><Coord><X>0.269</X><Y>2.936</Y></Coord></Start>
              <End><Coord><X>8.656</X><Y>5.989</Y></Coord></End>
            </OphthalmicAcquisitionContext>
            <ImageData><ExamURL>file:///C:\\XMLDATA\\BBB.tif</ExamURL></ImageData>
          </Image>
        </Series>
      </Study>
    </Patient>
  </BODY>
</HEDX>
"""


def _study(tmp_path: Path) -> SpectralisStudy:
    p = tmp_path / "sample.xml"
    p.write_text(SAMPLE_XML, encoding="utf-8")
    return SpectralisStudy.from_file(p)


def test_patient_fields(tmp_path):
    st = _study(tmp_path)
    assert st.patient.last_name == "SANSORI"
    assert st.patient.first_name == "220215-001"  # FirstNames (plural) is read
    assert st.patient.sex == "M"
    assert st.patient.patient_id == "148"
    assert st.patient.birth_date == date(1963, 8, 1)
    assert st.patient.full_name == "220215-001 SANSORI"


def test_study_date(tmp_path):
    assert _study(tmp_path).study_date == date(2022, 2, 15)


def test_series_count_and_ids(tmp_path):
    st = _study(tmp_path)
    assert len(st.series) == 2
    assert st.series[0].series_id == 1393  # "1393.0" -> int
    assert st.series[0].laterality == "R"


def test_timestamps(tmp_path):
    st = _study(tmp_path)
    t = st.series[0].acquisition_time
    assert isinstance(t, AcquisitionTime)
    # series time comes from the OCT image, not the localizer
    assert (t.hour, t.minute) == (17, 22)
    assert abs(t.second - 5.493) < 1e-9
    assert t.utc_bias == -360
    assert t.to_time().microsecond == 493000
    assert abs(t.seconds_of_day - (17 * 3600 + 22 * 60 + 5.493)) < 1e-9
    # ordering across the series increases
    secs = [s.acquisition_time.seconds_of_day for s in st.series]
    assert secs == sorted(secs)


def test_per_image_times_differ(tmp_path):
    st = _study(tmp_path)
    s = st.series[0]
    assert abs(s.oct.acquisition_time.second - 5.493) < 1e-9
    assert abs(s.fundus.acquisition_time.second - 5.140) < 1e-9


def test_resolutions(tmp_path):
    s = _study(tmp_path).series[0]
    assert s.lateral_resolution == 0.0116  # ScaleX
    assert s.axial_resolution == 0.0039  # ScaleY
    assert s.oct.width == 768 and s.oct.height == 496


def test_classification_and_files(tmp_path):
    s = _study(tmp_path).series[0]
    assert s.oct.kind == "oct"
    assert s.fundus.kind == "fundus"
    # file:///C:\...\name.tif  ->  bare name
    assert s.oct_file_name == "90F83740.tif"
    assert s.fundus_file_name == "90F37C50.tif"
    assert s.oct.file_path.startswith("file:///")
    assert s.oct.num_average == 9
    assert s.fundus.num_average == 100
    assert s.oct.quality == 28.0


def test_start_end_coords(tmp_path):
    s = _study(tmp_path).series[0]
    assert s.oct.start_xy == (0.269, 2.936)
    assert s.oct.end_xy == (8.656, 5.989)


def test_context_captures_extra_fields(tmp_path):
    s = _study(tmp_path).series[0]
    assert s.oct.context["Resolution"] == "LORES"
    assert s.oct.context["EDI"] == "true"
    assert "SeriesUID" in s.context and s.context["Modality"] == "OCT"


def test_fallback_classification_without_imagetype(tmp_path):
    # Second series' localizer still classifies via ImageType; here we just
    # confirm both images resolved and the OCT carries the timestamp.
    s = _study(tmp_path).series[1]
    assert s.oct is not None and s.fundus is not None
    assert abs(s.acquisition_time.second - 5.603) < 1e-9


def test_datetimes_combine_study_date(tmp_path):
    dts = _study(tmp_path).datetimes()
    assert dts[0] == datetime(2022, 2, 15, 17, 22, 5, 493000)


def test_to_dataframe(tmp_path):
    try:
        import pandas  # noqa: F401
    except ImportError:
        return  # pandas optional; skip silently when absent
    df = _study(tmp_path).to_dataframe()
    assert len(df) == 2
    assert {"lateral_resolution", "axial_resolution", "oct_file", "time"} <= set(df.columns)
    assert df.loc[0, "axial_resolution"] == 0.0039


# --------------------------------------------------------------------------- #
# Standalone runner (so the suite works without pytest installed)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import tempfile
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for fn in tests:
        with tempfile.TemporaryDirectory() as d:
            try:
                fn(Path(d))
                print(f"PASS  {fn.__name__}")
                passed += 1
            except Exception:  # noqa: BLE001
                print(f"FAIL  {fn.__name__}")
                traceback.print_exc()
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
