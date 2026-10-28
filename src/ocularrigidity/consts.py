import os
from pathlib import Path


_DEFAULT_DATA_ROOT = Path("/media/clement/HD/Santiago/OcularRigidity/outputs")
_DEFAULT_OUTPUT_ROOT = Path("/media/clement/HD/Santiago/OcularRigidity")
_DEFAULT_RUN = "FullPipeline"


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


# Memory/throughput only
REGISTRATION_BATCH_SIZE = _env_int("OCULARRIGIDITY_REGISTRATION_BATCH", 32)
SEGMENTATION_BATCH_SIZE = _env_int("OCULARRIGIDITY_SEGMENTATION_BATCH", 4)


DATA_ROOT = _env_path("OCULARRIGIDITY_DATA_ROOT", _DEFAULT_DATA_ROOT)
OUTPUT_ROOT = _env_path("OCULARRIGIDITY_OUTPUT_ROOT", _DEFAULT_OUTPUT_ROOT)
RUN_NAME = os.environ.get("OCULARRIGIDITY_RUN", _DEFAULT_RUN)

RUN_ROOT = OUTPUT_ROOT / RUN_NAME

MEASUREMENTS_PATH = Path("/home/clement/Documents/data/OcularRigidity/Biomechanics.db")

CLINICAL_VALUES_PATH = Path(
    "/home/clement/Documents/data/OcularRigidity/ClinicalValuesWithSteepest.db"
)

STUDY_PATH = Path("/home/clement/Documents/data/OcularRigidity/Studies.db")

# Third clinical source, parsed from the radial+circles PDFs by ocularrigidity.data.heyex.reports.
HEYEX_ONH_PATH = _env_path(
    "OCULARRIGIDITY_HEYEX_ONH",
    Path("/home/clement/Documents/data/OcularRigidity/heyex_data.pkl"),
)

OIMHS_ROOT = Path("/home/clement/Documents/data/OIMHS/Images/")


CHOROID_SEGMENTATION_ROOT = Path(
    "/home/clement/Documents/data/OcularRigidityAnnotated/dataset/ChoroidSegmentation/"
)

ROOT_DATA_MNT = Path("/mnt/smb/")

ROOT_MASKS = _env_path("OCULARRIGIDITY_MASKS", RUN_ROOT / "masks")

ROOT_COMPRESSED_VIDEO = DATA_ROOT / "compressed"


ROOT_CROSS_SECTIONAL_EXCEL = Path(
    "/home/clement/Documents/data/OcularRigidity/recrutement_alejandra.xlsx"
)
# Root passed to RegisteredVideo(cache_dir=...)
ROOT_REGISTERED_CACHE = _env_path("OCULARRIGIDITY_REGISTERED_CACHE", RUN_ROOT)

ROOT_CARDIAC_PIPELINE = _env_path("OCULARRIGIDITY_CARDIAC", RUN_ROOT)

# Cases rejected on visual QC in the gif viewer (one gif file name per entry).
QC_ERRORS_PATH = (
    Path(__file__).resolve().parents[2] / "notebooks/gif_viewer/errors.json"
)

# Hugging Face Hub revisions.
SEGMENTATION_REVISION = "version-3.0.0"
REGISTRATOR_REVISION = "reg_upernet_convnext_t_bmlines"
SINC_REVISION = "sinc_v0"

OUTPUT_FOLDER = ROOT_COMPRESSED_VIDEO

# Physical properties of the acquisition device, not study decisions
AXIAL_PIXEL_SIZE_MM = 1.95e-3
TRANVERSAL_PIXEL_SIZE_MM = 5.9e-3

BSCAN_WIDTH_PX = 1024
ASCAN_LENGTH_PX = 1536
# Columns dropped on each side of the choroid mask before tracking its
# boundary and measuring ΔCT / deltaA. The one trim setting: tracking, the
# cohort tables, the viewer and the Spectralis script all read it. 0 = the
# whole mask.
CHOROID_TRIM_PX = 0
