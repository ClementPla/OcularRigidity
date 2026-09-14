import os
from pathlib import Path

# A "run" is one pass of the whole pipeline: its masks, its registration cache
# and its cardiac outputs. All three depend on the weights, so each run gets its
# own tree under OUTPUT_ROOT/RUN rather than overwriting the previous one.
# Defaults match what scripts/pipeline.sh exports, so running a stage script
# directly resolves the same tree.

# The compressed source videos are the pipeline's *input*, shared by every run,
# so they are keyed on DATA_ROOT rather than on a run's output root.
_DEFAULT_DATA_ROOT = Path("/media/clement/HD/Santiago/OcularRigidity/outputs")
_DEFAULT_OUTPUT_ROOT = Path("/media/clement/HD/Santiago/OcularRigidity")
_DEFAULT_RUN = "FullPipeline"


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


# Memory/throughput only — neither changes the result. Registration is the
# hungry one: it grid_samples a (batch, 1 + C, H, W) float32 tensor plus an
# equally large sampling grid, several GB in one allocation at batch 256.
REGISTRATION_BATCH_SIZE = _env_int("OCULARRIGIDITY_REGISTRATION_BATCH", 32)
SEGMENTATION_BATCH_SIZE = _env_int("OCULARRIGIDITY_SEGMENTATION_BATCH", 4)


DATA_ROOT = _env_path("OCULARRIGIDITY_DATA_ROOT", _DEFAULT_DATA_ROOT)
OUTPUT_ROOT = _env_path("OCULARRIGIDITY_OUTPUT_ROOT", _DEFAULT_OUTPUT_ROOT)
RUN_NAME = os.environ.get("OCULARRIGIDITY_RUN", _DEFAULT_RUN)

# The individual OCULARRIGIDITY_* overrides below still win, which is how a run
# can point at another run's masks on purpose.
RUN_ROOT = OUTPUT_ROOT / RUN_NAME

MEASUREMENTS_PATH = Path("/home/clement/Documents/data/OcularRigidity/Biomechanics.db")

# Second clinical source: one wide row per visit, carrying the BMO-MRW /
# sector-RNFL / steepest-quadrant measures absent from Biomechanics.
CLINICAL_VALUES_PATH = Path(
    "/home/clement/Documents/data/OcularRigidity/ClinicalValuesWithSteepest.db"
)

STUDY_PATH = Path("/home/clement/Documents/data/OcularRigidity/Studies.db")

# Third clinical source, parsed from the radial+circles PDFs by
# ocularrigidity.data.heyex.reports. Keyed by the Heyex *file number*, which
# reaches PatientId through Biomechanics' Patients table. Overlaps
# CLINICAL_VALUES_PATH and extends its coverage.
HEYEX_ONH_PATH = _env_path(
    "OCULARRIGIDITY_HEYEX_ONH",
    Path("/home/clement/Documents/data/OcularRigidity/heyex_data.pkl"),
)

OIMHS_ROOT = Path("/home/clement/Documents/data/OIMHS/Images/")

ROOT_DATA_MNT = Path("/mnt/smb/")

ROOT_MASKS = _env_path("OCULARRIGIDITY_MASKS", RUN_ROOT / "masks")

ROOT_COMPRESSED_VIDEO = DATA_ROOT / "compressed"

# Root passed to RegisteredVideo(cache_dir=...); the cache lives in
# `registered_masks/` and `registered_frames/` below it. Per-run, because
# registration consumes the masks: masks from a different model must not
# resolve against a cache built from the old ones.
ROOT_REGISTERED_CACHE = _env_path("OCULARRIGIDITY_REGISTERED_CACHE", RUN_ROOT)

ROOT_CARDIAC_PIPELINE = _env_path("OCULARRIGIDITY_CARDIAC", RUN_ROOT)

# Cases rejected on visual QC in the gif viewer (one gif file name per entry).
QC_ERRORS_PATH = (
    Path(__file__).resolve().parents[2] / "notebooks/gif_viewer/errors.json"
)

# Used by *both* segmentation passes — the raw-video masks and the folded-cycle
# masks. They must agree: measuring a cycle with different weights than were
# registered is not comparable.
CHECKPOINT_PATH = _env_path(
    "OCULARRIGIDITY_CHECKPOINT",
    Path(__file__).resolve().parents[2]
    / "checkpoints/choroid-segmentation-epoch=29-dice=0.99-v1.ckpt",
)

# The RegistrationRegressor reads the *frozen* encoder of CHECKPOINT_PATH. The
# two are a trained pair — the regressor never saw any other encoder's features
# — so changing one without the other is a silent error, not a configuration
# choice.
REGISTRATOR_CHECKPOINT = _env_path(
    "OCULARRIGIDITY_REGISTRATOR",
    Path(__file__).resolve().parents[2] / "checkpoints/reg_cascade_v9/best.pt",
)

OUTPUT_FOLDER = ROOT_COMPRESSED_VIDEO

# Physical properties of the acquisition device, not study decisions — they live
# in this leaf module so library code can reach them without importing the
# study-level ``pipeline_config``.
AXIAL_PIXEL_SIZE_MM = 1.95e-3
TRANVERSAL_PIXEL_SIZE_MM = 5.9e-3
