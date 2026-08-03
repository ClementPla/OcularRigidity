import os
from pathlib import Path

# --- Run selection -----------------------------------------------------
# A "run" is one pass of the whole pipeline: its masks, its registration cache
# and its cardiac outputs. A re-run with a different segmentation model gets its
# own tree instead of overwriting the previous one. Set these in the environment
# (see scripts/pipeline.sh); the defaults reproduce the historical layout.
#
#   OCULARRIGIDITY_DATA_ROOT    shared *input*: the compressed source videos.
#                               Independent of the run — moving a run's outputs
#                               must not move the 1 TB of source material.
#   OCULARRIGIDITY_OUTPUT_ROOT  parent of this run's masks/, registered_*/ and
#                               CardiacPipeline_* dirs
#   OCULARRIGIDITY_RUN          name of the cardiac output dir under that root
#   OCULARRIGIDITY_CHECKPOINT   segmentation weights used by *both* the raw-video
#                               pass and the folded-cycle pass

_LEGACY_ROOT = Path("/media/clement/HD/Santiago/OcularRigidity/outputs")


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


# --- GPU batch sizes ---------------------------------------------------
# Overridable because they depend on the card and on what else is running,
# not on the study. Registration is the memory-hungry one: it grid_samples a
# (batch, 1 + C, H, W) float32 tensor and an equally large sampling grid, so at
# 1024x1536 a batch of 256 asks for several GB in a single allocation.
REGISTRATION_BATCH_SIZE = _env_int("OCULARRIGIDITY_REGISTRATION_BATCH", 32)
SEGMENTATION_BATCH_SIZE = _env_int("OCULARRIGIDITY_SEGMENTATION_BATCH", 8)


DATA_ROOT = _env_path("OCULARRIGIDITY_DATA_ROOT", _LEGACY_ROOT)
OUTPUT_ROOT = _env_path("OCULARRIGIDITY_OUTPUT_ROOT", _LEGACY_ROOT)
RUN_NAME = os.environ.get("OCULARRIGIDITY_RUN", "CardiacPipeline_V2")

MEASUREMENTS_PATH = Path("/home/clement/Documents/data/OcularRigidity/Biomechanics.db")

# Second clinical source: one wide row per visit (ClinicalValues), carrying the
# BMO-MRW / sector-RNFL / steepest-quadrant measures absent from Biomechanics.
CLINICAL_VALUES_PATH = Path(
    "/home/clement/Documents/data/OcularRigidity/ClinicalValuesWithSteepest.db"
)

STUDY_PATH = Path("/home/clement/Documents/data/OcularRigidity/Studies.db")

OIMHS_ROOT = Path("/home/clement/Documents/data/OIMHS/Images/")


ROOT_DATA_SMB = "smb://192.168.11.16/database/dataFiles/"

ROOT_DATA_MNT = Path("/mnt/smb/")

# Segmentation output. Per-run: change the model and these masks change, so a
# new run must not write over the old ones.
ROOT_MASKS = _env_path("OCULARRIGIDITY_MASKS", OUTPUT_ROOT / "masks")

ROOT_MEASURES = DATA_ROOT / "measures"

# Compressed source videos — the pipeline's *input*, shared by every run, so
# keyed on DATA_ROOT rather than on the run's output root.
ROOT_COMPRESSED_VIDEO = DATA_ROOT / "compressed"

# Root passed to RegisteredVideo(cache_dir=...). The registration cache lives in
# `registered_masks/` and `registered_frames/` subfolders below this root,
# mirroring the layout of ROOT_COMPRESSED_VIDEO and ROOT_MASKS.
# Per-run: registration consumes the masks, so masks from a different model must
# not resolve against a cache built from the old ones.
ROOT_REGISTERED_CACHE = _env_path("OCULARRIGIDITY_REGISTERED_CACHE", OUTPUT_ROOT)

# Root holding the per-method/phase pulsation outputs (one_cycle_*, measures_*,
# deltaY_*.pkl, misregistration_flags.csv) produced by the cohort scripts.
ROOT_CARDIAC_PIPELINE = _env_path(
    "OCULARRIGIDITY_CARDIAC", ROOT_REGISTERED_CACHE / RUN_NAME
)

ROOT_ONE_CYCLE = DATA_ROOT / "one_cycle"

# Cases rejected on visual QC in the gif viewer (one gif file name per entry).
QC_ERRORS_PATH = Path(__file__).resolve().parents[2] / "notebooks/gif_viewer/errors.json"


# Segmentation weights. Used by *both* passes: the raw-video masks
# (scripts/segmentation/inference_on_patients.py) and the folded-cycle masks
# (scripts/cohort_analysis/segment_n_cycles.py). They must agree — measuring a
# cycle with different weights than were registered is not comparable.
CHECKPOINT_PATH = _env_path(
    "OCULARRIGIDITY_CHECKPOINT",
    Path(__file__).resolve().parents[2]
    / "checkpoints/choroid-segmentation-epoch=29-dice=0.99.ckpt",
)

OUTPUT_FOLDER = ROOT_COMPRESSED_VIDEO


# --- Rig calibration ---------------------------------------------------
# Axial scale of the OCT, mm per pixel. Physical property of the acquisition
# device, not a study decision — lives here (a leaf module) so library code can
# use it without importing the study-level ``pipeline_config``.
AXIAL_PIXEL_SIZE_MM = 1.95e-3
TRANVERSAL_PIXEL_SIZE_MM = 5.9e-3  # mm per pixel, for the 5.9 µm/pixel scans
