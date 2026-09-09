# OcularRigidity

Choroid analysis on time-resolved OCT B-scan videos, for non-invasive estimation of ocular rigidity.

> **Work in progress.** APIs, file formats and module layout change without notice. Results are preliminary.

## What it does

Given an OCT video of a single B-scan location plus the matching IOP/OPA measurements, the pipeline runs:

1. **Segmentation + registration** (one GPU pass). A Segformer (`mit_b2` encoder) segments the choroid, from Bruch's membrane to the choroid–sclera interface. A regressor reads the same encoder features and predicts a global lateral shift `dx` and a per-column axial shift `dy`. One encode feeds both heads, so the pass costs roughly what segmentation alone used to.
2. **Pulsation.** Thickness traces from the registered masks, bandpassed on a cardiac band, coherence-selected, PCA-decomposed. Heart rate by Lomb-Scargle, phase by IQ demodulation on the best component. Frames are then folded into `N_CYCLES` cardiac cycles.
3. **Cycle segmentation.** The folded cycles are re-segmented at full resolution.
4. **ΔA / ΔCT.** Boundary displacement by optical flow, projected onto the CSI normal in physical units, giving peak-to-peak area and thickness change per cycle.
5. **Rigidity.** Friedenwald coefficient `K` from ΔCT (or ΔA) and the pressure pair, through a shell volume model.
6. **QC.** Per-video misregistration flags.

## Install

```bash
git clone https://github.com/ClementPla/OcularRigidity.git
cd OcularRigidity
pip install -e .
```

`pyproject.toml` does not pin dependencies yet. You need at least: `torch`, `pytorch-lightning`, `segmentation-models-pytorch`, `huggingface-hub`, `numpy`, `scipy`, `pandas`, `opencv-python`, `scikit-image`, `kornia`, `numba`, `zstandard`, `smbclient`, `imageio`, `av`, `decord`, `tqdm`, `matplotlib`, `seaborn`, `statsmodels`, `streamlit`, `eyepy`, `SimpleITK`.

A CUDA GPU is required in practice. `ffmpeg` must be on the system for the compressed-video readers — note that `data/compression.py` currently hardcodes a path to it.

## Quick start

Segmentation and registration in one pass ([notebook/demo_seg_and_register.ipynb](notebook/demo_seg_and_register.ipynb)):

```python
from pathlib import Path

from ocularrigidity.data.io import load_cube
from ocularrigidity.registration.config import RegistrationConfig
from ocularrigidity.registration.fused import segment_and_register
from ocularrigidity.segmentation.utils import (
    get_choroid_segmentation_model,
    get_registration_model,
)

DEVICE = "cuda"

# Weights are pulled from the Hugging Face Hub on first call.
# Move both models to the device yourself; segment_and_register only moves the data.
seg_model = get_choroid_segmentation_model().to(DEVICE)
reg_model = get_registration_model().to(DEVICE)

frames = load_cube(Path("/path/to/folder"))  # folder holding cube.bin (+ timestamp.txt)
frames = frames[20:-10]

config = RegistrationConfig(fused_batch_size=8, batch_size=16, keep_largest_cc=False)
result = segment_and_register(frames, seg_model, reg_model, config=config, device=DEVICE)

result.raw_masks          # (T, H, W) bool, unregistered
result.registered_masks   # (T, H, W) bool
result.registered_frames  # (T, H, W) uint8
result.transform          # {"dx": (T,), "dy": (T, W)}
```

`fused_batch_size` is the memory-sensitive knob: it holds encoder activations for a 1536x1024 input plus both heads. `batch_size` only affects the warping pass.

Segmentation alone, without registration:

```python
from ocularrigidity.segmentation.inference import infer
from ocularrigidity.data.io import save_mask

# The model was trained at an axial pixel size of 1.95 um and a lateral one of
# 5.95 um (1536x1024). Use scale_factor to match your acquisition.
mask = infer(model, data, scale_factor=0.5, batch_size=16, device="cuda:0")
save_mask(mask, path)  # bit-packed + zstd
```

## Running the cohort

Two entry points produce the same artifacts.

```bash
./scripts/pipeline.sh                     # six passes over the cohort, resumable per stage
./scripts/pipeline.sh --stage pulsation   # re-run one stage
OCULARRIGIDITY_SEG_GPUS=0 ./scripts/pipeline.sh

python -m ocularrigidity.scripts.run_cohort --dry-run   # single process, one video at a time
```

`pipeline.sh` shards segmentation+registration over several GPUs and re-reads each stage's output from disk. `run_cohort` carries one video through every stage while it is still in RAM. Use the script when a single stage has to be re-run over the whole cohort.

Both skip work whose output already exists, and both keep going past a failing video.

## Configuration

Paths and batch sizes come from [consts.py](src/ocularrigidity/consts.py), overridable by environment:

| Variable | Meaning |
| --- | --- |
| `OCULARRIGIDITY_DATA_ROOT` | compressed source videos (shared across runs) |
| `OCULARRIGIDITY_OUTPUT_ROOT`, `OCULARRIGIDITY_RUN` | outputs land in `OUTPUT_ROOT/RUN` |
| `OCULARRIGIDITY_MASKS`, `OCULARRIGIDITY_REGISTERED_CACHE`, `OCULARRIGIDITY_CARDIAC` | per-stage overrides |
| `OCULARRIGIDITY_CHECKPOINT`, `OCULARRIGIDITY_REGISTRATOR` | weights |
| `OCULARRIGIDITY_FUSED_BATCH`, `OCULARRIGIDITY_SEGMENTATION_BATCH`, `OCULARRIGIDITY_REGISTRATION_BATCH` | memory/throughput only |

A run is keyed on its weights, so masks and the registration cache are per-run and never mixed between models.

Algorithm parameters split in two: component configs (`RegistrationConfig`, `NCycleConfig`, ...) live next to the code they configure; the cohort-wide instances actually in force are the frozen singletons in [pipeline_config.py](src/ocularrigidity/pipeline_config.py).

## Layout

```
src/ocularrigidity/
  data/           readers (cube.bin, mp4/mkv, SMB), mask compression, clinical DBs
  segmentation/   Segformer module, inference, post-processing, fovea, vessels
  registration/   classical (rigid.py) and learned (deep_learning/) estimators
                  fused.py -- the single-pass segment + register
  motion/         pulsation (traces -> rate -> phase), cycle folding, displacement
  thickness/      delta CT from boundary tracking
  friedenwald.py  delta A / delta CT -> delta V -> K
  stats/          longitudinal and cohort statistics
  viewer/         streamlit cohort browser, gif/quiver renderers
  scripts/        batch entry points, one per pipeline stage
```

## Models

Weights are on the Hugging Face Hub and download on first use:

- `ClementP/ChoroidSegmentationModule`, revision `version-2.0.0` — Segformer / `mit_b2`, 1 input channel, 1 class.
- `ClementP/OCTVideoRegistration`, revision `cascade_v9` — cascade regressor over the frozen encoder pyramid.

Both are returned in eval mode on the CPU. Move them to your device before use.

## Cohort browser

```bash
streamlit run src/ocularrigidity/viewer/streamlit_explorer/Home.py
```

One row per rigidity visit, with the cardiac metrics, the clinical scalars, the Heyex ONH sectors and the diagnosis register merged onto it. Pages: cases, regression, viewer, inference, registration, fovea estimation, longitudinal.

The standalone pygame viewer described in earlier versions of this README is gone.

## Status

Done: choroid segmentation, learned registration, fused single-pass inference, compressed mask storage, cardiac phase extraction and cycle folding, ΔA/ΔCT measurement, Friedenwald fitting, cohort browser.

Open: temporal regularization at training time (warp-consistency, topology-preserving losses), synchronization with continuous tonometry, validation study.

The notebooks under [notebook/](notebook/) and [notebooks/](notebooks/) are exploratory and mostly out of date; [notebook/demo_seg_and_register.ipynb](notebook/demo_seg_and_register.ipynb) is the one kept current.
