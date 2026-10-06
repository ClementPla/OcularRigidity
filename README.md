# OcularRigidity

Choroid analysis on time-resolved OCT B-scan videos, for non-invasive estimation of ocular rigidity.

> **Work in progress.** APIs, file formats and module layout change without notice. Results are preliminary.

## What it does

Given an OCT video of a single B-scan location plus the matching IOP/OPA measurements, the pipeline runs:

1. **Segmentation + registration** (one GPU pass). A Segformer (`mit_b2` encoder) segments the choroid, from Bruch's membrane to the choroid–sclera interface. A regressor reads the same encoder features and predicts a global lateral shift `dx` and a per-column axial shift `dy`.
2. **Pulsation.** A SiNC network turns the registered Bruch's membrane and choroid–sclera boundaries into one pulse waveform. Heart rate comes from a Lomb-Scargle periodogram of that waveform and phase from IQ demodulation, with phase 0 anchored on minimal choroid thickness. Frames are then folded into `N_CYCLES` cardiac cycles.
3. **Cycle segmentation.** The folded cycles are segmented again, and the choroid–sclera interface is smoothed over time.
4. **ΔY.** Per-cycle thickness amplitude from a harmonic fit of each A-scan's thickness.
5. **ΔA / ΔCT.** Boundary displacement by optical flow, giving peak-to-peak area and thickness change per cycle.
6. **QC.** Per-video misregistration flags from the jitter and coverage of Bruch's membrane.

Rigidity is computed afterwards, at analysis time: the Friedenwald coefficient `K` from ΔCT (or ΔA) and the pressure pair, through a shell volume model ([friedenwald.py](src/ocularrigidity/friedenwald.py)).

## Install

```bash
git clone https://github.com/ClementPla/OcularRigidity.git
cd OcularRigidity
pip install -e ".[gpu]"
```

Dependencies are listed in `pyproject.toml`. The `gpu` extra installs `cupy` and `cucim` for CUDA 12; in a conda environment, install them from conda-forge / rapidsai instead. The `train` extra adds `wandb`.

A CUDA GPU is required in practice. `ffmpeg` must be on the system for the compressed-video readers. Note that `data/compression.py` currently hardcodes a path to it.

## Models

All weights are on the Hugging Face Hub and download on first use. The revisions are pinned in [consts.py](src/ocularrigidity/consts.py).

| Model | Repository | Revision | Loader |
| --- | --- | --- | --- |
| Choroid segmentation (Segformer / `mit_b2`) | `ClementP/ChoroidSegmentationModule` | `version-2.0.0` | `segmentation.utils.get_choroid_segmentation_model()` |
| Registration (cascade regressor on the frozen encoder) | `ClementP/OCTVideoRegistration` | `cascade_v9` | `segmentation.utils.get_registration_model()` |
| Pulse waveform (SiNC) | `ClementP/OCTVideoPulsationMeasure` | `sinc_v0` | `motion.pulsation.sinc.trace_source.load_sinc_module()` |

The first two are returned in eval mode on the CPU; move them to your device before use. The registration model was trained on the segmentation model's encoder, so the two revisions go together. The registration revision is part of the registration cache key.

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

seg_model = get_choroid_segmentation_model().to(DEVICE)
reg_model = get_registration_model().to(DEVICE)

frames = load_cube(Path("/path/to/folder"))  # folder holding cube.bin (+ timestamp.txt)
frames = frames[20:-10]

config = RegistrationConfig(fused_batch_size=8, batch_size=16, keep_largest_cc=False)
result = segment_and_register(frames, seg_model, reg_model, config=config, device=DEVICE)

result.raw_masks          # (T, H, W) bool, unregistered
result.registered_masks   # (T, H, W) bool
result.registered_frames  # (T, H, W) uint8
result.transform          # {"dx": (T,), "dy": (T, W), "bad_columns": (W,)}
```

`fused_batch_size` is the memory-sensitive setting: it holds encoder activations for a 1536x1024 input plus both heads. `batch_size` only affects the warping pass.

Segmentation alone, without registration:

```python
from ocularrigidity.data.io import save_mask
from ocularrigidity.segmentation.inference import infer

# The model was trained at an axial pixel size of 1.95 um and a lateral one of
# 5.9 um (1536x1024). Use scale_factor to match your acquisition.
mask = infer(seg_model, frames, scale_factor=0.5, batch_size=16, device="cuda:0")
save_mask(mask, path)  # bit-packed + zstd
```

`infer` returns the thresholded mask with its largest connected component kept (`post_process=False` skips that).

## Running the cohort

[run_cohort.py](src/ocularrigidity/scripts/run_cohort.py) is the single entry point. It carries one video through every stage (`segment`, `register`, `fold`, `cycles`, `deltaY`, `deltaA`, `qc`), skips work whose output already exists, and keeps going past a failing video.

```bash
python -m ocularrigidity.scripts.run_cohort --dry-run          # what would run, no GPU
python -m ocularrigidity.scripts.run_cohort                    # whole cohort, resumable
python -m ocularrigidity.scripts.run_cohort --limit 2          # smoke run
python -m ocularrigidity.scripts.run_cohort --force-from fold  # redo from one stage on
```

To start a new run on top of another run's masks and registration, point the two input folders at that run and pass `--reuse-segreg`:

```bash
OCULARRIGIDITY_RUN=<new run> \
OCULARRIGIDITY_MASKS=<output root>/<source run>/masks \
OCULARRIGIDITY_REGISTERED_CACHE=<output root>/<source run> \
python -m ocularrigidity.scripts.run_cohort --reuse-segreg
```

`run_cohort` uses one GPU. To spread segmentation+registration over several cards, run [segment_and_register.py](src/ocularrigidity/scripts/segmentation/segment_and_register.py) with `--num-shards N --shard i`, one process per GPU, then `run_cohort`.

### Outputs

Under `OUTPUT_ROOT/RUN`:

| Path | Content |
| --- | --- |
| `masks/<video>/mask.npz` | raw choroid masks |
| `registered_frames/<video>/cube.mp4`, `registered_masks/<video>/{mask,transform}.npz` | registration cache |
| `one_cycle/<video>/one_cycle.mkv` | folded cycles, lossless |
| `measures/<video>/measure.pkl` | heart rate, phase and the stage configs used |
| `measures/<video>/segmented_cycles.npz` | masks of the folded cycles |
| `measures/<video>/deltaA_per_cycle.pkl` | boundary displacement and area change |
| `deltaY.pkl`, `misregistration_flags.csv` | cohort tables |

## Configuration

Paths and batch sizes come from [consts.py](src/ocularrigidity/consts.py), overridable by environment:

| Variable | Meaning |
| --- | --- |
| `OCULARRIGIDITY_DATA_ROOT` | compressed source videos (shared across runs) |
| `OCULARRIGIDITY_OUTPUT_ROOT`, `OCULARRIGIDITY_RUN` | outputs land in `OUTPUT_ROOT/RUN` |
| `OCULARRIGIDITY_MASKS`, `OCULARRIGIDITY_REGISTERED_CACHE`, `OCULARRIGIDITY_CARDIAC` | per-stage folder overrides |
| `OCULARRIGIDITY_SEGMENTATION_BATCH`, `OCULARRIGIDITY_REGISTRATION_BATCH` | memory/throughput only |

What the pipeline computes is set in [pipeline_config.py](src/ocularrigidity/pipeline_config.py):

- `N_CYCLES`: cardiac cycles folded and measured.
- `REGISTRATION`: the learned registration, on the raw `cube.bin`.
- `PULSATION`: the trace → rate → phase chain and the fold. The default is the SiNC waveform with per-cycle thickness anchoring; `PulsationConfig()` gives the thickness-trace chain, bandpassed around the measured heart rate.
- `DELTA_Y`, `DELTA_A`: fit and tracking parameters.

The configs of individual components (`RegistrationConfig`, `NCycleConfig`, ...) live next to the code they configure.

## Layout

```
src/ocularrigidity/
  consts.py           paths, batch sizes, Hub revisions, pixel sizes
  pipeline_config.py  what the cohort pipeline runs with
  data/               readers (cube.bin, mp4/mkv, SMB), mask compression, clinical DBs
  segmentation/       Segformer module, inference, mask post-processing, fovea, vessels
  registration/       fused.py (single-pass segment + register), registration cache,
                      QC flags, learned (deep_learning/) and classical (rigid.py) estimators
  motion/             pulsation (traces -> rate -> phase, SiNC), cycle folding, displacement
  thickness/          thickness features, delta CT from boundary tracking
  friedenwald.py      delta A / delta CT -> delta V -> K
  stats/              longitudinal and cohort statistics
  viewer/             streamlit cohort browser, gif/quiver renderers
  scripts/            run_cohort.py, training scripts, other acquisitions (Spectralis, astronauts)
```

## Cohort browser

```bash
streamlit run src/ocularrigidity/viewer/streamlit_explorer/Home.py
```

One row per rigidity visit, with the cardiac metrics, the clinical scalars, the Heyex ONH sectors and the diagnosis register merged onto it. Pages: cases, regression, viewer, inference, registration, fovea estimation, longitudinal, modelling.

## Status

Done: choroid segmentation, learned registration, fused single-pass inference, compressed mask storage, SiNC pulse extraction and cycle folding, ΔA/ΔCT measurement, Friedenwald fitting, cohort browser.

Open: temporal regularization at training time (warp-consistency, topology-preserving losses), synchronization with continuous tonometry, validation study.

The notebooks under [notebook/](notebook/) and [notebooks/](notebooks/) are exploratory and many are out of date; [notebook/demo_seg_and_register.ipynb](notebook/demo_seg_and_register.ipynb) is the reference example.
