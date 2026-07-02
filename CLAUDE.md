# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo does

Pipeline for quantitative analysis of the choroid in time-resolved OCT (Spectralis) B-scan videos, toward
non-invasive estimation of ocular rigidity (the IOP–volume relationship). Stages: choroid segmentation
(U-Net) → video registration (lateral + vertical, optional A-scan/RPE 2nd pass) → cardiac-cycle motion
extraction → rigidity-coefficient fitting from pressure/area or pressure/thickness curves.

## Two parallel consumers of the same library

The `src/ocularrigidity` package is shared by two independent processing tracks that were developed on
different machines/branches — check which one you're touching before assuming file layout or config:

1. **Generic cohort pipeline** (Clement's track): `src/ocularrigidity/scripts/cohort_analysis/`
   (`segment_n_cycles.py`, `extract_deltaA.py`, `flag_misregistration.py`) + `motion/pulsation.py`
   (`CardiacCycleExtractor`) + `viewer/streamlit_explorer/`. Reads/writes under the hardcoded Linux paths
   in [src/ocularrigidity/consts.py](src/ocularrigidity/consts.py) (`ROOT_CARDIAC_PIPELINE`,
   `ROOT_REGISTERED_CACHE`, etc.) — these will not resolve on this machine without editing. Rigidity here
   is the **Friedenwald K** coefficient ([src/ocularrigidity/friedenwald.py](src/ocularrigidity/friedenwald.py)),
   derived from choroidal *area* change (`deltaA`, from `extract_deltaA.py`) via a spherical-shell model.
2. **SANSORI batch pipeline** (Nicolas's track, `Astronaut` branch): [Astronauts/](Astronauts/)
   (`register_files.py`, `segment_files.py`, `compute_rigidity_time_series.py`) + the Streamlit app in
   [testing_app/](testing_app/). Operates directly on `E:/SANSORI/<NN_id>/<...>_rigidity/<..._OD|OS...>/`
   (hardcoded `PATH_GENERAL` in each script). Rigidity here is the **Sayah et al. (2020) k** coefficient
   ([src/ocularrigidity/rigidity/features.py](src/ocularrigidity/rigidity/features.py)), derived from
   choroidal *thickness* pulsatility (`deltaY`/`delta_CT`) plus axial length/IOP/OPA from `sansori_db.db`.

Don't mix the two rigidity formulas or their config dataclasses up — `k` (Sayah, mm⁻³) and `K` (Friedenwald)
are different quantities computed from different signals.

## Commands

```bash
pip install -e .                       # editable install (poetry-core backend, pyproject.toml)

# Tests (only tests/test_spectralis.py exists; it also runs standalone)
pytest tests/test_spectralis.py
python tests/test_spectralis.py

# SANSORI batch pipeline (run in order; each iterates E:/SANSORI/*/*_rigidity/*)
python Astronauts/register_files.py              # segment + register raw .tif -> RawImages/registered/
python Astronauts/segment_files.py                # segment other MATLAB-produced video variants
python Astronauts/compute_rigidity_time_series.py # -> rigidity_time_series.csv + per-condition figures

# Interactive registration tuning (Streamlit, SANSORI data)
streamlit run testing_app/first_cc_registration.py

# Streamlit cohort explorer (generic pipeline)
streamlit run src/ocularrigidity/viewer/streamlit_explorer/Home.py

# Deprecated pygame viewer
python -m ocularrigidity.viewer
```

There is no lint/format config in the repo (no ruff/black/flake8 config, no pre-commit) and no CI workflow.

## Registration architecture (the part most scripts build on)

Registration always has two possible passes, in order:

1. **General recalage** — [registration/rigid.py](src/ocularrigidity/registration/rigid.py)
   `register_masks_by_displacement`: lateral (x) shift via `registration/horizontal/` (`xcorr` on vertical-mean
   profiles, or `fullframe` 2D phase correlation), then vertical (y) alignment of the segmented Bruch's-membrane
   boundary to a reference frame (optionally `flatten_rpe`). Operates on `(mask, frame)` pairs together so both
   get the same `grid_sample` transform.
2. **A-scan / RPE recalage (2nd pass, optional)** —
   [registration/axial/median_registration.py](src/ocularrigidity/registration/axial/median_registration.py)
   `register_ascans_to_median`: computes the temporal median of the *already-registered* volume, applies
   `correct_shadow` (Girard 2011 shadow compensation) and/or `laplacian_of_gaussian` (RPE enhancement,
   decoupled — either, both, or neither), then does a per-column (per-A-scan) 1D spectral phase correlation
   against the median to refine the vertical alignment. Preprocessing is only used to *estimate* the shift;
   it's applied to the raw registered pixels via `grid_sample`.

Both passes are wired together in
[motion/registered_video.py](src/ocularrigidity/motion/registered_video.py) `RegisteredVideo.compute_registration()`
(the generic-pipeline entry point, with optional on-disk caching under `ROOT_REGISTERED_CACHE`) and in
[registration/export.py](src/ocularrigidity/registration/export.py) `export_registered_video` (the SANSORI
entry point, called by `Astronauts/register_files.py` and by `testing_app/pages/2_Video_recalee.py`). Both
passes are governed by the single `RegistrationConfig` dataclass in
[pipeline_config.py](src/ocularrigidity/pipeline_config.py) (`median_registration` + `median_*` fields control
the A-scan pass; off by default). `export_registered_video(..., suffix=...)` writes the A-scan variant to
separate files (`registered_video_ascan.mp4`, `mask_ascan.npz`, `transform_ascan.npz`, ...) alongside the base
ones rather than overwriting, so both can be compared.

**Gotcha**: for the SANSORI dataset on `E:/SANSORI`, most conditions only have a *legacy MATLAB* general
registration under `RawImages/registeredBscans/` (masks + `compressedOCT_woOutliers_median_threshold.mj2`,
no A-scan variant). Only a handful of conditions have been reprocessed through the newer Python pipeline
into `RawImages/registered/` (which is where the A-scan `_ascan` variant support lives). Check which folder
a script is reading from before assuming A-scan registration is available for it.

## Segmentation

[segmentation/inference.py](src/ocularrigidity/segmentation/inference.py) `infer()` wraps
`ChoroidSegmentationModule` (PyTorch Lightning U-Net, weights auto-downloaded from Hugging Face via
`segmentation/utils.get_choroid_segmentation_model()`): optional resize (`scale_factor`), sigmoid + GPU
graph-cut postprocessing (`segmentation/postprocess/graphcut_gpu.py`), then largest-connected-component
filtering. Input is a `(T, H, W)` or `(T, C, H, W)` cube; uint8 input is auto-normalized.

## GPU dependency

Almost every numerical routine (segmentation inference, `register_masks_by_displacement`,
`register_ascans_to_median`, phase correlation, `temporal_median`) defaults to `device="cuda"` and several
default parameters are tuned assuming a GPU is present. `rigidity/geometry` and
`rigidity/features.extract_thickness_gpu` hard-import `cupy`/`cupyx` — these paths will fail to import on a
machine without CUDA/cupy even if you only wanted the CPU fallback functions in the same file.

## Data I/O

- [data/io.py](src/ocularrigidity/data/io.py): `save_mask`/`load_mask` (bit-packed + zstd-compressed boolean
  masks — always use these instead of raw `.npy` for masks), `load_cube` (raw `cube.bin` + `timestamp.txt`,
  local or `smb://` via `smbclient`), `save_mask`.
- [data/compression.py](src/ocularrigidity/data/compression.py): video codecs for cubes — `mp4_to_cube`,
  `read_gray` (decord-based `.mp4`/`.mkv` reader), `cube_to_mp4_fastest`/`cube_to_mkv_lossless`. Several of
  these default to a hardcoded Linux ffmpeg path (`/home/clement/.../ffmpeg`); on Windows/other machines pass
  `ffmpeg=...` explicitly or ensure `ffmpeg` is resolvable another way — `registration/export.py` has its own
  `resolve_ffmpeg()` (PATH, then falls back to `imageio_ffmpeg`) that scripts under `Astronauts/` rely on.
- [data/spectralis.py](src/ocularrigidity/data/spectralis.py): `SpectralisStudy` parses Heidelberg HEYEX XML
  exports (series, acquisition times, image quality) — this is how raw `.tif` frames get ordered/matched to
  metadata before segmentation/registration.

## Config dataclasses

[pipeline_config.py](src/ocularrigidity/pipeline_config.py) centralizes per-stage parameters as frozen
dataclasses (`RegistrationConfig`, `PulsationConfig`, `DeltaYConfig`, `SegmentationConfig`, `DeltaAConfig`,
`FriedenwaldConfig`, `MisregistrationConfig`), each with a singleton instance at module bottom
(`REGISTRATION`, `PULSATION`, etc.) imported by name elsewhere instead of being re-instantiated. For
`RegistrationConfig` specifically, every field except `batch_size` is part of `RegisteredVideo`'s cache key
(`_cache_meta()`) — changing one invalidates cached registrations for all downstream stages.
