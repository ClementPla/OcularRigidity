# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Write results to `E:`, never to `C:`.** `C:` is chronically full (it has hit 0 GB free, at which
> point even a Python `print` to a pipe fails with `No space left on device`); `E:` has ~1.4 TB. Any
> generated output — batch tables, per-condition `.npz`, figures, exported videos, caches — belongs
> under `E:/NASA_Rigidity/` (or `E:/SANSORI/` for data written next to its condition). Only source
> lives on `C:`. See "Where results go" below for the layout and the junction convention.

> The package went through a large restructuring (58 files, +14k/-2.2k lines) shortly before this
> revision of CLAUDE.md. Some consumers outside `src/ocularrigidity` (notably the gitignored
> `Astronauts/` scripts, see below) were **not** updated to match and currently have broken imports.
> Don't trust old references to `motion/registered_video.py`, `RegisteredVideo`,
> `registration/horizontal/`, `registration/export.py`, or `rigidity/features.py` — they're gone.

## Where results go

Everything generated lands on `E:`. The three roots in use:

```
E:/SANSORI/<NN_id>/<...>_rigidity/<..._OD|OS...>/   raw acquisitions + per-condition outputs
E:/NASA_Rigidity/SegmentationVariations/<variant>/  registered frames/masks + batch analysis tables
E:/NASA_Rigidity/quarto_results/                    figures of the Quarto report site
```

A batch script writes its tables and per-condition payloads under a named subdirectory of the mask
variant it consumed, so a result is always traceable to the segmentation it came from — e.g.
`Astronauts/compute_pulse_from_data.py` → `SegmentationVariations/model1_scale_1.0/pulse_from_data/`
(`conditions.csv`, `methods.csv`, `dmd_eigs.csv`, `ssa_sweep.csv`, `traces/<slug>.npz`). Follow that
pattern for new scripts rather than inventing a root: `SEGVAR_ROOT / MASK_VARIANT / OUTPUT_SUBDIR`.

**Quarto figures are junctions.** In `reveal_quarto_presentations/`, every `figures_*/` directory and
`firstPresentation_files/` is a Windows directory junction pointing into
`E:/NASA_Rigidity/quarto_results/`. Figure-generating scripts (`figures_*/make_figures.py`) therefore
write "into the repo" and land on `E:` transparently, and Quarto resolves the relative image paths
normally. To add a page, create the target directory on `E:` and junction it back
(`New-Item -ItemType Junction -Path <repo>\figures_<name> -Target E:\NASA_Rigidity\quarto_results\figures_<name>`
— no admin rights needed).

**Quarto figures must be interactive.** In every Quarto document — the report site *and* the reveal.js
slides — plots are produced as dynamic, interactive figures (Plotly preferred, Bokeh acceptable) and
embedded as self-contained HTML widgets, not as static PNG/SVG. Zoom, pan, hover tooltips and
series toggling are the point: these figures are read on screen, not printed. Fall back to a static
Matplotlib image only when interactivity is genuinely impossible or useless for that figure — e.g.
raw B-scan / mask image panels and video frames, very dense per-pixel maps whose HTML payload would
be prohibitive, or output consumed by a PDF/print target — and say so when you do. When writing a
new `figures_*/make_figures.py`, default to Plotly and write `.html` (or a Quarto code cell emitting
the figure directly) rather than saving `.png`.

`_site/` is the one deliberate exception and stays on `C:`: Quarto refuses to clean an output directory
that resolves outside the project (`WARN: Refusing to remove directory ... since it is not a
subdirectory of the main project directory`) and warns that strange behavior may result. It is a build
artifact, so it is cheap to keep local.

When `C:` needs reclaiming, the safe targets are the conda package cache (`conda clean -p`) and the pip
cache (`pip cache purge`) — together ~70 GB, both re-downloadable. **Never** touch
`Documents\Nicolas\mes-projets` (~1.2 TB of the user's data).

## What this repo does

Pipeline for quantitative analysis of the choroid in time-resolved OCT (Spectralis) B-scan videos, toward
non-invasive estimation of ocular rigidity (the IOP–volume relationship). Stages: choroid segmentation
(U-Net) → video registration (lateral + vertical, optional fovea correction + A-scan/RPE 2nd pass) →
cardiac-cycle rate/phase extraction → one-cycle folding → rigidity-coefficient fitting from
pressure/area or pressure/thickness curves.

## Two tracks sharing the same registration/pulsation core

The `src/ocularrigidity` package is shared by two processing tracks — check which one you're touching
before assuming file layout, config source, or output paths:

1. **Generic cohort pipeline** (Clement's track, `main`-derived): `src/ocularrigidity/scripts/cohort_analysis/`
   (`segment_n_cycles.py`, `extract_deltaA.py`, `flag_misregistration.py`) + `scripts/pulsation/infer.py`
   (calls `motion.pulsation.run_cardiac_pipeline`) + `scripts/registration/glaucoma.py` (registration entry
   point for this track) + `viewer/streamlit_explorer/`. Reads/writes under the hardcoded Linux paths in
   [src/ocularrigidity/consts.py](src/ocularrigidity/consts.py) (`ROOT_CARDIAC_PIPELINE`, `ROOT_REGISTERED_CACHE`,
   etc.) — will not resolve on Windows/SANSORI machines without editing. Rigidity here is the
   **Friedenwald K** coefficient ([src/ocularrigidity/friedenwald.py](src/ocularrigidity/friedenwald.py)),
   derived from choroidal *area* change (`deltaA`, from `extract_deltaA.py`) via a spherical-shell model.
2. **SANSORI batch pipeline** (Nicolas's track, `Astronaut` branch): [Astronauts/](Astronauts/) (gitignored,
   see below) + `src/ocularrigidity/scripts/registration/astronauts.py` + `scripts/one_cycle/astronauts.py`
   + the Streamlit app in [testing_app/](testing_app/). Operates directly on
   `E:/SANSORI/<NN_id>/<...>_rigidity/<..._OD|OS...>/` (hardcoded `PATH_GENERAL` in each script/page). Rigidity
   here is the **Sayah et al. (2020) k** coefficient, derived from choroidal *thickness* pulsatility
   (`deltaY`) plus axial length/IOP/OPA joined from `sansori_db.db` + `visit_data.csv`.

Both tracks now converge on the same registration engine (`registration.registration_engine.VideoRegistrator`)
and, for cardiac extraction, the same `motion.pulsation` package — the fork is really just "which script
calls them with which config and where it reads/writes," not two independent implementations. Don't mix up
`k` (Sayah, mm⁻³) and `K` (Friedenwald) — different quantities, different formulas, different config
dataclasses (`FriedenwaldConfig` vs the ad hoc bandpass+Hilbert method in `compute_rigidity_time_series.py`).

## `Astronauts/` is local-only and currently out of sync with the refactor

`Astronauts/` is entirely gitignored (`.gitignore:5`) — it lives only on this machine, isn't reviewed, and
was **not** touched by the recent restructuring commits. As of this writing:

- [Astronauts/register_files.py](Astronauts/register_files.py) imports `from ocularrigidity.registration.export
  import export_registered_video, DEFAULT_OUTPUT_SUBDIR` — that module was moved/renamed to
  `ocularrigidity.scripts.registration.astronauts`. It also prints `cfg.flatten` / `cfg.horizontal_alignment`,
  fields that don't exist on `RegistrationConfig` anymore (now `flatten_rpe` / `lateral_method`). **This
  script will raise `ImportError` before doing anything.**
- [Astronauts/compute_rigidity_time_series.py](Astronauts/compute_rigidity_time_series.py) imports
  `from ocularrigidity.rigidity.features import compute_deltaY_masks` — `rigidity/` no longer contains any
  Python source (moved to `thickness/features.py`); this import is also broken. It also still reads
  choroid data from the **legacy** `RawImages/registeredBscans/` (MATLAB registration) rather than the new
  `RawImages/registered/` produced by `export_registered_video`, and doesn't yet apply the A-scan/RPE
  refinement pass — wiring that in was the explicit goal of the most recent working session on this repo
  and is still in progress.
- [Astronauts/segment_files.py](Astronauts/segment_files.py) is unaffected (imports only stable
  `data`/`segmentation` modules) but still targets `RawImages/oneCycle_regAveBin/`, a MATLAB-era path.

Fix imports/attribute names here before running any of these three scripts.

## Registration architecture

**Entry point:** `VideoRegistrator` in
[registration/registration_engine.py](src/ocularrigidity/registration/registration_engine.py) — replaces the
old `motion/registered_video.py::RegisteredVideo` (stale name still floating around in some docstrings/comments,
including inside `astronauts.py`; the class itself no longer exists under that name/module). Lazily loads raw
frames (`data/compression.py`) + raw masks (`data/io.py`), calls `registration.rigid.register_videos(...)`,
exposes `registered_frames` / `registered_masks` / `thickness` (via `thickness.features.compute_deltaY_boundaries`
on the cleaned BM/CSI boundaries) as cached properties, and optionally persists/reloads results under
`cache_dir` (`_cache_meta()` — every registration-affecting field except `batch_size` is part of the cache key;
changing one invalidates prior caches).

Registration itself (`rigid.register_videos`, called by `VideoRegistrator.compute_registration`) is a 2-pass
pipeline:

1. **Lateral (x) + vertical (y) general alignment** — always runs.
   - Reference frame = the one whose mask area is closest to the temporal median.
   - **Lateral:** [registration/lateral/dx.py](src/ocularrigidity/registration/lateral/dx.py)
     `estimate_lateral_dx()` — `"xcorr"` (vertical-mean profile cross-correlation,
     `lateral/correlation.py::profile_correlation_dx`) or `"fullframe"` (2D phase correlation,
     `frame_correlation_dx`, Hann-windowed + spectrally band-passed to avoid low-freq drift artifacts) or
     `"both"` (averaged). `crop_factor` (fraction of frame width kept) and `scale_factor` (downscale before
     correlation) trade accuracy for speed/robustness; `transversal_bandpass` is the Hz band applied before
     `"fullframe"` correlation. Result is temporally outlier-rejected (`robust_temporal_dx`) and optionally
     smoothed (`smooth_translations`).
   - **Fovea correction** (`fovea_correction_enabled`): before lateral estimation, shifts frames/masks so the
     fovea sits at a fixed column, via `segmentation.fovea.from_ilm.estimate_fovea` (see below). Reduces
     lateral-registration error near the macula where the BM curvature confuses cross-correlation.
   - **Vertical (y):** per-column Bruch's-membrane boundary alignment to the reference frame's BM (or, if
     `flatten_rpe`, to a constant row instead).
   - **Postprocessing:** [registration/postprocess.py](src/ocularrigidity/registration/postprocess.py)
     `filter_bad_ascans_per_bms` flags A-scan columns with unreliable BM (jumps > 20px, bad in ≥50% of
     frames, ±15px dilation) and zeros them in both frames and masks, across *all* frames. (⚠️ at the time of
     this writing this call is commented out locally in `rigid.py` — uncommitted change, check `git diff`
     before assuming it's active.)
2. **A-scan / RPE refinement (2nd pass, optional — `axial_refinement`)** —
   [registration/axial/median_registration.py](src/ocularrigidity/registration/axial/median_registration.py)
   `register_ascans_to_median`: computes the temporal median of the pass-1-registered volume, applies
   `axial/shadow.py::correct_shadow` (Girard 2011) and/or `axial/log_filter.py::laplacian_of_gaussian` (RPE
   enhancement, independently togglable) *only to estimate the shift*, then does per-column 1D spectral phase
   correlation against the median (band-passed via `axial_bandpass`) to refine vertical alignment. Adds
   `dy_median` to the returned `params` dict; applied to the raw registered pixels via `grid_sample`.

`registration/horizontal/` (the pre-rename lateral folder) and `registration/rigidity/` no longer contain
source — empty leftover directories, ignore them if you see them in a listing.

`registration/sparse_demons.py` and `test/demons.py` (log-demons deformable registration, additive + BCH
diffeomorphic updates, multi-resolution) are a separate, **experimental, not-yet-integrated** registration
approach — don't assume anything under `test/` is wired into the main pipeline.

**Gotcha (SANSORI data on `E:/SANSORI`):** most conditions only have a *legacy MATLAB* registration under
`RawImages/registeredBscans/` (masks + `.mj2` video, no A-scan variant). Only conditions reprocessed through
`export_registered_video` have `RawImages/registered/` (the Python-pipeline output, which is where the
A-scan-refined variant — written with `suffix="_ascan"` alongside the base files — will live once that's wired
up). Check which folder a script reads from before assuming A-scan registration is available.

## Fovea estimation — two independent methods

[segmentation/fovea/](src/ocularrigidity/segmentation/fovea/):

- **`from_ilm.py`** (new, no training required): `estimate_fovea(frames, masks)` builds an ROI above the BM
  from the boundary + intensity threshold, extracts its upper boundary (ILM proxy) via
  `segmentation.postprocess.interfaces`, then `estimate_fovea_from_ilm(ilm, margin, center_bias)` — detrend →
  gap-fill → low-pass → peak search (optionally biased toward the frame center, since macula-adjacent
  artifacts can otherwise win) → parabolic subpixel vertex fit. Fully vectorized over a batch of curves. This
  is what `registration.lateral.dx.fovea_correction` calls during lateral registration.
- **`infer.py`**: `predict_fovea_x(model, frames, ...)` — DSNT-based (`dsnt.py`) trained keypoint model
  (`module.py::FoveaKeypointModule`, data loading in `data.py`, trained via `scripts/fovea/train.py`). Not
  currently the default path in registration but available as an alternative fovea-x source.

## Thickness vs. rigidity — extraction is separate from fitting

- [thickness/features.py](src/ocularrigidity/thickness/features.py) (moved out of the old `rigidity/`):
  pure feature extraction, no rigidity math. `compute_deltaY_boundaries(bm, csi)` (thickness from boundaries,
  what `VideoRegistrator.thickness` uses), `compute_deltaY_masks(mask)` (thickness from raw mask sum, used by
  `Astronauts/compute_rigidity_time_series.py`), `extract_thickness_gpu`/`extract_thickness_distance`
  (distance-transform thickness, CuPy/CPU respectively).
- [friedenwald.py](src/ocularrigidity/friedenwald.py) (top-level now, not under `rigidity/`): the actual K
  fitting — `cycle_amplitude` → `deltaA_to_deltaV_uL` (spherical-shell model) → `friedenwald_K`. Fully
  documented pipeline-glue docstring at the top of the file if you need the derivation.
- `rigidity/` itself is now an empty directory (only `__pycache__` survives) — don't add anything there or
  import from it; it's dead.

## Cardiac-cycle extraction: `motion/pulsation/` package

Replaces the old single-file `motion/pulsation.py`. Split into an abstract rate/phase estimator, a concrete
signal source, a separate folding step, and an orchestrator:

- [config.py](src/ocularrigidity/motion/pulsation/config.py): `PulseExtractionConfig` (physiological prior,
  ICA/PCA decomposition, Lomb-Scargle scoring, harmonic correction, IQ-demodulation phase) and `NCycleConfig`
  (folding: `n_cycle`, `n_bins`, `fold_method`, `phase_method`). **Don't confuse these with
  `pipeline_config.PulsationConfig`** — that's a separate, older config dataclass used by
  `scripts/pulsation/infer.py` for the generic cohort track; the two aren't interchangeable and cover
  overlapping but non-identical parameters.
- [abstract_pulse_extractor.py](src/ocularrigidity/motion/pulsation/abstract_pulse_extractor.py):
  `AbstractPulseExtractor` — source-agnostic engine (spatial smoothing → ICA/PCA → Lomb-Scargle frequency
  search → harmonic correction → IQ-demodulation *and* peak-locked phase). Subclasses just implement the
  abstract `signal` property.
- [mask_pulse_extractor.py](src/ocularrigidity/motion/pulsation/mask_pulse_extractor.py): `MaskPulseExtractor`
  — `signal` = registered-choroid thickness per A-scan (from `VideoRegistrator.thickness`), with hole-column
  trimming and outlier-frame rejection. The only concrete extractor currently implemented;
  `frame_pulse_extractor.py` (intensity-based signal) is an empty placeholder.
- [n_cycle_reconstructor.py](src/ocularrigidity/motion/pulsation/n_cycle_reconstructor.py):
  `NCycleReconstructor` — folds registered frames into averaged cardiac cycles given any extractor's
  phase/quality outputs. Deliberately decoupled from rate/phase estimation.
- [pipeline.py](src/ocularrigidity/motion/pulsation/pipeline.py): `run_cardiac_pipeline(...)` wires
  `VideoRegistrator → VideoTimelineAligner (motion/video_timeline_aligner.py) → MaskPulseExtractor →
  [NCycleReconstructor]` and returns a `CardiacPipelineResults` (`motion/pipeline_results.py`). Single entry
  point used by `scripts/pulsation/infer.py`.

`motion/` also still has older, separate signal-processing modules not part of this package —
`amplitude.py`, `displacement.py`, `one_cycle.py`, `pulsation_SVD_intensities.py`, `filters/`, `projection/` —
check whether a given script actually imports the new `motion.pulsation` package or one of these before
assuming which analysis path is live.

## Segmentation

[segmentation/inference.py](src/ocularrigidity/segmentation/inference.py) `infer()` wraps
`ChoroidSegmentationModule` (PyTorch Lightning U-Net, weights auto-downloaded from Hugging Face via
`segmentation/utils.get_choroid_segmentation_model()`): optional resize (`scale_factor`), sigmoid + GPU
graph-cut postprocessing (`segmentation/postprocess/graphcut_gpu.py`), then largest-connected-component
filtering (`postprocess/blob.py`). Input is a `(T, H, W)` or `(T, C, H, W)` cube; uint8 input is
auto-normalized. `segmentation/vessels/` (retinal vessel postprocessing) and `segmentation/trainer/` (Lightning
training module) are separate concerns, not part of the choroid inference path.

## GPU dependency

Almost every numerical routine (segmentation inference, lateral/axial registration, phase correlation,
`extract_thickness_gpu`) defaults to `device="cuda"` and several default parameters are tuned assuming a GPU
is present. `thickness/features.py` hard-imports `cupy`/`cupyx` at module level — importing it fails
immediately on a machine without CUDA/cupy, even if you only wanted a CPU-only function from the same file
(`extract_thickness_distance` is the CPU/scipy equivalent, but you can't reach it without cupy installed).

## Data I/O

- [data/io.py](src/ocularrigidity/data/io.py): `save_mask`/`load_mask` (bit-packed + zstd-compressed boolean
  masks — always use these instead of raw `.npy` for masks), `load_cube` (raw `cube.bin` + `timestamp.txt`,
  local or `smb://` via `smbclient`).
- [data/compression.py](src/ocularrigidity/data/compression.py): video codecs for cubes — `mp4_to_cube`,
  `read_gray` (decord-based `.mp4`/`.mkv` reader), `cube_to_mp4_fastest`/`cube_to_mkv_lossless`. Several
  default to a hardcoded Linux ffmpeg path; on Windows pass `ffmpeg=...` explicitly, or rely on
  `scripts/registration/astronauts.py::resolve_ffmpeg()` (PATH, then `imageio_ffmpeg` fallback), which also
  pops `IMAGEIO_FFMPEG_EXE` from the environment to defeat the hardcoded default some import chains set.
- [data/spectralis.py](src/ocularrigidity/data/spectralis.py): `SpectralisStudy` parses Heidelberg HEYEX XML
  exports (series, acquisition times, image quality) — how raw `.tif` frames get ordered/matched to metadata
  before segmentation/registration (`load_ordered_oct_series` in `scripts/registration/astronauts.py`).
- [consts.py](src/ocularrigidity/consts.py): hardcoded `/home/clement/...`, `/media/clement/...`,
  `smb://192.168.11.16/...` paths for the generic-track roots — won't resolve outside Clement's machine. The
  SANSORI track instead hardcodes `PATH_GENERAL = Path("E:/SANSORI")` independently in each script/page
  (`Astronauts/*.py`, `testing_app/sansori_nav.py`, `testing_app/_registration_common.py`).

## Config dataclasses

[pipeline_config.py](src/ocularrigidity/pipeline_config.py) centralizes per-stage parameters as frozen
dataclasses, each with a singleton instance at module bottom (`REGISTRATION`, `PULSATION`, `DELTA_Y`,
`SEGMENTATION`, `DELTA_A`, `FRIEDENWALD`, `MISREGISTRATION`) imported by name elsewhere instead of being
re-instantiated:

- `RegistrationConfig` — every field except `batch_size` is part of `VideoRegistrator`'s cache key; changing
  one invalidates cached registrations for all downstream stages. Recently gained `crop_factor`,
  `scale_factor`, `transversal_bandpass`, `axial_bandpass` (see Registration architecture above) — these are
  now user-tunable per condition rather than baked into `rigid.py`.
- `PulsationConfig` — for the generic cohort track only; see the pulsation-package section above for why this
  is distinct from `motion.pulsation.config.PulseExtractionConfig`.
- `FriedenwaldConfig` — rig-specific OCT calibration (axial scale, lateral coverage width, vitreous fraction,
  pressure convention). Must match the real segmentation geometry or `K` is silently biased.
- `MisregistrationConfig` — QC thresholds consumed by `scripts/cohort_analysis/flag_misregistration.py`.

The SANSORI interactive app (`testing_app/`) used to take a different path — per-condition
`experiments/experiment_*.json` files — but no longer does: `testing_app/_registration_common.py`'s
"Enregistrer" action (`save_registration_config`) now rewrites `RegistrationConfig`'s field defaults
DIRECTLY in `pipeline_config.py` source (AST-located literal replacement, touching only fields declared on
the dataclass, then `importlib.reload`s the module so the running app picks it up immediately) — there is no
longer any parameters file at all. `RegistrationConfig` is now a genuinely global singleton in every sense:
the same `pipeline_config.REGISTRATION` any script imports.

**Stale as of this rewrite**: `Astronauts/register_files.py` still reads `experiments/experiment_*.json`
(on top of its already-broken import noted above) — that file is never written anymore, so once its import
is fixed it will also need to read `pipeline_config.REGISTRATION` directly instead.

## Streamlit apps

Two separate apps, one per track:

- **`viewer/streamlit_explorer/`** (generic cohort track): `Home.py` + `pages/` —
  `1_Cases.py`, `2_Regression.py`, `3_Viewer.py`, `4_Inference.py` (segmentation tuning), and two new pages:
  `5_Registration.py` (tune lateral/axial/fovea-correction parameters on a selected case, calls
  `registration.rigid.register_videos` directly) and `6_Fovea_Estimation.py` (visualize
  `segmentation.fovea.from_ilm.estimate_fovea_from_ilm` output on a B-scan).
- **`testing_app/`** (SANSORI track), rewritten to persist to `pipeline_config.py` (no `.json` anywhere):
  `first_cc_registration.py` is now the SOLE place `RegistrationConfig` widgets are rendered — one widget per
  dataclass field (`w_reg_<field>` keys), seeded once per session from `pipeline_config.REGISTRATION`
  (`_registration_common.init_registration_state`) — plus a pair preview and the "Enregistrer" button.
  `pages/1_Correction_A-scan.py` and `pages/2_Video_recalee.py` are read-only consumers of that same shared
  state (`registration_config_from_state`); they render no config widgets of their own. Preview registration
  goes through `_registration_common.build_registrator`, which — like
  `scripts/registration/astronauts.py::export_registered_video` — injects in-memory frames/masks into a
  `VideoRegistrator` (`_raw_frames`/`_raw_masks` set directly, bypassing disk I/O), so the interactive preview
  and the production export share the exact same registration code. Since `VideoRegistrator` always picks its
  own reference frame (closest-to-median mask area — meaningful for a full video, not for one hand-picked
  pair), `register_pair` pads the 2-frame preview to `[ref, ref, ref, mov]`: with 3 identical copies the
  median (both the mask-area scalar used for auto ref-selection, and the per-pixel temporal median the
  A-scan pass aligns onto) is pinned to `ref` regardless of `mov`, so index `0`/`-1` of the result are exactly
  "reference" / "point recalé" as the user intended. `pages/3_Video_one_cycle.py` is unrelated to
  `RegistrationConfig` (one-cycle folding params, ad hoc kwargs to `export_one_cycle_video`) and untouched.
- Deprecated pygame viewer: `python -m ocularrigidity.viewer`.

## Commands

```bash
pip install -e .                       # editable install (poetry-core backend, pyproject.toml)

# Tests (only tests/test_spectralis.py exists; it also runs standalone)
pytest tests/test_spectralis.py
python tests/test_spectralis.py

# SANSORI batch pipeline (Astronauts/, gitignored — fix the broken imports noted above before running)
python Astronauts/register_files.py              # segment + register raw .tif -> RawImages/registered/
python Astronauts/segment_files.py                # segment oneCycle_regAveBin video variant
python Astronauts/compute_rigidity_time_series.py # -> rigidity_time_series.csv + per-condition figures

# Interactive registration tuning (Streamlit, SANSORI data)
streamlit run testing_app/first_cc_registration.py

# Streamlit cohort explorer (generic pipeline)
streamlit run src/ocularrigidity/viewer/streamlit_explorer/Home.py

# Deprecated pygame viewer
python -m ocularrigidity.viewer
```

`scripts/pipeline.sh` orchestrates the generic-track scripts end to end, but references
`scripts/registration/infer.py` — renamed to `scripts/registration/glaucoma.py` — so it's currently stale too;
fix the path before running it wholesale. `scripts/jupypeline.ipynb` is a notebook version of the same
pipeline, kept separately for interactive iteration.

There is no lint/format config in the repo (no ruff/black/flake8 config, no pre-commit) and no CI workflow.

## Tests vs. experiments

- `tests/` (singular) is the real pytest suite — currently just `test_spectralis.py`.
- `test/` (plural) is exploratory/experimental code, not a test suite in the pytest sense: FEM/biomechanics
  simulation (`fem_simulation.py`), animation generation, pulsation prototyping (`choroid_pulse.py`), and the
  standalone log-demons registration experiment (`demons.py`, `warp_core.py`) mentioned above. Don't expect
  `pytest test/` to do anything meaningful.
