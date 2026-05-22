# OcularRigidity

**Automated analysis of the choroid from OCT B-scan videos, toward non-invasive estimation of ocular rigidity.**

> ⚠️ **Work in progress.** This repository is under active development. APIs, file formats, and module structure are subject to change without notice. Results should be considered preliminary.

---

## Overview

This project develops a pipeline for the quantitative analysis of the choroid in time-resolved optical coherence tomography (OCT) B-scans. The long-term goal is to estimate **ocular rigidity** - the biomechanical relationship between intraocular pressure (IOP) and ocular volume - from pairs of continuous tonometry and OCT video recordings.

The pipeline is being developed in stages:

- ✅ **Choroid segmentation** - a deep learning model (U-Net) trained to delineate the choroidal layer (Bruch's membrane to choroid-sclera interface) on individual B-scans.
- 🚧 **Temporal analysis** - cycle-aware processing of segmented video to extract area/thickness time series.
- 🚧 **Rigidity estimation** - fitting pressure-area relationships to derive a rigidity coefficient, with appropriate propagation of segmentation uncertainty.

---

## Current features

- **Segmentation model** based on a U-Net architecture, trained on publicly available and in-house annotated OCT datasets.
- **Batch inference pipeline** with GPU acceleration, sliding-batch processing for long videos, and automatic post-processing (largest connected component).
- **Compressed mask storage** using bit-packing + zstd compression for efficient archival of boolean masks.
- **Interactive viewer** (pygame) for browsing and visualizing segmentation results across multiple recordings.
- **Flexible I/O layer** supporting local and SMB-mounted data sources.

---

## Installation

```bash
git clone https://github.com/ClementPla/OcularRigidity.git
cd OcularRigidity
pip install -e .
```

Dependencies include PyTorch, PyTorch Lightning, numpy, scipy, zstandard, and pygame. A CUDA-capable GPU is strongly recommended for inference.

---

## Usage

I strongly recommend to read the [example notebooks](notebook/example.ipynb).

### Running segmentation on a single recording

```python
from ocularrigidity.segmentation import ChoroidSegmentationModule, infer
from ocularrigidity.io import load_cube

model = ChoroidSegmentationModule.load_from_checkpoint("path/to/checkpoint.ckpt")
data = load_cube("path/to/recording/")
mask = infer(model, data, scale_factor=0.5, batch_size=128, device="cuda")
```

### Batch processing

A script is provided to run inference over a dataset and mirror the input folder structure in an output directory. Errors on individual recordings are logged without interrupting the batch.

### Visualization

A standalone viewer application lets you browse processed recordings and inspect segmentations over time:

```bash
python -m ocularrigidity.viewer
```

The viewer supports play/pause, frame stepping, speed control, and overlay toggling.

---

## Project structure

ocularrigidity/
├── segmentation/      # model, training, inference, post-processing
├── data/                # data loaders and mask (de)serialization
├── viewer/            # pygame UI for mask inspection. Soon to be deprecated
├── rigidity/          # pressure-area analysis and fitting
└── scripts/           # batch processing utilities


---

## Roadmap

- [x] Baseline U-Net segmentation of the choroid
- [x] Efficient inference and storage pipeline
- [x] Mask browser / viewer
- [ ] Temporal regularization (warp-consistency loss, topology-preserving losses)
- [x] Area / thickness time series extraction with artifact rejection
- [ ] Synchronization with continuous tonometry
- [x] Pressure-area curve fitting and rigidity estimation
- [~] Validation study

---