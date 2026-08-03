#!/usr/bin/env bash
#
# Full glaucoma-cohort run: segmentation -> registration -> pulsation ->
# per-cycle segmentation -> deltaA -> QC.
#
# Every output path is derived from the three variables below, so a run with
# different weights lands in its own tree and leaves earlier results untouched.
#
#   ./scripts/pipeline.sh                       # run with the defaults below
#   OCULARRIGIDITY_RUN=V4 ./scripts/pipeline.sh
#   ./scripts/pipeline.sh --stage pulsation     # re-run a single stage
#
# Stages are individually resumable: each script skips work whose output
# already exists, and each keeps going past a failing case rather than
# aborting the cohort. Failures are printed and land in the stage log.

set -uo pipefail   # deliberately NOT -e: a stage that fails on some cases still
                   # produces useful output for the rest, and the next stage
                   # skips whatever is missing.

# --- Run identity -----------------------------------------------------------
OUTPUT_ROOT="${OCULARRIGIDITY_OUTPUT_ROOT:-/media/clement/HD/Santiago/OcularRigidity/outputs_new_model}"
RUN="${OCULARRIGIDITY_RUN:-CardiacPipeline_V1}"
CHECKPOINT="${OCULARRIGIDITY_CHECKPOINT:-$(pwd)/checkpoints/choroid-segmentation-epoch=29-dice=0.99-v1.ckpt}"

# Masks and the registration cache are per-run: both depend on the weights, so
# reusing the previous run's would silently mix models. The compressed source
# videos are the shared input and stay where they are.
RUN_ROOT="${OUTPUT_ROOT}/${RUN}"

export OCULARRIGIDITY_OUTPUT_ROOT="${OUTPUT_ROOT}"
export OCULARRIGIDITY_RUN="${RUN}"
export OCULARRIGIDITY_CHECKPOINT="${CHECKPOINT}"
export OCULARRIGIDITY_MASKS="${RUN_ROOT}/masks"
export OCULARRIGIDITY_REGISTERED_CACHE="${RUN_ROOT}"
export OCULARRIGIDITY_CARDIAC="${RUN_ROOT}"

# GPU batch sizes. Memory/throughput only — neither changes the result, so
# raise them if the card has headroom and lower them on OOM. Registration is
# the hungry one: it allocates a (batch, 2, H, W) float32 tensor plus a
# same-sized sampling grid, ~3 GB each at batch 256 on 1024x1536.
export OCULARRIGIDITY_REGISTRATION_BATCH="${OCULARRIGIDITY_REGISTRATION_BATCH:-32}"
export OCULARRIGIDITY_SEGMENTATION_BATCH="${OCULARRIGIDITY_SEGMENTATION_BATCH:-24}"

LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${LOG_DIR}"

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
fi

# --- Stage selection --------------------------------------------------------
STAGE="all"
if [[ "${1:-}" == "--stage" ]]; then
    STAGE="${2:?--stage needs a name}"
fi

run_stage() {
    local name="$1" script="$2"
    if [[ "${STAGE}" != "all" && "${STAGE}" != "${name}" ]]; then
        return 0
    fi
    local log="${LOG_DIR}/${name}.log"
    echo "=== ${name} -> ${log}"
    python "${script}" 2>&1 | tee "${log}"
    local status="${PIPESTATUS[0]}"
    if [[ "${status}" -ne 0 ]]; then
        echo "!!! ${name} exited ${status} (continuing; see ${log})" >&2
    fi
}

echo "run        ${RUN}"
echo "root       ${RUN_ROOT}"
echo "checkpoint ${CHECKPOINT}"
echo

# 1. Segment the raw cohort videos -> ${RUN_ROOT}/masks
run_stage segmentation  src/ocularrigidity/scripts/segmentation/inference_on_patients.py
# 2. Register each video against those masks -> ${RUN_ROOT}/registered_{masks,frames}
run_stage registration  src/ocularrigidity/scripts/registration/glaucoma.py
# 3. Cardiac chain + fold -> ${RUN_ROOT}/one_cycle_pca_iq, measures_pca_iq
run_stage pulsation     src/ocularrigidity/scripts/pulsation/infer.py
# 4. Segment the folded cycles -> measures_pca_iq/**/segmented_cycles.npz
run_stage cycles        src/ocularrigidity/scripts/cohort_analysis/segment_n_cycles.py
# 5. Boundary displacement / area change -> deltaA_per_cycle.pkl
run_stage deltaA        src/ocularrigidity/scripts/cohort_analysis/extract_deltaA.py
# 6. QC -> ${RUN_ROOT}/misregistration_flags.csv
run_stage qc            src/ocularrigidity/scripts/cohort_analysis/flag_misregistration.py

echo
echo "Done. Outputs under ${RUN_ROOT}, logs in ${LOG_DIR}"
