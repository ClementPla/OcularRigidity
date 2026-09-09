#!/usr/bin/env bash
#
# Full glaucoma-cohort run: segmentation+registration -> pulsation ->
# per-cycle segmentation -> deltaA -> QC.
#
#   ./scripts/pipeline.sh                       # run with the defaults below
#   OCULARRIGIDITY_RUN=V4 ./scripts/pipeline.sh
#   ./scripts/pipeline.sh --stage pulsation     # re-run a single stage
#   OCULARRIGIDITY_SEG_GPUS=0 ./scripts/pipeline.sh   # one GPU only
#
# Segmentation and registration are ONE stage: a single encoder pass per frame
# feeds both the segmentation decoder and the registration heads (see
# src/ocularrigidity/registration/fused.py).
#
# Every output path derives from the run identity below, so a run with different
# weights lands in its own tree. Stages are individually resumable: each skips
# work whose output exists, and keeps going past a failing case.
#
# There is a single-process alternative that carries each video through every
# stage while it is still in RAM — same outputs, same env vars:
#
#   python -m ocularrigidity.scripts.run_cohort --dry-run
#
# Use this script instead when a stage has to be re-run over the whole cohort on
# its own, or when segmentation+registration is worth spreading over both GPUs.

set -uo pipefail   # deliberately NOT -e: a stage that fails on some cases still
                   # produces useful output for the rest, and the next stage
                   # skips whatever is missing.

# --- Run identity -----------------------------------------------------------
OUTPUT_ROOT="${OCULARRIGIDITY_OUTPUT_ROOT:-/media/clement/HD/Santiago/OcularRigidity}"
RUN="${OCULARRIGIDITY_RUN:-FullPipeline}"
# Weights. The defaults are *read from consts.py* rather than repeated here:
# a literal in this script is a second source of truth that silently drifts, and
# it already cost a cohort re-run — the script pinned reg_cascade_v2 while
# consts.py had moved to v9, which invalidated every registration cache (the
# checkpoint is part of the cache key) and would have redone all 1018 volumes
# with the older model. Set OCULARRIGIDITY_CHECKPOINT / OCULARRIGIDITY_REGISTRATOR
# to override; otherwise whatever consts.py says is what runs.
_consts() { python -c "from ocularrigidity.consts import $1 as p; print(p)"; }
CHECKPOINT="${OCULARRIGIDITY_CHECKPOINT:-$(_consts CHECKPOINT_PATH)}"
REGISTRATOR="${OCULARRIGIDITY_REGISTRATOR:-$(_consts REGISTRATOR_CHECKPOINT)}"


# Masks and the registration cache are per-run: both depend on the weights, so
# reusing the previous run's would silently mix models. The compressed source
# videos are the shared input and stay where they are.
RUN_ROOT="${OUTPUT_ROOT}/${RUN}"

export OCULARRIGIDITY_OUTPUT_ROOT="${OUTPUT_ROOT}"
export OCULARRIGIDITY_RUN="${RUN}"
export OCULARRIGIDITY_CHECKPOINT="${CHECKPOINT}"
export OCULARRIGIDITY_REGISTRATOR="${REGISTRATOR}"
export OCULARRIGIDITY_MASKS="${RUN_ROOT}/masks"
export OCULARRIGIDITY_REGISTERED_CACHE="${RUN_ROOT}"
export OCULARRIGIDITY_CARDIAC="${RUN_ROOT}"

# Memory/throughput only — neither changes the result. Registration is the
# hungry one: a (batch, 2, H, W) float32 tensor plus a same-sized sampling grid.
export OCULARRIGIDITY_REGISTRATION_BATCH="${OCULARRIGIDITY_REGISTRATION_BATCH:-32}"
export OCULARRIGIDITY_SEGMENTATION_BATCH="${OCULARRIGIDITY_SEGMENTATION_BATCH:-8}"
# Frames per fused forward pass. This one holds encoder activations for a
# 1536x1024 input as well as the two heads, so it is the memory-sensitive knob.
FUSED_BATCH="${OCULARRIGIDITY_FUSED_BATCH:-8}"

# GPUs the segmentation stage spreads the cohort over, one process per entry.
# Disjoint shards, so this changes how long the stage takes, never what it
# produces. Use a single index to leave the other card free for the display.
SEG_GPUS="${OCULARRIGIDITY_SEG_GPUS:-0,1}"

LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${LOG_DIR}"

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
fi
if [[ ! -f "${REGISTRATOR}" ]]; then
    echo "Registration checkpoint not found: ${REGISTRATOR}" >&2
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

## The one stage worth parallelising. Each shard takes a disjoint slice of the
## cohort on its own GPU, and all of them skip volumes that already have a mask
## and a valid registration cache, so an interrupted run resumes by re-running.
run_segreg() {
    local name="segreg"
    local script="src/ocularrigidity/scripts/segmentation/segment_and_register.py"
    if [[ "${STAGE}" != "all" && "${STAGE}" != "${name}" ]]; then
        return 0
    fi

    IFS=',' read -r -a gpus <<< "${SEG_GPUS}"
    local n="${#gpus[@]}"
    echo "=== ${name} -> ${LOG_DIR}/${name}*.log (${n} GPU(s): ${SEG_GPUS})"

    if [[ "${n}" -eq 1 ]]; then
        local log="${LOG_DIR}/${name}.log"
        CUDA_VISIBLE_DEVICES="${gpus[0]}" python "${script}" \
            --batch-size "${FUSED_BATCH}" 2>&1 | tee "${log}"
        local status="${PIPESTATUS[0]}"
        if [[ "${status}" -ne 0 ]]; then
            echo "!!! ${name} exited ${status} (continuing; see ${log})" >&2
        fi
        return 0
    fi

    local pids=()
    for i in "${!gpus[@]}"; do
        local log="${LOG_DIR}/${name}.shard${i}.log"
        # tee, not a bare redirect: a stage that only writes to a file prints
        # nothing for hours and is indistinguishable from a hung one. grep drops
        # the per-batch progress bars, which interleave into noise across shards.
        {
            CUDA_VISIBLE_DEVICES="${gpus[$i]}" python "${script}" \
                --num-shards "${n}" --shard "${i}" \
                --batch-size "${FUSED_BATCH}" 2>&1 \
                | tee "${log}" \
                | grep --line-buffered -E "INFO|ERROR|FAILED"
            exit "${PIPESTATUS[0]}"
        } &
        pids+=($!)
        echo "    shard ${i} on GPU ${gpus[$i]} (pid ${pids[-1]}) -> ${log}"
    done

    # Wait for every shard even if one dies: the survivors' outputs are still
    # wanted, and the next stage skips whatever is missing.
    for i in "${!pids[@]}"; do
        if ! wait "${pids[$i]}"; then
            echo "!!! ${name} shard ${i} failed (continuing; see ${LOG_DIR}/${name}.shard${i}.log)" >&2
        fi
    done
}

echo "run        ${RUN}"
echo "root       ${RUN_ROOT}"
echo "checkpoint ${CHECKPOINT}"
echo "registrator ${REGISTRATOR}"
echo "seg GPUs   ${SEG_GPUS}"
echo

# 1+2. Segment AND register every cohort video in one GPU pass
#      -> ${RUN_ROOT}/masks, registered_{frames,masks} (sharded over SEG_GPUS)
run_segreg
# 3. Cardiac chain + fold -> ${RUN_ROOT}/one_cycle, measures
run_stage pulsation     src/ocularrigidity/scripts/pulsation/infer.py
# 4. Segment the folded cycles -> measures/**/segmented_cycles.npz
run_stage cycles        src/ocularrigidity/scripts/cohort_analysis/segment_n_cycles.py
# 5. Boundary displacement / area change -> deltaA_per_cycle.pkl
run_stage deltaA        src/ocularrigidity/scripts/cohort_analysis/extract_deltaA.py
# 6. QC -> ${RUN_ROOT}/misregistration_flags.csv
run_stage qc            src/ocularrigidity/scripts/cohort_analysis/flag_misregistration.py

echo
echo "Done. Outputs under ${RUN_ROOT}, logs in ${LOG_DIR}"
