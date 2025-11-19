#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# run_all.sh — batch‑run `data_extractor_optuna_repo.main` for multiple TASK_IDs and
# multiple dataset shards ("parts").

#
# Usage:
#     bash run_all.sh
# -----------------------------------------------------------------------------
set -euo pipefail

##########################
# CONFIG
##########################
OUTPUT_DIR="/home/aschipper/projects/LLM_data_extractor_optuna_repo/output"
DATA_DIR="/home/aschipper/projects/LLM_data_extractor_optuna_repo/data"

MODEL="qwen2.5:14b-instruct"
N_RUNS=1
NUM_EXAMPLES=0

# List every Task ID you want to annotate
TASK_IDS=(028 047 055 059 078 081 085 099 106 113 126 136 144 148 151)

# Number of dataset shards
PART_COUNT=3

##########################
# Build PARTS array automatically
##########################
PARTS=()
for ((i=1; i<=PART_COUNT; i++)); do
  PARTS+=("part${i}")
done

##########################
# Main loop
##########################
for TASK_ID in "${TASK_IDS[@]}"; do
  for PART in "${PARTS[@]}"; do
    DATA_FILE="${DATA_DIR}/rst_2024_val_${PART}.jsonl"

    # Skip if the shard file is missing
    if [[ ! -f "${DATA_FILE}" ]]; then
      echo "⚠️  Skip: ${DATA_FILE} (missing)"
      continue
    fi

    RUN_NAME="task${TASK_ID}_${PART}"
    DST_DIR="${OUTPUT_DIR}/${RUN_NAME}/fold0"
    DST_FILE="rst_val_predictions_${PART}_task${TASK_ID}.json"
    DST_PATH="${DST_DIR}/${DST_FILE}"

    # Skip combo if the predictions file already exists
    if [[ -f "${DST_PATH}" ]]; then
      echo "⏭  ${DST_FILE} already exists — skipping"
      continue
    fi

    echo "▶ Task ${TASK_ID} / ${PART}"

    python -m data_extractor.main \
        --task_id        "${TASK_ID}" \
        --run_name       "${RUN_NAME}" \
        --n_runs         "${N_RUNS}" \
        --model_name     "${MODEL}" \
        --num_examples   "${NUM_EXAMPLES}" \
        --output_dir     "${OUTPUT_DIR}" \
        --data_dir       "${DATA_FILE}"

    SRC="${OUTPUT_DIR}/${RUN_NAME}/fold0/nlp-predictions-dataset.json"

    if [[ -f "${SRC}" ]]; then
      mkdir -p "${DST_DIR}"
      mv "${SRC}" "${DST_PATH}"
      echo "   ✔ Saved $(basename "${DST_PATH}")"
    else
      echo "   ⚠️  ${SRC} not found – nothing moved"
    fi
  done
done

echo "🎉 All done."