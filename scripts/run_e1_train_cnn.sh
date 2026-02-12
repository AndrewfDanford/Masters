#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

if [[ -z "${E1_COHORT_CSV:-}" ]]; then
  echo "Missing required env var: E1_COHORT_CSV"
  exit 1
fi

if [[ -z "${E1_IMAGE_ROOT:-}" ]]; then
  echo "Missing required env var: E1_IMAGE_ROOT"
  exit 1
fi

OUTPUT_DIR="${E1_CNN_OUTPUT_DIR:-outputs/reports/e1_cnn}"
CHECKPOINT_OUT="${E1_CNN_CHECKPOINT:-outputs/models/e1_cnn_checkpoint.pt}"

cmd=(
  "${PYTHON_BIN}" -m src.models.e1_train_cnn
  --cohort-csv "${E1_COHORT_CSV}"
  --image-root "${E1_IMAGE_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --checkpoint-out "${CHECKPOINT_OUT}"
  --study-col "${E1_STUDY_COL:-study_id}"
  --subject-col "${E1_SUBJECT_COL:-subject_id}"
  --dicom-col "${E1_DICOM_COL:-dicom_id}"
  --split-col "${E1_SPLIT_COL:-split}"
  --image-path-col "${E1_IMAGE_PATH_COL:-path}"
  --uncertain-policy "${E1_UNCERTAIN_POLICY:-u_ignore}"
  --arch "${E1_CNN_ARCH:-resnet18}"
  --device "${E1_CNN_DEVICE:-cpu}"
  --input-size "${E1_CNN_INPUT_SIZE:-320}"
  --batch-size "${E1_CNN_BATCH_SIZE:-16}"
  --num-workers "${E1_CNN_NUM_WORKERS:-0}"
  --epochs "${E1_CNN_EPOCHS:-6}"
  --learning-rate "${E1_CNN_LEARNING_RATE:-1e-4}"
  --weight-decay "${E1_CNN_WEIGHT_DECAY:-1e-5}"
  --seed "${E1_CNN_SEED:-17}"
  --ece-bins "${E1_CNN_ECE_BINS:-10}"
  --freeze-backbone-epochs "${E1_CNN_FREEZE_BACKBONE_EPOCHS:-0}"
)

if [[ -n "${E1_LABELS_CSV:-}" ]]; then
  IFS=',' read -r -a labels <<< "${E1_LABELS_CSV}"
  for index in "${!labels[@]}"; do
    labels[$index]="$(echo "${labels[$index]}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  done
  cmd+=(--labels "${labels[@]}")
fi

if [[ "${E1_CNN_PRETRAINED_BACKBONE:-0}" == "1" ]]; then
  cmd+=(--pretrained-backbone)
fi

if [[ -n "${E1_CNN_INIT_CHECKPOINT:-}" ]]; then
  cmd+=(--init-checkpoint "${E1_CNN_INIT_CHECKPOINT}")
fi

if [[ -n "${E1_CNN_MAX_SAMPLES_PER_SPLIT:-}" ]]; then
  cmd+=(--max-samples-per-split "${E1_CNN_MAX_SAMPLES_PER_SPLIT}")
fi

if [[ "${E1_CNN_TRAIN_AUGMENT:-0}" == "1" ]]; then
  cmd+=(--train-augment)
fi

if [[ "${E1_CNN_NO_CLASS_BALANCE:-0}" == "1" ]]; then
  cmd+=(--no-class-balance)
fi

"${cmd[@]}"

