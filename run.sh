#!/usr/bin/env bash
set -euo pipefail

IMAGE_DIR="${1:-./image}"
OUTPUT_ZIP="${2:-./results.zip}"

MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
REVISION="eed13092ef92e448dd6875b2a00151bd3f7db0ac"

echo "Running inference..."
echo "Image dir   : ${IMAGE_DIR}"
echo "Output zip  : ${OUTPUT_ZIP}"
echo "Model       : ${MODEL_NAME}"
echo "Revision    : ${REVISION}"

python src/inference.py \
  --image_dir "${IMAGE_DIR}" \
  --output_zip "${OUTPUT_ZIP}" \
  --model_name "${MODEL_NAME}" \
  --revision "${REVISION}"
