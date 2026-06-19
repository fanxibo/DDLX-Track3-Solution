#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python "$SCRIPT_DIR/src/inference.py" \
  --image_dir "${1:-$SCRIPT_DIR/image}" \
  --output_zip "${2:-$SCRIPT_DIR/submission_test.zip}"
