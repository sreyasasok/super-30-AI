#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"
MODEL_URL="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
MODEL_PATH="${MODEL_DIR}/pose_landmarker_lite.task"

mkdir -p "${MODEL_DIR}"

if [ -f "${MODEL_PATH}" ]; then
    echo "Model already present at ${MODEL_PATH}"
    exit 0
fi

echo "Downloading pose landmarker model to ${MODEL_PATH}..."
curl -L --fail -o "${MODEL_PATH}" "${MODEL_URL}"
echo "Done."
