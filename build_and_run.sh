#!/usr/bin/env bash
set -euo pipefail

# Usage:
# ./build_and_run.sh --output-dir ./results --gpu-enabled true --iters 150

OUTPUT_DIR="./results"
IMAGE_NAME="densenet_bench:latest"
GPU_ENABLED="true"

EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --gpu-enabled) GPU_ENABLED="$2"; shift 2;;
    --image) IMAGE_NAME="$2"; shift 2;;
    *) EXTRA_ARGS+=("$1"); shift;;   # 👈 collect all unknown args
  esac
done
mkdir -p "${OUTPUT_DIR}"

echo "Building docker image ${IMAGE_NAME}..."
docker build -t "${IMAGE_NAME}" .

# choose docker run flags
DOCKER_RUN_FLAGS=(
  "-v" "$(pwd)/benchmark_entry.py:/app/benchmark_entry.py"
  "-v" "${OUTPUT_DIR}:/app/results"
  "-w" "/app"
)

# expose tensorboard port
DOCKER_RUN_FLAGS+=("-p" "6006:6006")

# GPU flag + device
if [[ "${GPU_ENABLED}" == "true" ]]; then
  GPU_FLAG="--gpus all"
  DEVICE="cuda"
else
  GPU_FLAG=""
  DEVICE="cpu"
fi

echo "Running container (output -> ${OUTPUT_DIR})..."
set +e
# run container (blocking). When container exits, results will be in ${OUTPUT_DIR}
docker run --rm ${GPU_FLAG} "${DOCKER_RUN_FLAGS[@]}" "${IMAGE_NAME}" \
  --output-dir /app/results --device "${DEVICE}" "${EXTRA_ARGS[@]}"
RUN_RC=$?
set -e

if [[ ${RUN_RC} -ne 0 ]]; then
  echo "Container run failed with code ${RUN_RC}"
  exit ${RUN_RC}
fi

echo "Benchmark finished. Results in ${OUTPUT_DIR}:"
ls -la "${OUTPUT_DIR}"
echo "CSV summary:"
if [[ -f "${OUTPUT_DIR}/benchmark_results.csv" ]]; then
  cat "${OUTPUT_DIR}/benchmark_results.csv"
else
  echo "No CSV found."
fi

echo "Start tensorboard locally with: tensorboard --logdir ${OUTPUT_DIR}/logs/tensorboard --port 6006"
