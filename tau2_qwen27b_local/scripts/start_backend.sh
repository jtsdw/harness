#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

: "${QWEN_PACKAGE_ROOT:=${WORKSPACE_ROOT}/qwen36-offline-package}"
: "${QWEN_STACK_ROOT:=${QWEN_PACKAGE_ROOT}/source/qwen36-27b-single-5090}"
: "${QWEN_COMPOSE_DIR:=${QWEN_STACK_ROOT}/compose}"
: "${MODEL_CACHE_ROOT:=${HOME}/.cache/huggingface/hub/models--Lorbus--Qwen3.6-27B-int4-AutoRound}"

if [[ -z "${MODEL_DIR:-}" ]]; then
  require_file "${MODEL_CACHE_ROOT}/refs/main"
  model_revision="$(tr -d '[:space:]' < "${MODEL_CACHE_ROOT}/refs/main")"
  MODEL_DIR="${MODEL_CACHE_ROOT}/snapshots/${model_revision}"
fi
: "${MODEL_BLOBS_DIR:=${MODEL_CACHE_ROOT}/blobs}"

base_compose="${QWEN_COMPOSE_DIR}/docker-compose.yml"
offline_compose="${QWEN_COMPOSE_DIR}/docker-compose.offline.yml"

require_file "$base_compose"
require_file "$offline_compose"
require_file "${MODEL_DIR}/config.json"
require_file "${MODEL_DIR}/model.safetensors.index.json"
require_dir "$MODEL_BLOBS_DIR"

if ! docker image inspect qwen36-vllm-5090:offline-v1 >/dev/null 2>&1; then
  echo "ERROR: Docker image qwen36-vllm-5090:offline-v1 is not loaded" >&2
  exit 1
fi

if curl -fsS --max-time 3 "${VLLM_BASE_URL%/v1}/health" >/dev/null 2>&1; then
  echo "Backend is already healthy at ${VLLM_BASE_URL}."
  "${SCRIPT_DIR}/preflight.sh"
  exit 0
fi

echo "Starting qwen3.6-27b-autoround with the validated offline Compose stack..."
echo "  compose: ${QWEN_COMPOSE_DIR}"
echo "  model:   ${MODEL_DIR}"

MODEL_DIR="$MODEL_DIR" MODEL_BLOBS_DIR="$MODEL_BLOBS_DIR" \
  docker compose \
    -f "$base_compose" \
    -f "$offline_compose" \
    up -d

echo "Waiting for the API to become ready..."
deadline=$((SECONDS + 360))
until curl -fsS --max-time 5 "${VLLM_BASE_URL%/v1}/health" >/dev/null 2>&1; do
  if (( SECONDS >= deadline )); then
    echo "ERROR: backend did not become healthy within 360 seconds" >&2
    docker logs --tail=120 vllm-qwen36-27b >&2 || true
    exit 1
  fi
  if ! docker inspect vllm-qwen36-27b --format '{{.State.Running}}' 2>/dev/null | grep -qx true; then
    echo "ERROR: vllm-qwen36-27b stopped during startup" >&2
    docker logs --tail=120 vllm-qwen36-27b >&2 || true
    exit 1
  fi
  sleep 10
done

"${SCRIPT_DIR}/preflight.sh"
