#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
HARNESS_ROOT="$(cd -- "${PROJECT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${HARNESS_ROOT}/.." && pwd)"

PROFILE_FILE="${TAU2_PROFILE_FILE:-${PROJECT_DIR}/config.env}"
if [[ -f "$PROFILE_FILE" ]]; then
  # shellcheck source=/dev/null
  source "$PROFILE_FILE"
fi

: "${TAU2_DATA_DIR:=${WORKSPACE_ROOT}/dataset/datasets/tau2-bench-a1e85084a3960281cb06997594133e8f39ea42a7/data}"
: "${TAU2_ADAPTER_DIR:=${HARNESS_ROOT}/tau2_adapter}"
: "${VLLM_BASE_URL:=http://127.0.0.1:8020/v1}"
: "${VLLM_API_KEY:=local}"
: "${MODEL_NAME:=qwen3.6-27b-autoround}"
: "${TAU2_DOMAIN:=mock}"
: "${TAU2_TASK_SET:=}"
: "${TAU2_TASK_SPLIT:=auto}"
: "${TAU2_MAX_STEPS:=200}"
: "${TAU2_MAX_ERRORS:=10}"
: "${TAU2_EMPTY_RESPONSE_RETRIES:=2}"
: "${TAU2_SEED:=300}"
: "${TAU2_TIMEOUT:=900}"

VLLM_BASE_URL="${VLLM_BASE_URL%/}"
: "${INSPECT_TRACE_VLLM_METRICS_URL:=${VLLM_BASE_URL%/v1}/metrics}"
TASK_FILE="${TAU2_ADAPTER_DIR}/src/tau2_adapter/task.py"

LOCAL_NON_THINKING_LLM_ARGS="$(printf '{"api_base":"%s","api_key":"%s","temperature":0,"max_tokens":2048,"input_cost_per_token":0,"output_cost_per_token":0,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
  "$VLLM_BASE_URL" "$VLLM_API_KEY")"
: "${TAU2_USER_LLM_ARGS:=$LOCAL_NON_THINKING_LLM_ARGS}"
: "${TAU2_JUDGE_LLM_ARGS:=$LOCAL_NON_THINKING_LLM_ARGS}"

unset ALL_PROXY HTTPS_PROXY HTTP_PROXY
unset all_proxy https_proxy http_proxy
export NO_PROXY="127.0.0.1,localhost"
export TAU2_DATA_DIR
export LITELLM_LOCAL_MODEL_COST_MAP=True

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: required file not found: $path" >&2
    return 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "ERROR: required directory not found: $path" >&2
    return 1
  fi
}
