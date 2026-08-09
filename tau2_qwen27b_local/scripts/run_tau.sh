#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

if [[ ! -v NUM_TASKS ]]; then
  NUM_TASKS=1
fi
: "${TASK_IDS:=}"
: "${RUN_NAME:=tau2_${TAU2_DOMAIN}_qwen27b_$(date +%Y%m%d_%H%M%S)}"

"${SCRIPT_DIR}/preflight.sh"

export VLLM_BASE_URL VLLM_API_KEY MODEL_NAME
export USER_MODEL_NAME="$MODEL_NAME"
export JUDGE_MODEL_NAME="$MODEL_NAME"
export TAU2_DOMAIN TAU2_TASK_SET TAU2_TASK_SPLIT
export NUM_TASKS TASK_IDS RUN_NAME
export TAU2_USER_LLM_ARGS TAU2_JUDGE_LLM_ARGS
export INSPECT_TRACE_VLLM_METRICS_URL
export TAU2_MAX_STEPS TAU2_MAX_ERRORS TAU2_EMPTY_RESPONSE_RETRIES
export TAU2_SEED TAU2_TIMEOUT

exec "${TAU2_ADAPTER_DIR}/scripts/run_adapter.sh" native "$@"
