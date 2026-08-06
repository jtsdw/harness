#!/usr/bin/env bash
# Same pattern as run_bfcl_local_vllm.sh -- see that script's header comment for why this wrapper
# exists (properly `export`s the local-vLLM boilerplate instead of it being hand-edited into the
# shared run_gsm8k_benchmark.sh, which silently breaks env var propagation to child processes).
#
# Usage:
#   MODEL_NAME="Qwen/Qwen2.5-32B-Instruct" ./run_gsm8k_local_vllm.sh
#
# run_gsm8k_benchmark.sh's own overrides (LIMIT/OUTPUT_DIR) still apply -- see its header comment.
# Note: unlike run_bfcl_local_vllm.sh, the underlying script has no MAX_CONNECTIONS support.

set -euo pipefail

: "${MODEL_NAME:?ERROR: set MODEL_NAME to the model being served, e.g. Qwen/Qwen2.5-32B-Instruct}"

export MODEL="openai-api/vllm/${MODEL_NAME}"
export MODEL_ARGS="${MODEL_ARGS:-emulate_tools=true}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-not-needed}"

if ! curl -s -m 3 "${VLLM_BASE_URL}/models" >/dev/null 2>&1; then
  echo "ERROR: no vLLM server reachable at ${VLLM_BASE_URL} -- start one first:" >&2
  echo "  cd ../../local-model-server && MODEL=\"${MODEL_NAME}\" ./scripts/serve.sh" >&2
  exit 1
fi

exec "$(dirname "${BASH_SOURCE[0]}")/run_gsm8k_benchmark.sh"
