#!/usr/bin/env bash
# Reproduces the tau2-bench native-CLI baseline run documented in
# docs/tau2_bench_integration_findings.md -- the reference results the adapter path is compared
# against. Requires an OpenAI-compatible vLLM endpoint with native tool-calling enabled.
#
# Usage:
#   ./scripts/run_native_baseline.sh                    # mock domain, all 10 tasks
#   NUM_TASKS=1 SAVE_NAME=results_1task ./scripts/run_native_baseline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TAU2_SOURCE_DIR="${REPO_ROOT}/.deps/tau2-bench"

: "${TAU2_DATA_DIR:=}"
: "${DOMAIN:=mock}"
: "${AGENT_LLM:=openai/Qwen/Qwen2.5-3B-Instruct}"
: "${USER_LLM:=openai/Qwen/Qwen2.5-3B-Instruct}"
: "${JUDGE_LLM:=${USER_LLM}}"
: "${VLLM_BASE_URL:=http://localhost:8000/v1}"
: "${VLLM_API_KEY:=not-needed}"
: "${TEMPERATURE:=0.0}"
: "${NUM_TRIALS:=1}"
: "${NUM_TASKS:=10}"
: "${MAX_STEPS:=20}"
: "${SAVE_NAME:=results_mock_baseline}"

export PATH="$HOME/.local/bin:$PATH"
if [[ ! -f "${TAU2_SOURCE_DIR}/pyproject.toml" ]]; then
  echo "ERROR: tau2-bench source not found at $TAU2_SOURCE_DIR -- run scripts/setup_tau2_bench.sh first." >&2
  exit 1
fi

if [[ -z "$TAU2_DATA_DIR" || ! -d "${TAU2_DATA_DIR}/tau2/domains" ]]; then
  echo "ERROR: TAU2_DATA_DIR must point to a separately provisioned tau2 data directory." >&2
  echo "Expected to find <TAU2_DATA_DIR>/tau2/domains." >&2
  exit 1
fi

if ! curl -s -m 3 "${VLLM_BASE_URL}/models" >/dev/null 2>&1; then
  echo "ERROR: no vLLM server reachable at ${VLLM_BASE_URL}." >&2
  echo "Start the deployment backend and set VLLM_BASE_URL if it is not local." >&2
  exit 1
fi

LLM_ARGS=$(printf '{"temperature": %s, "api_base": "%s", "api_key": "%s"}' \
  "$TEMPERATURE" "$VLLM_BASE_URL" "$VLLM_API_KEY")
: "${JUDGE_LLM_ARGS:=$LLM_ARGS}"

RESULTS_DIR="${TAU2_DATA_DIR}/simulations/${SAVE_NAME}"
if [[ -e "$RESULTS_DIR" || -e "${RESULTS_DIR}.json" ]]; then
  echo "Removing existing results at ${SAVE_NAME} to avoid the interactive resume prompt..."
  rm -rf "$RESULTS_DIR" "${RESULTS_DIR}.json"
fi

cd "$TAU2_SOURCE_DIR"
TAU2_DATA_DIR="$TAU2_DATA_DIR" \
TAU2_LLM_NL_ASSERTIONS="$JUDGE_LLM" \
TAU2_LLM_NL_ASSERTIONS_ARGS="$JUDGE_LLM_ARGS" \
LITELLM_LOCAL_MODEL_COST_MAP=True \
uv run tau2 run \
  --domain "$DOMAIN" \
  --agent-llm "$AGENT_LLM" \
  --agent-llm-args "$LLM_ARGS" \
  --user-llm "$USER_LLM" \
  --user-llm-args "$LLM_ARGS" \
  --num-trials "$NUM_TRIALS" --num-tasks "$NUM_TASKS" --max-steps "$MAX_STEPS" \
  --save "$SAVE_NAME"

echo
echo "Done. Results: ${TAU2_DATA_DIR}/simulations/${SAVE_NAME}/"
