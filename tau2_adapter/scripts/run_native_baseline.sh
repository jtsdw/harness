#!/usr/bin/env bash
# Reproduces the tau2-bench native-CLI baseline run documented in
# docs/tau2_bench_integration_findings.md -- the reference results the adapter path is compared
# against. Requires a local vLLM server already running with native tool-calling enabled (see
# local-model-server/scripts/serve.sh's NATIVE_TOOL_CALLING option).
#
# Usage:
#   ./scripts/run_native_baseline.sh                    # mock domain, all 10 tasks
#   NUM_TASKS=1 SAVE_NAME=results_1task ./scripts/run_native_baseline.sh

set -euo pipefail

: "${TAU2_BENCH_DIR:=/home/liuyingen/code/tau2-bench}"
: "${DOMAIN:=mock}"
: "${AGENT_LLM:=openai/Qwen/Qwen2.5-3B-Instruct}"
: "${USER_LLM:=openai/Qwen/Qwen2.5-3B-Instruct}"
: "${VLLM_BASE_URL:=http://localhost:8000/v1}"
: "${VLLM_API_KEY:=not-needed}"
: "${TEMPERATURE:=0.0}"
: "${NUM_TRIALS:=1}"
: "${NUM_TASKS:=10}"
: "${MAX_STEPS:=20}"
: "${SAVE_NAME:=results_mock_baseline}"

export PATH="$HOME/.local/bin:$PATH"
if [[ ! -d "$TAU2_BENCH_DIR" ]]; then
  echo "ERROR: tau2-bench not found at $TAU2_BENCH_DIR -- run scripts/setup_tau2_bench.sh first." >&2
  exit 1
fi

if ! curl -s -m 3 "${VLLM_BASE_URL}/models" >/dev/null 2>&1; then
  echo "ERROR: no vLLM server reachable at ${VLLM_BASE_URL} -- start it first:" >&2
  echo "  cd ../local-model-server && NATIVE_TOOL_CALLING=true ./scripts/serve.sh" >&2
  exit 1
fi

LLM_ARGS=$(printf '{"temperature": %s, "api_base": "%s", "api_key": "%s"}' \
  "$TEMPERATURE" "$VLLM_BASE_URL" "$VLLM_API_KEY")

RESULTS_DIR="${TAU2_BENCH_DIR}/data/simulations/${SAVE_NAME}"
if [[ -e "$RESULTS_DIR" || -e "${RESULTS_DIR}.json" ]]; then
  echo "Removing existing results at ${SAVE_NAME} to avoid the interactive resume prompt..."
  rm -rf "$RESULTS_DIR" "${RESULTS_DIR}.json"
fi

cd "$TAU2_BENCH_DIR"
TAU2_DATA_DIR="${TAU2_BENCH_DIR}/data" uv run tau2 run \
  --domain "$DOMAIN" \
  --agent-llm "$AGENT_LLM" \
  --agent-llm-args "$LLM_ARGS" \
  --user-llm "$USER_LLM" \
  --user-llm-args "$LLM_ARGS" \
  --num-trials "$NUM_TRIALS" --num-tasks "$NUM_TASKS" --max-steps "$MAX_STEPS" \
  --save "$SAVE_NAME"

echo
echo "Done. Results: ${TAU2_BENCH_DIR}/data/simulations/${SAVE_NAME}/"
