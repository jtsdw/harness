#!/usr/bin/env bash
# Runs a tau2 domain through our inspect_ai adapter (tau2_adapter/src/tau2_adapter/),
# in either of the two variants documented in docs/tau2_bench_integration_findings.md:
#   - emulate: agent side uses inspect_ai's emulate_tools=true (client-side text parsing) --
#     the original workaround for Bug 3 (the "strict" tool field vLLM rejects).
#   - native: agent side uses the custom tau2-agent-vllm provider (tau2_adapter/_registry.py),
#     which strips the "strict" field so real native tool-calling works -- the proper fix.
# Both need an OpenAI-compatible vLLM endpoint with native tool-calling enabled;
# the user simulator always uses native tool-calling regardless of the agent variant.
#
# Usage:
#   ./scripts/run_adapter.sh emulate
#   ./scripts/run_adapter.sh native
#   NUM_TASKS=1 ./scripts/run_adapter.sh native   # single-task smoke test

set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "emulate" && "$MODE" != "native" ]]; then
  echo "Usage: $0 emulate|native" >&2
  echo "  emulate -- agent side uses emulate_tools=true (Bug 3 workaround)" >&2
  echo "  native  -- agent side uses the custom tau2-agent-vllm provider (Bug 3 proper fix)" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TASK_DIR="${PROJECT_DIR}/src/tau2_adapter"
REPO_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"
TAU2_SOURCE_DIR="${REPO_ROOT}/.deps/tau2-bench"

: "${TAU2_DATA_DIR:=}"
: "${VLLM_BASE_URL:=http://localhost:8000/v1}"
: "${VLLM_API_KEY:=not-needed}"
: "${MODEL_NAME:=Qwen/Qwen2.5-3B-Instruct}"
: "${USER_MODEL_NAME:=${MODEL_NAME}}"
: "${JUDGE_MODEL_NAME:=${MODEL_NAME}}"
: "${TAU2_DOMAIN:=mock}"
: "${TAU2_TASK_SET:=}"
: "${TAU2_TASK_SPLIT:=auto}"
: "${NUM_TASKS:=}"  # empty = all tasks in the domain
: "${RUN_NAME:=tau2_adapter_${TAU2_DOMAIN}_${MODE}}"

VLLM_BASE_URL="${VLLM_BASE_URL%/}"
: "${INSPECT_TRACE_VLLM_METRICS_URL:=${VLLM_BASE_URL%/v1}/metrics}"
DEFAULT_LITELLM_ARGS="$(printf \
  '{"temperature":0.0,"api_base":"%s","api_key":"%s"}' \
  "$VLLM_BASE_URL" "$VLLM_API_KEY")"
: "${TAU2_USER_LLM_ARGS:=$DEFAULT_LITELLM_ARGS}"
: "${TAU2_JUDGE_LLM_ARGS:=$DEFAULT_LITELLM_ARGS}"

if [[ "$MODE" == "emulate" ]]; then
  AGENT_MODEL="openai-api/vllm/${MODEL_NAME}"
  MODEL_FLAGS=(-M emulate_tools=true)
else
  AGENT_MODEL="tau2-agent-vllm/vllm/${MODEL_NAME}"
  MODEL_FLAGS=()
fi

if ! curl -s -m 3 "${VLLM_BASE_URL}/models" >/dev/null 2>&1; then
  echo "ERROR: no vLLM server reachable at ${VLLM_BASE_URL}." >&2
  echo "Start the deployment backend and set VLLM_BASE_URL if it is not local." >&2
  exit 1
fi

if [[ ! -f "${TAU2_SOURCE_DIR}/pyproject.toml" ]]; then
  echo "ERROR: tau2-bench source is not provisioned at ${TAU2_SOURCE_DIR}." >&2
  echo "Run ${PROJECT_DIR}/scripts/setup_tau2_bench.sh first." >&2
  exit 1
fi

if [[ -z "$TAU2_DATA_DIR" || ! -d "${TAU2_DATA_DIR}/tau2/domains" ]]; then
  echo "ERROR: TAU2_DATA_DIR must point to a separately provisioned tau2 data directory." >&2
  echo "Expected to find <TAU2_DATA_DIR>/tau2/domains." >&2
  exit 1
fi

OUTPUT_DIR="${REPO_ROOT}/runs/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR"

LIMIT_FLAGS=()
if [[ -n "$NUM_TASKS" ]]; then
  LIMIT_FLAGS+=(--limit "$NUM_TASKS")
fi

TASK_FLAGS=(-T "domain=${TAU2_DOMAIN}" -T "task_split=${TAU2_TASK_SPLIT}")
if [[ -n "$TAU2_TASK_SET" ]]; then
  TASK_FLAGS+=(-T "task_set=${TAU2_TASK_SET}")
fi

cd "$TASK_DIR"
export PATH="$HOME/.local/bin:$PATH"
TAU2_DATA_DIR="$TAU2_DATA_DIR" \
TAU2_USER_MODEL="openai/${USER_MODEL_NAME}" \
TAU2_USER_API_BASE="$VLLM_BASE_URL" \
TAU2_USER_API_KEY="$VLLM_API_KEY" \
TAU2_USER_LLM_ARGS="$TAU2_USER_LLM_ARGS" \
TAU2_LLM_NL_ASSERTIONS="openai/${JUDGE_MODEL_NAME}" \
TAU2_LLM_NL_ASSERTIONS_ARGS="$TAU2_JUDGE_LLM_ARGS" \
LITELLM_LOCAL_MODEL_COST_MAP=True \
VLLM_BASE_URL="$VLLM_BASE_URL" VLLM_API_KEY="$VLLM_API_KEY" \
INSPECT_TRACE_DIR="${OUTPUT_DIR}/.inspect_trace" \
INSPECT_TRACE_VLLM_METRICS_URL="$INSPECT_TRACE_VLLM_METRICS_URL" \
uv run --project "$PROJECT_DIR" inspect eval "task.py@tau2" \
  --model "$AGENT_MODEL" \
  "${TASK_FLAGS[@]}" \
  "${MODEL_FLAGS[@]}" \
  "${LIMIT_FLAGS[@]}" \
  --max-connections 1 \
  --max-samples 1 \
  --log-dir "${OUTPUT_DIR}/logs"

echo
echo "Done. .eval logs + inspect_trace JSONL: ${OUTPUT_DIR}/"
echo "Pull results to local analysis machine (if run remotely) with ../../scripts/pull_runs.sh"
