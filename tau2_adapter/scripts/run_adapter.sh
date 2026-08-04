#!/usr/bin/env bash
# Runs the tau2-bench mock domain through our inspect_ai adapter (tau2_adapter/src/tau2_adapter/),
# in either of the two variants documented in docs/tau2_bench_integration_findings.md:
#   - emulate: agent side uses inspect_ai's emulate_tools=true (client-side text parsing) --
#     the original workaround for Bug 3 (the "strict" tool field vLLM rejects).
#   - native: agent side uses the custom tau2-agent-vllm provider (tau2_adapter/_registry.py),
#     which strips the "strict" field so real native tool-calling works -- the proper fix.
# Both need a local vLLM server already running with native tool-calling enabled (see
# local-model-server/scripts/serve.sh's NATIVE_TOOL_CALLING option) -- the user simulator always
# uses native tool-calling regardless of which agent variant you pick.
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

ADAPTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../src/tau2_adapter" && pwd)"
REPO_ROOT="$(cd "$ADAPTER_DIR/../../.." && pwd)"

: "${TAU2_BENCH_DIR:=/home/liuyingen/code/tau2-bench}"
: "${VLLM_BASE_URL:=http://localhost:8000/v1}"
: "${VLLM_API_KEY:=not-needed}"
: "${MODEL_NAME:=Qwen/Qwen2.5-3B-Instruct}"
: "${NUM_TASKS:=}"  # empty = all tasks in the domain
: "${RUN_NAME:=tau2_adapter_${MODE}}"

if [[ "$MODE" == "emulate" ]]; then
  AGENT_MODEL="openai-api/vllm/${MODEL_NAME}"
  MODEL_FLAGS=(-M emulate_tools=true)
else
  AGENT_MODEL="tau2-agent-vllm/vllm/${MODEL_NAME}"
  MODEL_FLAGS=()
fi

if ! curl -s -m 3 "${VLLM_BASE_URL}/models" >/dev/null 2>&1; then
  echo "ERROR: no vLLM server reachable at ${VLLM_BASE_URL} -- start it first:" >&2
  echo "  cd ../local-model-server && NATIVE_TOOL_CALLING=true ./scripts/serve.sh" >&2
  exit 1
fi

OUTPUT_DIR="${REPO_ROOT}/runs/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR"

LIMIT_FLAGS=()
if [[ -n "$NUM_TASKS" ]]; then
  LIMIT_FLAGS+=(--limit "$NUM_TASKS")
fi

cd "$ADAPTER_DIR"
export PATH="$HOME/.local/bin:$PATH"
TAU2_DATA_DIR="${TAU2_BENCH_DIR}/data" \
TAU2_USER_MODEL="openai/${MODEL_NAME}" \
TAU2_USER_API_BASE="$VLLM_BASE_URL" \
TAU2_USER_API_KEY="$VLLM_API_KEY" \
VLLM_BASE_URL="$VLLM_BASE_URL" VLLM_API_KEY="$VLLM_API_KEY" \
INSPECT_TRACE_DIR="${OUTPUT_DIR}/.inspect_trace" \
uv run --project "$ADAPTER_DIR" inspect eval "task.py" \
  --model "$AGENT_MODEL" \
  "${MODEL_FLAGS[@]}" \
  "${LIMIT_FLAGS[@]}" \
  --max-connections 1 \
  --log-dir "${OUTPUT_DIR}/logs"

echo
echo "Done. .eval logs + inspect_trace JSONL: ${OUTPUT_DIR}/"
echo "Pull results to local analysis machine (if run remotely) with ../../scripts/pull_runs.sh"
