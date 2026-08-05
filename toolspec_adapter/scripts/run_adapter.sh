#!/usr/bin/env bash
# Runs the same API-Bank questions through our inspect_ai adapter (toolspec_adapter/src/
# toolspec_adapter/), via the custom `toolspec-hf` ModelAPI provider that calls ToolSpec's own
# baseline_forward()/toolspec_forward() in-process (see ../src/toolspec_adapter/provider.py).
# Unlike tau2-bench's adapter, no async/sync bridging trick is needed here -- this provider IS
# what inspect_ai calls to generate, so inspect_trace's Hooks fire without any special handling.
#
# Run ./run_native_repro.sh first -- this script compares against that run's baseline JSONL
# (TOOLSPEC_REFERENCE_JSONL) to check the adapter reproduces the same per-question output.
#
# Usage:
#   ./run_adapter.sh baseline
#   ./run_adapter.sh toolspec
#   NUM_QUESTIONS=10 ./run_adapter.sh toolspec   # quick smoke test

set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "baseline" && "$MODE" != "toolspec" ]]; then
  echo "Usage: $0 baseline|toolspec" >&2
  exit 1
fi

ADAPTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../src/toolspec_adapter" && pwd)"
REPO_ROOT="$(cd "$ADAPTER_DIR/../../.." && pwd)"

: "${TOOLSPEC_REPO_DIR:=/home/liuyingen/code/ToolSpec}"
: "${MODEL_NAME:=Qwen2.5-3B-Instruct}"
: "${MODEL_PATH:=Qwen/Qwen2.5-3B-Instruct}"
: "${NUM_QUESTIONS:=100}"
: "${RUN_NAME:=toolspec_adapter_${MODE}}"
: "${TOOLSPEC_REFERENCE_JSONL:=${TOOLSPEC_REPO_DIR}/output/APIBank/${MODEL_NAME}/${MODEL_NAME}-vanilla-float16-temp-0.0.jsonl}"

if [[ ! -f "$TOOLSPEC_REFERENCE_JSONL" ]]; then
  echo "ERROR: reference baseline JSONL not found at $TOOLSPEC_REFERENCE_JSONL" >&2
  echo "  Run ./run_native_repro.sh first (it produces this file)." >&2
  exit 1
fi

OUTPUT_DIR="${REPO_ROOT}/runs/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}/.inspect_trace"

cd "$ADAPTER_DIR"
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
TOOLSPEC_REPO_DIR="$TOOLSPEC_REPO_DIR" \
TOOLSPEC_REFERENCE_JSONL="$TOOLSPEC_REFERENCE_JSONL" \
INSPECT_TRACE_DIR="${OUTPUT_DIR}/.inspect_trace" \
uv run --project "$ADAPTER_DIR" inspect eval "task.py" \
  -T "limit=${NUM_QUESTIONS}" \
  --model "toolspec-hf/${MODEL_PATH}" \
  -M "mode=${MODE}" \
  --max-connections 1 \
  --log-dir "${OUTPUT_DIR}/logs"

echo
echo "Done. .eval logs + inspect_trace JSONL: ${OUTPUT_DIR}/"
echo "Pull results to local analysis machine (if run remotely) with ../../scripts/pull_runs.sh"
