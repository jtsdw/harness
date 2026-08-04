#!/usr/bin/env bash
# Same pattern as run_bfcl_benchmark.sh but for inspect_evals/gsm8k -- a single-turn, no-tool-call
# benchmark used as a sanity-check contrast against BFCL's dense multi-turn tool-calling trajectories
# (see /home/liuyingen/code/efficient-harness/docs/datasets.md).
#
# Usage (hosted):
#   OPENAI_BASE_URL="https://api.deepseek.com/" OPENAI_API_KEY="sk-..." ./run_gsm8k_benchmark.sh
# Usage (local vLLM):
#   MODEL="openai-api/vllm/Qwen/Qwen2.5-3B-Instruct" MODEL_ARGS="emulate_tools=true" \
#   VLLM_BASE_URL="http://localhost:8000/v1" VLLM_API_KEY="not-needed" ./run_gsm8k_benchmark.sh
#
# Optional overrides: MODEL, MODEL_ARGS, LIMIT (default 5), OUTPUT_DIR.

set -euo pipefail

# See run_bfcl_benchmark.sh for why this is split into two roots.
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$PKG_ROOT/.." && pwd)"

: "${MODEL:=openai/deepseek-chat}"
: "${MODEL_ARGS:=}"
: "${LIMIT:=5}"
: "${OUTPUT_DIR:=${REPO_ROOT}/runs/goal1_gsm8k}"

if [[ "$MODEL" == openai/* && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set." >&2
  exit 1
fi

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found. Install it first: python3 -m pip install --user uv" >&2
  exit 1
fi

MODEL_ARGS_FLAGS=()
for kv in $MODEL_ARGS; do
  MODEL_ARGS_FLAGS+=(-M "$kv")
done

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

echo "Model:      $MODEL"
echo "Model args: ${MODEL_ARGS:-<none>}"
echo "Limit:      $LIMIT"
echo "Output dir: $OUTPUT_DIR"
echo

INSPECT_TRACE_DIR="./.inspect_trace" uv run --project "$PKG_ROOT" inspect eval inspect_evals/gsm8k \
  --model "$MODEL" \
  "${MODEL_ARGS_FLAGS[@]}" \
  --limit "$LIMIT" \
  --log-dir ./logs

echo
echo "Done. Raw .eval log(s): ${OUTPUT_DIR}/logs/"
echo "      inspect_trace derived JSONL: ${OUTPUT_DIR}/.inspect_trace/"
