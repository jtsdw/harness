#!/usr/bin/env bash
# Reproduces the goal-1 real-benchmark validation run documented in
# /home/liuyingen/code/efficient-harness/docs/goal1_real_benchmark_findings.md:
# inspect_evals/bfcl (multi_turn_base) driven against a real, OpenAI-compatible model,
# with inspect_trace's Hooks enabled so it captures prefill_diff / segment_tokens / attempt_group
# derived facts alongside the normal .eval log.
#
# Usage (hosted, e.g. DeepSeek):
#   OPENAI_BASE_URL="https://api.deepseek.com/" OPENAI_API_KEY="sk-..." \
#     ./run_bfcl_benchmark.sh
#
# Usage (local vLLM server -- see /home/liuyingen/code/efficient-harness/local-model-server/, no API key needed):
#   MODEL="openai-api/vllm/Qwen/Qwen2.5-3B-Instruct" MODEL_ARGS="emulate_tools=true" \
#   VLLM_BASE_URL="http://localhost:8000/v1" VLLM_API_KEY="not-needed" \
#     ./run_bfcl_benchmark.sh
#
# Optional overrides (all have defaults matching the original DeepSeek run):
#   MODEL         (default: openai/deepseek-chat)
#   MODEL_ARGS    (default: empty)               -- space-separated key=value pairs, each becomes
#                                                    a separate `-M key=value` flag (e.g. needed for
#                                                    emulate_tools=true against a vLLM server too old
#                                                    to natively parse tool calls)
#   CATEGORIES    (default: multi_turn_base)     -- any inspect_evals/bfcl category, comma separated
#   LIMIT         (default: 2)                   -- number of samples
#   MAX_CONNECTIONS (default: unset -> inspect_ai's own default) -- pass a small number (e.g. 1) when
#                                                    targeting a fragile single-GPU local vLLM server:
#                                                    a real run against vllm==0.6.3.post1 crashed its
#                                                    MQLLMEngine when two /v1/chat/completions requests
#                                                    landed concurrently (see goal1_r3_r4_real_benchmark_findings.md)
#   OUTPUT_DIR    (default: runs/goal1_bfcl_<categories-with-underscores>, under the repo root)
#
# Does NOT hardcode any API key -- for a hosted model you must provide OPENAI_API_KEY (and
# OPENAI_BASE_URL if not plain OpenAI) yourself, e.g. by sourcing a private env file first:
#   source /path/to/your-secrets.env && ./run_bfcl_benchmark.sh
# For a local vLLM server (model string starting with "openai-api/"), no key check is enforced --
# set whatever <SERVICE>_BASE_URL/<SERVICE>_API_KEY pair matches the service prefix in MODEL.

set -euo pipefail

# PKG_ROOT is the inspect_trace/ project itself (uv run --project needs this to resolve
# its own pyproject.toml/venv, which declares inspect_ai/inspect_evals as real dependencies).
# REPO_ROOT is the efficient-harness/ repo root, one level up -- runs/ lives there, not
# inside inspect_trace/, since it's shared experiment data, not package-internal.
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$PKG_ROOT/.." && pwd)"

export XDG_CACHE_HOME=/home/users/ntu/n2505716/scratch/xdg_cache



: "${MODEL:="openai-api/vllm/Qwen/Qwen2.5-32B-Instruct"}"
: "${MODEL_ARGS:="emulate_tools=true"}"
: "${CATEGORIES:=multi_turn_base}"
: "${VLLM_BASE_URL="http://localhost:8000/v1"}"
: "${VLLM_API_KEY="not-needed"}"
export VLLM_BASE_URL VLLM_API_KEY
: "${LIMIT:=200}"
: "${MAX_CONNECTIONS:=1}"
: "${OUTPUT_DIR:=${REPO_ROOT}/runs/goal1_bfcl_${CATEGORIES//,/_}}"

if [[ "$MODEL" == openai/* && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set. Export it (and OPENAI_BASE_URL, if using a" >&2
  echo "non-OpenAI OpenAI-compatible endpoint) before running this script." >&2
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

MAX_CONNECTIONS_FLAGS=()
if [[ -n "$MAX_CONNECTIONS" ]]; then
  MAX_CONNECTIONS_FLAGS+=(--max-connections "$MAX_CONNECTIONS")
fi

# Build the -T categories="['a','b']" argument from a comma-separated CATEGORIES value.
IFS=',' read -ra CATEGORY_ARR <<< "$CATEGORIES"
CATEGORIES_PY="["
for c in "${CATEGORY_ARR[@]}"; do
  CATEGORIES_PY+="'${c}',"
done
CATEGORIES_PY+="]"

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

echo "Model:      $MODEL"
echo "Model args: ${MODEL_ARGS:-<none>}"
echo "Categories: $CATEGORIES_PY"
echo "Limit:      $LIMIT"
echo "Output dir: $OUTPUT_DIR"
echo

INSPECT_TRACE_DIR="./.inspect_trace" uv run --project "$PKG_ROOT" inspect eval inspect_evals/bfcl \
  -T "categories=${CATEGORIES_PY}" \
  --model "$MODEL" \
  "${MODEL_ARGS_FLAGS[@]}" \
  --limit "$LIMIT" \
  "${MAX_CONNECTIONS_FLAGS[@]}" \
  --log-dir ./logs

echo
echo "Done. Raw .eval log(s): ${OUTPUT_DIR}/logs/"
echo "      inspect_trace derived JSONL: ${OUTPUT_DIR}/.inspect_trace/"
echo "      View: uv run --project '$PKG_ROOT' inspect view --log-dir '${OUTPUT_DIR}/logs'"
