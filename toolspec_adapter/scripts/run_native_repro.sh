#!/usr/bin/env bash
# Runs ToolSpec's own native-repo baseline + toolspec (+ optionally pld/recycling/samd) eval
# commands on API-Bank, then prints the speedup table via ToolSpec's own evaluation/speed.py.
# This reproduces docs/toolspec_integration_findings.md's phase-1 numbers -- run this BEFORE
# run_adapter.sh so TOOLSPEC_REFERENCE_JSONL has something to compare the adapter against.
#
# Optional overrides:
#   TOOLSPEC_REPO_DIR   (default: /home/liuyingen/code/ToolSpec)
#   MODEL_PATH           (default: Qwen/Qwen2.5-3B-Instruct -- ungated, already cached locally)
#   MODEL_NAME            (default: Qwen2.5-3B-Instruct)
#   NUM_QUESTIONS          (default: 100 -- API-Bank has 399 total across its 3 levels)
#   METHODS                 (default: "baseline toolspec" -- space-separated subset of
#                             "baseline pld recycling samd toolspec")
#
# Usage:
#   ./run_native_repro.sh
#   NUM_QUESTIONS=50 METHODS="baseline toolspec pld" ./run_native_repro.sh

set -euo pipefail

: "${TOOLSPEC_REPO_DIR:=/home/liuyingen/code/ToolSpec}"
: "${MODEL_PATH:=Qwen/Qwen2.5-3B-Instruct}"
: "${MODEL_NAME:=Qwen2.5-3B-Instruct}"
: "${NUM_QUESTIONS:=100}"
: "${METHODS:=baseline toolspec}"

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

if [[ ! -x "$TOOLSPEC_REPO_DIR/.venv/bin/python" ]]; then
  echo "ERROR: ToolSpec venv not found at $TOOLSPEC_REPO_DIR/.venv -- run ./setup_toolspec.sh first." >&2
  exit 1
fi

cd "$TOOLSPEC_REPO_DIR"
PY=".venv/bin/python"

run_method() {
  local method="$1"
  case "$method" in
    baseline)
      $PY -m evaluation.inference_baseline \
        --model-path "$MODEL_PATH" --model-name "$MODEL_NAME" \
        --model-id "${MODEL_NAME}-vanilla-float16-temp-0.0" \
        --temperature 0.0 --dtype float16 --question-num "$NUM_QUESTIONS" --benchmark APIBank
      ;;
    toolspec)
      $PY -m evaluation.inference_toolspec \
        --model-path "$MODEL_PATH" --model-name "$MODEL_NAME" \
        --model-id "${MODEL_NAME}-toolspec-float16" \
        --dtype float16 --question-num "$NUM_QUESTIONS" --benchmark APIBank
      ;;
    pld)
      $PY -m evaluation.inference_pld \
        --model-path "$MODEL_PATH" --model-name "$MODEL_NAME" \
        --model-id "${MODEL_NAME}-pld-float16" \
        --dtype float16 --question-num "$NUM_QUESTIONS" --benchmark APIBank
      ;;
    recycling)
      $PY -m evaluation.inference_recycling \
        --model-path "$MODEL_PATH" --model-name "$MODEL_NAME" \
        --model-id "${MODEL_NAME}-recycling-float16-temp-0.0" \
        --dtype float16 --question-num "$NUM_QUESTIONS" --benchmark APIBank
      ;;
    samd)
      $PY -m evaluation.inference_samd \
        --model-path "$MODEL_PATH" --model-name "$MODEL_NAME" --model-id "${MODEL_NAME}-samd" \
        --benchmark APIBank --temperature 0.0 --dtype float16 \
        --samd_n_predicts 40 --samd_len_threshold 5 --samd_len_bias 5 \
        --tree_method token_recycle --attn_implementation sdpa --question-num "$NUM_QUESTIONS"
      ;;
    *)
      echo "Unknown method: $method (expected one of baseline pld recycling samd toolspec)" >&2
      exit 1
      ;;
  esac
}

for method in $METHODS; do
  echo "=== Running $method ==="
  run_method "$method"
done

echo
echo "=== Speedup vs baseline ==="
base_file="output/APIBank/${MODEL_NAME}/${MODEL_NAME}-vanilla-float16-temp-0.0.jsonl"
declare -A method_files=(
  [pld]="${MODEL_NAME}-pld-float16"
  [recycling]="${MODEL_NAME}-recycling-float16-temp-0.0"
  [samd]="${MODEL_NAME}-samd"
  [toolspec]="${MODEL_NAME}-toolspec-float16"
)
for method in $METHODS; do
  [[ "$method" == "baseline" ]] && continue
  f="output/APIBank/${MODEL_NAME}/${method_files[$method]}.jsonl"
  echo "--- $method ---"
  $PY -m evaluation.speed --file-path "$f" --base-path "$base_file" --tokenizer-path "$MODEL_PATH"
done

echo
echo "Reference baseline JSONL for run_adapter.sh's TOOLSPEC_REFERENCE_JSONL:"
echo "  $TOOLSPEC_REPO_DIR/$base_file"
