#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ADAPTER_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
APPWORLD_ROOT="${APPWORLD_ROOT:-$(realpath "$ADAPTER_ROOT/../.deps/appworld")}"
RUNS_ROOT="${RUNS_ROOT:-$ADAPTER_ROOT/../runs}"
MODEL="${MODEL:-tau2-agent-vllm/vllm/qwen3.6-27b-autoround}"
SPLIT="${SPLIT:-dev}"
unique_id="$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM"
RUN_NAME="${RUN_NAME:-appworld-$SPLIT-$unique_id}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-inspect-appworld-$SPLIT-$unique_id}"
RUN_DIR="$RUNS_ROOT/$RUN_NAME"
EXPERIMENT_DIR="$APPWORLD_ROOT/experiments/outputs/$EXPERIMENT_NAME"
INSPECT_BIN="${INSPECT_BIN:-$ADAPTER_ROOT/.venv/bin/inspect}"

if [[ -n "${LIMIT:-}" && ! "$LIMIT" =~ ^[1-9][0-9]*$ ]]; then
  printf 'LIMIT must be a positive integer.\n' >&2
  exit 2
fi

if [[ -e "$RUN_DIR" ]]; then
  printf 'Refusing to reuse harness run directory: %s\n' "$RUN_DIR" >&2
  exit 2
fi
if [[ -e "$EXPERIMENT_DIR" ]]; then
  printf 'Refusing to reuse AppWorld experiment directory: %s\n' "$EXPERIMENT_DIR" >&2
  exit 2
fi

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/trace"
cat >"$RUN_DIR/manifest.txt" <<EOF
run_directory=$RUN_DIR
experiment_name=$EXPERIMENT_NAME
split=$SPLIT
model=$MODEL
start_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

on_exit() {
  status=$?
  if (( status != 0 )); then
    printf '\nRun stopped with artifacts retained in %s.\n' "$RUN_DIR" >&2
    printf 'Where Inspect supports it, retry from logs in %s using a fresh output location; otherwise start this script again with fresh unique names. Never rerun into these directories.\n' "$RUN_DIR/logs" >&2
  fi
}
trap on_exit EXIT

args=(
  eval "$ADAPTER_ROOT/src/appworld_adapter/task.py@appworld"
  -T "split=$SPLIT" -T "experiment_name=$EXPERIMENT_NAME"
  --max-samples 1 --max-connections 1
  --display plain
  --model "$MODEL"
  --model-role "predictor=$MODEL"
  --model-role "main=$MODEL"
  --log-dir "$RUN_DIR/logs"
)
if [[ -n "${LIMIT:-}" ]]; then
  args+=(--limit "$LIMIT")
fi

export APPWORLD_ROOT INSPECT_TRACE_DIR="$RUN_DIR/trace"
"$INSPECT_BIN" "${args[@]}"
