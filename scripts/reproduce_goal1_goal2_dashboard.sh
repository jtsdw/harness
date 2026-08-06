#!/usr/bin/env bash
# Reproduces docs/goal1_r3_r4_dashboard.html (the goal1 execution-topology/action-parsing +
# goal2 three-layer-profiling dashboard) from scratch: runs the three real BFCL runs
# inspect_trace/scripts/build_r3_r4_dashboard.py needs, then builds the dashboard. See that
# script's own header comment for exactly which fields come from which run.
#
# This is real benchmark data, not a quick smoke test -- expect roughly an hour total on a local
# GPU, dominated by the 200-sample multi_turn_base run. Each of the three runs is skipped if its
# OUTPUT_DIR already has a .eval log (idempotent -- re-running this after a change to the
# dashboard script itself doesn't need to redo real experiments, just rebuild from what's already
# there). Set FORCE_RERUN=true to redo everything regardless.
#
# Usage:
#   ./scripts/reproduce_goal1_goal2_dashboard.sh
#   MODEL_NAME="Qwen/Qwen2.5-32B-Instruct" ./scripts/reproduce_goal1_goal2_dashboard.sh
#   FORCE_RERUN=true ./scripts/reproduce_goal1_goal2_dashboard.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_MODEL_SERVER_DIR="${REPO_ROOT}/local-model-server"
INSPECT_TRACE_DIR="${REPO_ROOT}/inspect_trace"
INSPECT_TRACE_SCRIPTS_DIR="${INSPECT_TRACE_DIR}/scripts"

: "${MODEL_NAME:=Qwen/Qwen2.5-3B-Instruct}"
: "${FORCE_RERUN:=}"

run_step() {
  local label="$1" categories="$2" limit="$3" output_dir="$4"
  if [[ -z "$FORCE_RERUN" ]] && compgen -G "${output_dir}/logs/*.eval" >/dev/null 2>&1; then
    echo "=== [$label] already has a .eval log under ${output_dir}/logs/ -- skipping (set FORCE_RERUN=true to redo) ==="
    return
  fi
  echo "=== [$label] running BFCL (categories=$categories, limit=$limit) ==="
  ( cd "$INSPECT_TRACE_SCRIPTS_DIR" && \
    MODEL_NAME="$MODEL_NAME" CATEGORIES="$categories" LIMIT="$limit" OUTPUT_DIR="$output_dir" \
    ./run_bfcl_local_vllm.sh )
}

NEED_SERVER=false
for output_dir in \
  "${REPO_ROOT}/runs/goal1_bfcl_multi_turn_base_full" \
  "${REPO_ROOT}/runs/goal1_bfcl_live_parallel_full" \
  "${REPO_ROOT}/runs/goal2_vllm_metrics_validation"
do
  if [[ -n "$FORCE_RERUN" ]] || ! compgen -G "${output_dir}/logs/*.eval" >/dev/null 2>&1; then
    NEED_SERVER=true
  fi
done

if [[ "$NEED_SERVER" == true ]]; then
  echo "=== starting vLLM ==="
  ( cd "$LOCAL_MODEL_SERVER_DIR" && MODEL="$MODEL_NAME" ./scripts/serve_baseline.sh )
fi

run_step "1/3 multi_turn_base full (200 samples, ~1hr, this is the slow one)" \
  "multi_turn_base" 200 "${REPO_ROOT}/runs/goal1_bfcl_multi_turn_base_full"

run_step "2/3 live_parallel full (15 samples)" \
  "live_parallel" 15 "${REPO_ROOT}/runs/goal1_bfcl_live_parallel_full"

run_step "3/3 goal2 vllm_metrics validation (8 samples)" \
  "multi_turn_base" 8 "${REPO_ROOT}/runs/goal2_vllm_metrics_validation"

if [[ "$NEED_SERVER" == true ]]; then
  echo "=== stopping vLLM ==="
  ( cd "$LOCAL_MODEL_SERVER_DIR" && ./scripts/stop.sh )
fi

echo "=== building dashboard ==="
( cd "$INSPECT_TRACE_DIR" && uv run python scripts/build_r3_r4_dashboard.py )

echo
echo "Done. Open docs/goal1_r3_r4_dashboard.html"
