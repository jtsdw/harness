#!/usr/bin/env bash
# 需求 B5: minimal robustness matrix -- concurrency x repetition, closed-loop only, against
# whatever OpenAI-compatible vLLM endpoint VLLM_BASE_URL points at (written for the NSCC
# deployment in ../../nscc_model_server/, not this dev machine's old vLLM pin -- see
# docs/next_phase_requirements.md's B5 and docs/serving_observability_b_howto.md).
#
# Generalizes ./run_concurrency_validation.sh (which stays as-is -- it's real historical evidence
# that concurrent requests crashed the old local vLLM outright, "MQLLMEngine already dead") into
# an actual grid: concurrency in {1, 4, 8}, extending to 16 only if every rep at 8 completed
# without error (per B5's "只有 8 未产生资源竞争时才增加 16"). Each cell also runs
# service_metrics_sampler.py alongside it (B4 服务层 time series) and is independently
# pass/fail-recorded in manifest.jsonl -- a crashed cell does NOT get retried and does NOT abort
# the rest of the matrix, because "did concurrency=8 crash the server" is itself the real answer
# B5 is trying to get, not a nuisance to route around.
#
# Sets INSPECT_EVAL_LOG_MODEL_API=true so inspect_ai retains ModelEvent.call.response (the raw
# per-request data vllm_per_request_metrics.py reads) for every call, not just a model's first 5
# -- see that module's docstring for why this default would otherwise silently starve B1/B2 of
# data past the first few calls per cell.
#
# Usage:
#   cd efficient-harness/inspect_trace
#   VLLM_BASE_URL="http://<nscc-node>:8000/v1" ./scripts/run_b5_matrix.sh
#
# Optional overrides:
#   MODEL            (default: openai-api/vllm/Qwen/Qwen3-32B -- match whatever's actually served)
#   MODEL_ARGS        (default: empty)
#   CATEGORIES        (default: multi_turn_base -- BFCL category, a real multi-turn agent trace)
#   LIMIT             (default: 20 samples per cell)
#   REPETITIONS       (default: 3 -- B5's "每个配置至少 3 次")
#   CONCURRENCIES     (default: "1 4 8", space separated)
#   ENABLE_16_IF_8_CLEAN (default: 1 -- set 0 to never try concurrency=16)
#   OUTPUT_ROOT       (default: runs/b5_matrix, under the repo root)
#   VLLM_BASE_URL, VLLM_API_KEY  (default: http://localhost:8000/v1, not-needed)

set -euo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$PKG_ROOT/.." && pwd)"

: "${MODEL:=openai-api/vllm/Qwen/Qwen3-32B}"
: "${MODEL_ARGS:=}"
: "${CATEGORIES:=multi_turn_base}"
: "${LIMIT:=20}"
: "${REPETITIONS:=3}"
: "${CONCURRENCIES:=1 4 8}"
: "${ENABLE_16_IF_8_CLEAN:=1}"
: "${VLLM_BASE_URL:=http://localhost:8000/v1}"
: "${VLLM_API_KEY:=not-needed}"
: "${OUTPUT_ROOT:=${REPO_ROOT}/runs/b5_matrix}"
export VLLM_BASE_URL VLLM_API_KEY

if ! curl -s -m 3 "${VLLM_BASE_URL}/models" >/dev/null 2>&1; then
  echo "ERROR: no vLLM server reachable at ${VLLM_BASE_URL}." >&2
  exit 1
fi

export INSPECT_EVAL_LOG_MODEL_API=true

mkdir -p "$OUTPUT_ROOT"
MANIFEST="${OUTPUT_ROOT}/manifest.jsonl"
: > "$MANIFEST"

run_cell() {
  local concurrency="$1" rep="$2"
  local cell_dir="${OUTPUT_ROOT}/concurrency_${concurrency}/rep_${rep}"
  local sampler_log="${cell_dir}/service_metrics.jsonl"
  mkdir -p "$cell_dir"

  uv run --project "$PKG_ROOT" python3 "${PKG_ROOT}/scripts/service_metrics_sampler.py" \
    --output "$sampler_log" --interval-seconds 1 &
  local sampler_pid=$!

  local started_at exit_code=0
  started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  set +e
  MODEL="$MODEL" MODEL_ARGS="$MODEL_ARGS" VLLM_BASE_URL="$VLLM_BASE_URL" VLLM_API_KEY="$VLLM_API_KEY" \
    CATEGORIES="$CATEGORIES" LIMIT="$LIMIT" MAX_CONNECTIONS="$concurrency" \
    OUTPUT_DIR="$cell_dir" "${PKG_ROOT}/scripts/run_bfcl_benchmark.sh" \
    > "${cell_dir}/stdout.log" 2> "${cell_dir}/stderr.log"
  exit_code=$?
  set -e
  local ended_at
  ended_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true

  local outcome="success"
  if [[ "$exit_code" -ne 0 ]]; then
    outcome="failed"
    echo "  concurrency=${concurrency} rep=${rep}: FAILED (exit ${exit_code}) -- see ${cell_dir}/stderr.log"
  else
    echo "  concurrency=${concurrency} rep=${rep}: ok"
  fi

  python3 -c "
import json, sys
concurrency, rep, outcome, exit_code, output_dir, started_at, ended_at = sys.argv[1:]
print(json.dumps({
    'concurrency': int(concurrency), 'repetition': int(rep), 'outcome': outcome,
    'exit_code': int(exit_code), 'output_dir': output_dir,
    'started_at': started_at, 'ended_at': ended_at,
}))
" "$concurrency" "$rep" "$outcome" "$exit_code" "$cell_dir" "$started_at" "$ended_at" >> "$MANIFEST"

  [[ "$outcome" == "success" ]]
}

# A `for x in $CONCURRENCIES` loop word-splits once at loop start, so appending "16" to that
# string mid-loop would never actually be visited -- use an index-walked array instead, which we
# can append to (queue[${#queue[@]}]=16) while still iterating over earlier elements.
read -ra concurrency_queue <<< "$CONCURRENCIES"
i=0
while [[ "$i" -lt "${#concurrency_queue[@]}" ]]; do
  concurrency="${concurrency_queue[$i]}"
  echo "== concurrency=${concurrency}, ${REPETITIONS} repetitions =="
  concurrency_clean=1
  for rep in $(seq 1 "$REPETITIONS"); do
    if ! run_cell "$concurrency" "$rep"; then
      concurrency_clean=0
    fi
  done
  if [[ "$concurrency" == "8" && "$ENABLE_16_IF_8_CLEAN" == "1" ]]; then
    if [[ "$concurrency_clean" == "1" ]]; then
      echo "== concurrency=8 was clean across all ${REPETITIONS} reps -- adding concurrency=16 =="
      concurrency_queue[${#concurrency_queue[@]}]=16
    else
      echo "== concurrency=8 had at least one failure -- NOT adding concurrency=16 (per B5) =="
    fi
  fi

  # B4/B5 both ask for "样本数和离散程度", not just a single number per cell -- pool every episode
  # from every successful rep at this concurrency and report n/mean/stddev, not just 3 separate
  # per-rep means (which would still hide episode-level spread within a rep).
  uv run --project "$PKG_ROOT" python3 - "$concurrency" "$OUTPUT_ROOT" "$REPETITIONS" <<'PYEOF'
import json
import statistics
import sys
from pathlib import Path

from inspect_trace.analysis.episode_layer import summarize_run

concurrency, output_root, repetitions = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
latencies: list[float] = []
for rep in range(1, repetitions + 1):
    trace_dir = output_root / f"concurrency_{concurrency}" / f"rep_{rep}" / ".inspect_trace"
    if not trace_dir.exists():
        continue  # failed cell, no trace produced
    run_summary = summarize_run(trace_dir)
    latencies.extend(
        e.end_to_end_latency_seconds
        for e in run_summary.per_episode
        if e.end_to_end_latency_seconds is not None
    )

summary = {
    "concurrency": concurrency,
    "n_episodes_pooled": len(latencies),
    "mean_end_to_end_latency_seconds": statistics.fmean(latencies) if latencies else None,
    "stddev_end_to_end_latency_seconds": (
        statistics.stdev(latencies) if len(latencies) >= 2 else None
    ),
}
out_path = output_root / f"concurrency_{concurrency}" / "dispersion_summary.json"
out_path.write_text(json.dumps(summary, indent=2))
print(f"  dispersion summary ({out_path}): {summary}")
PYEOF

  i=$((i + 1))
done

echo
echo "Done. Manifest: ${MANIFEST}"
echo "Per-cell outputs: ${OUTPUT_ROOT}/concurrency_<N>/rep_<R>/"
