#!/usr/bin/env bash
# NOTE: for 需求 B5's actual minimal robustness matrix (concurrency x repetition grid, targeting
# the NSCC deployment), see ./run_b5_matrix.sh instead -- this script is kept as-is because the
# crash finding below is real historical evidence from this dev machine's old vLLM, not because
# it's still the current way to run a B5-style check.
#
# Deliberately triggers real queueing/concurrency against the local vLLM server, to validate that
# vllm_metrics.py's queue_depth_running_at_start / queue_depth_waiting_at_start / preemptions_delta
# fields actually produce non-zero values under load.
#
# Why this script exists: every real run so far has used MAX_CONNECTIONS=1 (the crash-avoidance
# convention from goal1_r3_r4_real_benchmark_findings.md), so these three fields have been pinned
# at 0 across every single recorded call -- not because the fields are broken, but because nothing
# has ever actually made the server queue a request. That only proves "displays 0 correctly when
# there's no queueing", not "computes correctly when there is". See deployment_migration_guide.md's
# "顺便验证目标二的排队/并发字段" section for the full context.
#
# What it does: runs the same small BFCL slice twice against the currently-running local vLLM
# server -- once serialized (MAX_CONNECTIONS=1, the safe baseline) and once with real concurrency
# (MAX_CONNECTIONS=$CONCURRENT) -- then prints a before/after comparison of the vllm_metrics fields
# pulled straight from the recorded JSONL.
#
# Requires: local-model-server already running (cd local-model-server && ./scripts/serve.sh).
#
# WARNING: on the original hardware (RTX 2000 Ada, vllm==0.6.3.post1) real concurrent requests
# crashed the server outright ("MQLLMEngine already dead"). If the concurrent run below fails the
# same way, that IS the answer to "is MAX_CONNECTIONS>1 safe on this hardware yet" -- see
# deployment_migration_guide.md's 待现场验证清单.
#
# Usage:
#   cd efficient-harness/inspect_trace
#   ./scripts/run_concurrency_validation.sh                  # LIMIT=8, CONCURRENT=4 by default
#   LIMIT=20 CONCURRENT=8 ./scripts/run_concurrency_validation.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${LIMIT:=8}"
: "${CONCURRENT:=4}"
: "${MODEL:=openai-api/vllm/Qwen/Qwen2.5-3B-Instruct}"
: "${MODEL_ARGS:=emulate_tools=true}"
: "${VLLM_BASE_URL:=http://localhost:8000/v1}"
: "${VLLM_API_KEY:=not-needed}"

REPO_ROOT="$(cd .. && pwd)"
BASELINE_DIR="${REPO_ROOT}/runs/concurrency_validation_baseline"
CONCURRENT_DIR="${REPO_ROOT}/runs/concurrency_validation_concurrent"

if ! curl -s -m 3 "${VLLM_BASE_URL}/models" >/dev/null 2>&1; then
  echo "ERROR: no vLLM server reachable at ${VLLM_BASE_URL} -- start it first:" >&2
  echo "  cd ../local-model-server && ./scripts/serve.sh" >&2
  exit 1
fi

echo "== Baseline: MAX_CONNECTIONS=1, LIMIT=${LIMIT} =="
rm -rf "$BASELINE_DIR"
MODEL="$MODEL" MODEL_ARGS="$MODEL_ARGS" VLLM_BASE_URL="$VLLM_BASE_URL" VLLM_API_KEY="$VLLM_API_KEY" \
  CATEGORIES="multi_turn_base" LIMIT="$LIMIT" MAX_CONNECTIONS=1 \
  OUTPUT_DIR="$BASELINE_DIR" ./scripts/run_bfcl_benchmark.sh

echo
echo "== Concurrent: MAX_CONNECTIONS=${CONCURRENT}, LIMIT=${LIMIT} =="
echo "(if this crashes the server, that itself answers whether MAX_CONNECTIONS>1 is safe on this hardware yet)"
rm -rf "$CONCURRENT_DIR"
MODEL="$MODEL" MODEL_ARGS="$MODEL_ARGS" VLLM_BASE_URL="$VLLM_BASE_URL" VLLM_API_KEY="$VLLM_API_KEY" \
  CATEGORIES="multi_turn_base" LIMIT="$LIMIT" MAX_CONNECTIONS="$CONCURRENT" \
  OUTPUT_DIR="$CONCURRENT_DIR" ./scripts/run_bfcl_benchmark.sh

echo
echo "== Comparison =="
uv run python3 - "$BASELINE_DIR" "$CONCURRENT_DIR" <<'PYEOF'
import collections
import sys
from pathlib import Path

from inspect_trace.analysis._loader import load_records_by_sample, records_of_kind


def summarize(run_dir: str) -> None:
    trace_dir = Path(run_dir) / ".inspect_trace"
    by_sample = load_records_by_sample(trace_dir)
    all_records = [r for records in by_sample.values() for r in records]
    calls = records_of_kind(all_records, "vllm_metrics")
    if not calls:
        print(f"  no vllm_metrics records found under {trace_dir}")
        return
    confidence = collections.Counter(c["attribution_confidence"] for c in calls)
    running = [c["queue_depth_running_at_start"] for c in calls if c["queue_depth_running_at_start"] is not None]
    waiting = [c["queue_depth_waiting_at_start"] for c in calls if c["queue_depth_waiting_at_start"] is not None]
    preempt = [c["preemptions_delta"] for c in calls if c["preemptions_delta"] is not None]
    print(f"  n_calls: {len(calls)}")
    print(f"  attribution_confidence: {dict(confidence)}")
    print(
        f"  queue_depth_running_at_start: max={max(running, default='n/a')}, "
        f"nonzero={sum(1 for q in running if q > 0)}/{len(running)}"
    )
    print(
        f"  queue_depth_waiting_at_start: max={max(waiting, default='n/a')}, "
        f"nonzero={sum(1 for q in waiting if q > 0)}/{len(waiting)}"
    )
    print(
        f"  preemptions_delta: sum={sum(preempt) if preempt else 'n/a'}, "
        f"nonzero={sum(1 for p in preempt if p > 0)}/{len(preempt)}"
    )


print("Baseline (MAX_CONNECTIONS=1):")
summarize(sys.argv[1])
print()
print("Concurrent (MAX_CONNECTIONS>1):")
summarize(sys.argv[2])
print()
print("How to read this: the baseline should look exactly like every prior real run --")
print("attribution_confidence all 'exact', queue depths and preemptions all 0. The concurrent run")
print("is the actual test. If its queue depths/preemptions are STILL all 0, real concurrency")
print("didn't actually happen (requests may have finished too fast to overlap -- try a larger")
print("LIMIT or CONCURRENT). If they show real nonzero values and/or attribution_confidence drops")
print("to 'ambiguous' for some calls, these fields are validated under real load for the first time.")
PYEOF
