#!/usr/bin/env bash
# Orchestrates the real comparison from docs/toolspec_vllm_speculative_comparison.md: runs the
# same BFCL questions against vLLM baseline (no acceleration) and vLLM + ngram speculative
# decoding, then reports the real speedup -- not just "did both start", an actual tokens/s number
# for each, computed the same way (mean of per-question ratios) as ToolSpec's own
# evaluation/speed.py and this project's own dashboard scripts, so the number means the same thing
# every time it's reported.
#
# Starts and stops both vLLM configurations itself (calls serve_baseline.sh/
# serve_ngram_speculative.sh directly, which each block until the server is confirmed ready before
# returning -- no separate polling loop needed here). Only one vLLM server exists at a time
# throughout, matching this project's "no agent, single GPU" compute-node convention.
#
# Usage:
#   ./scripts/verify_ngram_speculative.sh
#   MODEL_NAME="Qwen/Qwen2.5-32B-Instruct" LIMIT=20 ./scripts/verify_ngram_speculative.sh
#
# Optional overrides:
#   MODEL_NAME  (default: Qwen/Qwen2.5-3B-Instruct)
#   LIMIT       (default: 20 -- BFCL multi_turn_base samples; too few and the speed comparison is
#               noise-dominated, this is not the place to save a few seconds)
#   CATEGORIES  (default: multi_turn_base, forwarded to run_bfcl_local_vllm.sh)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_MODEL_SERVER_DIR="${REPO_ROOT}/local-model-server"
INSPECT_TRACE_SCRIPTS_DIR="${REPO_ROOT}/inspect_trace/scripts"

: "${MODEL_NAME:=Qwen/Qwen2.5-3B-Instruct}"
: "${LIMIT:=20}"
: "${CATEGORIES:=multi_turn_base}"

BASELINE_OUTPUT_DIR="${REPO_ROOT}/runs/verify_ngram_speculative_baseline"
NGRAM_OUTPUT_DIR="${REPO_ROOT}/runs/verify_ngram_speculative_ngram"

# stop.sh returns as soon as it's sent the kill signal(s) -- it does NOT wait for the OS to
# actually release port 8000. Checking "does the port answer HTTP" isn't enough to prove it's
# safe to bind again either -- a socket can sit in TCP TIME_WAIT (not answering, but not yet
# free for a new bind) for longer than that check would ever see (real failure hit here even on
# the very FIRST server start of an otherwise-clean run, port already dead by the time this
# script's own liveness check would have looked at it). Retrying the bind itself with backoff is
# the only thing that's actually robust to this, since the exact TIME_WAIT duration isn't
# something this script can predict.
start_server_with_retry() {
  local serve_script="$1"
  local attempt=1 max_attempts=6
  while (( attempt <= max_attempts )); do
    if ( cd "$LOCAL_MODEL_SERVER_DIR" && MODEL="$MODEL_NAME" "./scripts/${serve_script}" ); then
      return 0
    fi
    if grep -q "Address already in use" "${LOCAL_MODEL_SERVER_DIR}/logs/vllm_server.log" 2>/dev/null; then
      echo "Port 8000 still tied up from a previous server (attempt ${attempt}/${max_attempts}) -- waiting 10s and retrying..." >&2
      ( cd "$LOCAL_MODEL_SERVER_DIR" && ./scripts/stop.sh ) || true
      sleep 10
      attempt=$((attempt + 1))
    else
      echo "ERROR: server failed to start for a reason other than the port -- see log above." >&2
      exit 1
    fi
  done
  echo "ERROR: port 8000 still unavailable after ${max_attempts} attempts." >&2
  exit 1
}

run_one() {
  local label="$1" serve_script="$2" output_dir="$3"
  echo "=== [$label] starting vLLM ==="
  start_server_with_retry "$serve_script"

  echo "=== [$label] running BFCL (limit=$LIMIT, categories=$CATEGORIES) ==="
  ( cd "$INSPECT_TRACE_SCRIPTS_DIR" && \
    MODEL_NAME="$MODEL_NAME" LIMIT="$LIMIT" CATEGORIES="$CATEGORIES" OUTPUT_DIR="$output_dir" \
    ./run_bfcl_local_vllm.sh )

  echo "=== [$label] stopping vLLM ==="
  ( cd "$LOCAL_MODEL_SERVER_DIR" && ./scripts/stop.sh )
}

run_one "baseline" "serve_baseline.sh" "$BASELINE_OUTPUT_DIR"
run_one "ngram"    "serve_ngram_speculative.sh" "$NGRAM_OUTPUT_DIR"

echo
echo "=== Comparing real tokens/s (mean of per-question ratios) ==="
uv run --project "${REPO_ROOT}/inspect_trace" python3 -c "
from inspect_ai.log import read_eval_log
import glob
import numpy as np

def speed(run_dir):
    eval_files = sorted(glob.glob(f'{run_dir}/logs/*.eval'))
    assert eval_files, f'no .eval file found under {run_dir}/logs/'
    log = read_eval_log(eval_files[-1])
    speeds = []
    for sample in log.samples:
        for event in sample.events:
            if event.event == 'model':
                wt = event.working_time
                tok = event.output.usage.output_tokens if event.output and event.output.usage else None
                if wt and tok:
                    speeds.append(tok / wt)
    return np.mean(speeds), eval_files[-1]

base_tps, base_file = speed('$BASELINE_OUTPUT_DIR')
ngram_tps, ngram_file = speed('$NGRAM_OUTPUT_DIR')
print(f'baseline ({base_file}): {base_tps:.2f} tokens/s')
print(f'ngram    ({ngram_file}): {ngram_tps:.2f} tokens/s')
print(f'speedup: {ngram_tps/base_tps:.2f}x')
"
