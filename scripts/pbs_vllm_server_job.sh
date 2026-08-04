#!/bin/bash
# Batch-mode alternative to scripts/nscc_interactive_gpu_session.sh: submit this as a PBS job
# (`qsub scripts/pbs_vllm_server_job.sh`) and it runs unattended for the walltime -- starts vLLM,
# runs whatever benchmark command you point it at, tears vLLM down, exits. No live shell needed,
# no tmux/screen to babysit. Use this for "let it run overnight", use the interactive session
# script for real debugging.
#
# Modeled on /home/liuyingen/code/quant/unimq/bash/run_native.sh's structure (#PBS header,
# module load, cd into project dir, run) -- same cluster, same conventions, so the mental model
# transfers if you've used that project before.
#
# *** Fill in -P before submitting *** -- there's no sane default, see
# scripts/nscc_interactive_gpu_session.sh's comment for how to find your project code.
#PBS -P REPLACE_ME_WITH_YOUR_PROJECT_CODE
#PBS -q normal
#PBS -N efficient_harness_bench
#PBS -l select=1:ngpus=1:ncpus=4:mem=80gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o /home/users/ntu/n2505716/scratch/harness/logs

set -euo pipefail

# Adjust to your own checkout (see docs/team_collaboration.md -- each person keeps a separate
# subdirectory under scratch/, this must point at yours, not a shared one).
CHECKOUT_DIR="${CHECKOUT_DIR:-/home/users/ntu/n2505716/scratch/harness-$(whoami)}"

# --- Environment setup ---
# This project manages its own venvs with uv (see docs/environment_checklist.md), not conda --
# unlike quant's `module load miniforge3 && source activate <env>` pattern. If this cluster's
# compute nodes need an explicit CUDA driver module loaded before vLLM will see the GPU, add it
# here after checking `module avail` on an allocated node -- not verified yet, see
# docs/remote_compute_workflow.md's "待现场验证" list.
# module load cuda/xxx   # uncomment and fill in if needed

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found on PATH. Install it first (see docs/environment_checklist.md)." >&2
  exit 1
fi

echo "[$(date)] Starting vLLM server..."
cd "${CHECKOUT_DIR}/local-model-server"
./scripts/serve.sh &
SERVE_PID=$!

# serve.sh itself blocks until the server is ready (or fails) and returns -- but it's launched
# with `&` here so this job script can also tear it down afterward. Wait on it directly: serve.sh
# exits once startup finishes (it doesn't stay foregrounded blocking on the server process), so
# this wait returns quickly once vLLM is confirmed up, not just once vLLM fully shuts down.
wait "$SERVE_PID"

echo "[$(date)] vLLM ready. Running benchmark..."
cd "${CHECKOUT_DIR}/inspect_trace"

# Override any of these via `qsub -v VAR=value` or by editing before submitting.
: "${BENCHMARK_MODEL:=openai-api/vllm/Qwen/Qwen2.5-3B-Instruct}"
: "${BENCHMARK_MODEL_ARGS:=emulate_tools=true}"
: "${BENCHMARK_CATEGORIES:=multi_turn_base}"
: "${BENCHMARK_LIMIT:=200}"
: "${BENCHMARK_OUTPUT_DIR:=${CHECKOUT_DIR}/runs/goal1_bfcl_${BENCHMARK_CATEGORIES}_pbs}"

VLLM_BASE_URL="http://localhost:8000/v1" VLLM_API_KEY="not-needed" \
MODEL="$BENCHMARK_MODEL" MODEL_ARGS="$BENCHMARK_MODEL_ARGS" \
CATEGORIES="$BENCHMARK_CATEGORIES" LIMIT="$BENCHMARK_LIMIT" MAX_CONNECTIONS=1 \
OUTPUT_DIR="$BENCHMARK_OUTPUT_DIR" \
./scripts/run_bfcl_benchmark.sh

echo "[$(date)] Benchmark done. Stopping vLLM..."
cd "${CHECKOUT_DIR}/local-model-server"
./scripts/stop.sh

echo "[$(date)] Job complete. Pull results back with scripts/pull_runs.sh from the local machine."
