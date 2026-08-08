#!/usr/bin/env bash
# Starts vLLM on the NSCC H100 node. See ../README.md -- this session still has no direct access
# to that machine, so this remains largely a best-effort reading of vLLM's own docs/source rather
# than a known-working recipe, but as of 2026-08-08 it has real feedback from someone who can run
# it: model + EAGLE-3 draft load fine, but kernel warmup crashed on a missing CUDA toolkit (no
# nvcc/CUDA_HOME on this node) -- worked around below via VLLM_USE_FLASHINFER_SAMPLER=0. Hasn't
# gotten past that point yet. Run this, then ./scripts/verify_eagle3.sh, and report back what
# actually happens.
#
# Optional overrides:
#   MODEL                     (default: Qwen/Qwen3-32B)
#   PORT                      (default: 8000)
#   GPU_MEMORY_UTILIZATION    (default: 0.9)
#   MAX_MODEL_LEN             (default: 16384)
#   TENSOR_PARALLEL_SIZE      (default: 1 -- a single H100 80GB should hold a 32B model in bf16
#                             (~64GB weights) plus KV cache plus the EAGLE-3 draft head, but this
#                             is arithmetic, not a measurement -- if it OOMs, try
#                             TENSOR_PARALLEL_SIZE=2 if a second GPU is available in the same job)
#   SPECULATIVE_MODE          (default: unset -- no speculative decoding. Set to "eagle3" to
#                             enable, see EAGLE3_DRAFT_MODEL/NUM_SPECULATIVE_TOKENS below)
#   EAGLE3_DRAFT_MODEL        (default: RedHatAI/Qwen3-32B-speculator.eagle3 -- only used when
#                             SPECULATIVE_MODE=eagle3. Must be a draft trained specifically
#                             against whatever MODEL is set to -- these are not interchangeable
#                             across model sizes/families, see ../README.md)
#   NUM_SPECULATIVE_TOKENS    (default: 3, only used when SPECULATIVE_MODE=eagle3)
#   HF_HOME                   (default: $HOME/scratch/model/.hf-cache if unset -- NSCC convention,
#                             same reasoning as local-model-server/scripts/serve.sh's HF_HOME
#                             handling: large files belong under $HOME/scratch/, not $HOME itself)
#   VLLM_USE_FLASHINFER_SAMPLER (default: 0 -- disabled. This node has no nvcc/CUDA_HOME, and the
#                             FlashInfer fused sampler needs to JIT-compile a kernel the first time
#                             it's used, which crashes without a CUDA toolkit. Set to 1 to re-enable
#                             once/if a real CUDA toolkit is available on this node -- should be
#                             faster than the fallback, just untested here)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${HF_HOME:=$HOME/scratch/model/.hf-cache}"
export HF_HOME
export PATH="$HOME/.local/bin:$PATH"
: "${MODEL:=Qwen/Qwen3-32B}"
: "${PORT:=8000}"
: "${GPU_MEMORY_UTILIZATION:=0.9}"
: "${MAX_MODEL_LEN:=16384}"
: "${TENSOR_PARALLEL_SIZE:=1}"
: "${SPECULATIVE_MODE:=}"
: "${EAGLE3_DRAFT_MODEL:=RedHatAI/Qwen3-32B-speculator.eagle3}"
: "${NUM_SPECULATIVE_TOKENS:=3}"
# 2026-08-08 real finding: this node has no discoverable CUDA toolkit (nvcc not found,
# /usr/local/cuda doesn't exist -- only the driver seems to be present), which crashed vLLM for
# real during kernel warmup ("RuntimeError: Could not find nvcc...") the first time it tried to
# JIT-compile FlashInfer's fused top-k/top-p sampling kernel. VLLM_USE_FLASHINFER_SAMPLER=0 (real
# env var, confirmed in vLLM's own envs.py) skips that kernel entirely, falling back to vLLM's
# native PyTorch/Triton sampling path -- no JIT compile needed. Trade-off: this is presumably
# somewhat slower than the fused kernel would be, real cost not measured here. If CUDA_HOME/nvcc
# ever get properly set up on this node (e.g. via `module load cuda` if this cluster uses
# Environment Modules -- worth checking), this can go back to the default (unset this override)
# for the real fused-kernel performance.
: "${VLLM_USE_FLASHINFER_SAMPLER:=0}"
export VLLM_USE_FLASHINFER_SAMPLER

# Same defensive fix as local-model-server/scripts/serve.sh -- PBS/Slurm clusters can set
# CUDA_VISIBLE_DEVICES to GPU UUIDs instead of plain integers. Confirmed to break the OLD vLLM
# this project deliberately does NOT use; kept here as a precaution since it's harmless when
# CUDA_VISIBLE_DEVICES is already numeric (or unset), and untested whether newer vLLM actually
# fixed the underlying bug.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && ! "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  IFS=',' read -ra _cvd_entries <<< "$CUDA_VISIBLE_DEVICES"
  _cvd_new=$(seq -s, 0 $((${#_cvd_entries[@]} - 1)))
  echo "CUDA_VISIBLE_DEVICES=\"$CUDA_VISIBLE_DEVICES\" uses non-numeric device identifiers," >&2
  echo "remapping to \"$_cvd_new\" -- see local-model-server/scripts/serve.sh's comment for why." >&2
  export CUDA_VISIBLE_DEVICES="$_cvd_new"
fi

mkdir -p logs
if [[ -f logs/vllm_server.pid ]] && kill -0 "$(cat logs/vllm_server.pid)" 2>/dev/null; then
  echo "A server is already running (pid $(cat logs/vllm_server.pid)). Run ./scripts/stop.sh first."
  exit 1
fi

SPECULATIVE_FLAGS=()
if [[ "$SPECULATIVE_MODE" == "eagle3" ]]; then
  SPECULATIVE_FLAGS+=(
    --speculative-config "{\"method\": \"eagle3\", \"model\": \"${EAGLE3_DRAFT_MODEL}\", \"num_speculative_tokens\": ${NUM_SPECULATIVE_TOKENS}}"
  )
elif [[ -n "$SPECULATIVE_MODE" ]]; then
  echo "Unsupported SPECULATIVE_MODE: $SPECULATIVE_MODE (only \"eagle3\" is wired up)" >&2
  exit 1
fi

echo "Starting vLLM serving $MODEL on port $PORT (this downloads the model on first run -- a 32B"
echo "model is a large download, expect it to take a while) ..."
nohup uv run vllm serve "$MODEL" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  "${SPECULATIVE_FLAGS[@]}" \
  > logs/vllm_server.log 2>&1 &
echo $! > logs/vllm_server.pid
echo "pid $(cat logs/vllm_server.pid), logs at logs/vllm_server.log"

echo "Waiting for startup (or failure) -- a 32B model load + EAGLE-3 draft load can take longer"
echo "than local-model-server's 3B-model timeout, this waits up to 10 minutes ..."
# 2026-08-08 real finding: a bare "Traceback|OSError|RuntimeError" grep is too loose -- vLLM's own
# optional-dependency import failures (e.g. deep_gemm's _find_cuda_home() AssertionError when
# CUDA_HOME isn't set, seen for real on the NSCC node) get caught and logged as a WARNING that
# still contains the literal string "Traceback" (Python's own formatting for a caught exception).
# That single line used to satisfy this grep and made the script report "failed to start" while
# the server was still normally starting up (and the nohup'd process kept running in the
# background, unaffected -- this script just gave up watching it and printed a false alarm).
# Real fatal errors from vLLM/uvicorn are not wrapped in a "WARNING ...:" prefix the way a caught,
# non-fatal one is -- excluding lines that contain "WARNING" is what actually distinguishes them.
if timeout 600 bash -c '
  until grep -qE "Uvicorn running|Application startup complete" "'"$PWD"'/logs/vllm_server.log" 2>/dev/null \
     || grep -E "Traceback|OSError|RuntimeError" "'"$PWD"'/logs/vllm_server.log" 2>/dev/null | grep -qv "WARNING"; do
    sleep 8
  done
'; then
  if grep -q "Uvicorn running" logs/vllm_server.log; then
    echo
    echo "Server is up: http://localhost:${PORT}/v1"
    if [[ "$SPECULATIVE_MODE" == "eagle3" ]]; then
      echo "EAGLE-3 speculative decoding requested (draft: ${EAGLE3_DRAFT_MODEL}, num_speculative_tokens=${NUM_SPECULATIVE_TOKENS})."
      echo "This does NOT confirm it's actually active -- run ./scripts/verify_eagle3.sh next."
    fi
  else
    echo "Server failed to start -- see logs/vllm_server.log" >&2
    tail -60 logs/vllm_server.log >&2
    exit 1
  fi
else
  echo "Timed out waiting for startup -- check logs/vllm_server.log" >&2
  exit 1
fi
