#!/usr/bin/env bash
# Starts the vLLM OpenAI-compatible server in the background, waits for it to be ready (or fail),
# and prints how to connect from inspect_ai.
#
# Optional overrides:
#   MODEL                     (default: Qwen/Qwen2.5-3B-Instruct)
#   PORT                      (default: 8000)
#   GPU_MEMORY_UTILIZATION    (default: 0.85)
#   MAX_MODEL_LEN             (default: 16384)
#   NATIVE_TOOL_CALLING       (default: unset/false) -- set to any non-empty value to add
#                             --enable-auto-tool-choice --tool-call-parser hermes.
#   SPECULATIVE_MODE          (default: unset -- no speculative decoding). Set to "ngram" to
#                             enable vLLM's built-in n-gram/prompt-lookup speculative decoding
#                             (no separate draft model needed -- vllm serve's own
#                             --speculative-model "[ngram]"). See
#                             docs/toolspec_vllm_speculative_comparison.md for why this exists:
#                             a real comparison point against the ToolSpec paper's own
#                             training-free speculative decoding (docs/toolspec_integration_findings.md).
#   NUM_SPECULATIVE_TOKENS    (default: 5, only used when SPECULATIVE_MODE=ngram)
#   NGRAM_PROMPT_LOOKUP_MAX   (default: 4, only used when SPECULATIVE_MODE=ngram)
#   NGRAM_PROMPT_LOOKUP_MIN   (default: 1, only used when SPECULATIVE_MODE=ngram)
#   HF_HOME                   (default: unset -- HuggingFace's own default, $HOME/.cache/huggingface)
#                             On clusters with a small $HOME quota and a much larger scratch quota
#                             (hit for real on an NSCC DGX node downloading Qwen2.5-32B-Instruct --
#                             "OSError: Disk quota exceeded"), set this to a scratch-space path
#                             BEFORE running this script, e.g.:
#                               export HF_HOME=~/scratch/hf_cache   # NSCC convention: large files
#                               go under $HOME/scratch/, not $HOME itself -- see
#                               docs/remote_compute_workflow.md
#                             This script does NOT default HF_HOME to a project-relative path
#                             itself, on purpose -- doing so would silently orphan whatever's
#                             already cached at the machine's real default on every machine that
#                             already has a working cache there (this dev machine included). It
#                             only warns (see below) when $HOME and this project's own directory
#                             look like they're on different filesystems, since that's the
#                             concrete condition under which this class of quota problem occurs.
#
# Note: this vLLM version (0.6.3.post1) does not expose any speculative-decoding-specific
# Prometheus metrics on /metrics (checked directly -- only the generic num_preemptions_total
# counter exists), so inspect_trace's vllm_metrics.py cannot report an "accepted tokens"-style
# field for this path the way ToolSpec's own eval does. Speed has to be measured end-to-end
# (tokens/wall_time), not decomposed into accept-length like ToolSpec's own JSONL output.
#
# An earlier version of this comment claimed vllm==0.6.3.post1 (pinned here for GPU-driver
# compatibility -- see setup.sh) has no native tool-calling support -- that was never actually
# tested and turned out to be wrong (see docs/tau2_bench_integration_findings.md's Bug 1): a real
# request with --enable-auto-tool-choice --tool-call-parser hermes returns clean structured
# tool_calls on this exact pinned version. Default here is still OFF (emulate_tools=true remains
# the most battle-tested path across this project's real benchmark runs), but native tool-calling
# is a real, working option -- set NATIVE_TOOL_CALLING=true to use it. Note: inspect_ai's
# openai-api provider adds a "strict" field to tool definitions this vLLM version rejects (Bug 3,
# same doc) -- either keep using emulate_tools=true with native-mode servers too, or use the
# tau2_adapter/_registry.py-style custom ModelAPI provider that strips that field.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PATH="$HOME/.local/bin:$PATH"
: "${MODEL:=Qwen/Qwen2.5-3B-Instruct}"
: "${PORT:=8000}"
: "${GPU_MEMORY_UTILIZATION:=0.85}"
: "${MAX_MODEL_LEN:=16384}"
: "${NATIVE_TOOL_CALLING:=}"
: "${SPECULATIVE_MODE:=}"
: "${NUM_SPECULATIVE_TOKENS:=5}"
: "${NGRAM_PROMPT_LOOKUP_MAX:=4}"
: "${NGRAM_PROMPT_LOOKUP_MIN:=1}"

# Some PBS/Slurm-managed multi-GPU clusters set CUDA_VISIBLE_DEVICES to GPU UUIDs
# (e.g. "GPU-2a95c117-1313-cb8f-323f-ea75d126060d") instead of plain integer indices, as part of
# cgroup-level device isolation. This vLLM version (0.6.3.post1) is too old to handle that --
# vllm/platforms/cuda.py::device_id_to_physical_device_id() does a bare int() on each entry and
# crashes with "invalid literal for int()" before the engine even starts. Remapping to sequential
# indices (0, 1, ...) is safe here specifically because the scheduler's cgroup restriction already
# means this process can only see its allocated GPU(s) -- which physical GPU(s) are visible was
# decided upstream, this only fixes the label format vLLM's older code expects.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && ! "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  IFS=',' read -ra _cvd_entries <<< "$CUDA_VISIBLE_DEVICES"
  _cvd_new=$(seq -s, 0 $((${#_cvd_entries[@]} - 1)))
  echo "CUDA_VISIBLE_DEVICES=\"$CUDA_VISIBLE_DEVICES\" uses non-numeric device identifiers," >&2
  echo "which this vLLM version can't parse. Remapping to \"$_cvd_new\" (safe under PBS/Slurm" >&2
  echo "cgroup-based GPU isolation -- see this comment in serve.sh for why)." >&2
  export CUDA_VISIBLE_DEVICES="$_cvd_new"
fi

# Heuristic warning for the disk-quota class of problem (see HF_HOME in the header comment): if
# $HOME and this project's own directory are on different filesystems, a small $HOME quota is a
# real risk for large model downloads (hit for real: Qwen2.5-32B-Instruct is ~65GB, blew a DGX
# node's $HOME quota mid-download). Only warns -- does not set HF_HOME itself, see header comment
# for why.
if [[ -z "${HF_HOME:-}" ]] && command -v stat >/dev/null 2>&1; then
  _home_dev="$(stat -c %d "$HOME" 2>/dev/null || echo "")"
  _proj_dev="$(stat -c %d "$PWD" 2>/dev/null || echo "")"
  if [[ -n "$_home_dev" && -n "$_proj_dev" && "$_home_dev" != "$_proj_dev" ]]; then
    echo "NOTE: \$HOME ($HOME) and this project's directory ($PWD) are on different filesystems." >&2
    echo "If \$HOME has a small quota (common on shared clusters), a large model download can" >&2
    echo "fail with \"Disk quota exceeded\" -- HuggingFace caches to \$HOME/.cache/huggingface by" >&2
    echo "default. Set HF_HOME to a path with more space before running this script if that" >&2
    echo "happens, e.g.: export HF_HOME=~/scratch/hf_cache (NSCC convention)" >&2
  fi
fi

mkdir -p logs
if [[ -f logs/vllm_server.pid ]] && kill -0 "$(cat logs/vllm_server.pid)" 2>/dev/null; then
  echo "A server is already running (pid $(cat logs/vllm_server.pid)). Run ./scripts/stop.sh first."
  exit 1
fi

TOOL_CALLING_FLAGS=()
if [[ -n "$NATIVE_TOOL_CALLING" ]]; then
  TOOL_CALLING_FLAGS+=(--enable-auto-tool-choice --tool-call-parser hermes)
fi

SPECULATIVE_FLAGS=()
if [[ "$SPECULATIVE_MODE" == "ngram" ]]; then
  SPECULATIVE_FLAGS+=(
    --speculative-model "[ngram]"
    --num-speculative-tokens "$NUM_SPECULATIVE_TOKENS"
    --ngram-prompt-lookup-max "$NGRAM_PROMPT_LOOKUP_MAX"
    --ngram-prompt-lookup-min "$NGRAM_PROMPT_LOOKUP_MIN"
  )
elif [[ -n "$SPECULATIVE_MODE" ]]; then
  echo "Unsupported SPECULATIVE_MODE: $SPECULATIVE_MODE (only \"ngram\" is wired up so far)" >&2
  exit 1
fi

echo "Starting vLLM serving $MODEL on port $PORT (this downloads the model on first run)..."
nohup uv run vllm serve "$MODEL" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  "${TOOL_CALLING_FLAGS[@]}" \
  "${SPECULATIVE_FLAGS[@]}" \
  > logs/vllm_server.log 2>&1 &
echo $! > logs/vllm_server.pid
echo "pid $(cat logs/vllm_server.pid), logs at logs/vllm_server.log"

echo "Waiting for startup (or failure)..."
if timeout 280 bash -c '
  until grep -qE "Uvicorn running|Application startup complete|Traceback|OSError|RuntimeError" "'"$PWD"'/logs/vllm_server.log" 2>/dev/null; do
    sleep 8
  done
'; then
  if grep -q "Uvicorn running" logs/vllm_server.log; then
    echo
    echo "Server is up: http://localhost:${PORT}/v1"
    echo
    if [[ -n "$NATIVE_TOOL_CALLING" ]]; then
      echo "Native tool-calling is enabled server-side. NOTE: inspect_ai's stock openai-api"
      echo "provider still won't work as-is against this server -- it always adds a \"strict\""
      echo "field to tool definitions that this vLLM version rejects (Bug 3, see header comment"
      echo "above). Either keep using -M emulate_tools=true anyway (works fine, just doesn't"
      echo "exercise native tool-calling), or use a custom ModelAPI provider that strips that"
      echo "field -- see tau2_adapter/src/tau2_adapter/_registry.py (registers"
      echo "\"tau2-agent-vllm\") for a working example to copy the pattern from."
    else
      echo "Connect from inspect_ai (tool calling emulated client-side):"
      echo "  VLLM_BASE_URL=\"http://localhost:${PORT}/v1\" VLLM_API_KEY=\"not-needed\" \\"
      echo "    uv run --project /home/liuyingen/code/efficient-harness/inspect_trace inspect eval <task> \\"
      echo "    --model \"openai-api/vllm/${MODEL}\" -M emulate_tools=true"
    fi
    if [[ "$SPECULATIVE_MODE" == "ngram" ]]; then
      echo
      echo "Speculative decoding: ngram mode ON (num_speculative_tokens=${NUM_SPECULATIVE_TOKENS},"
      echo "prompt_lookup_max=${NGRAM_PROMPT_LOOKUP_MAX}, prompt_lookup_min=${NGRAM_PROMPT_LOOKUP_MIN})."
      echo "No separate draft model needed. This version's /metrics has no spec-decode-specific"
      echo "fields (checked directly) -- speed has to be measured end-to-end, not via an"
      echo "accept-length histogram like ToolSpec's own output gives you."
    fi
  else
    echo "Server failed to start -- see logs/vllm_server.log" >&2
    tail -40 logs/vllm_server.log >&2
    exit 1
  fi
else
  echo "Timed out waiting for startup -- check logs/vllm_server.log" >&2
  exit 1
fi
