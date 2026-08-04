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

mkdir -p logs
if [[ -f logs/vllm_server.pid ]] && kill -0 "$(cat logs/vllm_server.pid)" 2>/dev/null; then
  echo "A server is already running (pid $(cat logs/vllm_server.pid)). Run ./scripts/stop.sh first."
  exit 1
fi

TOOL_CALLING_FLAGS=()
if [[ -n "$NATIVE_TOOL_CALLING" ]]; then
  TOOL_CALLING_FLAGS+=(--enable-auto-tool-choice --tool-call-parser hermes)
fi

echo "Starting vLLM serving $MODEL on port $PORT (this downloads the model on first run)..."
nohup uv run vllm serve "$MODEL" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  "${TOOL_CALLING_FLAGS[@]}" \
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
  else
    echo "Server failed to start -- see logs/vllm_server.log" >&2
    tail -40 logs/vllm_server.log >&2
    exit 1
  fi
else
  echo "Timed out waiting for startup -- check logs/vllm_server.log" >&2
  exit 1
fi
