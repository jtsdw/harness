#!/usr/bin/env bash
# Starts the vLLM OpenAI-compatible server in the background, waits for it to be ready (or fail),
# and prints how to connect from inspect_ai.
#
# Optional overrides:
#   MODEL                     (default: Qwen/Qwen2.5-3B-Instruct)
#   PORT                      (default: 8000)
#   GPU_MEMORY_UTILIZATION    (default: 0.85)
#   MAX_MODEL_LEN             (default: 16384)
#
# vllm==0.6.3.post1 (pinned here for GPU-driver compatibility -- see setup.sh) has no
# --enable-auto-tool-choice / --tool-call-parser support, so it cannot natively parse structured
# tool calls. Connect from inspect_ai using the openai-api provider with emulate_tools=true, which
# makes inspect_ai itself prompt for and parse tool calls instead of relying on the server:
#
#   VLLM_BASE_URL="http://localhost:${PORT:-8000}/v1" VLLM_API_KEY="not-needed" \
#     uv run --project /home/liuyingen/code/efficient-harness/inspect_trace inspect eval <task> \
#     --model "openai-api/vllm/${MODEL:-Qwen/Qwen2.5-3B-Instruct}" -M emulate_tools=true

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PATH="$HOME/.local/bin:$PATH"
: "${MODEL:=Qwen/Qwen2.5-3B-Instruct}"
: "${PORT:=8000}"
: "${GPU_MEMORY_UTILIZATION:=0.85}"
: "${MAX_MODEL_LEN:=16384}"

mkdir -p logs
if [[ -f logs/vllm_server.pid ]] && kill -0 "$(cat logs/vllm_server.pid)" 2>/dev/null; then
  echo "A server is already running (pid $(cat logs/vllm_server.pid)). Run ./scripts/stop.sh first."
  exit 1
fi

echo "Starting vLLM serving $MODEL on port $PORT (this downloads the model on first run)..."
nohup uv run vllm serve "$MODEL" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
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
    echo "Connect from inspect_ai (tool calling emulated client-side, see header comment above):"
    echo "  VLLM_BASE_URL=\"http://localhost:${PORT}/v1\" VLLM_API_KEY=\"not-needed\" \\"
    echo "    uv run --project /home/liuyingen/code/efficient-harness/inspect_trace inspect eval <task> \\"
    echo "    --model \"openai-api/vllm/${MODEL}\" -M emulate_tools=true"
  else
    echo "Server failed to start -- see logs/vllm_server.log" >&2
    tail -40 logs/vllm_server.log >&2
    exit 1
  fi
else
  echo "Timed out waiting for startup -- check logs/vllm_server.log" >&2
  exit 1
fi
