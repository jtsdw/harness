#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -f logs/vllm_server.pid ]]; then
  pid="$(cat logs/vllm_server.pid)"
  if kill -0 "$pid" 2>/dev/null; then
    echo "Stopping vLLM server (pid $pid)..."
    kill "$pid" 2>/dev/null || true
    sleep 2
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f logs/vllm_server.pid
fi
pkill -f "vllm serve" 2>/dev/null || true
echo "Stopped."
