#!/usr/bin/env bash
# Thin wrapper over run_bfcl_benchmark.sh for the "local vLLM server" case specifically --
# defaults (and properly `export`s) the boilerplate that's the same every time you're pointing at
# a local vLLM server (VLLM_BASE_URL/VLLM_API_KEY/MAX_CONNECTIONS/MODEL_ARGS), so you don't have
# to retype it or (worse) hardcode it into run_bfcl_benchmark.sh itself -- editing that shared
# script directly causes real problems (a local edit that conflicts with the next `git pull`, and
# variables set with `: "${VAR=value}"` inside a script are NOT exported to child processes the
# way command-line-prefixed `VAR=value command` is, which silently breaks anything downstream that
# reads the env var -- both hit for real on an NSCC DGX node).
#
# MODEL_NAME is the one thing that legitimately varies by run -- everything else defaults to the
# standard local-vLLM setup and can still be overridden if you need to.
#
# Usage:
#   MODEL_NAME="Qwen/Qwen2.5-32B-Instruct" ./run_bfcl_local_vllm.sh
#   MODEL_NAME="Qwen/Qwen2.5-32B-Instruct" CATEGORIES=live_parallel LIMIT=10 ./run_bfcl_local_vllm.sh
#
# All of run_bfcl_benchmark.sh's own overrides (CATEGORIES/LIMIT/OUTPUT_DIR) still apply --
# see that script's own header comment for the full list.

set -euo pipefail

: "${MODEL_NAME:?ERROR: set MODEL_NAME to the model being served, e.g. Qwen/Qwen2.5-32B-Instruct}"

export MODEL="openai-api/vllm/${MODEL_NAME}"
export MODEL_ARGS="${MODEL_ARGS:-emulate_tools=true}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-not-needed}"
export MAX_CONNECTIONS="${MAX_CONNECTIONS:-1}"

if ! curl -s -m 3 "${VLLM_BASE_URL}/models" >/dev/null 2>&1; then
  echo "ERROR: no vLLM server reachable at ${VLLM_BASE_URL} -- start one first:" >&2
  echo "  cd ../../local-model-server && MODEL=\"${MODEL_NAME}\" ./scripts/serve.sh" >&2
  exit 1
fi

exec "$(dirname "${BASH_SOURCE[0]}")/run_bfcl_benchmark.sh"
