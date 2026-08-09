#!/usr/bin/env bash
# Parameterized NSCC/PBS job for Qwen3.6-27B + tau2 core domains.
# Before qsub, replace the project code below and run tau2 setup on the login node.
# Required qsub -v values: CHECKOUT_DIR, TAU2_DATA_DIR, MODEL_PATH, VLLM_PYTHON.

#PBS -P REPLACE_ME_WITH_YOUR_PROJECT_CODE
#PBS -q normal
#PBS -N tau2_qwen36
#PBS -l select=1:ngpus=1:ncpus=8:mem=80gb
#PBS -l walltime=12:00:00
#PBS -j oe

set -euo pipefail

require_value() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: required environment variable is missing: $name" >&2
    exit 2
  fi
}

: "${CHECKOUT_DIR:=${PBS_O_WORKDIR:-}}"
require_value CHECKOUT_DIR
require_value TAU2_DATA_DIR
require_value MODEL_PATH
require_value VLLM_PYTHON

: "${MODEL_NAME:=Qwen3.6-27B}"
: "${VLLM_PORT:=8000}"
: "${VLLM_API_KEY:=local}"
: "${VLLM_MAX_MODEL_LEN:=65536}"
: "${VLLM_MAX_NUM_SEQS:=300}"
: "${VLLM_GPU_MEMORY_UTILIZATION:=0.9}"
: "${VLLM_STARTUP_TIMEOUT:=900}"
: "${VLLM_TOOL_CALL_PARSER:=qwen3_xml}"
: "${VLLM_EXTRA_ARGS:=}"
: "${TAU2_DOMAINS:=mock airline retail telecom telecom-workflow}"
: "${RUN_SMOKE:=1}"
: "${RESUME:=1}"
: "${FULL_RUN_ID:=nscc_tau2_core_$(date +%Y%m%d_%H%M%S)}"

ADAPTER_DIR="${CHECKOUT_DIR}/tau2_adapter"
RUN_ROOT="${CHECKOUT_DIR}/runs/${FULL_RUN_ID}"
VLLM_LOG="${VLLM_LOG:-${RUN_ROOT}/vllm.log}"
VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
INSPECT_TRACE_VLLM_METRICS_URL="http://127.0.0.1:${VLLM_PORT}/metrics"

if [[ ! -x "$VLLM_PYTHON" ]]; then
  echo "ERROR: VLLM_PYTHON is not executable: $VLLM_PYTHON" >&2
  exit 1
fi
if [[ ! -x "${ADAPTER_DIR}/.venv/bin/python" ||
      ! -f "${CHECKOUT_DIR}/.deps/tau2-bench/pyproject.toml" ]]; then
  echo "ERROR: tau2 adapter is not provisioned." >&2
  echo "Run tau2_adapter/scripts/setup_tau2_bench.sh on the login node first." >&2
  exit 1
fi
if [[ ! -d "${TAU2_DATA_DIR}/tau2/domains" ]]; then
  echo "ERROR: TAU2_DATA_DIR does not contain tau2/domains: $TAU2_DATA_DIR" >&2
  exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: MODEL_PATH is not a directory: $MODEL_PATH" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT"
VLLM_ENV_DIR="$(cd -- "$(dirname -- "$VLLM_PYTHON")/.." && pwd)"
export PATH="${VLLM_ENV_DIR}/bin:${PATH}"
if [[ -d "${VLLM_ENV_DIR}/lib" ]]; then
  export LD_LIBRARY_PATH="${VLLM_ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
fi

read -r -a extra_vllm_args <<<"$VLLM_EXTRA_ARGS"
echo "Starting Qwen3.6 vLLM backend"
echo "  model: $MODEL_PATH"
echo "  endpoint: $VLLM_BASE_URL"
echo "  log: $VLLM_LOG"
"$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
  --enable-auto-tool-choice \
  --tool-call-parser "$VLLM_TOOL_CALL_PARSER" \
  --port "$VLLM_PORT" \
  "${extra_vllm_args[@]}" \
  >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

cleanup() {
  if kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

deadline=$((SECONDS + VLLM_STARTUP_TIMEOUT))
until curl -fsS --max-time 5 "${VLLM_BASE_URL%/v1}/health" >/dev/null 2>&1; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "ERROR: vLLM exited during startup. Last log lines:" >&2
    tail -n 120 "$VLLM_LOG" >&2 || true
    exit 1
  fi
  if ((SECONDS >= deadline)); then
    echo "ERROR: vLLM did not become ready within ${VLLM_STARTUP_TIMEOUT}s." >&2
    tail -n 120 "$VLLM_LOG" >&2 || true
    exit 1
  fi
  sleep 10
done

model_catalog="$(curl -fsS --max-time 10 \
  -H "Authorization: Bearer ${VLLM_API_KEY}" \
  "${VLLM_BASE_URL}/models")"
MODEL_CATALOG="$model_catalog" EXPECTED_MODEL="$MODEL_NAME" \
  "${ADAPTER_DIR}/.venv/bin/python" - <<'PY'
import json
import os

models = [item.get("id") for item in json.loads(os.environ["MODEL_CATALOG"]).get("data", [])]
expected = os.environ["EXPECTED_MODEL"]
if expected not in models:
    raise SystemExit(f"expected model {expected!r}; served models are {models!r}")
print("served model:", expected)
PY
curl -fsS --max-time 10 "$INSPECT_TRACE_VLLM_METRICS_URL" >/dev/null

DEFAULT_LLM_ARGS="$(printf \
  '{"temperature":0,"max_tokens":2048,"api_base":"%s","api_key":"%s","input_cost_per_token":0,"output_cost_per_token":0,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
  "$VLLM_BASE_URL" "$VLLM_API_KEY")"
: "${TAU2_USER_LLM_ARGS:=$DEFAULT_LLM_ARGS}"
: "${TAU2_JUDGE_LLM_ARGS:=$DEFAULT_LLM_ARGS}"

export TAU2_DATA_DIR VLLM_BASE_URL VLLM_API_KEY MODEL_NAME
export USER_MODEL_NAME="${USER_MODEL_NAME:-$MODEL_NAME}"
export JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-$MODEL_NAME}"
export INSPECT_TRACE_VLLM_METRICS_URL
export TAU2_USER_LLM_ARGS TAU2_JUDGE_LLM_ARGS
export TAU2_DOMAINS FULL_RUN_ID RESUME

if [[ "$RUN_SMOKE" == "1" ]]; then
  echo "Running one-sample smoke tests: $TAU2_DOMAINS"
  read -r -a smoke_domains <<<"$TAU2_DOMAINS"
  for domain in "${smoke_domains[@]}"; do
    safe_domain="${domain//-/_}"
    smoke_name="${FULL_RUN_ID}_smoke_${safe_domain}"
    TAU2_DOMAIN="$domain" \
      TAU2_TASK_SET="$domain" \
      TAU2_TASK_SPLIT=auto \
      NUM_TASKS=1 \
      RUN_NAME="$smoke_name" \
      "${ADAPTER_DIR}/scripts/run_adapter.sh" --display plain
    "${ADAPTER_DIR}/.venv/bin/python" \
      "${ADAPTER_DIR}/scripts/validate_run.py" \
      "${CHECKOUT_DIR}/runs/${smoke_name}" 1
  done
fi

"${ADAPTER_DIR}/scripts/run_full_core.sh"
