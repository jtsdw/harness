#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

require_dir "${TAU2_DATA_DIR}/tau2/domains"
require_file "$TASK_FILE"
require_file "${TAU2_ADAPTER_DIR}/pyproject.toml"

health_url="${VLLM_BASE_URL%/v1}/health"
models_url="${VLLM_BASE_URL}/models"

echo "Checking vLLM health at ${health_url}..."
curl -fsS --max-time 5 "$health_url" >/dev/null

echo "Checking served model at ${models_url}..."
model_catalog="$(curl -fsS --max-time 5 \
  -H "Authorization: Bearer ${VLLM_API_KEY}" \
  "$models_url")"

MODEL_CATALOG="$model_catalog" EXPECTED_MODEL="$MODEL_NAME" \
  uv run --project "$TAU2_ADAPTER_DIR" python - <<'PY'
import json
import os

payload = json.loads(os.environ["MODEL_CATALOG"])
expected = os.environ["EXPECTED_MODEL"]
models = [item.get("id") for item in payload.get("data", [])]
if expected not in models:
    raise SystemExit(f"expected model {expected!r}, served models are {models!r}")
print("served model:", expected)
PY

echo "Checking vLLM metrics at ${INSPECT_TRACE_VLLM_METRICS_URL}..."
metrics_catalog="$(curl -fsS --max-time 5 "$INSPECT_TRACE_VLLM_METRICS_URL")"
grep -q '^# HELP vllm:' <<<"$metrics_catalog"

METRICS_CATALOG="$metrics_catalog" \
  uv run --project "$TAU2_ADAPTER_DIR" python - <<'PY'
import os
import re

metrics = os.environ["METRICS_CATALOG"]
busy = {}
for name in ("running", "waiting"):
    match = re.search(
        rf"^vllm:num_requests_{name}\{{[^}}]*\}}\s+([0-9.]+)$",
        metrics,
        re.MULTILINE,
    )
    busy[name] = float(match.group(1)) if match else 0.0
if any(busy.values()):
    raise SystemExit(
        "vLLM is not idle before evaluation: "
        f"running={busy['running']:g}, waiting={busy['waiting']:g}"
    )
print("vLLM queue: idle")
PY

echo "Checking adapter imports and selected tau2 domain..."
TAU2_PREFLIGHT_DOMAIN="$TAU2_DOMAIN" \
TAU2_PREFLIGHT_TASK_SET="$TAU2_TASK_SET" \
TAU2_PREFLIGHT_TASK_SPLIT="$TAU2_TASK_SPLIT" \
uv run --project "$TAU2_ADAPTER_DIR" python - <<'PY'
import os

from tau2_adapter.agent import InspectAIAgent
from tau2_adapter.runtime import (
    build_domain_environment,
    load_domain_tasks,
    resolved_selection,
)

domain = os.environ["TAU2_PREFLIGHT_DOMAIN"]
task_set = os.environ["TAU2_PREFLIGHT_TASK_SET"] or None
task_split = os.environ["TAU2_PREFLIGHT_TASK_SPLIT"]
tasks = load_domain_tasks(domain, task_set, task_split)
resolved_task_set, resolved_task_split = resolved_selection(
    domain, task_set, task_split
)
environment = build_domain_environment(domain, tasks[0])
assert tasks, f"{domain} task set is empty"
assert environment.get_tools(), f"{domain} environment exposes no agent tools"
try:
    user_tool_count = len(environment.get_user_tools())
except ValueError:
    user_tool_count = 0
print(f"domain: {domain}")
print(f"task set: {resolved_task_set}")
print(f"task split: {resolved_task_split or 'all'}")
print(f"tasks: {len(tasks)}")
print(f"agent tools: {len(environment.get_tools())}")
print(f"user tools: {user_tool_count}")
print("InspectAIAgent:", InspectAIAgent.__name__)
PY

echo "Preflight complete."
