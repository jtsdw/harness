#!/usr/bin/env bash
# Run the five tau2 core selections sequentially with per-domain logs,
# validation, and resumable output directories. Suitable for local or NSCC use.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${PROJECT_DIR}/.." && pwd)"
RUNNER="${TAU2_RUNNER:-${SCRIPT_DIR}/run_adapter.sh}"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
VALIDATOR="${SCRIPT_DIR}/validate_run.py"

: "${FULL_RUN_ID:=tau2_core_$(date +%Y%m%d_%H%M%S)}"
: "${TAU2_DOMAINS:=mock airline retail telecom telecom-workflow}"
: "${RESUME:=1}"
: "${TAU2_CONTROLLER_DIR:=${REPO_ROOT}/runs/${FULL_RUN_ID}}"

if [[ ! -x "$RUNNER" ]]; then
  echo "ERROR: tau2 runner is not executable: $RUNNER" >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: adapter environment is missing; run scripts/setup_tau2_bench.sh first." >&2
  exit 1
fi

declare -A EXPECTED_TASKS=(
  [mock]=10
  [airline]=50
  [retail]=114
  [telecom]=114
  [telecom-workflow]=114
)
read -r -a domains <<<"$TAU2_DOMAINS"

controller_dir="$TAU2_CONTROLLER_DIR"
status_file="${controller_dir}/domain-status.tsv"
mkdir -p "$controller_dir"
printf 'domain\texpected_tasks\tprocess_exit\tvalidation_exit\tresult\n' >"$status_file"

validate_run() {
  local run_dir="$1"
  local expected="$2"
  "$PYTHON" "$VALIDATOR" "$run_dir" "$expected"
}

overall_status=0
echo "Starting tau2 core controller"
echo "  run id:  $FULL_RUN_ID"
echo "  domains: $TAU2_DOMAINS"
echo "  resume:  $RESUME"
echo "  status:  $status_file"

for domain in "${domains[@]}"; do
  expected="${EXPECTED_TASKS[$domain]:-}"
  if [[ -z "$expected" ]]; then
    echo "ERROR: unsupported core domain: $domain" >&2
    exit 2
  fi

  safe_domain="${domain//-/_}"
  run_name="${FULL_RUN_ID}_${safe_domain}"
  run_dir="${REPO_ROOT}/runs/${run_name}"
  console_log="${controller_dir}/${safe_domain}.console.log"

  if [[ "$RESUME" == "1" && -d "$run_dir" ]] && validate_run "$run_dir" "$expected"; then
    echo "Skipping already validated domain: $domain"
    printf '%s\t%s\tskipped\t0\tok\n' "$domain" "$expected" >>"$status_file"
    continue
  fi

  echo
  echo "[$(date --iso-8601=seconds)] Starting $domain ($expected tasks)"
  TAU2_DOMAIN="$domain" \
    TAU2_TASK_SET="$domain" \
    TAU2_TASK_SPLIT=auto \
    NUM_TASKS=all \
    RUN_NAME="$run_name" \
    "$RUNNER" --display plain "$@" \
    2>&1 | tee "$console_log"
  process_status=${PIPESTATUS[0]}

  validation_status=0
  if ! validate_run "$run_dir" "$expected"; then
    validation_status=1
  fi

  if ((process_status == 0 && validation_status == 0)); then
    result=ok
  else
    result=failed
    overall_status=1
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$domain" "$expected" "$process_status" "$validation_status" "$result" \
    >>"$status_file"
done

echo
echo "Tau2 core controller finished:"
column -t -s $'\t' "$status_file" 2>/dev/null || sed -n '1,20p' "$status_file"
exit "$overall_status"
