#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

: "${FULL_RUN_ID:=tau2_core_full_qwen27b_$(date +%Y%m%d_%H%M%S)}"
export FULL_RUN_ID

TAU2_RUNNER="${SCRIPT_DIR}/run_tau.sh" \
  exec "${TAU2_ADAPTER_DIR}/scripts/run_full_core.sh" "$@"
