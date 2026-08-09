#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export TAU2_DOMAIN=mock
export TAU2_TASK_SET=mock
export TAU2_TASK_SPLIT=auto
: "${RUN_NAME:=tau2_mock_qwen27b_$(date +%Y%m%d_%H%M%S)}"
export RUN_NAME

exec "${SCRIPT_DIR}/run_tau.sh" "$@"
