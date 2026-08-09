#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

require_dir "$TAU2_ADAPTER_DIR"

exec "${TAU2_ADAPTER_DIR}/scripts/setup_tau2_bench.sh" "$@"
