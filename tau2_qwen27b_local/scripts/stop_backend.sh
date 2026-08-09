#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

: "${QWEN_PACKAGE_ROOT:=${WORKSPACE_ROOT}/qwen36-offline-package}"
: "${QWEN_COMPOSE_DIR:=${QWEN_PACKAGE_ROOT}/source/qwen36-27b-single-5090/compose}"

base_compose="${QWEN_COMPOSE_DIR}/docker-compose.yml"
require_file "$base_compose"

docker compose \
  -f "$base_compose" \
  down
