#!/usr/bin/env bash
# Pulls the NSCC compute node's runs/ (real .eval logs + inspect_trace JSONL, gitignored --
# doesn't travel with git) back to this local machine, into nscc_runs/ -- deliberately NOT into
# this repo's own runs/, which is for anything executed directly on this machine. See
# nscc_runs/README.md for why that split exists. Once pulled, the dashboard-generation scripts
# (inspect_trace/scripts/build_*.py) can point at nscc_runs/<name>/ same as they would runs/<name>/.
#
# Modeled directly on /home/liuyingen/code/quant/nscc2local.sh (same cluster, same rsync
# preview/pull/delete pattern) -- see docs/remote_compute_workflow.md for the full picture of
# how code (git) and data (this script) flow between the two nodes.
#
# 2026-08-08 real finding: the default used to assume each team member keeps a separate checkout
# under scratch/ (harness-$(whoami), per docs/team_collaboration.md's stated convention), but the
# actual real checkout on the NSCC node right now is a single shared one at plain
# scratch/harness/ (no per-user suffix) -- confirmed against a real terminal prompt
# (`~/scratch/harness/nscc_model_server/scripts$`). Defaulting to that; override
# NSCC_REMOTE_SUBDIR if/when the team actually moves to separate per-person checkouts.
#
# Usage:
#   ./scripts/pull_runs.sh preview   # dry run, see what would transfer
#   ./scripts/pull_runs.sh pull      # actual sync
#   ./scripts/pull_runs.sh delete    # actual sync, and delete local files no longer on remote

set -euo pipefail

REMOTE_HOST="${NSCC_SSH_ALIAS:-nscc}"
REMOTE_SUBDIR="${NSCC_REMOTE_SUBDIR:-harness}"
REMOTE_DIR="/home/users/ntu/n2505716/scratch/${REMOTE_SUBDIR}/runs/"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/nscc_runs/"

MODE="${1:-preview}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/pull_runs.sh preview   # dry run
  ./scripts/pull_runs.sh pull      # actual sync
  ./scripts/pull_runs.sh delete    # actual sync and delete local extra files

Env vars:
  NSCC_SSH_ALIAS     SSH alias/host for the compute node (default: nscc, see ~/.ssh/config)
  NSCC_REMOTE_SUBDIR Checkout subdirectory name under scratch/ (default: harness -- the actual
                      shared checkout confirmed on the real node 2026-08-08; override if your
                      setup uses a different/per-person subdirectory)
EOF
}

if ! command -v rsync >/dev/null 2>&1; then
  echo "Error: rsync is not installed." >&2
  exit 1
fi

mkdir -p "$LOCAL_DIR"

RSYNC_ARGS=(
  "-avzP"
  "--info=progress2"
)

case "$MODE" in
  preview)
    RSYNC_ARGS+=("-n")
    ;;
  pull)
    ;;
  delete)
    RSYNC_ARGS+=("--delete")
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "Error: unknown mode: $MODE" >&2
    usage >&2
    exit 1
    ;;
esac

echo "Mode: $MODE"
echo "From: ${REMOTE_HOST}:${REMOTE_DIR}"
echo "To:   $LOCAL_DIR"
echo

rsync "${RSYNC_ARGS[@]}" \
  "${REMOTE_HOST}:${REMOTE_DIR}" \
  "$LOCAL_DIR"
