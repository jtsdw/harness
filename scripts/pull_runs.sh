#!/usr/bin/env bash
# Pulls runs/ (real .eval logs + inspect_trace JSONL, gitignored -- doesn't travel with git) back
# from the NSCC compute node to this local machine, so the dashboard-generation scripts
# (inspect_trace/scripts/build_*.py) can read them and produce a real-data report locally.
#
# Modeled directly on /home/liuyingen/code/quant/nscc2local.sh (same cluster, same rsync
# preview/pull/delete pattern) -- see docs/remote_compute_workflow.md for the full picture of
# how code (git) and data (this script) flow between the two nodes.
#
# Each person on the team keeps their OWN checkout under scratch/ (see docs/team_collaboration.md
# for why) -- REMOTE_SUBDIR defaults to your local username so this points at your own checkout
# by default; override it if your remote directory is named differently.
#
# Usage:
#   ./scripts/pull_runs.sh preview   # dry run, see what would transfer
#   ./scripts/pull_runs.sh pull      # actual sync
#   ./scripts/pull_runs.sh delete    # actual sync, and delete local files no longer on remote

set -euo pipefail

REMOTE_HOST="${NSCC_SSH_ALIAS:-nscc}"
REMOTE_SUBDIR="${NSCC_REMOTE_SUBDIR:-harness-$(whoami)}"
REMOTE_DIR="/home/users/ntu/n2505716/scratch/${REMOTE_SUBDIR}/runs/"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/runs/"

MODE="${1:-preview}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/pull_runs.sh preview   # dry run
  ./scripts/pull_runs.sh pull      # actual sync
  ./scripts/pull_runs.sh delete    # actual sync and delete local extra files

Env vars:
  NSCC_SSH_ALIAS     SSH alias/host for the compute node (default: nscc, see ~/.ssh/config)
  NSCC_REMOTE_SUBDIR Your own checkout's subdirectory name under scratch/
                      (default: harness-$(whoami) -- each team member keeps a separate
                      checkout, see docs/team_collaboration.md)
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
