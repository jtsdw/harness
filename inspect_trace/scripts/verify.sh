#!/usr/bin/env bash
# Runs the three checks that must pass before any change to inspect_trace is considered done:
# tests, lint, and type-check. Exists so there's one command to run instead of three separately
# remembered ones -- run this after `uv sync --extra dev`, or after pulling changes on a new
# machine, to confirm the environment is actually working before trusting any real benchmark run.
#
# Usage:
#   cd efficient-harness/inspect_trace
#   ./scripts/verify.sh
#
# Exits non-zero on the first failing step (set -e), printing which step failed. All three steps
# are read-only except ruff, which is run with --fix (auto-fixes safe issues like import order;
# anything it can't fix on its own is reported as a normal error).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found. Install it first: python3 -m pip install --user uv" >&2
  exit 1
fi

echo "== 1/3: pytest =="
uv run pytest tests -q

echo
echo "== 2/3: ruff (format check + lint, --fix for safe auto-fixes) =="
uv run ruff format --check . || {
  echo "ruff format found unformatted files -- run 'uv run ruff format .' to fix, then re-run this script." >&2
  exit 1
}
uv run ruff check --fix .

echo
echo "== 3/3: mypy =="
uv run mypy --exclude tests/test_package src

echo
echo "All checks passed."
