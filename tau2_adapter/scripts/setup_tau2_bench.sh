#!/usr/bin/env bash
# One-shot setup for the tau2-bench dependency: fetch the pinned upstream commit,
# apply the local compatibility patch, and synchronize the tau2 and adapter
# Python 3.12 environments. The version check makes repeated runs deterministic.
#
# This does NOT touch efficient-harness's own git history -- tau2-bench is a separate upstream
# repo (github.com/sierra-research/tau2-bench) we depend on via a path dependency
# (tau2_adapter/pyproject.toml's [tool.uv.sources]), not something we vendor into this repo. The
# patch file (tau2_bench_bug2_fix.patch, tracked here) is the only piece of tau2-bench-specific
# state this repo owns.
#
# Usage:
#   ./scripts/setup_tau2_bench.sh
#
# Env vars:
#   TAU2_BENCH_REPO   Upstream repository
#                     (default: https://github.com/sierra-research/tau2-bench.git)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"
TAU2_SOURCE_DIR="${REPO_ROOT}/.deps/tau2-bench"
: "${TAU2_BENCH_REPO:=https://github.com/sierra-research/tau2-bench.git}"
TAU2_BENCH_REF="a1e85084a3960281cb06997594133e8f39ea42a7"

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found. Install it first: python3 -m pip install --user uv" >&2
  exit 1
fi

if [[ ! -e "$TAU2_SOURCE_DIR" ]]; then
  echo "== Initializing tau2-bench source checkout at $TAU2_SOURCE_DIR =="
  mkdir -p "$TAU2_SOURCE_DIR"
  git -C "$TAU2_SOURCE_DIR" init
  git -C "$TAU2_SOURCE_DIR" remote add origin "$TAU2_BENCH_REPO"
elif [[ ! -d "${TAU2_SOURCE_DIR}/.git" ]]; then
  echo "ERROR: $TAU2_SOURCE_DIR is not a Git checkout." >&2
  echo "Move that path aside, then rerun this setup script." >&2
  exit 1
fi

actual_ref="$(git -C "$TAU2_SOURCE_DIR" rev-parse --verify HEAD 2>/dev/null || true)"
if [[ -z "$actual_ref" ]]; then
  remote_url="$(git -C "$TAU2_SOURCE_DIR" remote get-url origin 2>/dev/null || true)"
  if [[ -z "$remote_url" ]]; then
    git -C "$TAU2_SOURCE_DIR" remote add origin "$TAU2_BENCH_REPO"
  elif [[ "$remote_url" != "$TAU2_BENCH_REPO" ]]; then
    echo "ERROR: tau2-bench origin is $remote_url; expected $TAU2_BENCH_REPO." >&2
    exit 1
  fi
  echo "== Fetching pinned tau2-bench commit $TAU2_BENCH_REF =="
  git -C "$TAU2_SOURCE_DIR" sparse-checkout init --cone
  git -C "$TAU2_SOURCE_DIR" sparse-checkout set src
  git -C "$TAU2_SOURCE_DIR" fetch --depth 1 --filter=blob:none origin "$TAU2_BENCH_REF"
  git -C "$TAU2_SOURCE_DIR" checkout --detach FETCH_HEAD
  actual_ref="$(git -C "$TAU2_SOURCE_DIR" rev-parse HEAD)"
fi

if [[ "$actual_ref" != "$TAU2_BENCH_REF" ]]; then
  echo "ERROR: tau2-bench is at $actual_ref; expected $TAU2_BENCH_REF." >&2
  echo "Move the existing checkout aside, then rerun this setup script." >&2
  exit 1
fi

echo
echo "== Applying Bug 2 fix (extra 'name' field in to_litellm_messages) =="
if (cd "$TAU2_SOURCE_DIR" && git apply --check "$SCRIPT_DIR/tau2_bench_bug2_fix.patch" 2>/dev/null); then
  (cd "$TAU2_SOURCE_DIR" && git apply "$SCRIPT_DIR/tau2_bench_bug2_fix.patch")
  echo "Patch applied."
elif (cd "$TAU2_SOURCE_DIR" && git apply --reverse --check "$SCRIPT_DIR/tau2_bench_bug2_fix.patch" 2>/dev/null); then
  echo "Patch already applied, skipping."
else
  echo "WARNING: patch didn't apply cleanly and doesn't look already-applied either." >&2
  echo "tau2-bench's own source may have changed upstream -- check src/tau2/utils/llm_utils.py" >&2
  echo "manually against docs/tau2_bench_integration_findings.md's Bug 2 description." >&2
  exit 1
fi

echo
echo "== Synchronizing tau2-bench (Python 3.12) =="
(cd "$TAU2_SOURCE_DIR" && uv sync --frozen --python 3.12)

echo
echo "== Synchronizing tau2_adapter (Python 3.12) =="
(cd "$PROJECT_DIR" && uv sync --frozen --extra dev --python 3.12)

echo
echo "Setup complete. tau2-bench source is ready at $TAU2_SOURCE_DIR."
