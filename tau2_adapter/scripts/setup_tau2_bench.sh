#!/usr/bin/env bash
# One-shot setup for the tau2-bench dependency: clone it if missing, pin Python 3.12 (tau2's
# voice module unconditionally imports `audioop`, which Python 3.13 removed -- see
# docs/tau2_bench_integration_findings.md, environment section), sync its own venv, and apply the
# local bug fix for the extra "name" field in to_litellm_messages() (Bug 2 in the same doc) --
# idempotent, safe to re-run after a fresh clone or a `git pull` inside tau2-bench.
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
#   TAU2_BENCH_DIR   Where tau2-bench lives (default: /home/liuyingen/code/tau2-bench). Cloned
#                     here if it doesn't exist yet.
#   TAU2_BENCH_REPO   Upstream repo to clone if missing
#                     (default: https://github.com/sierra-research/tau2-bench.git)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${TAU2_BENCH_DIR:=/home/liuyingen/code/tau2-bench}"
: "${TAU2_BENCH_REPO:=https://github.com/sierra-research/tau2-bench.git}"

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found. Install it first: python3 -m pip install --user uv" >&2
  exit 1
fi

if [[ ! -d "$TAU2_BENCH_DIR" ]]; then
  echo "== Cloning tau2-bench to $TAU2_BENCH_DIR =="
  git clone "$TAU2_BENCH_REPO" "$TAU2_BENCH_DIR"
else
  echo "== tau2-bench already present at $TAU2_BENCH_DIR, skipping clone =="
fi

echo
echo "== uv sync (Python 3.12 -- 3.13 removed the audioop module tau2's voice import chain needs) =="
(cd "$TAU2_BENCH_DIR" && uv sync --python 3.12)

echo
echo "== Applying Bug 2 fix (extra 'name' field in to_litellm_messages) =="
if (cd "$TAU2_BENCH_DIR" && git apply --check "$SCRIPT_DIR/tau2_bench_bug2_fix.patch" 2>/dev/null); then
  (cd "$TAU2_BENCH_DIR" && git apply "$SCRIPT_DIR/tau2_bench_bug2_fix.patch")
  echo "Patch applied."
elif (cd "$TAU2_BENCH_DIR" && git apply --reverse --check "$SCRIPT_DIR/tau2_bench_bug2_fix.patch" 2>/dev/null); then
  echo "Patch already applied, skipping."
else
  echo "WARNING: patch didn't apply cleanly and doesn't look already-applied either." >&2
  echo "tau2-bench's own source may have changed upstream -- check src/tau2/utils/llm_utils.py" >&2
  echo "manually against docs/tau2_bench_integration_findings.md's Bug 2 description." >&2
  exit 1
fi

echo
echo "Setup complete. tau2-bench is ready at $TAU2_BENCH_DIR."
echo "Next: cd /path/to/efficient-harness/tau2_adapter && uv sync --extra dev --python 3.12"
