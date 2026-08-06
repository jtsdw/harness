#!/usr/bin/env bash
# One-time environment setup for nscc_model_server -- see ../README.md for why this is a separate
# project from ../../local-model-server/ (short version: that one's vllm==0.6.3.post1 pin exists
# only to satisfy THIS dev machine's old GPU driver, which doesn't apply on the NSCC H100 node).
#
# Idempotent -- safe to re-run.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found. Install it first: python3 -m pip install --user uv" >&2
  exit 1
fi

echo "Checking GPU/driver ..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
  echo "WARNING: nvidia-smi not found -- can't confirm GPU/driver before installing." >&2
fi

echo
echo "Installing dependencies (vllm>=0.9.0, no upper pin -- this project intentionally does NOT"
echo "inherit local-model-server's old CUDA-12.2-driver-constrained pin, see ../README.md) ..."
uv sync

echo
echo "Installed vllm version (this is real ground truth -- report this back, the exact"
echo "--speculative-config syntax in serve_eagle3.sh was written from documentation, not"
echo "verified against whatever version actually resolves here):"
uv run vllm --version

echo
echo "Done. Next: ./scripts/serve_baseline.sh or ./scripts/serve_eagle3.sh"
