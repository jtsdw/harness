#!/usr/bin/env bash
# One-shot environment setup for serving a local model with vLLM on this machine.
#
# Why this script exists (read before changing version pins): this machine's NVIDIA driver
# (535.230.02) only supports up to CUDA 12.2. `uv add vllm` with no version pin resolves to the
# latest vllm, which pulls torch built against a much newer CUDA (12.8+) -- that fails at engine
# startup with "The NVIDIA driver on your system is too old". vllm==0.6.3.post1 is the newest
# release whose declared torch pin (2.4.0) still defaults to a CUDA-12.1 wheel, which this driver
# can run. Do not bump vllm/torch here without first checking `nvidia-smi` for the driver's CUDA
# ceiling and confirming the target torch version's default PyPI wheel doesn't exceed it.
#
# Two more things this script works around, both discovered the hard way:
#  1. `uv add`'s incremental resolution left `transformers`/`torchaudio` at whatever loose-latest
#     version satisfied vllm's ">=" constraints (5.14.1 / 2.11.0) instead of the 0.6.3.post1-era
#     versions -- those pull in a torchaudio extension that wants libcudart.so.13, which doesn't
#     exist for a CUDA-12.1 install. Pinned explicitly in pyproject.toml.
#  2. `pyairports==0.0.1` on PyPI is a metadata-only wheel with no actual code (a long-abandoned
#     package), but `outlines` (an indirect vllm dependency, needed even for plain unstructured
#     generation) imports `pyairports.airports.AIRPORT_LIST` unconditionally. Replaced with a local
#     shim package (vendor/pyairports/) backed by the `airportsdata` package, wired in via
#     `[tool.uv.sources]` in pyproject.toml.
#  3. Some of the large nvidia-*-cu12 wheels (cudnn, nccl specifically, so far) have installed from
#     a corrupted/incomplete cache at least once on this machine, leaving empty stub directories
#     with no actual .so files -- silent until a request actually exercises that code path. This
#     script checks every nvidia-*-cu12 package for at least one .so file after sync and force
#     reinstalls (--no-cache) any that come up empty.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  python3 -m pip install --user uv
fi

echo "== nvidia-smi driver/CUDA ceiling =="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
nvidia-smi | grep "CUDA Version" || true
echo

echo "== uv sync =="
uv sync

echo
echo "== verifying every nvidia-*-cu12 package actually shipped its .so files =="
broken=()
for dir in .venv/lib/python3.12/site-packages/nvidia/*/; do
  name="$(basename "$dir")"
  count=$(find "$dir" -iname "*.so*" 2>/dev/null | wc -l)
  if [[ "$count" -eq 0 ]]; then
    echo "  BROKEN (0 .so files): $name"
    broken+=("$name")
  fi
done

if [[ ${#broken[@]} -gt 0 ]]; then
  echo
  echo "Reinstalling ${#broken[@]} broken package(s) with --no-cache: ${broken[*]}"
  for name in "${broken[@]}"; do
    pkg="nvidia-${name//_/-}-cu12"
    version=$(uv pip show "$pkg" 2>/dev/null | awk '/^Version:/{print $2}')
    uv pip install --reinstall --no-cache "${pkg}==${version}"
  done
  echo
  echo "Re-checking..."
  for dir in .venv/lib/python3.12/site-packages/nvidia/*/; do
    name="$(basename "$dir")"
    count=$(find "$dir" -iname "*.so*" 2>/dev/null | wc -l)
    if [[ "$count" -eq 0 ]]; then
      echo "  STILL BROKEN: $name -- rerun this script, or investigate manually." >&2
      exit 1
    fi
  done
  echo "All good after reinstall."
else
  echo "All nvidia-*-cu12 packages have real .so files."
fi

echo
echo "== sanity check: torch CUDA + cuDNN both actually work =="
uv run python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print('torch', torch.__version__, 'cuda', torch.version.cuda, '-', torch.cuda.get_device_name(0))
x = torch.randn(2000, 2000, device='cuda')
(x @ x).sum().item()
import torch.nn.functional as F
c = torch.randn(8, 16, 32, 32, device='cuda')
w = torch.randn(16, 16, 3, 3, device='cuda')
F.conv2d(c, w, padding=1).sum().item()
print('matmul + cudnn conv both OK')
"

echo
echo "Setup complete. Next: ./scripts/serve.sh"
