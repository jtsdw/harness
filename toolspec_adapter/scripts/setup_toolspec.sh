#!/usr/bin/env bash
# Sets up the ToolSpec repo's own venv (needed by both the native reproduction and by this
# adapter, which imports ToolSpec's real model/evaluation code via TOOLSPEC_REPO_DIR at runtime
# rather than reimplementing it -- see ../src/toolspec_adapter/provider.py's module docstring).
#
# Idempotent: re-running skips steps that already succeeded.
#
# Optional overrides:
#   TOOLSPEC_REPO_DIR  (default: /home/liuyingen/code/ToolSpec)
#   TOOLSPEC_REPO_URL  (default: unset -- only used if TOOLSPEC_REPO_DIR doesn't exist yet)

set -euo pipefail

: "${TOOLSPEC_REPO_DIR:=/home/liuyingen/code/ToolSpec}"
: "${TOOLSPEC_REPO_URL:=}"

export PATH="$HOME/.local/bin:$PATH"

if [[ ! -d "$TOOLSPEC_REPO_DIR" ]]; then
  if [[ -z "$TOOLSPEC_REPO_URL" ]]; then
    echo "ERROR: $TOOLSPEC_REPO_DIR does not exist and TOOLSPEC_REPO_URL is not set." >&2
    echo "  Either point TOOLSPEC_REPO_DIR at an existing checkout, or set TOOLSPEC_REPO_URL to clone from." >&2
    exit 1
  fi
  echo "Cloning ToolSpec into $TOOLSPEC_REPO_DIR ..."
  git clone "$TOOLSPEC_REPO_URL" "$TOOLSPEC_REPO_DIR"
fi

cd "$TOOLSPEC_REPO_DIR"

if [[ ! -d .venv ]]; then
  echo "Creating ToolSpec venv (Python 3.12) ..."
  uv venv --python 3.12 .venv
fi

# The GPU driver on this machine caps at CUDA 12.2 (see docs/local_model_deployment.md) -- torch's
# default `pip install torch==2.5.1` resolves a CUDA 12.4 wheel, which the driver rejects. Pin to
# the cu121 index explicitly, same fix as local-model-server's setup.sh.
if ! .venv/bin/python -c "import torch" 2>/dev/null; then
  echo "Installing torch==2.5.1 (cu121 wheel) ..."
  uv pip install --python .venv/bin/python torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
fi

echo "Installing the rest of requirements.txt ..."
uv pip install --python .venv/bin/python \
  transformers==4.51.1 accelerate==1.13.0 fschat==0.2.31 gradio==3.50.2 \
  openai==0.28.0 anthropic==0.5.0 sentencepiece==0.2.0 protobuf==3.19.0 \
  datasets==3.4.1 shortuuid tqdm

echo "Checking CUDA is visible ..."
.venv/bin/python -c "
import torch
print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
"

echo
echo "Done. ToolSpec repo ready at $TOOLSPEC_REPO_DIR"
echo "Next: ./run_native_repro.sh  (native-repo baseline vs toolspec reproduction)"
echo "  or: ./run_adapter.sh       (same comparison run inside our inspect_ai harness)"
