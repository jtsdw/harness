#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
uv sync
uv pip install --python .venv/bin/python --no-deps openai==2.53.0 -e ../inspect_trace -e ../tau2_adapter
