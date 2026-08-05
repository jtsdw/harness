#!/usr/bin/env bash
# vLLM baseline: no native tool-calling, no speculative decoding. This is the most
# battle-tested configuration across this project's real benchmark runs (emulate_tools=true
# on the inspect_ai side). See ../README.md's script table for the other variants.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec ./scripts/serve.sh
