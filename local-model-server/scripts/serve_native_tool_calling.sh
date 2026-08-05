#!/usr/bin/env bash
# vLLM with native tool-calling (--enable-auto-tool-choice --tool-call-parser hermes) instead of
# client-side emulate_tools=true. NOTE: inspect_ai's stock openai-api provider still can't talk to
# this as-is (Bug 3 in docs/tau2_bench_integration_findings.md -- it adds a "strict" field this
# vLLM version rejects). Use -M emulate_tools=true anyway, or a custom ModelAPI provider that
# strips that field (see tau2_adapter/src/tau2_adapter/_registry.py for a working example).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
NATIVE_TOOL_CALLING=true exec ./scripts/serve.sh
