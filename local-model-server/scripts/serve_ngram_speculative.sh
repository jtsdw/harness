#!/usr/bin/env bash
# vLLM with built-in n-gram/prompt-lookup speculative decoding (--speculative-model "[ngram]"),
# no separate draft model needed. This is the configuration used in
# docs/toolspec_vllm_speculative_comparison.md's real comparison against ToolSpec: 1.87x speedup
# vs vLLM's own baseline, but a real, worse-than-ToolSpec 23/100 output divergence rate -- read
# that doc before trusting this mode's outputs for anything correctness-sensitive.
#
# Optional overrides (same defaults as serve.sh):
#   NUM_SPECULATIVE_TOKENS, NGRAM_PROMPT_LOOKUP_MAX, NGRAM_PROMPT_LOOKUP_MIN
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SPECULATIVE_MODE=ngram exec ./scripts/serve.sh
