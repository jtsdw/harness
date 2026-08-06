#!/usr/bin/env bash
# Checks whether EAGLE-3 speculative decoding is actually active on a running server started with
# ./scripts/serve_eagle3.sh -- a server starting successfully does NOT by itself confirm the
# draft model loaded and is being used, the same way it didn't for ngram mode on the old vLLM
# (see docs/toolspec_vllm_speculative_comparison.md).
#
# UNTESTED (see ../README.md): this checks the two signals that seem most likely to exist based
# on vLLM's own docs -- (1) the startup log mentioning the speculative config, (2) whatever
# /metrics exposes now (this project's old vLLM pin had zero spec-decode-specific metrics, see
# docs/toolspec_vllm_speculative_comparison.md's "一个真实发现" section -- newer vLLM may or may
# not be better here, this script doesn't assume either way, it just reports what it finds).
# Neither of these alone proves real speedup -- for that, run the same BFCL comparison
# methodology already used for ngram (docs/toolspec_vllm_speculative_comparison.md), swapping in
# this project's serve_baseline.sh/serve_eagle3.sh instead of local-model-server's.
#
# Optional overrides:
#   VLLM_BASE_URL  (default: http://localhost:8000/v1)
#   MODEL          (default: Qwen/Qwen3-32B -- must match whatever the server is serving)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${VLLM_BASE_URL:=http://localhost:8000/v1}"
: "${MODEL:=Qwen/Qwen3-32B}"

if ! curl -s -m 3 "${VLLM_BASE_URL}/models" >/dev/null 2>&1; then
  echo "ERROR: no vLLM server reachable at ${VLLM_BASE_URL}." >&2
  echo "  Start one with EAGLE-3 first: ./scripts/serve_eagle3.sh" >&2
  exit 1
fi

echo "=== [1] Startup log: does it mention the speculative config? ==="
if grep -iE "speculative|eagle" logs/vllm_server.log 2>/dev/null; then
  echo "(found the lines above -- read them, don't just trust that a match exists)"
else
  echo "NOTHING FOUND mentioning speculative/eagle in logs/vllm_server.log -- this is a bad sign,"
  echo "the server may have silently ignored --speculative-config. Check the full log."
fi

echo
echo "=== [2] /metrics: what spec-decode-related fields (if any) does this vLLM version expose? ==="
METRICS_URL="${VLLM_BASE_URL%/v1}/metrics"
SPEC_METRICS="$(curl -s -m 3 "$METRICS_URL" 2>/dev/null | grep -iE "spec|draft|accept" || true)"
if [[ -n "$SPEC_METRICS" ]]; then
  echo "$SPEC_METRICS"
else
  echo "No spec/draft/accept-related metric names found on $METRICS_URL."
  echo "(The old vLLM this project's dev machine uses has none at all either -- see"
  echo "docs/toolspec_vllm_speculative_comparison.md. Not necessarily a problem, just means"
  echo "you'll need the same tokens/s-comparison methodology used for ngram, not a built-in metric.)"
fi

echo
echo "=== [3] Real generation request (read the output, does it look coherent -- lossy bugs in"
echo "    speculative decoding show up as garbled/repeated text, not just as an HTTP error) ==="
curl -s "${VLLM_BASE_URL}/chat/completions" -H "Content-Type: application/json" -d @- <<EOF | python3 -m json.tool
{
  "model": "${MODEL}",
  "messages": [{"role": "user", "content": "Count from 1 to 20, one number per line."}],
  "max_tokens": 200,
  "temperature": 0
}
EOF
