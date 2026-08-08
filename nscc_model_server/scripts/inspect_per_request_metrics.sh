#!/usr/bin/env bash
# Fires a real request against a running vLLM server and prints the raw response verbatim, to
# settle what requirement B1's per-request metrics actually look like on this vLLM version --
# see docs/next_phase_requirements.md's B1 ("是否启用、版本下具体响应结构和性能开销必须现场验证，
# 不能只根据文档假设").
#
# History, since the mechanism took two real findings to pin down: the official vLLM
# per-request-metrics doc page kept 429-ing, so 2026-08-07 this switched to reading vLLM's GitHub
# source instead -- found `ChatCompletionResponse.metrics: PerRequestTimingMetrics | None` with
# fields `time_to_first_token_ms`/`queue_time_ms`/`generation_time_ms`/`mean_itl_ms`/
# `tokens_per_second`, but guessed (wrongly) that it was populated automatically. A real run
# against the actual NSCC node on 2026-08-08 (`vllm-0.26.0-8cfe525c`) got back `"metrics": null`
# -- confirming the container key is real but showing the automatic-population guess was wrong.
# Reading further into vLLM's serving code found the real answer: `metrics` is only populated when
# the server is started with `--enable-per-request-metrics` (a server-side CLI flag, not a
# request-side field -- the `return_metrics` request flag this script used to also try never did
# anything, confirmed empirically, so that test was removed). `nscc_model_server/scripts/serve.sh`
# now passes that flag.
#
# What's still unconfirmed: whether `metrics` actually comes back populated now that the flag is
# passed, and whether its field names really are what vLLM's GitHub source said. That's what this
# script checks. Paste the output back so vllm_per_request_metrics.py's field-candidate list can
# be corrected/confirmed against what this specific deployed version actually returns.
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
  echo "  Start one first: ./scripts/serve_eagle3.sh (or serve_baseline.sh)" >&2
  exit 1
fi

echo "=== Real request -- with --enable-per-request-metrics now passed by serve.sh, \"metrics\""
echo "    should no longer be null. Does it have time_to_first_token_ms / queue_time_ms /"
echo "    generation_time_ms / mean_itl_ms / tokens_per_second, or something else? ==="
curl -s "${VLLM_BASE_URL}/chat/completions" -H "Content-Type: application/json" -d @- <<EOF | python3 -m json.tool
{
  "model": "${MODEL}",
  "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
  "max_tokens": 20,
  "temperature": 0
}
EOF
