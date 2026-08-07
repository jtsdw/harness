#!/usr/bin/env bash
# Fires real requests against a running vLLM server and prints the raw response verbatim, to
# settle what requirement B1's per-request metrics actually look like on this vLLM version --
# see docs/next_phase_requirements.md's B1 ("是否启用、版本下具体响应结构和性能开销必须现场验证，
# 不能只根据文档假设"). This is a discovery script, not a working recipe: the official vLLM
# per-request-metrics doc page kept 429-ing (rate-limited) both times it was tried.
#
# Since then, vLLM's actual GitHub source (main branch, 2026-08-07) was read directly instead --
# see inspect_trace/src/inspect_trace/vllm_per_request_metrics.py's module docstring for the full
# story. What that found: `ChatCompletionResponse.metrics` is a real, non-opt-in field (populated
# automatically, `None` when unavailable) holding a `PerRequestTimingMetrics` object with
# `time_to_first_token_ms`/`queue_time_ms`/`generation_time_ms`/`mean_itl_ms`/`tokens_per_second`.
# That's a real step up from a blind guess, but it's still `main` branch today, not necessarily
# the exact version `nscc_model_server`'s `vllm>=0.9.0` floor resolves to on the real node -- so
# step [1] below is now the important one (does `response["metrics"]` actually look like that),
# step [2]'s flag is kept mostly as a harmless cross-check since the mechanism doesn't look
# opt-in anymore.
#
# What to do with the output: paste it back so vllm_per_request_metrics.py's field-candidate list
# (currently ordered with these GitHub-sourced names first, older blind guesses as fallback) can
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

PLAIN_HEADERS_FILE="$(mktemp)"
FLAGGED_HEADERS_FILE="$(mktemp)"
trap 'rm -f "$PLAIN_HEADERS_FILE" "$FLAGGED_HEADERS_FILE"' EXIT

echo "=== [1] Plain request -- look for a top-level \"metrics\" key. If it's there, does it have"
echo "    time_to_first_token_ms / queue_time_ms / generation_time_ms / mean_itl_ms /"
echo "    tokens_per_second (the field names found in vLLM's GitHub source, see this script's"
echo "    header), or something else? This is the important check now, not [2] below. ==="
PLAIN_RESPONSE="$(curl -s -D "$PLAIN_HEADERS_FILE" \
  "${VLLM_BASE_URL}/chat/completions" -H "Content-Type: application/json" -d @- <<EOF
{
  "model": "${MODEL}",
  "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
  "max_tokens": 20,
  "temperature": 0
}
EOF
)"
echo "$PLAIN_RESPONSE" | python3 -m json.tool
echo
echo "--- response headers ---"
cat "$PLAIN_HEADERS_FILE"

echo
echo "=== [2] Same request with a guessed flag (extra_body: return_metrics=true) -- kept as a"
echo "    cross-check even though the real mechanism looks non-opt-in (see header): if [1]"
echo "    already had \"metrics\" populated, this should look identical; if [1] didn't, this"
echo "    flag very likely won't change that either, but costs nothing to confirm. ==="
FLAGGED_RESPONSE="$(curl -s -D "$FLAGGED_HEADERS_FILE" \
  "${VLLM_BASE_URL}/chat/completions" -H "Content-Type: application/json" -d @- <<EOF
{
  "model": "${MODEL}",
  "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
  "max_tokens": 20,
  "temperature": 0,
  "return_metrics": true
}
EOF
)"
echo "$FLAGGED_RESPONSE" | python3 -m json.tool
echo
echo "--- response headers ---"
cat "$FLAGGED_HEADERS_FILE"

echo
echo "=== [3] Top-level key diff between [1] and [2] (new keys here = the flag did something) ==="
python3 - "$PLAIN_RESPONSE" "$FLAGGED_RESPONSE" <<'PYEOF'
import json
import sys

plain = json.loads(sys.argv[1])
flagged = json.loads(sys.argv[2])
plain_keys = set(plain.keys()) | set((plain.get("choices") or [{}])[0].keys())
flagged_keys = set(flagged.keys()) | set((flagged.get("choices") or [{}])[0].keys())
new_keys = flagged_keys - plain_keys
if new_keys:
    print(f"New top-level/choice keys with the flag set: {sorted(new_keys)}")
else:
    print("No new top-level/choice keys -- the guessed 'return_metrics' flag likely isn't the")
    print("real opt-in mechanism (or per-request metrics are on by default and [1] already had")
    print("everything -- re-check [1]'s output above by hand).")
PYEOF
