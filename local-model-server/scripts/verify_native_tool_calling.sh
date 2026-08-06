#!/usr/bin/env bash
# Verifies that a running vLLM server actually returns structured tool_calls, not just that it
# started -- for use after ./scripts/serve_native_tool_calling.sh. Bypasses inspect_ai entirely
# (its stock openai-api provider hits Bug 3 against this vLLM version -- see
# docs/tau2_bench_integration_findings.md), talking to the server's raw OpenAI-compatible HTTP
# API directly, which is exactly how Bug 1's original finding (this vLLM version DOES support
# --enable-auto-tool-choice, contrary to an earlier undocumented assumption) was confirmed.
#
# Does NOT start a server itself -- run ./scripts/serve_native_tool_calling.sh first.
#
# Optional overrides:
#   VLLM_BASE_URL  (default: http://localhost:8000/v1)
#   MODEL          (default: Qwen/Qwen2.5-3B-Instruct -- must match whatever the server is serving)

set -euo pipefail

: "${VLLM_BASE_URL:=http://localhost:8000/v1}"
: "${MODEL:=Qwen/Qwen2.5-3B-Instruct}"

if ! curl -s -m 3 "${VLLM_BASE_URL}/models" >/dev/null 2>&1; then
  echo "ERROR: no vLLM server reachable at ${VLLM_BASE_URL}." >&2
  echo "  Start one with native tool-calling first: ./scripts/serve_native_tool_calling.sh" >&2
  exit 1
fi

echo "Sending a request with a tool definition to ${VLLM_BASE_URL}/chat/completions ..."
RESPONSE="$(curl -s "${VLLM_BASE_URL}/chat/completions" -H "Content-Type: application/json" -d @- <<EOF
{
  "model": "${MODEL}",
  "messages": [{"role": "user", "content": "What is the weather in Beijing?"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get the current weather for a city",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
      }
    }
  }],
  "tool_choice": "auto"
}
EOF
)"

TOOL_CALLS="$(echo "$RESPONSE" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
except json.JSONDecodeError:
    print('INVALID_JSON')
    sys.exit(0)
choice = data.get('choices', [{}])[0]
tool_calls = choice.get('message', {}).get('tool_calls')
print(json.dumps(tool_calls) if tool_calls else 'NONE')
")"

echo
echo "Raw response:"
echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
echo

if [[ "$TOOL_CALLS" == "INVALID_JSON" ]]; then
  echo "FAIL: response wasn't valid JSON -- server may not be up or errored, see raw response above." >&2
  exit 1
elif [[ "$TOOL_CALLS" == "NONE" ]]; then
  echo "FAIL: no structured tool_calls in the response -- native tool-calling is NOT working." >&2
  echo "  (the model may have written the call as plain text in \"content\" instead -- check the raw response above)" >&2
  exit 1
else
  echo "PASS: server returned structured tool_calls: $TOOL_CALLS"
fi
