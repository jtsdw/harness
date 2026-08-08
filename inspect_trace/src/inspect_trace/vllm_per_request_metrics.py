"""需求 B1/B2/B3: real per-request metrics read from the new vLLM's own per-request metrics API.

Unlike `vllm_metrics.py` (Prometheus histogram-delta, needs serialized execution to attribute
correctly and degrades to `ambiguous` under real concurrency -- see that module's docstring), this
reads whatever the model's own HTTP response carried for *this exact request*: no separate
`/metrics` scrape, no before/after window, no correlation guesswork. That is the entire point of
B1 ("不允许只靠时间窗口匹配").

**A real constraint this depends on, found while building this (not documented anywhere in this
project before now): inspect_ai only retains `ModelEvent.call.response` (the raw response dict) by
default for the first `DEFAULT_LOG_MODEL_API_CALLS` (=5, see
`inspect_ai.log._transcript.Transcript._process_event`) calls *per model* -- every call after that
gets `event.call = None`, and this applies to hooks too (`_process_event` prunes before
`_notify_subscribers`), not just to what ends up in the persisted `.eval` log.** None of this
project's existing run scripts set the flag that disables this, so every run before this file
existed effectively only had raw response data for a model's first 5 calls, silently. To get raw
response data for every call (required for this collector to produce anything beyond the first 5
calls per model), the eval run needs `log_model_api=True` in inspect_ai's `eval()`/`eval_set()`,
or equivalently the `INSPECT_EVAL_LOG_MODEL_API=true` environment variable -- `run_b5_matrix.sh`
sets this. Trade-off worth knowing: this keeps the full raw request+response payload for every
call in the transcript for the run's duration (not just written to disk), which is real memory/log
size overhead on top of whatever this collector's own parsing costs -- exactly the kind of thing
B4's "指标采集本身的 CPU/latency overhead" line item asks to have measured, not assumed away.

**Field names below come from reading vLLM's actual current source on GitHub (2026-08-07), not
docs.vllm.ai** (that page 429'd every time it was fetched -- both while first writing this module
and again just now). `vllm/entrypoints/openai/chat_completion/protocol.py`'s `ChatCompletionResponse`
and `ChatCompletionStreamResponse` both carry a `metrics: PerRequestTimingMetrics | None` field,
and `vllm/entrypoints/openai/engine/protocol.py`'s `PerRequestTimingMetrics` class has exactly
these fields, **in milliseconds**: `time_to_first_token_ms`, `generation_time_ms`,
`queue_time_ms`, `mean_itl_ms`, `tokens_per_second` (this last one already a rate, not
raw -- not derived from the others). There is no `prefill_time` field in that class at all in the
current source -- B4 itself hedges this one with "服务端版本可用时" (when the server version makes
it available), which now reads as prescient rather than just cautious. There's also nothing for
speculative draft/accepted tokens or guided-decoding overhead in this class; those candidates
below are kept as a long shot (maybe a different serving path or a version this project hasn't
seen exposes them), not because they were found anywhere.

**2026-08-08 update: partially confirmed against a real NSCC response.** `inspect_per_request_metrics.sh`
ran against a real server (`vllm-0.26.0-8cfe525c`) and got back a response with a top-level
`"metrics"` key -- the container name is real, not a guess. But its value was `null`, not the
object this module expected. Read the real serving code
(`vllm/entrypoints/openai/chat_completion/serving.py`) to find out why: `metrics` is only
populated when the server was started with `--enable-per-request-metrics` (a CLI flag,
`enable_per_request_metrics: bool = False` by default) -- it is **not** request-side opt-in the
way the earlier guess in this docstring assumed, and the `return_metrics` request field
`inspect_per_request_metrics.sh` tried really doesn't do anything (confirmed empirically now, not
just suspected). `nscc_model_server/scripts/serve.sh` now passes `--enable-per-request-metrics`.
The actual field names inside `PerRequestTimingMetrics` (`time_to_first_token_ms` etc.) are still
unconfirmed against a real populated response -- that needs one more real run with the flag now
being passed, and `inspect_per_request_metrics.sh` should be run again to get that.

`usage.prompt_tokens`, `usage.completion_tokens`, `usage.prompt_tokens_details.cached_tokens`, and
`choices[0].finish_reason` are unaffected by any of this -- standard OpenAI chat-completion schema,
independently confirmed earlier via `inspect_ai.model._openai.model_output_from_openai`.
"""

from __future__ import annotations

from typing import Any

from inspect_ai.event import ModelEvent

_METRICS_CONTAINER_CANDIDATES = ("metrics", "vllm_metrics", "timing")
""""metrics" is the real key per `ChatCompletionResponse.metrics` (vLLM source, see module
docstring) -- listed first so it wins over the other two, which are unconfirmed leftover guesses
kept only as a fallback for older/different vLLM versions."""

_FIELD_CANDIDATES: dict[str, tuple[str, ...]] = {
    # Confirmed field names from PerRequestTimingMetrics (see module docstring), in ms -- listed
    # first in each tuple; older guessed names kept as fallback candidates after them. Unit
    # conversion (ms -> seconds, to match this schema's `*_seconds` naming) happens below in
    # `_extract`, not here -- these are just the raw JSON key names to look for.
    "queue_time_seconds": ("queue_time_ms", "queue_time", "time_in_queue"),
    "ttft_seconds": ("time_to_first_token_ms", "ttft", "time_to_first_token"),
    "prefill_time_seconds": ("prefill_time_ms", "prefill_time", "prompt_time"),
    "decode_time_seconds": ("generation_time_ms", "decode_time", "generation_time"),
    "mean_itl_seconds": ("mean_itl_ms", "mean_itl", "inter_token_latency"),
    # Not present in PerRequestTimingMetrics as of the source read above -- long-shot candidates,
    # not source-confirmed like the ones above.
    "speculative_draft_tokens": (
        "draft_tokens",
        "num_draft_tokens",
        "speculative_draft_tokens",
    ),
    "speculative_accepted_tokens": (
        "accepted_tokens",
        "num_accepted_tokens",
        "speculative_accepted_tokens",
    ),
    "speculative_acceptance_rate": ("acceptance_rate", "spec_acceptance_rate"),
    "guided_decoding_overhead_seconds": (
        "guided_decoding_time",
        "structured_output_time",
        "guided_decoding_overhead_seconds",
    ),
}
_MS_FIELDS = frozenset(
    {
        "queue_time_seconds",
        "ttft_seconds",
        "prefill_time_seconds",
        "decode_time_seconds",
        "mean_itl_seconds",
    }
)
"""Which _FIELD_CANDIDATES keys hold a value found in milliseconds (per PerRequestTimingMetrics)
that needs /1000 to match this schema's *_seconds naming -- the fallback candidate names in the
same tuples are unconfirmed and might not actually be in ms, but there's no way to tell without a
real response, so the same conversion is applied uniformly; flagged here rather than silently."""
_TOKENS_PER_SECOND_FIELD_CANDIDATES = ("tokens_per_second", "request_tokens_per_second")
"""vLLM's PerRequestTimingMetrics.tokens_per_second is already a computed rate, not something this
module needs to derive itself from decode_time/generated_tokens -- preferred over our own division
(see _extract) when present, since it comes straight from the server."""

_CALL_NOT_RETAINED = (
    "event.call is None (raw model call not retained -- run with "
    "INSPECT_EVAL_LOG_MODEL_API=true, otherwise inspect_ai only keeps it for a model's first "
    "few calls; see this module's docstring)"
)


def _find_metrics_container(response: dict[str, Any]) -> dict[str, Any] | None:
    for key in _METRICS_CONTAINER_CANDIDATES:
        value = response.get(key)
        if isinstance(value, dict):
            return value
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        for key in _METRICS_CONTAINER_CANDIDATES:
            value = choices[0].get(key)
            if isinstance(value, dict):
                return value
    return None


def _extract(response: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Best-effort field extraction. Returns (payload, names of expected fields not found)."""
    missing: list[str] = []
    payload: dict[str, Any] = {}

    payload["serving_request_id"] = response.get("id")
    if payload["serving_request_id"] is None:
        missing.append("serving_request_id")

    choices = response.get("choices")
    finish_reason = None
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        finish_reason = choices[0].get("finish_reason")
    payload["finish_reason"] = finish_reason
    if finish_reason is None:
        missing.append("finish_reason")

    raw_usage = response.get("usage")
    usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
    payload["prompt_tokens"] = usage.get("prompt_tokens")
    payload["generated_tokens"] = usage.get("completion_tokens")
    if payload["prompt_tokens"] is None:
        missing.append("prompt_tokens")
    if payload["generated_tokens"] is None:
        missing.append("generated_tokens")

    prompt_details = usage.get("prompt_tokens_details")
    cached_tokens = (
        prompt_details.get("cached_tokens")
        if isinstance(prompt_details, dict)
        else None
    )
    payload["cached_tokens"] = cached_tokens
    if cached_tokens is None:
        missing.append("cached_tokens")

    container = _find_metrics_container(response)
    for field_name, candidates in _FIELD_CANDIDATES.items():
        value = None
        if container is not None:
            for candidate in candidates:
                if candidate in container:
                    value = container[candidate]
                    break
        if value is not None and field_name in _MS_FIELDS:
            value = value / 1000.0
        payload[field_name] = value
        if value is None:
            missing.append(field_name)

    # vLLM computes this itself (PerRequestTimingMetrics.tokens_per_second) -- use it directly
    # rather than re-deriving from decode_time/generated_tokens when it's there.
    server_tps = None
    if container is not None:
        for candidate in _TOKENS_PER_SECOND_FIELD_CANDIDATES:
            if candidate in container:
                server_tps = container[candidate]
                break
    if server_tps is not None:
        payload["request_tokens_per_second"] = server_tps
    else:
        decode_time = payload.get("decode_time_seconds")
        generated = payload.get("generated_tokens")
        payload["request_tokens_per_second"] = (
            generated / decode_time if decode_time and generated else None
        )
        if payload["request_tokens_per_second"] is None:
            missing.append("request_tokens_per_second")

    return payload, missing


def extract_per_request_metrics(event: ModelEvent) -> dict[str, Any]:
    """Pure function over a completed `ModelEvent` -- no I/O, no tracker state needed.

    Unlike `VLLMMetricsTracker`, there is no `before_model_generate` step: everything comes from
    this one event's own `call.response`, so there's nothing to correlate across calls.
    """
    if event.call is None or event.call.response is None:
        return dict(
            confidence="unattributed",
            serving_request_id=None,
            queue_time_seconds=None,
            ttft_seconds=None,
            prefill_time_seconds=None,
            decode_time_seconds=None,
            mean_itl_seconds=None,
            prompt_tokens=None,
            generated_tokens=None,
            cached_tokens=None,
            request_tokens_per_second=None,
            finish_reason=None,
            speculative_draft_tokens=None,
            speculative_accepted_tokens=None,
            speculative_acceptance_rate=None,
            guided_decoding_overhead_seconds=None,
            raw_fields_missing=[_CALL_NOT_RETAINED],
        )

    payload, missing = _extract(event.call.response)
    confidence = (
        "exact" if payload.get("serving_request_id") is not None else "unattributed"
    )
    return dict(confidence=confidence, raw_fields_missing=missing, **payload)
