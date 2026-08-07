"""Unit tests for vllm_per_request_metrics.py's tolerant field extraction.

No live NSCC/vLLM dependency (there isn't one to have -- see that module's docstring): these only
check the pure-function extraction logic against synthetic `ModelEvent`/`ModelCall` fixtures,
covering the cases the design is actually meant to handle -- call not retained, standard-schema-
only response, and a response with a matching extension-field container.
"""

import pytest
from inspect_ai.event import ModelEvent
from inspect_ai.model import ModelOutput
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._model_call import ModelCall

from inspect_trace.schema import VLLMPerRequestMetricsRecord
from inspect_trace.vllm_per_request_metrics import (
    _CALL_NOT_RETAINED,
    extract_per_request_metrics,
)


def _event(*, call: ModelCall | None) -> ModelEvent:
    return ModelEvent(
        model="openai-api/vllm/Qwen/Qwen3-32B",
        input=[],
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
        output=ModelOutput.from_content("openai-api/vllm/Qwen/Qwen3-32B", "hi"),
        call=call,
    )


def test_call_not_retained_is_unattributed() -> None:
    payload = extract_per_request_metrics(_event(call=None))
    assert payload["confidence"] == "unattributed"
    assert payload["serving_request_id"] is None
    assert _CALL_NOT_RETAINED in payload["raw_fields_missing"]
    assert payload["ttft_seconds"] is None


def test_standard_openai_schema_only_extracts_known_fields() -> None:
    response = {
        "id": "chatcmpl-abc123",
        "choices": [{"finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 40},
        },
    }
    event = _event(call=ModelCall(request={}, response=response))
    payload = extract_per_request_metrics(event)

    assert payload["confidence"] == "exact"
    assert payload["serving_request_id"] == "chatcmpl-abc123"
    assert payload["finish_reason"] == "stop"
    assert payload["prompt_tokens"] == 100
    assert payload["generated_tokens"] == 20
    assert payload["cached_tokens"] == 40
    # Nothing in _FIELD_CANDIDATES' container keys ("metrics"/"vllm_metrics"/etc.) is present in
    # this response at all, so every speculative/timing field must be reported missing, not
    # guessed at.
    assert payload["ttft_seconds"] is None
    assert "ttft_seconds" in payload["raw_fields_missing"]
    assert payload["queue_time_seconds"] is None
    assert payload["request_tokens_per_second"] is None


def test_real_per_request_timing_metrics_shape_is_read_and_converted_from_ms() -> None:
    """Fixture shaped like vLLM's actual `PerRequestTimingMetrics`.

    Confirmed from vLLM's GitHub source 2026-08-07, see vllm_per_request_metrics.py's module
    docstring -- values in ms, a `tokens_per_second` the server already computed, no
    prefill_time/speculative fields at all (that class genuinely doesn't have them).
    """
    response = {
        "id": "chatcmpl-xyz",
        "choices": [{"finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        "metrics": {
            "time_to_first_token_ms": 50.0,
            "queue_time_ms": 10.0,
            "generation_time_ms": 200.0,
            "mean_itl_ms": 25.0,
            "tokens_per_second": 25.0,
        },
    }
    event = _event(call=ModelCall(request={}, response=response))
    payload = extract_per_request_metrics(event)

    assert payload["ttft_seconds"] == pytest.approx(0.05)
    assert payload["queue_time_seconds"] == pytest.approx(0.01)
    assert payload["decode_time_seconds"] == pytest.approx(0.2)
    assert payload["mean_itl_seconds"] == pytest.approx(0.025)
    # server's own rate is used directly, not re-derived from decode_time/generated_tokens
    assert payload["request_tokens_per_second"] == 25.0
    assert "ttft_seconds" not in payload["raw_fields_missing"]
    # genuinely not part of PerRequestTimingMetrics -- stays missing, not guessed at
    assert payload["prefill_time_seconds"] is None
    assert "prefill_time_seconds" in payload["raw_fields_missing"]
    assert payload["speculative_accepted_tokens"] is None


def test_legacy_fallback_field_names_still_work_and_get_ms_conversion() -> None:
    """The older, unconfirmed guessed names are still tried as fallbacks.

    Same ms->seconds conversion applies to them too, since there's no way to know their unit
    without a real response (see _MS_FIELDS' docstring).
    """
    response = {
        "id": "chatcmpl-xyz",
        "choices": [{"finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        "metrics": {"queue_time": 10.0, "num_accepted_tokens": 3},
    }
    event = _event(call=ModelCall(request={}, response=response))
    payload = extract_per_request_metrics(event)

    assert payload["queue_time_seconds"] == pytest.approx(0.01)
    assert payload["speculative_accepted_tokens"] == 3


def test_no_serving_request_id_is_unattributed_even_with_a_response() -> None:
    response = {"choices": [{"finish_reason": "stop"}], "usage": {}}
    event = _event(call=ModelCall(request={}, response=response))
    payload = extract_per_request_metrics(event)
    assert payload["confidence"] == "unattributed"


def test_extracted_payload_round_trips_through_the_schema_record() -> None:
    """This is exactly what hooks.py does: payload dict -> VLLMPerRequestMetricsRecord(...)."""
    payload = extract_per_request_metrics(_event(call=None))
    record = VLLMPerRequestMetricsRecord(
        eval_id="e1",
        sample_uuid="s1",
        recorded_at="2026-08-07T00:00:00Z",
        model_event_uuid="m1",
        **payload,
    )
    assert record.kind == "vllm_per_request_metrics"
    assert record.source == "server_per_request"
    assert record.confidence == "unattributed"
