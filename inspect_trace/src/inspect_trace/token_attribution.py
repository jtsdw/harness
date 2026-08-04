"""token-level attribution (efficient-harness.md 目标一 需求一).

Stateless recombination of `prefill_diff` (input side) and `segment_tokens` (output side) into the
5-category attribution required by 需求一: system template / tool schema / conversation (input side,
agent-agnostic) and reasoning / tool-calling / final-response (output side, ReAct-stage). No new
tokenization happens here -- every number is either copied straight from `event.output.usage` (real,
billed) or derived by summing fields the two upstream trackers already computed (estimates). There is
no native per-segment *real* prefill/decode split from any provider; the estimate fields are the
ceiling of what's achievable without provider-side instrumentation, and callers must not treat them as
billed values.
"""

from __future__ import annotations

from typing import Any

from inspect_ai.event import ModelEvent


def compose(
    event: ModelEvent, prefill_payload: dict[str, Any], segment_payload: dict[str, Any]
) -> dict[str, Any]:
    system_template_tokens_estimate = (
        prefill_payload["system_template_new_tokens_estimate"]
        + prefill_payload["system_template_reused_tokens_estimate"]
    )
    tool_schema_tokens_estimate = (
        prefill_payload["new_tool_tokens_estimate"]
        + prefill_payload["reused_tool_tokens_estimate"]
    )
    conversation_tokens_estimate = (
        prefill_payload["new_tokens_estimate"]
        + prefill_payload["reused_tokens_estimate"]
        - system_template_tokens_estimate
    )

    usage = event.output.usage

    return dict(
        model_event_uuid=event.uuid,
        model_name=event.model,
        step_index=prefill_payload["step_index"],
        billed_input_tokens=usage.input_tokens if usage else None,
        billed_output_tokens=usage.output_tokens if usage else None,
        billed_reasoning_tokens=usage.reasoning_tokens if usage else None,
        billed_total_tokens=usage.total_tokens if usage else None,
        system_template_tokens_estimate=system_template_tokens_estimate,
        tool_schema_tokens_estimate=tool_schema_tokens_estimate,
        conversation_tokens_estimate=conversation_tokens_estimate,
        reasoning_tokens_estimate=segment_payload["reasoning_estimated_tokens"],
        tool_calling_tokens_estimate=(
            segment_payload["tool_call_estimated_tokens"]
            + segment_payload["server_tool_use_estimated_tokens"]
        ),
        final_response_tokens_estimate=segment_payload["text_estimated_tokens"],
    )
