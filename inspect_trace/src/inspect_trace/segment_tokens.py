"""缺口B: segment-level token estimation (efficient-harness.md 目标一 Q3).

`ModelUsage` only records whole-call aggregates (`output_tokens`, `reasoning_tokens`, ...); it does
not split a turn's output across reasoning / tool-call / server-tool-use / final-text content. This
module estimates that split with a standalone tokenizer (see `token_estimate.py`) and reports it
alongside — never in place of — the real billed usage inspect_ai already captured.
"""

from __future__ import annotations

from typing import Any

from inspect_ai._util.content import ContentReasoning, ContentText, ContentToolUse
from inspect_ai.event import ModelEvent

from .token_estimate import estimate_text_tokens, estimate_tool_call_tokens


def observe(event: ModelEvent) -> dict[str, Any]:
    message = event.output.message

    reasoning_tokens = 0
    reasoning_blocks = 0
    server_tool_tokens = 0
    server_tool_blocks = 0
    text_tokens = 0
    text_blocks = 0
    other_tokens = 0

    for block in message.content_list:
        if isinstance(block, ContentReasoning):
            text = block.reasoning if not block.redacted else (block.summary or "")
            reasoning_tokens += estimate_text_tokens(text)
            reasoning_blocks += 1
        elif isinstance(block, ContentText):
            text_tokens += estimate_text_tokens(block.text)
            text_blocks += 1
        elif isinstance(block, ContentToolUse):
            server_tool_tokens += estimate_text_tokens(
                block.name
            ) + estimate_text_tokens(block.context)
            server_tool_blocks += 1
        else:
            other_tokens += estimate_text_tokens(getattr(block, "text", "") or "")

    tool_call_tokens = 0
    tool_call_count = 0
    for call in message.tool_calls or []:
        tool_call_tokens += estimate_tool_call_tokens(call.function, call.arguments)
        tool_call_count += 1

    usage = event.output.usage

    return dict(
        model_event_uuid=event.uuid,
        model_name=event.model,
        stop_reason=event.output.stop_reason,
        cache=event.cache,
        reasoning_estimated_tokens=reasoning_tokens,
        reasoning_block_count=reasoning_blocks,
        tool_call_estimated_tokens=tool_call_tokens,
        tool_call_count=tool_call_count,
        server_tool_use_estimated_tokens=server_tool_tokens,
        server_tool_use_count=server_tool_blocks,
        text_estimated_tokens=text_tokens,
        text_block_count=text_blocks,
        other_estimated_tokens=other_tokens,
        estimated_output_tokens_total=(
            reasoning_tokens
            + tool_call_tokens
            + server_tool_tokens
            + text_tokens
            + other_tokens
        ),
        billed_input_tokens=usage.input_tokens if usage else None,
        billed_output_tokens=usage.output_tokens if usage else None,
        billed_reasoning_tokens=usage.reasoning_tokens if usage else None,
        billed_total_tokens=usage.total_tokens if usage else None,
        retries=event.retries,
    )
