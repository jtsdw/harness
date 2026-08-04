"""action-parsing and observation-write-back tracing (efficient-harness.md 目标一 需求四).

Stateless: reads a single `ToolEvent`'s own `error` field (populated by inspect_ai's own
`parse_tool_call()`/`validate_tool_input()` when a tool call's JSON or arguments fail to validate --
`ToolCallError(type="parsing", ...)`) plus `parent_model_event_uuid` resolved by the caller via
`ExecutionTopologyTracker.parent_model_event_uuid` (shared index, not duplicated here).

`tool_call_id` (== `ToolEvent.id` == `ChatMessageTool.tool_call_id`) is the join key forward into the
next step's `prefill_diff` record: the `PrefillDiffMessageDetail` entry with a matching `tool_call_id`
is the observation this tool call's result was written back as.
"""

from __future__ import annotations

from typing import Any

from inspect_ai.event import ToolEvent


def observe(event: ToolEvent, parent_model_event_uuid: str | None) -> dict[str, Any]:
    error = event.error
    return dict(
        tool_event_uuid=event.uuid,
        tool_call_id=event.id,
        function=event.function,
        parent_model_event_uuid=parent_model_event_uuid,
        error_present=error is not None,
        error_type=error.type if error else None,
        error_message=error.message if error else None,
        is_parse_or_validation_error=error is not None and error.type == "parsing",
        observation_message_id=event.message_id,
        working_time_seconds=event.working_time,
        started_at=event.timestamp.isoformat(),
        completed_at=event.completed.isoformat() if event.completed else None,
    )
