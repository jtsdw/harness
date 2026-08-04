"""repeated-prefill detection (efficient-harness.md 目标一 需求二，及需求一里 system template 的归因输入).

For every completed `ModelEvent` (including failed retry attempts — each attempt is a real API call
and therefore a real "step" for prefill-cost purposes), classifies each message in `event.input`
against the *entire prior history* of that sample (not a sliding window, since a message can drop out
of context via compaction/truncation and reappear later) as `new`, `reused`, or `dup_in_step`, and
aggregates reuse by tool name to answer "which observations keep getting re-sent."

Also does the same new/reused/dup_in_step classification for `event.tools` (the tool schema
definitions offered on every call), tracked in a separate pass with its own seen-set. This was added
after running against a real multi-turn benchmark (BFCL `multi_turn_base`) and finding that a fixed
list of ~30 tool schemas -- serialized to ~23K characters -- was being resent byte-for-byte on every
one of 41 calls in a sample, completely unaccounted for by message-only tracking. Tool schemas are a
structurally different kind of repeated content from conversation history (a largely static blob, not
growing turn-by-turn), so they get their own fields rather than being folded into the message counts.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from inspect_ai.event import ModelEvent
from inspect_ai.model import ChatMessageTool
from inspect_ai.tool import ToolInfo

from .hashing import hash_message, hash_tool
from .token_estimate import estimate_text_tokens, estimate_tool_schema_tokens


@dataclass
class _SeenInfo:
    first_step_index: int
    first_model_event_uuid: str


class PrefillDiffTracker:
    def __init__(self) -> None:
        self._seen: dict[str, dict[str, _SeenInfo]] = {}
        self._seen_tools: dict[str, dict[str, _SeenInfo]] = {}
        self._step_index: dict[str, int] = {}

    def reset_sample(self, sample_uuid: str) -> None:
        self._seen.pop(sample_uuid, None)
        self._seen_tools.pop(sample_uuid, None)
        self._step_index.pop(sample_uuid, None)

    def observe(self, sample_uuid: str, event: ModelEvent) -> dict[str, Any]:
        assert event.uuid is not None, (
            "uuid is only unset before an event is first recorded"
        )
        step_index = self._step_index.get(sample_uuid, 0) + 1
        self._step_index[sample_uuid] = step_index
        seen = self._seen.setdefault(sample_uuid, {})

        details: list[dict[str, Any]] = []
        seen_this_step: set[str] = set()
        new_count = reused_count = dup_count = 0
        new_tokens = reused_tokens = 0
        tool_reuse_count: Counter[str] = Counter()
        tool_reuse_tokens: Counter[str] = Counter()
        system_template_messages = 0
        system_template_new_tokens = system_template_reused_tokens = 0

        for index, message in enumerate(event.input):
            content_hash = hash_message(message)
            tokens = estimate_text_tokens(message.text)
            function = (
                message.function if isinstance(message, ChatMessageTool) else None
            )
            tool_call_id = (
                message.tool_call_id if isinstance(message, ChatMessageTool) else None
            )
            content_category = (
                "system_template" if message.role == "system" else "conversation"
            )
            if content_category == "system_template":
                system_template_messages += 1

            if content_hash in seen_this_step:
                status = "dup_in_step"
                dup_count += 1
                info = seen.get(content_hash)
            elif content_hash in seen:
                status = "reused"
                reused_count += 1
                reused_tokens += tokens
                if content_category == "system_template":
                    system_template_reused_tokens += tokens
                info = seen[content_hash]
                if function:
                    tool_reuse_count[function] += 1
                    tool_reuse_tokens[function] += tokens
            else:
                status = "new"
                new_count += 1
                new_tokens += tokens
                if content_category == "system_template":
                    system_template_new_tokens += tokens
                info = _SeenInfo(
                    first_step_index=step_index, first_model_event_uuid=event.uuid
                )
                seen[content_hash] = info

            seen_this_step.add(content_hash)
            details.append(
                dict(
                    index=index,
                    role=message.role,
                    message_id=message.id,
                    tool_call_id=tool_call_id,
                    content_hash=content_hash,
                    status=status,
                    content_category=content_category,
                    function=function,
                    estimated_tokens=tokens,
                    first_seen_model_event_uuid=info.first_model_event_uuid
                    if info
                    else None,
                    first_seen_step_index=info.first_step_index if info else None,
                )
            )

        tool_fields = self._observe_tools(sample_uuid, event, step_index)

        return dict(
            model_event_uuid=event.uuid,
            model_name=event.model,
            step_index=step_index,
            total_messages=len(event.input),
            new_messages=new_count,
            reused_messages=reused_count,
            dup_in_step_messages=dup_count,
            new_tokens_estimate=new_tokens,
            reused_tokens_estimate=reused_tokens,
            cumulative_distinct_messages=len(seen),
            tool_reuse_breakdown=[
                dict(
                    function=fn,
                    reused_count=count,
                    reused_tokens_estimate=tool_reuse_tokens[fn],
                )
                for fn, count in tool_reuse_count.items()
            ],
            messages=details,
            system_template_messages=system_template_messages,
            system_template_new_tokens_estimate=system_template_new_tokens,
            system_template_reused_tokens_estimate=system_template_reused_tokens,
            **tool_fields,
        )

    def _observe_tools(
        self, sample_uuid: str, event: ModelEvent, step_index: int
    ) -> dict[str, Any]:
        assert event.uuid is not None, (
            "uuid is only unset before an event is first recorded"
        )
        seen = self._seen_tools.setdefault(sample_uuid, {})
        tools: list[ToolInfo] = event.tools

        details: list[dict[str, Any]] = []
        seen_this_step: set[str] = set()
        new_count = reused_count = dup_count = 0
        new_tokens = reused_tokens = 0

        for index, tool in enumerate(tools):
            content_hash = hash_tool(tool)
            tokens = estimate_tool_schema_tokens(tool)

            if content_hash in seen_this_step:
                status = "dup_in_step"
                dup_count += 1
                info = seen.get(content_hash)
            elif content_hash in seen:
                status = "reused"
                reused_count += 1
                reused_tokens += tokens
                info = seen[content_hash]
            else:
                status = "new"
                new_count += 1
                new_tokens += tokens
                info = _SeenInfo(
                    first_step_index=step_index, first_model_event_uuid=event.uuid
                )
                seen[content_hash] = info

            seen_this_step.add(content_hash)
            details.append(
                dict(
                    index=index,
                    tool_name=tool.name,
                    content_hash=content_hash,
                    status=status,
                    estimated_tokens=tokens,
                    first_seen_model_event_uuid=info.first_model_event_uuid
                    if info
                    else None,
                    first_seen_step_index=info.first_step_index if info else None,
                )
            )

        return dict(
            tools_total=len(tools),
            tools_new=new_count,
            tools_reused=reused_count,
            dup_in_step_tools=dup_count,
            new_tool_tokens_estimate=new_tokens,
            reused_tool_tokens_estimate=reused_tokens,
            tools=details,
        )
