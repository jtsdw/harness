"""execution topology reconstruction (efficient-harness.md 目标一 需求三).

Reconstructs, per sample, whether tool-call execution was linear or had observed parallelism, and how
much time was spent on each side waiting on the other. Parallelism is inferred purely from whether
sibling `ToolEvent` `[timestamp, completed]` windows actually overlap -- not from inspect_ai's own
`ToolDef.parallel` declaration (which isn't exposed on the event anyway) -- because for efficiency
research, what actually happened concurrently matters more than what was merely declared eligible.

There is no rollback field here, deliberately: inspect_ai has no mechanism that reverts `TaskState`
and discards a branch. `fork()` (`solver/_fork.py`) runs every branch to completion and returns all of
them -- nothing is thrown away. `BranchEvent`/`AnchorEvent` are timeline-viewer markers, not state
reverts. `CheckpointEvent` is disk persistence for crash-resume, not an in-memory rollback. The closest
real analog available in this framework is a retry (attempt discarded, request redone), which is
already captured by `attempt_group` records -- join on `sample_uuid` if you need it alongside topology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from inspect_ai.event import ModelEvent, ToolEvent


@dataclass
class _StageBuilder:
    parent_model_event_uuid: str
    tool_events: list[ToolEvent] = field(default_factory=list)


class ExecutionTopologyTracker:
    def __init__(self) -> None:
        self._tool_call_parent: dict[str, dict[str, str]] = {}
        self._model_timings: dict[str, list[tuple[str, datetime, datetime | None]]] = {}
        self._tool_events: dict[str, list[ToolEvent]] = {}

    def reset_sample(self, sample_uuid: str) -> None:
        self._tool_call_parent.pop(sample_uuid, None)
        self._model_timings.pop(sample_uuid, None)
        self._tool_events.pop(sample_uuid, None)

    def observe_model_event(self, sample_uuid: str, event: ModelEvent) -> None:
        assert event.uuid is not None, (
            "uuid is only unset before an event is first recorded"
        )
        parent_index = self._tool_call_parent.setdefault(sample_uuid, {})
        for call in event.output.message.tool_calls or []:
            parent_index[call.id] = event.uuid

        timings = self._model_timings.setdefault(sample_uuid, [])
        timings.append((event.uuid, event.timestamp, event.completed))

    def parent_model_event_uuid(
        self, sample_uuid: str, tool_call_id: str
    ) -> str | None:
        return self._tool_call_parent.get(sample_uuid, {}).get(tool_call_id)

    def observe_tool_event(self, sample_uuid: str, event: ToolEvent) -> None:
        self._tool_events.setdefault(sample_uuid, []).append(event)

    def finalize(self, sample_uuid: str) -> dict[str, Any] | None:
        tool_events = self._tool_events.get(sample_uuid, [])
        model_timings = self._model_timings.get(sample_uuid, [])
        if not model_timings:
            return None

        parent_index = self._tool_call_parent.get(sample_uuid, {})
        stages_by_parent: dict[str, _StageBuilder] = {}
        stage_order: list[str] = []
        unmatched: list[str] = []

        for tool_event in tool_events:
            parent_uuid = parent_index.get(tool_event.id)
            if parent_uuid is None:
                unmatched.append(tool_event.uuid or tool_event.id)
                continue
            if parent_uuid not in stages_by_parent:
                stages_by_parent[parent_uuid] = _StageBuilder(
                    parent_model_event_uuid=parent_uuid
                )
                stage_order.append(parent_uuid)
            stages_by_parent[parent_uuid].tool_events.append(tool_event)

        model_timing_by_uuid = {
            uuid: (start, end) for uuid, start, end in model_timings
        }
        model_order = [uuid for uuid, _, _ in model_timings]

        stages: list[dict[str, Any]] = []
        for stage_index, parent_uuid in enumerate(stage_order, start=1):
            builder = stages_by_parent[parent_uuid]
            events = builder.tool_events

            observed_parallel = _has_overlap(events)
            max_concurrency = _max_concurrency(events)

            parent_start, parent_end = model_timing_by_uuid.get(
                parent_uuid, (None, None)
            )
            tool_waiting_for_model_seconds = [
                max(0.0, (e.timestamp - parent_end).total_seconds())
                if parent_end is not None
                else None
                for e in events
            ]

            stage_end = max(
                (e.completed for e in events if e.completed is not None), default=None
            )
            model_waiting_for_tool_seconds = None
            try:
                next_model_uuid = model_order[model_order.index(parent_uuid) + 1]
            except (ValueError, IndexError):
                next_model_uuid = None
            if next_model_uuid is not None and stage_end is not None:
                next_start, _ = model_timing_by_uuid[next_model_uuid]
                model_waiting_for_tool_seconds = max(
                    0.0, (next_start - stage_end).total_seconds()
                )

            stages.append(
                dict(
                    stage_index=stage_index,
                    parent_model_event_uuid=parent_uuid,
                    tool_call_ids=[e.id for e in events],
                    tool_functions=[e.function for e in events],
                    tool_count=len(events),
                    stage_started_at=min(e.timestamp for e in events).isoformat(),
                    stage_ended_at=stage_end.isoformat() if stage_end else None,
                    observed_parallel=observed_parallel,
                    max_observed_concurrency=max_concurrency,
                    tool_waiting_for_model_seconds=[
                        t for t in tool_waiting_for_model_seconds if t is not None
                    ],
                    tool_semaphore_wait_seconds=[
                        (
                            (e.completed - e.timestamp).total_seconds() - e.working_time
                            if e.completed is not None and e.working_time is not None
                            else None
                        )
                        for e in events
                    ],
                    model_waiting_for_tool_seconds=model_waiting_for_tool_seconds,
                )
            )

        total_model_calls = len(model_timings)
        total_tool_calls = sum(len(s["tool_call_ids"]) for s in stages)
        linear = all(
            s["tool_count"] <= 1 and not s["observed_parallel"] for s in stages
        )

        return dict(
            total_model_calls=total_model_calls,
            total_tool_calls=total_tool_calls,
            total_stages=len(stages),
            linear=linear,
            stages=stages,
            unmatched_tool_event_uuids=unmatched,
        )


def _has_overlap(events: list[ToolEvent]) -> bool:
    for i, a in enumerate(events):
        if a.completed is None:
            continue
        for b in events[i + 1 :]:
            if b.completed is None:
                continue
            if a.timestamp < b.completed and b.timestamp < a.completed:
                return True
    return False


def _max_concurrency(events: list[ToolEvent]) -> int:
    boundaries: list[tuple[datetime, int]] = []
    for e in events:
        if e.completed is None:
            continue
        boundaries.append((e.timestamp, 1))
        boundaries.append((e.completed, -1))
    boundaries.sort(key=lambda b: (b[0], -b[1]))

    current = peak = 0
    for _, delta in boundaries:
        current += delta
        peak = max(peak, current)
    return peak
