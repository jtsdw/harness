"""缺口C: retry attempt grouping (efficient-harness.md 目标一 Q5).

inspect_ai gives an independent `ModelEvent` per HTTP/API attempt (failed attempts keep their error
and are never silently dropped), but no field ties "attempt 1, 2, 3 of the same logical request"
together. Three hooks combine to reconstruct that grouping:

- `on_before_model_generate` fires at the top of every attempt (incl. retries) but its
  `BeforeModelGenerate.sample_id` is `ActiveSample.id` — a per-attempt-object id, NOT the stable
  `sample_uuid` every other hook uses (verified in `inspect_ai/hooks/_hooks.py::emit_before_model_generate`
  vs. `emit_model_retry`/`emit_sample_event`, which use `active.sample_uuid`). It carries no attempt
  counter either.
- `on_model_retry` fires only when an attempt fails *and will be retried*, and does carry
  `ModelRetry.attempt: int` (1-indexed) plus `sample_uuid`-space `sample_id`.
- `on_sample_event` delivers **every** attempt's `ModelEvent` once it completes — including a failed
  one — not just the terminal one. Per `Model._generate()`, a failed attempt's event completes (and
  is therefore delivered here) *before* tenacity's `before_sleep=log_model_retry` runs and fires
  `on_model_retry` for it. If retries are exhausted, tenacity re-raises without ever calling
  `before_sleep`, so a terminal failure produces a completed `ModelEvent` (seen here) with **no**
  corresponding `on_model_retry` call at all.

Because `BeforeModelGenerate`/`ModelRetry` for the same attempt chain run inside the same coroutine
(tenacity's retry loop never spawns a new task), `anyio.get_current_task().id` — the wrapper's own
stable task identifier, NOT `id()` of the `TaskInfo` object itself, which is a fresh wrapper allocated
on every call and therefore not a stable key — is a reliable, private-API-free correlation key between
them. The per-attempt `ModelEvent` (delivered on a *different* task — the sample's dedicated
event-emitter task) is matched back to that chain by content identity: a retried attempt resends the
same message list, so `(model_name, hash(input))` is indexed at `on_before_model_generate` time and
looked up at `on_sample_event` time.

Because a failed attempt's `ModelEvent` arrives before we know whether it will be retried, a failed
`on_sample_event` does not finalize the group — it stashes the event uuid/error on the group and waits
for either `on_model_retry` (merges into one `AttemptDetail`, group stays open for the next attempt) or
`on_sample_end`'s abandoned-sweep (retries were exhausted; finalize as `resolution="error"`).

`BeforeModelGenerate.sample_id` being unusable (see above) also means we can't use it to know which
sample owns a given task for the abandoned-sweep. Instead `bind_sample()` is called once from
`on_sample_attempt_start` (which *does* run on the sample's own task, since it's a plain `await` in the
sample's execution coroutine, not a spawned child task) to record `task -> sample_uuid`. This covers the
common linear agent-loop case, where every subsequent `generate()` call for that sample runs on the same
task. It does **not** cover model calls made from a child task spawned for parallel tool calls or
`inspect_ai.util.collect()` subagents — those groups still resolve correctly via `(model_name,
input_hash)` on the success/normal-retry path, they just won't be found (and cleaned up) by the
abandoned-sweep if the sample ends while one is still in flight. Documented as a known limitation
rather than solved with a fragile parent-task heuristic; revisit if it proves to matter in practice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import anyio
from inspect_ai.event import ModelEvent
from inspect_ai.hooks import BeforeModelGenerate, ModelRetry

from .hashing import hash_message_list


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _current_task_key() -> int:
    # anyio.get_current_task() returns a fresh TaskInfo wrapper on every call (not a cached
    # singleton), so `id()` of that wrapper is not a stable key -- it just happens to often
    # collide with a just-freed address in a tight loop. `.id` is the wrapper's own stable
    # identifier for the underlying task, and is what stays constant across calls.
    return anyio.get_current_task().id


@dataclass
class _InflightGroup:
    model_name: str
    sample_uuid: str | None
    input_hashes_seen: set[str] = field(default_factory=set)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    current_attempt_number: int = 1
    started_at: str = field(default_factory=_now_iso)
    pending_failed_event_uuid: str | None = None
    pending_failed_error: str | None = None


class AttemptGroupTracker:
    def __init__(self) -> None:
        self._inflight_by_task: dict[int, _InflightGroup] = {}
        self._resolve_index: dict[tuple[str, str], int] = {}
        self._task_keys_by_sample_uuid: dict[str, set[int]] = {}
        self._task_to_sample: dict[int, str] = {}

    def bind_sample(self, sample_uuid: str) -> None:
        """Call from on_sample_attempt_start (runs on the sample's own task)."""
        self._task_to_sample[_current_task_key()] = sample_uuid

    def _register_task_for_sample(self, task_key: int, sample_uuid: str | None) -> None:
        if sample_uuid is None:
            return
        self._task_keys_by_sample_uuid.setdefault(sample_uuid, set()).add(task_key)

    def before_model_generate(self, data: BeforeModelGenerate) -> None:
        task_key = _current_task_key()
        sample_uuid = self._task_to_sample.get(task_key)
        input_hash = hash_message_list(data.input)

        group = self._inflight_by_task.get(task_key)
        if group is None:
            group = _InflightGroup(model_name=data.model_name, sample_uuid=sample_uuid)
            self._inflight_by_task[task_key] = group
            self._register_task_for_sample(task_key, sample_uuid)
        else:
            group.current_attempt_number += 1
            if sample_uuid is not None:
                group.sample_uuid = sample_uuid
                self._register_task_for_sample(task_key, sample_uuid)

        group.input_hashes_seen.add(input_hash)
        self._resolve_index[(data.model_name, input_hash)] = task_key

    def model_retry(self, data: ModelRetry) -> None:
        task_key = _current_task_key()
        group = self._inflight_by_task.get(task_key)
        if group is None:
            # Defensive: hooks attached mid-flight, or a retry we never saw the
            # matching before_model_generate for. Drop rather than fabricate state.
            return
        group.attempts.append(
            dict(
                attempt_number=data.attempt,
                outcome="error",
                model_event_uuid=group.pending_failed_event_uuid,
                error_type=data.exception_type,
                status_code=data.status_code,
                wait_time_after=data.wait_time,
            )
        )
        group.pending_failed_event_uuid = None
        group.pending_failed_error = None

    def _pop_group(self, task_key: int, group: _InflightGroup) -> None:
        self._inflight_by_task.pop(task_key, None)
        for input_hash in group.input_hashes_seen:
            self._resolve_index.pop((group.model_name, input_hash), None)
        if group.sample_uuid is not None:
            self._task_keys_by_sample_uuid.get(group.sample_uuid, set()).discard(
                task_key
            )

    def sample_event(
        self, sample_uuid: str, event: ModelEvent
    ) -> dict[str, Any] | None:
        input_hash = hash_message_list(event.input)
        task_key = self._resolve_index.pop((event.model, input_hash), None)

        if task_key is None:
            # Either a first-attempt cache hit (the common case, not a fallback: cache reads
            # go through before_model_generate too) with no tracked group, or a genuine
            # orphan (hooks attached mid-run). Either way, don't drop the data.
            return dict(
                model_name=event.model,
                resolution="orphan_no_open_match",
                total_attempts=1,
                total_wasted_wait_time=0.0,
                final_model_event_uuid=event.uuid,
                attempts=[
                    dict(
                        attempt_number=1,
                        outcome="error" if event.error else "success",
                        model_event_uuid=event.uuid,
                        error_type=None,
                        status_code=None,
                        wait_time_after=None,
                    )
                ],
                first_attempt_started_at=_now_iso(),
                resolved_at=_now_iso(),
            )

        group = self._inflight_by_task[task_key]

        if event.error is not None:
            # Failed attempt: don't know yet whether tenacity will retry it. Stash and wait
            # for on_model_retry (retry happens) or on_sample_end's sweep (retries exhausted).
            group.pending_failed_event_uuid = event.uuid
            group.pending_failed_error = event.error
            return None

        # Success: finalize the group now.
        group.attempts.append(
            dict(
                attempt_number=group.current_attempt_number,
                outcome="success",
                model_event_uuid=event.uuid,
                error_type=None,
                status_code=None,
                wait_time_after=None,
            )
        )
        self._pop_group(task_key, group)
        wasted = sum(a["wait_time_after"] or 0.0 for a in group.attempts)

        return dict(
            model_name=group.model_name,
            resolution="success",
            total_attempts=len(group.attempts),
            total_wasted_wait_time=wasted,
            final_model_event_uuid=event.uuid,
            attempts=group.attempts,
            first_attempt_started_at=group.started_at,
            resolved_at=_now_iso(),
        )

    def sweep_abandoned(self, sample_uuid: str) -> list[dict[str, Any]]:
        """Finalize anything still open for this sample and free its state.

        Called from on_sample_end: finalize anything still open for this sample (either a
        terminal failure whose retries were exhausted, or a genuinely interrupted request), then
        free all state for it so nothing survives past its owning sample's lifetime.
        """
        results: list[dict[str, Any]] = []
        for task_key, bound_sample in list(self._task_to_sample.items()):
            if bound_sample == sample_uuid:
                self._task_to_sample.pop(task_key, None)

        for task_key in list(self._task_keys_by_sample_uuid.pop(sample_uuid, set())):
            group = self._inflight_by_task.get(task_key)
            if group is None:
                continue
            self._pop_group(task_key, group)

            if group.pending_failed_event_uuid is not None:
                group.attempts.append(
                    dict(
                        attempt_number=group.current_attempt_number,
                        outcome="error",
                        model_event_uuid=group.pending_failed_event_uuid,
                        error_type=group.pending_failed_error,
                        status_code=None,
                        wait_time_after=None,
                    )
                )
                resolution = "error"
            else:
                resolution = "abandoned"

            if not group.attempts:
                continue

            wasted = sum(a["wait_time_after"] or 0.0 for a in group.attempts)
            results.append(
                dict(
                    model_name=group.model_name,
                    resolution=resolution,
                    total_attempts=len(group.attempts),
                    total_wasted_wait_time=wasted,
                    final_model_event_uuid=None,
                    attempts=group.attempts,
                    first_attempt_started_at=group.started_at,
                    resolved_at=_now_iso() if resolution == "error" else None,
                )
            )
        return results
