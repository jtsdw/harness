"""Shared eval/sample identity bookkeeping used by all three detectors and the writer.

Populated from the hook payloads that reliably carry the stable identifiers (`TaskStart`/`TaskEnd` for
eval-level identity, `SampleAttemptStart` for sample-level identity — both use the stable
`state.uuid`/`sample_uuid` id space, unlike `BeforeModelGenerate.sample_id` which uses the transient
per-`ActiveSample` id; see `attempt_groups.py` for how that mismatch is handled). Cleared per
`sample_uuid` at `on_sample_end` and per `eval_id` at `on_task_end` so memory does not grow unbounded
across a long, many-sample eval run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EvalContext:
    run_id: str | None
    eval_set_id: str | None
    task_name: str | None


@dataclass
class SampleContext:
    eval_id: str
    dataset_id: int | str | None
    epoch: int | None
    sample_attempt: int


class TraceContext:
    def __init__(self) -> None:
        self.eval_context: dict[str, EvalContext] = {}
        self.sample_context: dict[str, SampleContext] = {}

    def set_eval(self, eval_id: str, ctx: EvalContext) -> None:
        self.eval_context[eval_id] = ctx

    def clear_eval(self, eval_id: str) -> None:
        self.eval_context.pop(eval_id, None)

    def set_sample(self, sample_uuid: str, ctx: SampleContext) -> None:
        self.sample_context[sample_uuid] = ctx

    def update_sample_attempt(self, sample_uuid: str, attempt: int) -> None:
        existing = self.sample_context.get(sample_uuid)
        if existing is not None:
            existing.sample_attempt = attempt

    def clear_sample(self, sample_uuid: str) -> None:
        self.sample_context.pop(sample_uuid, None)

    def envelope_fields(
        self, sample_uuid: str, eval_id: str | None = None
    ) -> dict[str, Any]:
        sample_ctx = self.sample_context.get(sample_uuid)
        resolved_eval_id = eval_id or (sample_ctx.eval_id if sample_ctx else None) or ""
        eval_ctx = self.eval_context.get(resolved_eval_id)
        return dict(
            run_id=eval_ctx.run_id if eval_ctx else None,
            eval_set_id=eval_ctx.eval_set_id if eval_ctx else None,
            eval_id=resolved_eval_id,
            task_name=eval_ctx.task_name if eval_ctx else None,
            sample_id=sample_ctx.dataset_id if sample_ctx else None,
            sample_uuid=sample_uuid,
            epoch=sample_ctx.epoch if sample_ctx else None,
            sample_attempt=sample_ctx.sample_attempt if sample_ctx else None,
        )
