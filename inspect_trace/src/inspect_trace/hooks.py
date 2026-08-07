"""`Hooks` subclass wiring inspect_ai's callback stream to the goal-1 detectors and the JSONL writer.

This is the only class registered via entry points (see `_registry.py`); everything else in this
package is a plain, hook-agnostic library used from here.
"""

from __future__ import annotations

from datetime import datetime, timezone

import anyio
from inspect_ai.event import ModelEvent, ToolEvent
from inspect_ai.hooks import (
    BeforeModelGenerate,
    Hooks,
    ModelRetry,
    SampleAttemptStart,
    SampleEnd,
    SampleEvent,
    TaskEnd,
    TaskStart,
)

from . import action_parsing, segment_tokens, token_attribution
from .attempt_groups import AttemptGroupTracker
from .context import EvalContext, SampleContext, TraceContext
from .execution_topology import ExecutionTopologyTracker
from .prefill_diff import PrefillDiffTracker
from .schema import (
    ActionParsingRecord,
    AttemptGroupRecord,
    ExecutionTopologyRecord,
    ManifestRecord,
    PrefillDiffRecord,
    SegmentTokenRecord,
    TokenAttributionRecord,
    VLLMMetricsRecord,
    VLLMPerRequestMetricsRecord,
)
from .vllm_metrics import VLLMMetricsTracker
from .vllm_per_request_metrics import extract_per_request_metrics
from .writer import TraceWriter


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TraceHooks(Hooks):
    def __init__(self) -> None:
        self._context = TraceContext()
        self._prefill = PrefillDiffTracker()
        self._attempts = AttemptGroupTracker()
        self._topology = ExecutionTopologyTracker()
        self._vllm_metrics = VLLMMetricsTracker()
        self._writer = TraceWriter()
        self._manifest_lock = anyio.Lock()

    async def on_task_start(self, data: TaskStart) -> None:
        self._context.set_eval(
            data.eval_id,
            EvalContext(
                run_id=data.run_id,
                eval_set_id=data.eval_set_id,
                task_name=data.spec.task,
            ),
        )

    async def on_task_end(self, data: TaskEnd) -> None:
        record = ManifestRecord(
            run_id=data.run_id,
            eval_set_id=data.eval_set_id,
            eval_id=data.eval_id,
            task_name=data.log.eval.task if data.log.eval else None,
            log_location=data.log.location or None,
            status=data.log.status,
        )
        async with self._manifest_lock:
            self._writer.write_manifest(record)
        self._context.clear_eval(data.eval_id)

    async def on_sample_attempt_start(self, data: SampleAttemptStart) -> None:
        sample_uuid = data.sample_id
        self._context.set_sample(
            sample_uuid,
            SampleContext(
                eval_id=data.eval_id,
                dataset_id=data.summary.id,
                epoch=data.summary.epoch,
                sample_attempt=data.attempt,
            ),
        )
        self._attempts.bind_sample(sample_uuid)

    async def on_before_model_generate(self, data: BeforeModelGenerate) -> None:
        self._attempts.before_model_generate(data)
        await self._vllm_metrics.before_model_generate(data)

    async def on_model_retry(self, data: ModelRetry) -> None:
        self._attempts.model_retry(data)

    async def on_sample_event(self, data: SampleEvent) -> None:
        event = data.event
        sample_uuid = data.sample_id

        if isinstance(event, ModelEvent):
            assert event.uuid is not None, (
                "uuid is only unset before an event is first recorded"
            )
            envelope = self._context.envelope_fields(sample_uuid, eval_id=data.eval_id)
            envelope["recorded_at"] = _now_iso()
            run_id = envelope.get("run_id")

            # Must run before this ModelEvent's tool_calls are indexed by anything downstream:
            # execute_tools() (and hence the ToolEvents referencing these tool_calls) only runs
            # after task_generate() has already appended this assistant message, so the index is
            # guaranteed populated by the time any ToolEvent for this call arrives.
            self._topology.observe_model_event(sample_uuid, event)

            prefill_payload = self._prefill.observe(sample_uuid, event)
            prefill_record = PrefillDiffRecord(**envelope, **prefill_payload)
            self._writer.write(run_id, data.eval_id, sample_uuid, prefill_record)

            segment_payload = segment_tokens.observe(event)
            segment_record = SegmentTokenRecord(**envelope, **segment_payload)
            self._writer.write(run_id, data.eval_id, sample_uuid, segment_record)

            attribution_payload = token_attribution.compose(
                event, prefill_payload, segment_payload
            )
            attribution_record = TokenAttributionRecord(
                **envelope, **attribution_payload
            )
            self._writer.write(run_id, data.eval_id, sample_uuid, attribution_record)

            attempt_payload = self._attempts.sample_event(sample_uuid, event)
            if attempt_payload is not None:
                attempt_record = AttemptGroupRecord(**envelope, **attempt_payload)
                self._writer.write(run_id, data.eval_id, sample_uuid, attempt_record)

            vllm_payload = await self._vllm_metrics.sample_event(event)
            if vllm_payload is not None:
                vllm_record = VLLMMetricsRecord(
                    **envelope, model_event_uuid=event.uuid, **vllm_payload
                )
                self._writer.write(run_id, data.eval_id, sample_uuid, vllm_record)

            # Same "is this actually vLLM" gate as VLLMMetricsRecord above uses (there, a
            # reachable /metrics endpoint; here, no network probe available since this is a pure
            # read of the event's own response, so the model name is the only signal) -- avoids
            # writing a "vllm_*" record kind for hosted-provider calls where it's structurally
            # never going to have anything but nulls in it.
            if "vllm" in event.model.lower():
                per_request_payload = extract_per_request_metrics(event)
                per_request_record = VLLMPerRequestMetricsRecord(
                    **envelope, model_event_uuid=event.uuid, **per_request_payload
                )
                self._writer.write(
                    run_id, data.eval_id, sample_uuid, per_request_record
                )

        elif isinstance(event, ToolEvent):
            envelope = self._context.envelope_fields(sample_uuid, eval_id=data.eval_id)
            envelope["recorded_at"] = _now_iso()
            run_id = envelope.get("run_id")

            self._topology.observe_tool_event(sample_uuid, event)
            parent_uuid = self._topology.parent_model_event_uuid(sample_uuid, event.id)
            action_payload = action_parsing.observe(event, parent_uuid)
            action_record = ActionParsingRecord(**envelope, **action_payload)
            self._writer.write(run_id, data.eval_id, sample_uuid, action_record)

    async def on_sample_end(self, data: SampleEnd) -> None:
        sample_uuid = data.sample_id
        envelope = self._context.envelope_fields(sample_uuid, eval_id=data.eval_id)
        run_id = envelope.get("run_id")

        for attempt_payload in self._attempts.sweep_abandoned(sample_uuid):
            envelope["recorded_at"] = _now_iso()
            record = AttemptGroupRecord(**envelope, **attempt_payload)
            self._writer.write(run_id, data.eval_id, sample_uuid, record)

        topology_payload = self._topology.finalize(sample_uuid)
        if topology_payload is not None:
            envelope["recorded_at"] = _now_iso()
            topology_record = ExecutionTopologyRecord(**envelope, **topology_payload)
            self._writer.write(run_id, data.eval_id, sample_uuid, topology_record)

        self._writer.close_sample(run_id, data.eval_id, sample_uuid)
        self._prefill.reset_sample(sample_uuid)
        self._topology.reset_sample(sample_uuid)
        self._context.clear_sample(sample_uuid)
