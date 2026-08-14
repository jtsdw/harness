"""Shared JSONL loading for the offline analysis modules (token_layer.py/episode_layer.py)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from inspect_ai.log import EvalSample, read_eval_log


def load_records_by_sample(trace_dir: Path) -> dict[str, list[dict]]:
    """All records (any `kind`) across every `sample-*.jsonl` under `trace_dir`, grouped by sample.

    `trace_dir` is whatever `INSPECT_TRACE_DIR` pointed at for the run(s) being analyzed -- pass
    the specific run's directory to scope to one run, or a parent directory to aggregate across
    several.
    """
    by_sample: dict[str, list[dict]] = {}
    for path in Path(trace_dir).glob("**/sample-*.jsonl"):
        for line in path.read_text().splitlines():
            if not line:
                continue
            record = json.loads(line)
            by_sample.setdefault(record["sample_uuid"], []).append(record)
    return by_sample


def load_manifest(trace_dir: Path) -> list[dict]:
    """Every `_manifest.jsonl` record under `trace_dir`, one per completed task.

    Gives `eval_id -> log_location` so episode_layer.py can join back to the real `.eval` log for
    timestamps and scorer results that inspect_trace itself never copies.
    """
    records = []
    for path in Path(trace_dir).glob("**/_manifest.jsonl"):
        for line in path.read_text().splitlines():
            if line:
                records.append(json.loads(line))
    return records


@dataclass(frozen=True)
class ModelEventIndexEntry:
    sample_uuid: str
    role: str
    interval: tuple[float, float] | None
    working_time: float | None


@dataclass
class EvalSampleIndex:
    samples: dict[str, EvalSample]
    model_events: dict[str, ModelEventIndexEntry]

    @property
    def roles(self) -> dict[str, str]:
        return {uuid: event.role for uuid, event in self.model_events.items()}


def load_eval_sample_index(trace_dir: Path) -> EvalSampleIndex:
    """Load each manifest log once and index samples and every model event UUID."""
    samples: dict[str, EvalSample] = {}
    model_events: dict[str, ModelEventIndexEntry] = {}
    loaded_locations: set[str] = set()
    for manifest in load_manifest(trace_dir):
        location = manifest.get("log_location")
        if not location or location in loaded_locations:
            continue
        loaded_locations.add(location)
        log = read_eval_log(location)
        for sample in log.samples or []:
            if sample.uuid is None:
                continue
            samples[sample.uuid] = sample
            for event in sample.events:
                if event.event == "model" and event.uuid is not None:
                    interval = (
                        (event.timestamp.timestamp(), event.completed.timestamp())
                        if event.completed is not None
                        else None
                    )
                    model_events[event.uuid] = ModelEventIndexEntry(
                        sample_uuid=sample.uuid,
                        role=(event.role or "").strip() or "unknown",
                        interval=interval,
                        working_time=event.working_time,
                    )
    return EvalSampleIndex(samples=samples, model_events=model_events)


def records_of_kind(records: list[dict], kind: str) -> list[dict]:
    return [r for r in records if r["kind"] == kind]
