"""Shared JSONL loading for the offline analysis modules (token_layer.py/episode_layer.py)."""

from __future__ import annotations

import json
from pathlib import Path


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


def records_of_kind(records: list[dict], kind: str) -> list[dict]:
    return [r for r in records if r["kind"] == kind]
