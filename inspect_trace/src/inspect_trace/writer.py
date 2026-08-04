"""JSONL persistence, one open file handle per (run_id, eval_id, sample_uuid).

Also maintains a small process-wide manifest mapping each `eval_id` to the real `.eval` log
location. Writes are plain synchronous `write()` calls with no `await` in between compute and
write, which -- per this repo's own concurrency reasoning for a single event-loop thread -- needs
no lock; per-sample writes are in any case already serialized by inspect_ai itself (one background
event-emitter task per `ActiveSample`).
"""

from __future__ import annotations

from typing import TextIO

from pydantic import BaseModel

from . import config


class TraceWriter:
    def __init__(self) -> None:
        self._handles: dict[str, TextIO] = {}

    def _key(self, run_id: str | None, eval_id: str, sample_uuid: str) -> str:
        return f"{run_id or '_no_run'}:{eval_id}:{sample_uuid}"

    def write(
        self, run_id: str | None, eval_id: str, sample_uuid: str, record: BaseModel
    ) -> None:
        key = self._key(run_id, eval_id, sample_uuid)
        handle = self._handles.get(key)
        if handle is None:
            path = config.sample_file(run_id or "_no_run", eval_id, sample_uuid)
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = open(path, "a", encoding="utf-8")
            self._handles[key] = handle
        handle.write(record.model_dump_json() + "\n")
        handle.flush()

    def close_sample(self, run_id: str | None, eval_id: str, sample_uuid: str) -> None:
        key = self._key(run_id, eval_id, sample_uuid)
        handle = self._handles.pop(key, None)
        if handle is not None:
            handle.close()

    def write_manifest(self, record: BaseModel) -> None:
        path = config.manifest_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")
