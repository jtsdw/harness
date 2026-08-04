"""Output directory resolution.

inspect_ai's hook payloads do not carry a per-`eval()` log directory that would let us co-locate
derived facts next to the original `.eval` log (only `EvalSetStart/End.log_dir` exists, and only for
`eval_set()`). Instead derived facts go under a configurable directory of their own, keyed by
`run_id`/`eval_id` (both globally unique per invocation), and a manifest records the real `.eval` log
location for each `eval_id` so downstream tooling can join the two.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_TRACE_DIR = ".inspect_trace"


def trace_dir() -> Path:
    return Path(os.environ.get("INSPECT_TRACE_DIR", DEFAULT_TRACE_DIR))


def eval_dir(run_id: str, eval_id: str) -> Path:
    return trace_dir() / run_id / eval_id


def sample_file(run_id: str, eval_id: str, sample_uuid: str) -> Path:
    return eval_dir(run_id, eval_id) / f"sample-{sample_uuid}.jsonl"


def manifest_file() -> Path:
    return trace_dir() / "_manifest.jsonl"
