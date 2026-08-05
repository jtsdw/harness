"""Builds inspect_ai Samples directly from ToolSpec's own bundled API-Bank data.

Reuses ToolSpec's data files unmodified (data/API-Bank/level-{1,2,3}-api_processed.json), same
question_id numbering as evaluation/eval.py's own `load_questions()` (count across the three
levels in order), so question_id lines up with the native-repo runs from phase 1 for a direct
per-question comparison.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser

TOOLSPEC_REPO_DIR = os.environ.get("TOOLSPEC_REPO_DIR", "/home/liuyingen/code/ToolSpec")


def api_bank_samples(limit: int | None = None) -> list[Sample]:
    samples: list[Sample] = []
    count = 0
    for level in ("1", "2", "3"):
        path = Path(TOOLSPEC_REPO_DIR) / "data" / "API-Bank" / f"level-{level}-api_processed.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data:
            samples.append(
                Sample(
                    id=count,
                    input=[
                        ChatMessageSystem(content=item["system"]),
                        ChatMessageUser(content=item["user"]),
                    ],
                    metadata={"question_id": count, "level": level},
                )
            )
            count += 1
            if limit is not None and count >= limit:
                return samples
    return samples
