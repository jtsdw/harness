"""Stable content hashing for `ChatMessage`s.

`Model._generate()` runs several transforms on `input` before `ModelEvent.input` is captured
(reasoning-history resolution, tool-result-media rewriting, consecutive-message collapsing) that can
produce new `ChatMessage` objects — with a new `.id` — for content whose semantic value is unchanged.
Object identity and `.id` equality are therefore not reliable across steps; hashing excludes `id` so
two structurally-identical messages always hash the same regardless of provenance.
"""

from __future__ import annotations

import hashlib
import json

from inspect_ai.model import ChatMessage
from inspect_ai.tool import ToolInfo


def hash_message(message: ChatMessage) -> str:
    payload = message.model_dump(mode="json", exclude={"id"})
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hash_message_list(messages: list[ChatMessage]) -> str:
    joined = "\n".join(hash_message(m) for m in messages)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def hash_tool(tool: ToolInfo) -> str:
    # ToolInfo has no volatile identity field (unlike ChatMessage.id), so the full dump is hashed.
    payload = tool.model_dump(mode="json")
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
