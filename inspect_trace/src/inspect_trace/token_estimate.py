"""Standalone tiktoken-based token estimator.

`inspect_ai.model._tokens.count_text_tokens` is private (not re-exported from any public
`inspect_ai` module) and is deliberately tuned to over-count for context-compaction safety, so it is
not reused here. This module estimates independently, using the same `o200k_base` encoding as
inspect_ai's internal estimator, and every value it produces is labelled `estimated_*` by callers —
never merged into or presented as a real provider-billed token count.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from inspect_ai.tool import ToolInfo

if TYPE_CHECKING:
    import tiktoken

_encoding: "tiktoken.Encoding | None" = None


def _get_encoding() -> "tiktoken.Encoding":
    global _encoding
    if _encoding is None:
        import tiktoken

        _encoding = tiktoken.get_encoding("o200k_base")
    return _encoding


def estimate_text_tokens(text: str | None) -> int:
    if not text:
        return 0
    return len(_get_encoding().encode(text, disallowed_special=()))


def estimate_tool_call_tokens(function: str, arguments: dict[str, Any]) -> int:
    payload = f"{function}({json.dumps(arguments, sort_keys=True, default=str)})"
    return estimate_text_tokens(payload)


def estimate_tool_schema_tokens(tool: ToolInfo) -> int:
    """Estimate the serialized size of a `ToolInfo` definition.

    Name + description + JSON Schema parameters -- this is what gets resent, unchanged, on every
    call that offers the same tool.
    """
    payload = tool.model_dump(mode="json")
    return estimate_text_tokens(json.dumps(payload, sort_keys=True, default=str))
