"""End-to-end validation for 需求四 (action_parsing), driven through a real `eval()` run.

The "malformed" scenario omits the tool's required argument, which deterministically triggers
inspect_ai's own real `validate_tool_input()` -> `ToolParsingError` -> `ToolCallError("parsing", ...)`
path (`model/_call_tools.py`) rather than simulating an error out-of-band.
"""

import json
from pathlib import Path

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ModelOutput,
    get_model,
)
from inspect_ai.scorer import includes
from inspect_ai.solver import basic_agent
from inspect_ai.tool import ToolCall, tool


@tool
def tool_a():
    async def execute(value: str):
        """Echo a value.

        Args:
            value (str): Value to echo.

        Returns:
            The same value.
        """
        return value

    return execute


def _assistant_output(*, tool_calls: list[ToolCall], text: str) -> ModelOutput:
    return ModelOutput(
        model="mockllm/model",
        choices=[
            ChatCompletionChoice(
                message=ChatMessageAssistant(
                    content=text,
                    model="mockllm/model",
                    source="generate",
                    tool_calls=tool_calls,
                ),
                stop_reason="tool_calls",
            )
        ],
    )


def test_action_parsing_records_a_real_validation_error(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("INSPECT_TRACE_DIR", str(tmp_path))

    task = Task(
        dataset=[
            Sample(
                input="call tool_a with a missing argument, then submit", target="done"
            )
        ],
        solver=basic_agent(tools=[tool_a()]),
        scorer=includes(),
        message_limit=10,
    )

    model = get_model(
        "mockllm/model",
        custom_outputs=[
            _assistant_output(
                # required "value" argument omitted -> real schema validation failure
                tool_calls=[ToolCall(id="call-1", function="tool_a", arguments={})],
                text="calling tool_a with a missing argument",
            ),
            ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="submit",
                tool_arguments={"answer": "done"},
            ),
        ],
    )

    logs = eval(task, model=model)
    assert logs[0].status == "success"

    jsonl_files = list(tmp_path.rglob("sample-*.jsonl"))
    assert len(jsonl_files) == 1
    records = [json.loads(line) for line in jsonl_files[0].read_text().splitlines()]
    action_records = [r for r in records if r["kind"] == "action_parsing"]
    prefill_records = [r for r in records if r["kind"] == "prefill_diff"]

    # basic_agent's auto-injected `submit` tool call produces its own action_parsing record too.
    assert len(action_records) == 2
    record = next(r for r in action_records if r["tool_call_id"] == "call-1")
    assert record["error_present"] is True
    assert record["error_type"] == "parsing"
    assert record["is_parse_or_validation_error"] is True
    assert record["error_message"]
    assert record["parent_model_event_uuid"] == prefill_records[0]["model_event_uuid"]

    # Join key check: the write-back message in the next step's prefill_diff has a matching
    # tool_call_id.
    next_step_messages = prefill_records[1]["messages"]
    matching = [m for m in next_step_messages if m["tool_call_id"] == "call-1"]
    assert len(matching) == 1
    assert matching[0]["function"] == "tool_a"


def test_action_parsing_no_error_on_clean_call(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("INSPECT_TRACE_DIR", str(tmp_path))

    task = Task(
        dataset=[Sample(input="call tool_a cleanly, then submit", target="done")],
        solver=basic_agent(tools=[tool_a()]),
        scorer=includes(),
        message_limit=10,
    )

    model = get_model(
        "mockllm/model",
        custom_outputs=[
            _assistant_output(
                tool_calls=[
                    ToolCall(id="call-1", function="tool_a", arguments={"value": "x"})
                ],
                text="calling tool_a",
            ),
            ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="submit",
                tool_arguments={"answer": "done"},
            ),
        ],
    )

    logs = eval(task, model=model)
    assert logs[0].status == "success"

    jsonl_files = list(tmp_path.rglob("sample-*.jsonl"))
    records = [json.loads(line) for line in jsonl_files[0].read_text().splitlines()]
    action_records = [r for r in records if r["kind"] == "action_parsing"]

    assert len(action_records) == 2
    assert all(r["error_present"] is False for r in action_records)
    assert all(r["error_type"] is None for r in action_records)
