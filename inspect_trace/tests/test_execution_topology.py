"""End-to-end validation for 需求三 (execution_topology), driven through a real `eval()` run.

Scenario A forces genuinely overlapping `ToolEvent` windows (not just "fast enough to look
parallel") by having a `parallel=True` tool sleep past the moment its sibling call starts, so
`observed_parallel` reflects real concurrency rather than a timing coincidence.
"""

import json
from pathlib import Path

import anyio
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


@tool(parallel=True)
def tool_slow():
    async def execute(value: str):
        """Do something slowly.

        Args:
            value (str): Value to process.

        Returns:
            The same value.
        """
        await anyio.sleep(0.05)
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


def test_execution_topology_detects_real_parallelism(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("INSPECT_TRACE_DIR", str(tmp_path))

    task = Task(
        dataset=[
            Sample(
                input="call tool_slow twice in parallel, then once more", target="done"
            )
        ],
        solver=basic_agent(tools=[tool_slow()]),
        scorer=includes(),
        message_limit=10,
    )

    model = get_model(
        "mockllm/model",
        custom_outputs=[
            _assistant_output(
                tool_calls=[
                    ToolCall(
                        id="call-1", function="tool_slow", arguments={"value": "a"}
                    ),
                    ToolCall(
                        id="call-2", function="tool_slow", arguments={"value": "b"}
                    ),
                ],
                text="calling tool_slow twice",
            ),
            _assistant_output(
                tool_calls=[
                    ToolCall(
                        id="call-3", function="tool_slow", arguments={"value": "c"}
                    )
                ],
                text="calling tool_slow once more",
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
    topology_records = [r for r in records if r["kind"] == "execution_topology"]

    assert len(topology_records) == 1
    record = topology_records[0]
    # basic_agent's auto-injected `submit` tool call is its own trailing stage, on top of the
    # two tool_slow stages this test is actually about.
    assert record["total_stages"] == 3
    assert record["linear"] is False
    assert record["unmatched_tool_event_uuids"] == []

    stage1, stage2, stage3 = record["stages"]
    assert stage1["tool_count"] == 2
    assert stage1["observed_parallel"] is True
    assert stage1["max_observed_concurrency"] == 2
    assert all(t >= 0 for t in stage1["tool_waiting_for_model_seconds"])
    assert stage1["model_waiting_for_tool_seconds"] is not None
    assert stage1["model_waiting_for_tool_seconds"] >= 0

    assert stage2["tool_count"] == 1
    assert stage2["observed_parallel"] is False
    assert stage2["max_observed_concurrency"] == 1
    assert stage2["model_waiting_for_tool_seconds"] is not None

    assert stage3["tool_functions"] == ["submit"]
    # Last stage: no following ModelEvent to wait for.
    assert stage3["model_waiting_for_tool_seconds"] is None


def test_execution_topology_pure_linear(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("INSPECT_TRACE_DIR", str(tmp_path))

    task = Task(
        dataset=[
            Sample(input="call tool_slow three times sequentially", target="done")
        ],
        solver=basic_agent(tools=[tool_slow()]),
        scorer=includes(),
        message_limit=10,
    )

    model = get_model(
        "mockllm/model",
        custom_outputs=[
            _assistant_output(
                tool_calls=[
                    ToolCall(
                        id=f"call-{i}", function="tool_slow", arguments={"value": "a"}
                    )
                ],
                text=f"call {i}",
            )
            for i in range(1, 4)
        ]
        + [
            ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="submit",
                tool_arguments={"answer": "done"},
            )
        ],
    )

    logs = eval(task, model=model)
    assert logs[0].status == "success"

    jsonl_files = list(tmp_path.rglob("sample-*.jsonl"))
    records = [json.loads(line) for line in jsonl_files[0].read_text().splitlines()]
    topology_records = [r for r in records if r["kind"] == "execution_topology"]

    assert len(topology_records) == 1
    record = topology_records[0]
    assert record["linear"] is True
    # 3 tool_slow stages + 1 trailing `submit` stage.
    assert record["total_stages"] == 4
    assert all(s["tool_count"] == 1 for s in record["stages"])
    assert all(s["observed_parallel"] is False for s in record["stages"])
    assert record["stages"][-1]["tool_functions"] == ["submit"]
    assert record["unmatched_tool_event_uuids"] == []
