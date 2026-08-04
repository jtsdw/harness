"""End-to-end validation for 缺口C (retry attempt grouping).

Driven through a real `eval()` run: a `mockllm/model` callable that sleeps past `attempt_timeout`
on its first invocation (deterministically triggering the real tenacity retry path via
`AttemptTimeoutError`) and returns immediately afterward.
"""

import json
from pathlib import Path

import anyio
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatMessage, GenerateConfig, ModelOutput, get_model
from inspect_ai.scorer import includes
from inspect_ai.solver import basic_agent
from inspect_ai.tool import ToolChoice, ToolInfo


def test_attempt_group_records_a_real_retry(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("INSPECT_TRACE_DIR", str(tmp_path))

    calls = {"n": 0}

    async def outputs(
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        calls["n"] += 1
        if calls["n"] == 1:
            await anyio.sleep(
                2
            )  # exceeds attempt_timeout=1, gets cancelled -> AttemptTimeoutError
        return ModelOutput.for_tool_call(
            model="mockllm/model", tool_name="submit", tool_arguments={"answer": "done"}
        )

    model = get_model(
        "mockllm/model",
        config=GenerateConfig(max_retries=1, attempt_timeout=1),
        custom_outputs=outputs,
    )

    task = Task(
        dataset=[Sample(input="say done", target="done")],
        solver=basic_agent(tools=[]),
        scorer=includes(),
        message_limit=5,
    )

    logs = eval(task, model=model)
    assert logs[0].status == "success"

    jsonl_files = list(tmp_path.rglob("sample-*.jsonl"))
    assert len(jsonl_files) == 1
    records = [json.loads(line) for line in jsonl_files[0].read_text().splitlines()]
    attempt_records = [r for r in records if r["kind"] == "attempt_group"]

    assert len(attempt_records) == 1
    record = attempt_records[0]
    assert record["resolution"] == "success"
    assert record["total_attempts"] == 2
    assert record["attempts"][0]["outcome"] == "error"
    assert record["attempts"][1]["outcome"] == "success"
    assert record["attempts"][1]["model_event_uuid"] is not None
    assert record["total_wasted_wait_time"] == record["attempts"][0]["wait_time_after"]

    # Cross-check against ground truth: every ModelEvent inspect_ai itself logged for this
    # sample must be accounted for by exactly one attempt in our derived records -- no
    # double-counting, nothing dropped.
    eval_log = read_eval_log(logs[0].location)
    assert eval_log.samples is not None
    sample = eval_log.samples[0]
    model_event_count = sum(1 for e in sample.events if e.event == "model")
    assert model_event_count == sum(r["total_attempts"] for r in attempt_records)
