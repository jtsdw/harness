"""End-to-end validation for 目标二 token_layer.py / episode_layer.py.

Same style as the other inspect_trace tests: drive a real `eval()` run through
`mockllm/model` so the offline analysis is exercised against genuine Hooks output, not a
hand-built fixture.
"""

import json
from pathlib import Path

import anyio
import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ModelOutput,
    ModelUsage,
    get_model,
)
from inspect_ai.scorer import includes
from inspect_ai.solver import basic_agent, solver
from inspect_ai.tool import ToolCall, tool

from inspect_trace.analysis import episode_layer, token_layer


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


def _assistant_output(*, reasoning: str, tool_call_id: str) -> ModelOutput:
    from inspect_ai._util.content import ContentReasoning, ContentText

    return ModelOutput(
        model="mockllm/model",
        choices=[
            ChatCompletionChoice(
                message=ChatMessageAssistant(
                    content=[
                        ContentReasoning(reasoning=reasoning),
                        ContentText(text="calling tool_a"),
                    ],
                    model="mockllm/model",
                    source="generate",
                    tool_calls=[
                        ToolCall(
                            id=tool_call_id, function="tool_a", arguments={"value": "x"}
                        )
                    ],
                ),
                stop_reason="tool_calls",
            )
        ],
    )


def _run(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("INSPECT_TRACE_DIR", str(tmp_path))

    task = Task(
        dataset=[Sample(input="call tool_a twice then submit", target="done")],
        solver=basic_agent(tools=[tool_a()]),
        scorer=includes(),
        message_limit=10,
    )
    model = get_model(
        "mockllm/model",
        custom_outputs=[
            _assistant_output(reasoning="first call", tool_call_id="call-1"),
            _assistant_output(reasoning="second call", tool_call_id="call-2"),
            ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="submit",
                tool_arguments={"answer": "done"},
            ),
        ],
    )
    logs = eval(task, model=model)
    assert logs[0].status == "success"
    return tmp_path


def test_token_layer_summarizes_real_hooks_output(tmp_path: Path, monkeypatch) -> None:
    trace_dir = _run(tmp_path, monkeypatch)
    run_summary = token_layer.summarize_run(trace_dir)

    assert run_summary.n_episodes == 1
    episode = run_summary.per_episode[0]
    assert episode.n_model_calls == 3
    # step 1 all-new tool schema, steps 2-3 reused -- reuse must be > 0.
    assert run_summary.total_reused_tool_schema_tokens_estimate > 0
    assert episode.billed_output_tokens is not None and episode.billed_output_tokens > 0
    # no retries in this trajectory -> nothing wasted.
    assert episode.retry_wasted_output_tokens_estimate == 0


def test_episode_layer_summarizes_real_hooks_output(
    tmp_path: Path, monkeypatch
) -> None:
    trace_dir = _run(tmp_path, monkeypatch)
    run_summary = episode_layer.summarize_run(trace_dir)

    assert run_summary.n_episodes == 1
    episode = run_summary.per_episode[0]
    assert episode.n_llm_calls == 3
    assert (
        episode.n_tool_calls == 3
    )  # tool_a x2 + basic_agent's auto-injected submit call
    assert episode.success is True
    assert run_summary.success_rate == 1.0
    # sequential mock trajectory -- no concurrent tool execution.
    assert episode.observed_parallel is False
    assert episode.sample_end_to_end_latency_seconds is not None
    assert episode.sample_working_time_seconds is not None
    assert episode.model_tool_window_seconds is not None
    assert episode.total_busy_seconds is not None
    assert (
        episode.sample_end_to_end_latency_seconds
        >= episode.model_tool_window_seconds
        >= episode.total_busy_seconds
    )
    assert episode.concurrency_savings_seconds == 0.0
    assert episode.n_retries == 0


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


def test_episode_layer_detects_real_concurrency(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("INSPECT_TRACE_DIR", str(tmp_path))

    task = Task(
        dataset=[Sample(input="call tool_slow twice in parallel", target="done")],
        solver=basic_agent(tools=[tool_slow()]),
        scorer=includes(),
        message_limit=10,
    )
    model = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput(
                model="mockllm/model",
                choices=[
                    ChatCompletionChoice(
                        message=ChatMessageAssistant(
                            content="calling tool_slow twice",
                            model="mockllm/model",
                            source="generate",
                            tool_calls=[
                                ToolCall(
                                    id="call-1",
                                    function="tool_slow",
                                    arguments={"value": "a"},
                                ),
                                ToolCall(
                                    id="call-2",
                                    function="tool_slow",
                                    arguments={"value": "b"},
                                ),
                            ],
                        ),
                        stop_reason="tool_calls",
                    )
                ],
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

    run_summary = episode_layer.summarize_run(tmp_path)
    episode = run_summary.per_episode[0]
    assert episode.observed_parallel is True
    assert episode.sample_end_to_end_latency_seconds is not None
    assert episode.model_tool_window_seconds is not None
    assert episode.total_busy_seconds is not None
    assert (
        episode.sample_end_to_end_latency_seconds
        >= episode.model_tool_window_seconds
        >= episode.total_busy_seconds
    )
    # Real overlap must show up as positive concurrency savings.
    assert episode.concurrency_savings_seconds > 0.0
    assert episode.concurrency_savings_seconds == pytest.approx(
        episode.exclusive_model_seconds
        + episode.exclusive_tool_seconds
        - episode.total_busy_seconds
    )
    assert (
        episode.total_busy_seconds
        < episode.exclusive_model_seconds + episode.exclusive_tool_seconds
    )


def test_role_rollups_and_usage_validation_real_eval(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("INSPECT_TRACE_DIR", str(tmp_path))

    def output(text: str, input_tokens: int, output_tokens: int) -> ModelOutput:
        result = ModelOutput.from_content(model="mockllm/model", content=text)
        result.usage = ModelUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
        return result

    predictor = get_model("mockllm/model", custom_outputs=[output("predict", 11, 2)])
    main = get_model("mockllm/model", custom_outputs=[output("main", 13, 3)])
    unknown = get_model("mockllm/model", custom_outputs=[output("done", 17, 5)])

    @solver
    def role_calls():
        async def solve(state, generate):
            await get_model(role="predictor").generate(state.messages)
            await get_model(role="main").generate(state.messages)
            state.output = await unknown.generate(state.messages)
            return state

        return solve

    task = Task(
        dataset=[Sample(input="exercise roles", target="done")],
        solver=role_calls(),
        scorer=includes(),
        model_roles={"predictor": predictor, "main": main},
    )
    logs = eval(task, model=unknown)
    assert logs[0].status == "success"

    tokens = token_layer.summarize_run(tmp_path)
    episodes = episode_layer.summarize_run(tmp_path)
    assert {role: summary.n_model_calls for role, summary in tokens.per_role.items()} == {
        "predictor": 1,
        "main": 1,
        "unknown": 1,
    }
    assert {
        role: (summary.billed_input_tokens, summary.billed_output_tokens)
        for role, summary in tokens.per_role.items()
    } == {"predictor": (11, 2), "main": (13, 3), "unknown": (17, 5)}
    assert episodes.per_role_model_calls == {"predictor": 1, "main": 1, "unknown": 1}
    assert all(value > 0 for value in episodes.per_role_model_latency_seconds.values())

    trace_file = next(tmp_path.rglob("sample-*.jsonl"))
    lines = trace_file.read_text().splitlines()
    for index, line in enumerate(lines):
        record = json.loads(line)
        if record.get("kind") == "token_attribution" and record["billed_input_tokens"] == 11:
            record["billed_input_tokens"] = 12
            lines[index] = json.dumps(record)
            break
    else:
        raise AssertionError("predictor token attribution not found")
    trace_file.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="role_usage mismatch for predictor"):
        token_layer.summarize_run(tmp_path)
