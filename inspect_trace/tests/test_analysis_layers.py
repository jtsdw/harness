"""End-to-end validation for 目标二 token_layer.py / episode_layer.py.

Same style as the other inspect_trace tests: drive a real `eval()` run through
`mockllm/model` so the offline analysis is exercised against genuine Hooks output, not a
hand-built fixture.
"""

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
    assert episode.end_to_end_latency_seconds is not None
    # critical path == end-to-end always (same number by definition, see episode_layer.py).
    assert episode.critical_path_latency_seconds == episode.end_to_end_latency_seconds
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
    # end-to-end is still the same total wall-clock number...
    assert episode.critical_path_latency_seconds == episode.end_to_end_latency_seconds
    # ...but real overlap must show up as positive concurrency savings (naive exclusive sum
    # double-counts the overlapping window; total_busy_seconds doesn't).
    assert episode.concurrency_savings_seconds > 0.0
    assert episode.total_busy_seconds is not None
    assert (
        episode.total_busy_seconds
        < episode.exclusive_model_seconds + episode.exclusive_tool_seconds
    )
