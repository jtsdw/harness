import json
from datetime import datetime, timezone
from pathlib import Path

from appworld import AppWorld
from appworld.common.datetime import DateTime
from appworld.task import Task as AppWorldTask
from appworld_agents.code.simplified.agent import ExecutionIO
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.event import ModelEvent, ToolEvent
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ModelOutput,
    get_model,
)
from inspect_ai.solver import solver
from inspect_ai.tool import ToolCall

from appworld_adapter.solver import appworld_solver, execute_execution_inputs
from appworld_adapter.task import appworld_scorer


def test_appworld_freeze_does_not_freeze_inspect_timestamps(tmp_path, monkeypatch):
    (tmp_path / "data").symlink_to(
        Path("/data/agents-research/harness/.deps/appworld/data"),
        target_is_directory=True,
    )
    monkeypatch.setenv("APPWORLD_ROOT", str(tmp_path))
    monkeypatch.setenv("INSPECT_TRACE_DIR", str(tmp_path / "trace"))
    captured = {}

    @solver
    def timestamp_solver():
        async def solve(state, generate):
            with AppWorld("50e1ac9_1", experiment_name="timestamps") as world:
                captured["appworld_datetime"] = world.task.datetime
                captured["appworld_now"] = DateTime.now()
                await generate(state)
                await execute_execution_inputs(
                    world,
                    [
                        ExecutionIO(
                            content="print(supervisor__show_active_task())",
                            metadata={
                                "id": "timestamp-call",
                                "function_name": "supervisor__show_active_task",
                            },
                        )
                    ],
                    {
                        "tool_calls": [
                            {
                                "id": "timestamp-call",
                                "function": {
                                    "name": "supervisor__show_active_task",
                                    "arguments": "{}",
                                },
                            }
                        ]
                    },
                    [],
                )
            state.completed = True
            return state

        return solve

    before = datetime.now(timezone.utc)
    logs = eval(
        Task(
            dataset=[Sample(input="deterministic AppWorld timestamp boundary")],
            solver=timestamp_solver(),
        ),
        model="mockllm/model",
        max_samples=1,
        log_dir=str(tmp_path / "logs"),
    )
    after = datetime.now(timezone.utc)

    events = logs[0].samples[0].events
    runtime_events = [
        event for event in events if isinstance(event, (ModelEvent, ToolEvent))
    ]
    assert {type(event) for event in runtime_events} == {ModelEvent, ToolEvent}
    assert all(before <= event.timestamp <= after for event in runtime_events)

    trace_records = [
        json.loads(line)
        for path in (tmp_path / "trace").rglob("sample-*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert trace_records
    assert all(
        before <= datetime.fromisoformat(record["recorded_at"]) <= after
        for record in trace_records
    )
    assert captured["appworld_now"] == captured["appworld_datetime"]
    assert captured["appworld_datetime"] == AppWorldTask.load("50e1ac9_1").datetime
    assert captured["appworld_datetime"] != before.replace(tzinfo=None)


def test_execute_execution_inputs_matches_native_appworld(tmp_path, monkeypatch):
    (tmp_path / "data").symlink_to(
        Path("/data/agents-research/harness/.deps/appworld/data"),
        target_is_directory=True,
    )
    monkeypatch.setenv("APPWORLD_ROOT", str(tmp_path))

    task_id = "50e1ac9_1"
    execution_inputs = [
        ExecutionIO(
            content="print(supervisor__show_active_task(unexpected=True))",
            metadata={"id": "call-1", "function_name": "supervisor__show_active_task"},
        ),
        ExecutionIO(
            content="print(supervisor__complete_task(status='fail'))",
            metadata={"id": "call-2", "function_name": "supervisor__complete_task"},
        ),
        ExecutionIO(
            content="print(supervisor__show_active_task())",
            metadata={"id": "call-3", "function_name": "supervisor__show_active_task"},
        ),
        ExecutionIO(
            content="print(unknown__function())",
            metadata={"id": "call-4", "function_name": "unknown__function"},
        ),
    ]
    raw_assistant_message = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "supervisor__show_active_task",
                    "arguments": json.dumps({"unexpected": True}),
                },
            },
            {
                "id": "call-2",
                "type": "function",
                "function": {
                    "name": "supervisor__complete_task",
                    "arguments": json.dumps({"status": "fail"}),
                },
            },
            {
                "id": "call-3",
                "type": "function",
                "function": {
                    "name": "supervisor__show_active_task",
                    "arguments": json.dumps({}),
                },
            },
            {
                "id": "call-4",
                "type": "function",
                "function": {
                    "name": "unknown__function",
                    "arguments": json.dumps({}),
                },
            },
        ],
    }
    captured = {}

    @solver
    def execution_solver():
        async def solve(state, generate):
            native_world = AppWorld(
                task_id,
                experiment_name="native",
                include_direct_functions=True,
                direct_function_separator="__",
            )
            try:
                schemas = [
                    schema
                    for schema in native_world.task.api_docs.function_calling()
                    if schema["function"]["name"] == "supervisor__show_active_task"
                ]
                native_outputs = native_world.batch_execute(
                    [execution_input.content for execution_input in execution_inputs]
                )
                native_facts = (
                    list(native_world.environment_io),
                    native_world.num_interactions,
                    native_world.task_completed(),
                )
            finally:
                native_world.close()

            adapter_world = AppWorld(
                task_id,
                experiment_name="adapter",
                include_direct_functions=True,
                direct_function_separator="__",
            )
            try:
                adapter_outputs = await execute_execution_inputs(
                    adapter_world,
                    execution_inputs,
                    raw_assistant_message,
                    schemas,
                )
                adapter_facts = (
                    list(adapter_world.environment_io),
                    adapter_world.num_interactions,
                    adapter_world.task_completed(),
                )
            finally:
                adapter_world.close()

            native_contents = native_outputs
            assert [output.content for output in adapter_outputs] == native_contents
            assert adapter_facts[0] == native_facts[0]
            assert native_facts[1] == adapter_facts[1] == 1
            assert [item["number"] for item in adapter_facts[0]] == [
                "1.1",
                "1.2",
                "1.3",
                "1.4",
            ]
            assert native_facts[2] is adapter_facts[2] is True
            captured["native_contents"] = native_contents
            state.completed = True
            return state

        return solve

    logs = eval(
        Task(
            dataset=[Sample(input="deterministic AppWorld boundary")],
            solver=execution_solver(),
        ),
        model="mockllm/model",
        max_samples=1,
        log_dir=str(tmp_path / "logs"),
    )

    assert len(logs) == 1
    assert logs[0].status == "success"
    tool_events = [
        event for event in logs[0].samples[0].events if isinstance(event, ToolEvent)
    ]
    assert [event.id for event in tool_events] == [
        "call-1",
        "call-2",
        "call-3",
        "call-4",
    ]
    assert [event.function for event in tool_events] == [
        "supervisor__show_active_task",
        "supervisor__complete_task",
        "supervisor__show_active_task",
        "unknown__function",
    ]
    assert [event.result for event in tool_events] == captured["native_contents"]
    assert all(event.completed > event.timestamp for event in tool_events)
    assert all(event.working_time > 0 for event in tool_events)
    assert all(event.error is None for event in tool_events)


def test_solver_call_ledger_conserves_calls_and_observations(tmp_path, monkeypatch):
    (tmp_path / "data").symlink_to(
        Path("/data/agents-research/harness/.deps/appworld/data"),
        target_is_directory=True,
    )
    monkeypatch.setenv("APPWORLD_ROOT", str(tmp_path))
    monkeypatch.setattr(AppWorld.init_defaults, "max_api_calls_per_interaction", 2)

    predictor = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.from_content(
                model="mockllm/model",
                content="supervisor__show_active_task\nsupervisor__complete_task",
            )
        ],
    )
    main = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput(
                model="mockllm/model",
                choices=[
                    ChatCompletionChoice(
                        message=ChatMessageAssistant(
                            content="",
                            tool_calls=[
                                ToolCall(
                                    id="accepted",
                                    function="supervisor__show_active_task",
                                    arguments={"unexpected": True},
                                ),
                                ToolCall(id="invalid", function="invalid", arguments={}),
                                ToolCall(
                                    id="truncated",
                                    function="supervisor__show_active_task",
                                    arguments={},
                                ),
                            ],
                        ),
                        stop_reason="tool_calls",
                    )
                ],
            )
        ],
    )
    task = Task(
        dataset=[
            Sample(
                input="exercise the AppWorld call ledger",
                metadata={"appworld_task_id": "50e1ac9_1"},
            )
        ],
        solver=appworld_solver(experiment_name="ledger", max_steps=2),
        scorer=appworld_scorer(),
        model=main,
        model_roles={"predictor": predictor, "main": main},
    )

    logs = eval(task, log_dir=str(tmp_path / "logs"))

    assert logs[0].status == "success"
    sample = logs[0].samples[0]
    score = next(iter(sample.scores.values()))
    ledger = score.metadata["call_ledger"]
    assert score.metadata["termination"] == "max_steps"
    assert len(ledger) == 1
    round_entry = ledger[0]
    assert len(round_entry["proposed"]) == len(round_entry["accepted"]) + len(
        round_entry["dropped"]
    )
    assert [call["id"] for call in round_entry["accepted"]] == ["accepted"]
    assert {call["id"]: call["reason"] for call in round_entry["dropped"]} == {
        "invalid": "invalid_function_name",
        "truncated": "max_call_truncation",
    }
    assert round_entry["executed"] == [
        {
            "id": "accepted",
            "function_name": "supervisor__show_active_task",
            "inspect_outcome": "success",
            "appworld_outcome": "environment_failure",
            "observation_status": "not_consumed_after_termination",
        }
    ]
    tool_events = [event for event in sample.events if isinstance(event, ToolEvent)]
    assert [event.id for event in tool_events] == ["accepted"]
    assert tool_events[0].error is None

    predictor = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.from_content(
                model="mockllm/model", content="supervisor__show_active_task"
            )
        ],
    )
    first_call = ModelOutput.for_tool_call(
        model="mockllm/model",
        tool_name="supervisor__show_active_task",
        tool_arguments={},
    )
    first_call.message.tool_calls[0].id = "written-back"
    main = get_model(
        "mockllm/model",
        custom_outputs=[
            first_call,
            ModelOutput(model="mockllm/model", error="provider unavailable"),
        ],
    )
    task = Task(
        dataset=[
            Sample(
                input="exercise provider failure after an observation",
                metadata={"appworld_task_id": "50e1ac9_1"},
            )
        ],
        solver=appworld_solver(experiment_name="ledger_provider_error", max_steps=3),
        scorer=appworld_scorer(),
        model=main,
        model_roles={"predictor": predictor, "main": main},
    )

    logs = eval(task, log_dir=str(tmp_path / "provider_logs"))

    sample = logs[0].samples[0]
    score = next(iter(sample.scores.values()))
    ledger = score.metadata["call_ledger"]
    assert score.metadata["termination"] == "model_error"
    assert ledger[0]["executed"][0]["observation_status"] == "written_back"
    assert ledger[1]["provider_error"] == "provider unavailable"
    assert ledger[1]["proposed"] == ledger[1]["accepted"] == []
    assert ledger[1]["dropped"] == ledger[1]["executed"] == []
