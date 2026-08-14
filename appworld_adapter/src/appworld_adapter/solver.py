import json
import os
from collections.abc import Sequence
from typing import Any

import anyio
import appworld
from appworld import evaluate_task
from appworld.common.path_store import path_store
from appworld_agents.code.common.usage_tracker import CostPerToken
from appworld_agents.code.simplified.agent import ExecutionIO
from appworld_agents.code.simplified.function_calling_agent import SimplifiedFunctionCallingAgent
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, execute_tools, get_model
from inspect_ai.solver import Solver, TaskState, solver
from inspect_ai.tool import ToolCall, ToolDef, ToolParams

from .model import InspectLanguageModel


_APPWORLD_SAMPLE_LOCK = anyio.Lock()


def _build_agent(
    predictor, main, max_steps: int, temperature: float, seed: int, max_tokens: int
) -> SimplifiedFunctionCallingAgent:
    zero_cost = {
        "input_cache_miss": 0.0,
        "input_cache_hit": 0.0,
        "input_cache_write": 0.0,
        "output": 0.0,
    }
    placeholder_model_config = {
        "name": "inspect-placeholder",
        "client_name": "litellm",
        "use_cache": False,
        "cost_per_token": zero_cost,
    }
    previous = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = "unused-placeholder"
    try:
        agent = SimplifiedFunctionCallingAgent(
            model_config=placeholder_model_config,
            api_predictor_config={
                "mode": "predicted",
                "model_config": placeholder_model_config.copy(),
                "prompt_file_path": os.path.join(path_store.experiment_prompts, "api_predictor.txt"),
                "demo_task_ids": ["82e2fac_1", "29caf6f_1", "d0b1f43_1"],
                "max_predicted_apis": 20,
            },
            prompt_file_path=os.path.join(
                path_store.experiment_prompts, "function_calling_agent", "instructions.txt"
            ),
            demo_messages_file_path=os.path.join(
                path_store.experiment_prompts, "function_calling_agent", "demos.json"
            ),
            appworld_config={
                "random_seed": seed,
                "raise_on_extra_parameters": True,
                "include_direct_functions": True,
                "direct_function_separator": "__",
            },
            logger_config={"color": False, "verbose": True},
            usage_tracker_config={
                "max_cost_overall": 1000,
                "max_cost_per_task": 10,
                "max_output_tokens_per_task": 100000,
            },
            max_steps=max_steps,
            log_lm_calls=False,
            skip_if_finished=False,
        )
    finally:
        if previous is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = previous
    agent.api_predictor.language_model = InspectLanguageModel(
        predictor,
        CostPerToken(**zero_cost),
        {"temperature": temperature, "seed": seed, "max_tokens": max_tokens},
    )
    agent.language_model = InspectLanguageModel(
        main,
        CostPerToken(**zero_cost),
        {
            "temperature": temperature,
            "seed": seed,
            "max_tokens": max_tokens,
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        },
        retain_call_audit=True,
    )
    return agent


async def execute_execution_inputs(
    world: appworld.AppWorld,
    execution_inputs: Sequence[ExecutionIO],
    raw_assistant_message: dict[str, Any],
    functions: list[dict[str, Any]],
) -> list[ExecutionIO]:
    if not execution_inputs:
        raise ValueError("execution_inputs must not be empty")

    raw_tool_calls = raw_assistant_message.get("tool_calls", []) or []
    accepted_ids: set[str] = set()
    tool_calls: list[ToolCall] = []
    required_names: list[str] = []
    for execution_input in execution_inputs:
        call_id = execution_input.metadata["id"]
        function_name = execution_input.metadata["function_name"]
        if call_id in accepted_ids:
            raise ValueError(f"Duplicate accepted tool call ID: {call_id!r}")
        accepted_ids.add(call_id)

        matches = [
            raw_tool_call
            for raw_tool_call in raw_tool_calls
            if raw_tool_call.get("call_id", raw_tool_call.get("id")) == call_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one raw tool call for accepted ID {call_id!r}, "
                f"found {len(matches)}"
            )
        raw_tool_call = matches[0]
        raw_function = raw_tool_call["function"]
        if raw_function["name"] != function_name:
            raise ValueError(
                f"Function name mismatch for tool call {call_id!r}: "
                f"{raw_function['name']!r} != {function_name!r}"
            )
        try:
            arguments = json.loads(raw_function["arguments"])
        except json.JSONDecodeError:
            arguments = {}
        if not isinstance(arguments, dict):
            raise TypeError(f"Tool call arguments for {call_id!r} must decode to an object")
        tool_calls.append(ToolCall(id=call_id, function=function_name, arguments=arguments))
        if function_name not in required_names:
            required_names.append(function_name)

    schemas_by_name: dict[str, dict[str, Any]] = {}
    for schema in functions:
        function_schema = schema["function"]
        name = function_schema["name"]
        if name not in required_names:
            continue
        if name in schemas_by_name:
            raise ValueError(f"Duplicate function schema for required function {name!r}")
        schemas_by_name[name] = function_schema
    missing_names = [name for name in required_names if name not in schemas_by_name]
    if missing_names:
        fallback_names = set(missing_names)
        for schema in world.task.api_docs.function_calling():
            function_schema = schema["function"]
            name = function_schema["name"]
            if name not in fallback_names:
                continue
            if name in schemas_by_name:
                raise ValueError(f"Duplicate function schema for required function {name!r}")
            schemas_by_name[name] = function_schema
        missing_names = [name for name in required_names if name not in schemas_by_name]
        if missing_names:
            for name in missing_names:
                schemas_by_name[name] = {"name": name}

    cursor = 0
    starting_num_interactions = world.num_interactions
    batch_size = len(execution_inputs)

    def make_wrapper(expected_name: str) -> Any:
        async def wrapper(**kwargs: Any) -> str:
            nonlocal cursor
            if cursor >= batch_size:
                raise RuntimeError(f"Unexpected extra tool execution for {expected_name!r}")
            execution_input = execution_inputs[cursor]
            actual_name = execution_input.metadata["function_name"]
            if actual_name != expected_name:
                raise RuntimeError(
                    f"Tool execution order mismatch at index {cursor}: "
                    f"{expected_name!r} != {actual_name!r}"
                )
            index = cursor
            cursor += 1
            world.num_interactions = starting_num_interactions
            world.num_sub_interactions = index + 1 if batch_size > 1 else 0
            return world.execute(execution_input.content)

        return wrapper

    tool_defs = [
        ToolDef(
            tool=make_wrapper(name),
            name=name,
            description=schemas_by_name[name].get("description", ""),
            parameters=ToolParams(additionalProperties=True),
            parallel=False,
        )
        for name in required_names
    ]
    rebuilt_assistant = ChatMessageAssistant(content="", tool_calls=tool_calls)

    try:
        result = await execute_tools([rebuilt_assistant], tool_defs)
    finally:
        world.num_sub_interactions = 0

    if cursor != batch_size:
        raise RuntimeError(f"Executed {cursor} of {batch_size} accepted tool calls")
    if result.output is not None or len(result.messages) != batch_size:
        raise RuntimeError("Unexpected execute_tools output shape")

    outputs: list[ExecutionIO] = []
    for index, message in enumerate(result.messages):
        execution_input = execution_inputs[index]
        expected_id = execution_input.metadata["id"]
        if not isinstance(message, ChatMessageTool):
            raise TypeError(f"Expected ChatMessageTool at output index {index}")
        if message.tool_call_id != expected_id:
            raise RuntimeError(
                f"Tool output ID mismatch at index {index}: "
                f"{message.tool_call_id!r} != {expected_id!r}"
            )
        metadata = dict(execution_input.metadata)
        metadata["inspect_outcome"] = "transport_failure" if message.error else "success"
        metadata["appworld_outcome"] = (
            "environment_failure"
            if not message.error and message.text.startswith("Execution failed")
            else "not_reached" if message.error else "success"
        )
        outputs.append(ExecutionIO(content=message.text, metadata=metadata))
    return outputs


@solver
def appworld_solver(
    experiment_name="inspect_appworld",
    max_steps=50,
    temperature=0.0,
    seed=100,
    max_tokens=2048,
) -> Solver:
    async def solve(state: TaskState, generate):
        # ponytail: AppWorld has process-global DB/time state; use process isolation
        # if parallel sample throughput becomes necessary.
        async with _APPWORLD_SAMPLE_LOCK:
            return await solve_serial(state)

    async def solve_serial(state: TaskState):
        task_id = state.metadata["appworld_task_id"]
        base = get_model()
        agent = _build_agent(
            get_model(role="predictor", default=base),
            get_model(role="main", default=base),
            max_steps,
            temperature,
            seed,
            max_tokens,
        )
        agent.logger.initialize(experiment_name, 1, 1, 0)
        termination = "max_steps"
        last_outputs = []
        ledger: list[dict[str, Any]] = []
        pending_observations: dict[str, dict[str, Any]] = {}
        steps = interactions = 0
        with appworld.AppWorld(
            task_id, experiment_name=experiment_name, **agent.appworld_config
        ) as world:
            agent.initialize(world)
            for _ in range(max_steps):
                agent.step_number += 1
                execution_inputs, usage, status = await anyio.to_thread.run_sync(
                    agent.next_execution_inputs_usage_and_status, last_outputs
                )
                audit = agent.language_model.call_audits[-1] if agent.language_model.call_audits else None
                if audit:
                    consumed = set(audit["input_tool_call_ids"])
                    for call_id in consumed:
                        if call_id in pending_observations:
                            pending_observations.pop(call_id)["observation_status"] = "written_back"
                    proposed = audit["tool_calls"]
                    accepted_ids = [item.metadata["id"] for item in execution_inputs]
                    accepted_set = set(accepted_ids)
                    limit = world.max_api_calls_per_interaction
                    round_entry = {
                        "round": agent.step_number - 1,
                        "provider_error": audit["provider_error"],
                        "proposed": proposed,
                        "accepted": [call for call in proposed if call["id"] in accepted_set],
                        "dropped": [],
                        "executed": [],
                    }
                    for index, call in enumerate(proposed):
                        if call["id"] in accepted_set:
                            continue
                        reason = (
                            "max_call_truncation"
                            if index >= limit
                            else "invalid_function_name"
                        )
                        round_entry["dropped"].append({**call, "reason": reason})
                    ledger.append(round_entry)
                if status.failed:
                    agent.logger.show_message(role="termination", content=status.message)
                    termination = "model_error"
                    break
                last_outputs = (
                    await execute_execution_inputs(
                        world, execution_inputs, agent.messages[-1], agent.functions
                    )
                    if execution_inputs
                    else []
                )
                if audit:
                    for output in last_outputs:
                        execution = {
                            "id": output.metadata["id"],
                            "function_name": output.metadata["function_name"],
                            "inspect_outcome": output.metadata["inspect_outcome"],
                            "appworld_outcome": output.metadata["appworld_outcome"],
                            "observation_status": "pending",
                        }
                        round_entry["executed"].append(execution)
                        pending_observations[execution["id"]] = execution
                agent.usage_tracker.add(task_id, usage)
                agent.log_usage()
                if world.task_completed():
                    termination = "task_completed"
                    break
                if agent.usage_tracker.exceeded(task_id):
                    termination = "usage_limit"
                    break
            steps = agent.step_number
            interactions = world.num_interactions
        for observation in pending_observations.values():
            observation["observation_status"] = "not_consumed_after_termination"
        agent.logger.complete_task()
        tracker = evaluate_task(task_id=task_id, experiment_name=experiment_name)
        state.store.set("appworld_test_tracker", tracker.to_dict())
        state.store.set("appworld_task_id", task_id)
        state.store.set("appworld_experiment_name", experiment_name)
        state.store.set("appworld_termination_reason", termination)
        state.store.set("appworld_steps", steps)
        state.store.set("appworld_interactions", interactions)
        state.store.set("appworld_call_ledger", ledger)
        state.completed = True
        return state

    return solve
