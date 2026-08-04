"""The inspect_ai Solver that drives a real tau2-bench simulation for one Sample.

Per sample: builds a real tau2 `Environment`, a real tau2 `UserSimulator` (unmodified, its model
calls go through tau2's own LiteLLM path -- we deliberately do not trace the user simulator, only
the agent-under-test), and an `InspectAIAgent` (agent.py) whose model calls go through
`inspect_ai.model.generate()`. Runs `tau2.runner.simulation.run_simulation()` -- tau2's own
library entry point that runs the orchestrator AND evaluates the result with tau2's real
evaluator, unmodified -- inside a worker thread (`anyio.to_thread.run_sync`), because
`Orchestrator.run()` is synchronous. See agent.py's module docstring for why the agent's own
`generate_next_message()` then has to hop back to this coroutine via `anyio.from_thread.run()`.
"""

from __future__ import annotations

import os

import anyio.to_thread
from inspect_ai.solver import Generate, Solver, TaskState, solver
from tau2.data_model.tasks import Task
from tau2.domains.mock.environment import get_environment, get_tasks
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.runner.simulation import run_simulation
from tau2.user.user_simulator import UserSimulator

from tau2_adapter.agent import InspectAIAgent

_TASKS_BY_ID: dict[str, Task] | None = None


def _tasks_by_id() -> dict[str, Task]:
    global _TASKS_BY_ID
    if _TASKS_BY_ID is None:
        _TASKS_BY_ID = {t.id: t for t in get_tasks()}
    return _TASKS_BY_ID


def _user_llm_args() -> dict:
    args: dict = {"temperature": 0.0}
    api_base = os.environ.get("TAU2_USER_API_BASE")
    if api_base:
        args["api_base"] = api_base
    api_key = os.environ.get("TAU2_USER_API_KEY")
    if api_key:
        args["api_key"] = api_key
    return args


@solver
def tau2_mock_solver() -> Solver:
    max_steps = int(os.environ.get("TAU2_MAX_STEPS", "20"))
    user_model = os.environ.get("TAU2_USER_MODEL", "openai/Qwen/Qwen2.5-3B-Instruct")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        task_id = state.metadata["tau2_task_id"]
        task = _tasks_by_id()[task_id]

        environment = get_environment()
        # Agent model comes from inspect_ai's own active model (--model/-M on the eval() CLI),
        # not a separate adapter-specific config -- see InspectAIAgent's docstring.
        agent = InspectAIAgent(
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
        )
        user = UserSimulator(
            llm=user_model,
            instructions=task.user_scenario.instructions,
            tools=environment.get_user_tools(),
            llm_args=_user_llm_args(),
        )
        orchestrator = Orchestrator(
            domain="mock",
            agent=agent,
            user=user,
            environment=environment,
            task=task,
            max_steps=max_steps,
        )

        # run_simulation() is tau2's own library entry point: runs the orchestrator loop AND
        # evaluates the result with tau2's real evaluator (DB-hash/communicate/nl_assertions),
        # unmodified -- same code path the native `tau2 run` CLI uses. It's synchronous, so it
        # runs in a worker thread; agent.py's generate_next_message() hops back to *this*
        # coroutine (via anyio.from_thread.run) whenever it needs a real model call, which is
        # what keeps that call inside inspect_ai's own Sample execution context.
        simulation = await anyio.to_thread.run_sync(run_simulation, orchestrator)

        reward_info = simulation.reward_info
        state.store.set("tau2_reward", reward_info.reward if reward_info else 0.0)
        state.store.set(
            "tau2_reward_basis",
            [r.value for r in reward_info.reward_basis] if reward_info and reward_info.reward_basis else [],
        )
        state.store.set("tau2_termination_reason", simulation.termination_reason.value)
        state.store.set("tau2_duration_seconds", simulation.duration)
        state.completed = True
        return state

    return solve
