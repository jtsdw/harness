"""The inspect_ai Solver that drives a real tau2 simulation for one Sample.

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
from tau2.config import DEFAULT_MAX_ERRORS, DEFAULT_MAX_STEPS, DEFAULT_SEED
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.runner.build import build_user
from tau2.runner.simulation import run_simulation

from tau2_adapter.agent import InspectAIAgent
from tau2_adapter.nl_assertions import configure_tau2_nl_assertions
from tau2_adapter.runtime import (
    AUTO_TASK_SPLIT,
    build_domain_environment,
    json_object_from_env,
    load_domain_tasks,
)


EMPTY_RESPONSE_RETRIES_ENV = "TAU2_AGENT_MAX_EMPTY_RETRIES"
LEGACY_EMPTY_RESPONSE_RETRIES_ENV = "TAU2_EMPTY_RESPONSE_RETRIES"


def _user_llm_args() -> dict:
    args: dict = {"temperature": 0.0, **json_object_from_env("TAU2_USER_LLM_ARGS")}
    api_base = os.environ.get("TAU2_USER_API_BASE")
    if api_base:
        args["api_base"] = api_base
    api_key = os.environ.get("TAU2_USER_API_KEY")
    if api_key:
        args["api_key"] = api_key
    return args


def _empty_response_retries() -> int:
    raw = os.environ.get(EMPTY_RESPONSE_RETRIES_ENV)
    if raw is None:
        raw = os.environ.get(LEGACY_EMPTY_RESPONSE_RETRIES_ENV, "3")
    retries = int(raw)
    if retries < 0:
        raise ValueError(f"{EMPTY_RESPONSE_RETRIES_ENV} must be non-negative")
    return retries


@solver
def tau2_solver(
    domain: str = "mock",
    task_set: str | None = None,
    task_split: str | None = AUTO_TASK_SPLIT,
) -> Solver:
    """Create a solver for any registry-backed half-duplex tau2 domain."""
    max_steps = int(os.environ.get("TAU2_MAX_STEPS", str(DEFAULT_MAX_STEPS)))
    max_errors = int(os.environ.get("TAU2_MAX_ERRORS", str(DEFAULT_MAX_ERRORS)))
    empty_response_retries = _empty_response_retries()
    configure_tau2_nl_assertions()
    seed = int(os.environ.get("TAU2_SEED", str(DEFAULT_SEED)))
    timeout_raw = os.environ.get("TAU2_TIMEOUT", "").strip()
    timeout = float(timeout_raw) if timeout_raw else None
    user_model = os.environ.get("TAU2_USER_MODEL", "openai/Qwen/Qwen2.5-3B-Instruct")
    tasks_by_id = {
        task.id: task for task in load_domain_tasks(domain, task_set, task_split)
    }

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        task_id = state.metadata["tau2_task_id"]
        if state.metadata.get("tau2_domain") != domain:
            raise ValueError(
                f"Sample domain {state.metadata.get('tau2_domain')!r} does not match "
                f"solver domain {domain!r}"
            )
        task = tasks_by_id[task_id]

        environment = build_domain_environment(domain, task)
        # Agent model comes from inspect_ai's own active model (--model/-M on the eval() CLI),
        # not a separate adapter-specific config -- see InspectAIAgent's docstring.
        agent = InspectAIAgent(
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
            empty_response_retries=empty_response_retries,
        )
        user = build_user(
            "user_simulator",
            environment,
            task,
            llm=user_model,
            llm_args=_user_llm_args(),
        )
        orchestrator = Orchestrator(
            domain=domain,
            agent=agent,
            user=user,
            environment=environment,
            task=task,
            max_steps=max_steps,
            max_errors=max_errors,
            seed=seed,
            timeout=timeout,
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
            [r.value for r in reward_info.reward_basis]
            if reward_info and reward_info.reward_basis
            else [],
        )
        state.store.set("tau2_termination_reason", simulation.termination_reason.value)
        state.store.set("tau2_duration_seconds", simulation.duration)
        state.store.set("tau2_domain", domain)
        state.completed = True
        return state

    return solve


@solver
def tau2_mock_solver() -> Solver:
    """Backward-compatible mock-domain solver alias."""
    return tau2_solver(domain="mock")
