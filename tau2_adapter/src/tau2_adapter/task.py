"""Assemble a registry-backed tau2 Inspect task and its real reward scorer."""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState

from tau2_adapter.dataset import tau2_dataset
from tau2_adapter.runtime import AUTO_TASK_SPLIT
from tau2_adapter.solver import tau2_solver


@scorer(metrics=[accuracy(), stderr()])
def tau2_reward_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        reward = state.store.get("tau2_reward", 0.0)
        return Score(
            value=1.0 if reward >= 1.0 else 0.0,
            answer=None,
            explanation=(
                f"tau2 reward={reward}, basis={state.store.get('tau2_reward_basis')}, "
                f"termination={state.store.get('tau2_termination_reason')}"
            ),
            metadata={
                "tau2_reward": reward,
                "tau2_reward_basis": state.store.get("tau2_reward_basis"),
                "tau2_termination_reason": state.store.get("tau2_termination_reason"),
                "tau2_duration_seconds": state.store.get("tau2_duration_seconds"),
                "tau2_domain": state.store.get("tau2_domain"),
            },
        )

    return score


@task
def tau2(
    domain: str = "mock",
    task_set: str | None = None,
    task_split: str | None = AUTO_TASK_SPLIT,
) -> Task:
    """Run a tau2 domain while tracing only the agent-under-test model calls."""
    return Task(
        dataset=tau2_dataset(domain, task_set, task_split),
        solver=tau2_solver(domain, task_set, task_split),
        scorer=tau2_reward_scorer(),
    )
