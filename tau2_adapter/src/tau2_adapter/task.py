"""Assembles the inspect_ai Task: tau2 mock-domain dataset + the sync/async-bridged solver +
a scorer that reports the real reward tau2's own evaluator computed (not a re-derived one).
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState

from tau2_adapter.dataset import mock_dataset
from tau2_adapter.solver import tau2_mock_solver


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
            },
        )

    return score


@task
def tau2_mock() -> Task:
    return Task(
        dataset=mock_dataset(),
        solver=tau2_mock_solver(),
        scorer=tau2_reward_scorer(),
    )
