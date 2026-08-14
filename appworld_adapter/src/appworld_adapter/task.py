import appworld as appworld_lib
from appworld.task import Task as AppWorldTask
from inspect_ai import Task as InspectTask, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState

from appworld_adapter.solver import appworld_solver


def appworld_dataset(split: str = "dev") -> MemoryDataset:
    samples = []
    for task_id in appworld_lib.load_task_ids(split):
        instruction = AppWorldTask.load(task_id).instruction
        samples.append(
            Sample(
                input=instruction,
                id=task_id,
                metadata={"appworld_task_id": task_id, "appworld_split": split},
            )
        )
    return MemoryDataset(samples=samples, name=f"appworld_{split}")


@scorer(metrics=[accuracy(), stderr()])
def appworld_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        tracker = state.store.get("appworld_test_tracker")
        if not isinstance(tracker, dict):
            raise RuntimeError("Missing or invalid appworld_test_tracker")

        task_id = state.store.get("appworld_task_id")
        experiment = state.store.get("appworld_experiment_name")
        termination = state.store.get("appworld_termination_reason")
        steps = state.store.get("appworld_steps")
        interactions = state.store.get("appworld_interactions")
        call_ledger = state.store.get("appworld_call_ledger")
        success = tracker["success"]
        passed_count = len(tracker.get("passes", []))
        failed_count = len(tracker.get("failures", []))

        return Score(
            value=1.0 if success else 0.0,
            answer=None,
            explanation=(
                f"success={success}; passed={passed_count}; failed={failed_count}; "
                f"termination={termination}"
            ),
            metadata={
                "tracker": tracker,
                "task_id": task_id,
                "experiment": experiment,
                "termination": termination,
                "steps": steps,
                "interactions": interactions,
                "call_ledger": call_ledger,
            },
        )

    return score


@task
def appworld(
    split: str = "dev",
    experiment_name: str = "inspect_appworld",
    max_steps: int = 50,
    temperature: float = 0.0,
    seed: int = 100,
    max_tokens: int = 2048,
) -> InspectTask:
    return InspectTask(
        dataset=appworld_dataset(split),
        solver=appworld_solver(
            experiment_name=experiment_name,
            max_steps=max_steps,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
        ),
        scorer=appworld_scorer(),
    )
