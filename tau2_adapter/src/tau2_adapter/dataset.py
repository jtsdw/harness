"""Build inspect_ai datasets from tau2's registry-backed task sets."""

from __future__ import annotations

from inspect_ai.dataset import Dataset, MemoryDataset, Sample

from tau2_adapter.runtime import AUTO_TASK_SPLIT, load_domain_tasks, resolved_selection


def tau2_dataset(
    domain: str = "mock",
    task_set: str | None = None,
    task_split: str | None = AUTO_TASK_SPLIT,
) -> Dataset:
    """Create one Inspect sample per task selected by tau2 itself."""
    tasks = load_domain_tasks(domain, task_set, task_split)
    resolved_task_set, resolved_task_split = resolved_selection(
        domain, task_set, task_split
    )
    samples = [
        Sample(
            input=task.ticket
            or (task.description.purpose if task.description else None)
            or task.id,
            id=task.id,
            metadata={
                "tau2_domain": domain,
                "tau2_task_id": task.id,
                "tau2_task_set": resolved_task_set,
                "tau2_task_split": resolved_task_split or "all",
            },
        )
        for task in tasks
    ]
    return MemoryDataset(samples=samples, name=f"tau2_{domain}")


def mock_dataset() -> Dataset:
    """Backward-compatible mock-domain dataset alias."""
    return tau2_dataset(domain="mock")
