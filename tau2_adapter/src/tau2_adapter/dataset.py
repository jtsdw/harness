"""Builds an inspect_ai Dataset from tau2-bench's own `mock` domain tasks.

Reuses `tau2.domains.mock.environment.get_tasks()` (tau2's own task loader, reads
`data/tau2/domains/mock/tasks.json`) rather than re-parsing the JSON ourselves, so there's no risk
of the Sample's task diverging from what tau2's own CLI would load for the same domain.

Each Sample only carries the task id in metadata -- the solver re-fetches the full tau2 `Task`
object from `get_tasks()` by id rather than round-tripping it through Sample.metadata
(model_dump/model_validate), since the task objects are small, fixed, and cheap to reload, and
this avoids any risk of a serialization mismatch silently producing a different task.
"""

from __future__ import annotations

from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from tau2.domains.mock.environment import get_tasks


def mock_dataset() -> Dataset:
    tasks = get_tasks()
    samples = [
        Sample(
            input=task.ticket or (task.description.purpose if task.description else task.id),
            id=task.id,
            metadata={"tau2_task_id": task.id},
        )
        for task in tasks
    ]
    return MemoryDataset(samples=samples, name="tau2_mock")
