"""Domain selection and environment construction for the tau2 adapter.

The adapter deliberately resolves domains and task sets through tau2's public
registry instead of importing individual domain modules. New registry-backed
text domains therefore become available without another adapter code change.
"""

from __future__ import annotations

import json
import os
from functools import cache
from inspect import signature
from typing import Any

from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.registry import registry
from tau2.runner.build import build_environment
from tau2.runner.helpers import load_task_splits

AUTO_TASK_SPLIT = "auto"


def resolve_task_set(domain: str, task_set: str | None = None) -> str:
    """Return and validate the registry task-set name for a domain."""
    if domain not in registry.get_domains():
        raise ValueError(
            f"Unknown tau2 domain {domain!r}; available domains: "
            f"{', '.join(registry.get_domains())}"
        )
    resolved = task_set or domain
    if resolved not in registry.get_task_sets():
        raise ValueError(
            f"Unknown tau2 task set {resolved!r}; available task sets: "
            f"{', '.join(registry.get_task_sets())}"
        )
    return resolved


def resolve_task_split(task_set: str, task_split: str | None) -> str | None:
    """Resolve ``auto`` to ``base`` when the task set provides that split.

    ``all``, ``none``, and an empty string explicitly select the complete task
    file. This also handles domains such as mock that do not register splits.
    """
    if task_split is None:
        return None
    normalized = task_split.strip().lower()
    if normalized in {"", "all", "none", "null"}:
        return None
    if normalized != AUTO_TASK_SPLIT:
        return task_split
    splits = load_task_splits(task_set)
    return "base" if splits and "base" in splits else None


@cache
def load_domain_tasks(
    domain: str,
    task_set: str | None = None,
    task_split: str | None = AUTO_TASK_SPLIT,
) -> tuple[Task, ...]:
    """Load immutable task selection metadata through tau2's registry."""
    resolved_task_set = resolve_task_set(domain, task_set)
    resolved_task_split = resolve_task_split(resolved_task_set, task_split)
    task_loader = registry.get_tasks_loader(resolved_task_set)
    if signature(task_loader).parameters:
        tasks = tuple(task_loader(task_split_name=resolved_task_split))
    else:
        if resolved_task_split is not None:
            raise ValueError(
                f"Task set {resolved_task_set!r} does not support task splits"
            )
        tasks = tuple(task_loader())
    if not tasks:
        raise ValueError(
            f"No tasks found for domain={domain!r}, task_set={resolved_task_set!r}, "
            f"task_split={resolved_task_split!r}"
        )
    return tasks


def resolved_selection(
    domain: str,
    task_set: str | None = None,
    task_split: str | None = AUTO_TASK_SPLIT,
) -> tuple[str, str | None]:
    """Return the concrete task set and split used by a run."""
    resolved_task_set = resolve_task_set(domain, task_set)
    return resolved_task_set, resolve_task_split(resolved_task_set, task_split)


def json_object_from_env(name: str) -> dict[str, Any]:
    """Parse an optional JSON-object environment variable."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError(f"{name} must contain a JSON object")
    return value


def _derive_read_log_allowlist(task: Task) -> set[str]:
    """Mirror tau2's banking-knowledge read-log selection for correct scoring."""
    allowlist: set[str] = set()
    criteria = task.evaluation_criteria
    if criteria is None:
        return allowlist
    for action in criteria.actions or []:
        if action.name == "call_discoverable_agent_tool":
            name = (action.arguments or {}).get("agent_tool_name")
            if name:
                allowlist.add(name)
    return allowlist


def build_domain_environment(domain: str, task: Task) -> Environment:
    """Build a fresh environment for one task using tau2's own builder."""
    env_kwargs = json_object_from_env("TAU2_ENV_ARGS")
    if domain == "banking_knowledge":
        if "retrieval_variant" in env_kwargs:
            env_kwargs.setdefault("task", task)
        env_kwargs.setdefault("read_log_allowlist", _derive_read_log_allowlist(task))
    return build_environment(domain, env_kwargs=env_kwargs)
