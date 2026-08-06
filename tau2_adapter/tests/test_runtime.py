import pytest
from tau2_adapter.dataset import tau2_dataset
from tau2_adapter.runtime import (
    build_domain_environment,
    load_domain_tasks,
    resolved_selection,
)
from tau2_adapter.task import tau2


@pytest.mark.parametrize(
    ("domain", "expected_split"),
    [
        ("mock", None),
        ("airline", "base"),
        ("retail", "base"),
        ("telecom", "base"),
        ("telecom-workflow", "base"),
    ],
)
def test_registry_domains_load_tasks_and_environment(domain, expected_split):
    task_set, task_split = resolved_selection(domain, task_split="auto")
    tasks = load_domain_tasks(domain, task_split="auto")
    environment = build_domain_environment(domain, tasks[0])

    assert task_set == domain
    assert task_split == expected_split
    assert tasks
    assert environment.get_domain_name() == domain
    assert environment.get_tools()


@pytest.mark.parametrize("domain", ["mock", "airline", "retail", "telecom"])
def test_dataset_carries_domain_selection_metadata(domain):
    dataset = tau2_dataset(domain, task_split="auto")

    assert dataset
    assert dataset[0].input
    assert dataset[0].metadata["tau2_domain"] == domain
    assert dataset[0].metadata["tau2_task_set"] == domain
    assert dataset[0].metadata["tau2_task_split"] == (
        "all" if domain == "mock" else "base"
    )


def test_generic_inspect_task_defaults_to_mock():
    inspect_task = tau2()

    assert len(inspect_task.dataset) == len(load_domain_tasks("mock"))


@pytest.mark.parametrize(
    ("task_set", "expected_count"),
    [("telecom_small", 20), ("telecom_full", 2285)],
)
def test_legacy_task_sets_without_split_parameter(task_set, expected_count):
    tasks = load_domain_tasks("telecom", task_set=task_set, task_split="auto")

    assert len(tasks) == expected_count


def test_unknown_domain_fails_with_available_options():
    with pytest.raises(ValueError, match="Unknown tau2 domain"):
        load_domain_tasks("not-a-domain")
