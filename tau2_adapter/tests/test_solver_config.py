import pytest

from tau2_adapter.solver import _empty_response_retries


def test_cloud_retry_variable_takes_precedence(monkeypatch):
    monkeypatch.setenv("TAU2_AGENT_MAX_EMPTY_RETRIES", "4")
    monkeypatch.setenv("TAU2_EMPTY_RESPONSE_RETRIES", "1")

    assert _empty_response_retries() == 4


def test_local_retry_variable_remains_compatible(monkeypatch):
    monkeypatch.delenv("TAU2_AGENT_MAX_EMPTY_RETRIES", raising=False)
    monkeypatch.setenv("TAU2_EMPTY_RESPONSE_RETRIES", "2")

    assert _empty_response_retries() == 2


def test_retry_budget_defaults_to_three(monkeypatch):
    monkeypatch.delenv("TAU2_AGENT_MAX_EMPTY_RETRIES", raising=False)
    monkeypatch.delenv("TAU2_EMPTY_RESPONSE_RETRIES", raising=False)

    assert _empty_response_retries() == 3


def test_negative_retry_budget_is_rejected(monkeypatch):
    monkeypatch.setenv("TAU2_AGENT_MAX_EMPTY_RETRIES", "-1")

    with pytest.raises(ValueError, match="must be non-negative"):
        _empty_response_retries()
