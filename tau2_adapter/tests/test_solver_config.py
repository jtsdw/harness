import pytest

from tau2_adapter.solver import _empty_response_retries


def test_retry_budget_can_be_configured(monkeypatch):
    monkeypatch.setenv("TAU2_EMPTY_RESPONSE_RETRIES", "4")

    assert _empty_response_retries() == 4


def test_current_retry_variable_takes_precedence(monkeypatch):
    monkeypatch.setenv("TAU2_EMPTY_RESPONSE_RETRIES", "4")
    monkeypatch.setenv("TAU2_AGENT_MAX_EMPTY_RETRIES", "1")

    assert _empty_response_retries() == 4


def test_nscc_legacy_retry_variable_warns(monkeypatch, caplog):
    monkeypatch.delenv("TAU2_EMPTY_RESPONSE_RETRIES", raising=False)
    monkeypatch.setenv("TAU2_AGENT_MAX_EMPTY_RETRIES", "2")

    assert _empty_response_retries() == 2
    assert "TAU2_AGENT_MAX_EMPTY_RETRIES is deprecated" in caplog.text


def test_retry_budget_defaults_to_three(monkeypatch):
    monkeypatch.delenv("TAU2_EMPTY_RESPONSE_RETRIES", raising=False)
    monkeypatch.delenv("TAU2_AGENT_MAX_EMPTY_RETRIES", raising=False)

    assert _empty_response_retries() == 3


def test_negative_retry_budget_is_rejected(monkeypatch):
    monkeypatch.setenv("TAU2_EMPTY_RESPONSE_RETRIES", "-1")

    with pytest.raises(ValueError, match="must be non-negative"):
        _empty_response_retries()
