from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tau2_adapter import agent as agent_module


@pytest.fixture(autouse=True)
def _stub_history_conversion(monkeypatch):
    monkeypatch.setattr(agent_module, "tau2_message_to_chat_message", lambda message: message)


def _output(*, text: str = "", tool_calls=None):
    return SimpleNamespace(message=SimpleNamespace(text=text, tool_calls=tool_calls))


def _bare_agent(outputs):
    agent = object.__new__(agent_module.InspectAIAgent)
    agent._generate = Mock()
    return agent, iter(outputs)


def test_empty_generation_is_retried_before_conversion(monkeypatch):
    agent, outputs = _bare_agent([_output(), _output(text="done")])
    monkeypatch.setattr(agent_module, "MAX_EMPTY_RETRIES", 3)
    monkeypatch.setattr(agent_module.anyio.from_thread, "run", lambda *_: next(outputs))
    converted = object()
    monkeypatch.setattr(
        agent_module, "model_output_to_tau2_assistant_message", lambda output: converted
    )
    state = SimpleNamespace(system_messages=[], messages=[])

    message, returned_state = agent.generate_next_message(
        SimpleNamespace(), state
    )

    assert message is converted
    assert returned_state is state
    assert state.messages[-1] is converted


def test_empty_generation_raises_after_retry_budget(monkeypatch):
    agent, outputs = _bare_agent([_output(), _output(), _output()])
    monkeypatch.setattr(agent_module, "MAX_EMPTY_RETRIES", 2)
    monkeypatch.setattr(agent_module.anyio.from_thread, "run", lambda *_: next(outputs))
    convert = Mock()
    monkeypatch.setattr(agent_module, "model_output_to_tau2_assistant_message", convert)
    state = SimpleNamespace(system_messages=[], messages=[])

    with pytest.raises(RuntimeError, match="after 2 retries"):
        agent.generate_next_message(SimpleNamespace(), state)

    convert.assert_not_called()


def test_zero_retry_budget_makes_one_model_call(monkeypatch):
    agent, outputs = _bare_agent([_output()])
    monkeypatch.setattr(agent_module, "MAX_EMPTY_RETRIES", 0)
    model_call = Mock(side_effect=lambda *_: next(outputs))
    monkeypatch.setattr(agent_module.anyio.from_thread, "run", model_call)
    state = SimpleNamespace(system_messages=[], messages=[])

    with pytest.raises(RuntimeError, match="after 0 retries"):
        agent.generate_next_message(SimpleNamespace(), state)

    assert model_call.call_count == 1
