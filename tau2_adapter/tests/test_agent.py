from inspect_ai.model import ModelOutput
import pytest

from tau2_adapter.agent import InspectAIAgent
from tau2_adapter.convert import EmptyAssistantResponseError


class StubModel:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = 0

    async def generate(self, *, input, tools):
        self.calls += 1
        return next(self.outputs)


def agent_with_outputs(outputs, retries=2):
    agent = object.__new__(InspectAIAgent)
    agent._model = StubModel(outputs)
    agent._tool_infos = []
    agent._empty_response_retries = retries
    return agent


@pytest.mark.asyncio
async def test_empty_assistant_response_is_retried():
    agent = agent_with_outputs(
        [
            ModelOutput.from_content(model="stub", content=""),
            ModelOutput.from_content(model="stub", content="recovered"),
        ]
    )

    message = await agent._generate_assistant([])

    assert message.content == "recovered"
    assert agent._model.calls == 2


@pytest.mark.asyncio
async def test_empty_assistant_response_stops_after_retry_budget():
    agent = agent_with_outputs(
        [
            ModelOutput.from_content(model="stub", content=""),
            ModelOutput.from_content(model="stub", content=""),
        ],
        retries=1,
    )

    with pytest.raises(EmptyAssistantResponseError, match="after 2 attempts"):
        await agent._generate_assistant([])

    assert agent._model.calls == 2


@pytest.mark.asyncio
async def test_zero_retry_budget_makes_one_model_call():
    agent = agent_with_outputs(
        [ModelOutput.from_content(model="stub", content="")], retries=0
    )

    with pytest.raises(EmptyAssistantResponseError, match="after 1 attempts"):
        await agent._generate_assistant([])

    assert agent._model.calls == 1


@pytest.mark.asyncio
async def test_tool_call_response_is_not_retried():
    output = ModelOutput.for_tool_call(
        model="stub",
        tool_name="lookup",
        tool_arguments={"query": "value"},
        tool_call_id="call-1",
    )
    agent = agent_with_outputs([output])

    message = await agent._generate_assistant([])

    assert message.tool_calls[0].name == "lookup"
    assert agent._model.calls == 1


@pytest.mark.asyncio
async def test_whitespace_only_response_is_retried():
    agent = agent_with_outputs(
        [
            ModelOutput.from_content(model="stub", content=" \n\t"),
            ModelOutput.from_content(model="stub", content="recovered"),
        ]
    )

    message = await agent._generate_assistant([])

    assert message.content == "recovered"
    assert agent._model.calls == 2


def test_negative_retry_budget_is_rejected():
    agent = object.__new__(InspectAIAgent)

    with pytest.raises(ValueError, match="must be non-negative"):
        InspectAIAgent.__init__(
            agent,
            tools=[],
            domain_policy="",
            empty_response_retries=-1,
        )
