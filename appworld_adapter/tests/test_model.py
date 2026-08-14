import json

import anyio
import pytest
from appworld_agents.code.common.usage_tracker import CostPerToken
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentReasoning,
    ContentText,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.tool import ToolCall

from appworld_adapter import InspectLanguageModel


@pytest.mark.anyio
async def test_generate_converts_request_response_and_usage():
    captured = {}

    async def generate(*, input, tools, tool_choice, config):
        captured.update(
            input=input, tools=tools, tool_choice=tool_choice, config=config
        )
        message = ChatMessageAssistant(
            content=[ContentReasoning(reasoning="because"), ContentText(text="done")],
            tool_calls=[ToolCall(id="call-2", function="lookup", arguments={"id": 7})],
        )
        return ModelOutput(
            model="fake",
            choices=[ChatCompletionChoice(message=message)],
            usage=ModelUsage(
                input_tokens=11,
                output_tokens=5,
                input_tokens_cache_read=3,
                input_tokens_cache_write=2,
            ),
        )

    model = type("FakeModel", (), {"generate": staticmethod(generate)})()
    adapter = InspectLanguageModel(
        model,
        CostPerToken(
            input_cache_miss=0.1,
            input_cache_hit=0.2,
            input_cache_write=0.3,
            output=0.4,
        ),
        {},
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "checking",
            "reasoning_content": "prior thought",
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {"name": "lookup", "arguments": '{"id": 6}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "name": "lookup", "content": "six"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up an id",
                "parameters": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                    "required": ["id"],
                },
            },
        }
    ]

    result = await anyio.to_thread.run_sync(
        lambda: adapter.generate(
            messages,
            tools,
            temperature=0.2,
            seed=4,
            max_tokens=30,
            top_p=0.8,
            stop="STOP",
            tool_choice="auto",
            parallel_tool_calls=True,
        )
    )

    assert [type(message) for message in captured["input"]] == [
        ChatMessageSystem,
        ChatMessageUser,
        ChatMessageAssistant,
        ChatMessageTool,
    ]
    prior = captured["input"][2]
    assert isinstance(prior.content[0], ContentReasoning)
    assert prior.content[0].reasoning == "prior thought"
    assert prior.tool_calls[0].arguments == {"id": 6}
    assert captured["input"][3].tool_call_id == "call-1"
    assert captured["tools"][0].name == "lookup"
    assert captured["tools"][0].parameters.required == ["id"]
    assert captured["tools"][0].parameters.properties["id"].type == "integer"
    assert captured["tool_choice"] == "auto"
    assert captured["config"].model_dump(exclude_none=True) == {
        "temperature": 0.2,
        "top_p": 0.8,
        "max_tokens": 30,
        "stop_seqs": ["STOP"],
        "seed": 4,
        "parallel_tool_calls": True,
    }
    assert result["role"] == "assistant"
    assert result["reasoning_content"] == "because"
    assert result["tool_calls"][0]["id"] == "call-2"
    assert result["tool_calls"][0]["function"] == {
        "name": "lookup",
        "arguments": json.dumps({"id": 7}),
    }
    assert result["standardized_usage"].dict() == {
        "tokens": {
            "input_cache_miss": 8,
            "input_cache_hit": 3,
            "input_cache_write": 2,
            "output": 5,
        },
        "cost_per_token": {
            "input_cache_miss": 0.1,
            "input_cache_hit": 0.2,
            "input_cache_write": 0.3,
            "output": 0.4,
        },
        "cost": {
            "input_cache_miss": 0.8,
            "input_cache_hit": 0.6,
            "input_cache_write": 0.6,
            "output": 2,
        },
    }

    with pytest.raises(ValueError, match="Unsupported generate kwargs"):
        await anyio.to_thread.run_sync(lambda: adapter.generate([], unsupported=True))


@pytest.mark.anyio
async def test_generate_records_provider_error_without_a_choice():
    async def generate(**kwargs):
        return ModelOutput(model="fake", error="provider unavailable")

    adapter = InspectLanguageModel(
        type("FakeModel", (), {"generate": staticmethod(generate)})(),
        CostPerToken(),
        {},
        retain_call_audit=True,
    )
    result = await anyio.to_thread.run_sync(
        lambda: adapter.generate(
            [{"role": "tool", "tool_call_id": "call-1", "content": "result"}]
        )
    )

    assert result["error"] == "provider unavailable"
    assert adapter.call_audits == [
        {
            "input_tool_call_ids": ["call-1"],
            "tool_calls": [],
            "provider_error": "provider unavailable",
        }
    ]
