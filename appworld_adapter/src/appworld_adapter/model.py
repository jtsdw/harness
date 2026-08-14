from __future__ import annotations

import json
from typing import Any

import anyio
from appworld_agents.code.common.usage_tracker import Cost, CostPerToken, Tokens, Usage
from freezegun import configure
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentReasoning,
    ContentText,
    GenerateConfig,
    Model,
    ModelUsage,
)
from inspect_ai.tool import ToolCall, ToolInfo, ToolParams


_CONFIG_KEYS = {
    "temperature": "temperature",
    "seed": "seed",
    "max_tokens": "max_tokens",
    "top_p": "top_p",
    "stop": "stop_seqs",
    "parallel_tool_calls": "parallel_tool_calls",
}
_SPECIAL_KEYS = {"tool_choice"}

configure(extend_ignore_list=["inspect_ai", "inspect_trace"])


class InspectLanguageModel:
    def __init__(
        self,
        model: Model,
        cost_per_token: CostPerToken,
        supported_config: dict[str, Any],
        retain_call_audit: bool = False,
    ) -> None:
        unsupported = set(supported_config) - _CONFIG_KEYS.keys() - _SPECIAL_KEYS
        if unsupported:
            raise ValueError(f"Unsupported generation config keys: {sorted(unsupported)}")
        self.model = model
        self.cost_per_token = cost_per_token
        self.supported_config = dict(supported_config)
        self.tool_parser = None
        self.retain_call_audit = retain_call_audit
        self.call_audits: list[dict[str, Any]] = []

    def log_calls_to(
        self, file_path: str | None = None, world: Any | None = None
    ) -> None:
        # Inspect's .eval log is authoritative; do not duplicate raw model logs.
        return None

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        cache_control_at: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del cache_control_at
        return anyio.from_thread.run(self._generate, messages, tools, kwargs)

    async def _generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        unsupported = set(kwargs) - _CONFIG_KEYS.keys() - _SPECIAL_KEYS
        if unsupported:
            raise ValueError(f"Unsupported generate kwargs: {sorted(unsupported)}")

        options = self.supported_config | kwargs
        if isinstance(options.get("stop"), str):
            options["stop"] = [options["stop"]]
        config = GenerateConfig(
            **{
                _CONFIG_KEYS[key]: value
                for key, value in options.items()
                if key in _CONFIG_KEYS
            }
        )
        input_tool_call_ids = [
            message["tool_call_id"]
            for message in messages
            if message.get("role") == "tool" and "tool_call_id" in message
        ]
        output = await self.model.generate(
            input=[self._message(message) for message in messages],
            tools=[self._tool(tool) for tool in tools] if tools else [],
            tool_choice=options.get("tool_choice"),
            config=config,
        )
        if output.error:
            if self.retain_call_audit:
                self.call_audits.append(
                    {
                        "input_tool_call_ids": input_tool_call_ids,
                        "tool_calls": [],
                        "provider_error": output.error,
                    }
                )
            return {
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
                "tool_calls": None,
                "error": output.error,
            }
        message = output.message
        tool_calls = (
            [
                {
                    "id": call.id,
                    "call_id": call.id,
                    "function": {
                        "name": call.function,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in message.tool_calls
            ]
            if message.tool_calls
            else None
        )
        if self.retain_call_audit:
            self.call_audits.append(
                {
                    "input_tool_call_ids": input_tool_call_ids,
                    "tool_calls": tool_calls or [],
                    "provider_error": output.error,
                }
            )
        model_usage = output.usage or ModelUsage()
        cache_read = model_usage.input_tokens_cache_read or 0
        cache_write = model_usage.input_tokens_cache_write or 0
        tokens = Tokens(
            input_cache_miss=max(model_usage.input_tokens - cache_read, 0),
            input_cache_hit=cache_read,
            input_cache_write=cache_write,
            output=model_usage.output_tokens,
        )
        usage = Usage(
            tokens=tokens,
            cost_per_token=self.cost_per_token,
            cost=Cost.build(tokens, self.cost_per_token),
        )
        reasoning_content = ""
        if isinstance(message.content, list):
            reasoning_content = "\n".join(
                reasoning
                for block in message.content
                if isinstance(block, ContentReasoning)
                and (reasoning := block.reasoning or block.summary)
            )
        return {
            "role": "assistant",
            "content": message.text,
            "reasoning_content": reasoning_content,
            "tool_calls": tool_calls,
            "standardized_usage": usage,
            **({"error": output.error} if output.error else {}),
        }

    @staticmethod
    def _message(message: dict[str, Any]) -> ChatMessage:
        role = message["role"]
        content = message.get("content") or ""
        if role == "system":
            return ChatMessageSystem(content=content)
        if role == "user":
            return ChatMessageUser(content=content)
        if role == "assistant":
            calls = []
            for call in message.get("tool_calls") or []:
                function = call["function"]
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                calls.append(
                    ToolCall(
                        id=call.get("call_id") or call["id"],
                        function=function["name"],
                        arguments=arguments,
                    )
                )
            assistant_content: str | list[ContentReasoning | ContentText] = content
            reasoning = message.get("reasoning_content")
            if reasoning is not None:
                assistant_content = [
                    ContentReasoning(reasoning=reasoning),
                    ContentText(text=content),
                ]
            return ChatMessageAssistant(content=assistant_content, tool_calls=calls or None)
        if role == "tool":
            return ChatMessageTool(
                content=content,
                tool_call_id=message["tool_call_id"],
                function=message.get("name"),
                error=None,
            )
        raise ValueError(f"Unsupported message role: {role}")

    @staticmethod
    def _tool(tool: dict[str, Any]) -> ToolInfo:
        function = tool["function"]
        return ToolInfo(
            name=function["name"],
            description=function.get("description", ""),
            parameters=ToolParams.model_validate(
                function.get("parameters", {"type": "object", "properties": {}})
            ),
        )
