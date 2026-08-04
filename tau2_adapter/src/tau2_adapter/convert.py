"""Message/tool conversion between tau2-bench's own types and inspect_ai's.

tau2 and inspect_ai both model chat as OpenAI-style role/content/tool_calls, so this is a
straightforward field-by-field mapping, not a semantic translation -- the goal is that nothing
is lost or reinterpreted crossing the boundary, since that's exactly what would make a
"same bench, different harness" comparison unfair.
"""

from __future__ import annotations

import json

from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageTool, ChatMessageUser
from inspect_ai.model._model_output import ModelOutput
from inspect_ai.tool import ToolCall, ToolInfo
from inspect_ai.tool._tool_params import ToolParams
from tau2.data_model.message import (
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.message import (
    ToolCall as Tau2ToolCall,
)
from tau2.environment.tool import Tool as Tau2Tool


def tau2_tool_to_tool_info(tool: Tau2Tool) -> ToolInfo:
    """tau2's `Tool.openai_schema` is already an OpenAI function-calling schema dict --
    `{"type": "function", "function": {"name", "description", "parameters"}}` -- the same
    shape `ToolParams` expects, so this is a direct field lift, not a re-derivation.
    """
    fn = tool.openai_schema["function"]
    return ToolInfo(
        name=fn["name"],
        description=fn.get("description", ""),
        parameters=ToolParams.model_validate(
            fn.get("parameters", {"type": "object", "properties": {}})
        ),
    )


def tau2_message_to_chat_message(
    message: SystemMessage | UserMessage | AssistantMessage | ToolMessage,
) -> ChatMessage:
    """Converts one tau2 history message into the inspect_ai equivalent.

    Only used for the agent-under-test's own message history (system_messages + messages in
    `LLMAgentState`) -- never for the user simulator's side, which stays entirely on tau2's own
    LiteLLM path and is never traced.
    """
    if isinstance(message, SystemMessage):
        from inspect_ai.model import ChatMessageSystem

        return ChatMessageSystem(content=message.content or "")
    elif isinstance(message, UserMessage):
        return ChatMessageUser(content=message.content or "")
    elif isinstance(message, AssistantMessage):
        tool_calls = (
            [
                ToolCall(id=tc.id, function=tc.name, arguments=tc.arguments)
                for tc in message.tool_calls
            ]
            if message.tool_calls
            else None
        )
        return ChatMessageAssistant(content=message.content or "", tool_calls=tool_calls)
    elif isinstance(message, ToolMessage):
        return ChatMessageTool(
            content=message.content or "",
            tool_call_id=message.id,
            error=None,
        )
    else:
        raise TypeError(f"Unsupported tau2 message type: {type(message)}")


def model_output_to_tau2_assistant_message(output: ModelOutput) -> AssistantMessage:
    """Converts inspect_ai's real ModelOutput back into tau2's AssistantMessage, so the rest of
    tau2's own orchestrator/environment/evaluator machinery can keep operating on it unchanged.
    """
    msg = output.message
    tool_calls = (
        [
            Tau2ToolCall(
                id=tc.id,
                name=tc.function,
                arguments=tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments),
            )
            for tc in msg.tool_calls
        ]
        if msg.tool_calls
        else None
    )
    return AssistantMessage(role="assistant", content=msg.text or None, tool_calls=tool_calls)
