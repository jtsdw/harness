"""The agent-under-test: a tau2 HalfDuplexAgent whose model calls go through inspect_ai.

Why this needs a sync/async bridge: `tau2.orchestrator.orchestrator.Orchestrator.run()`/`step()`
are synchronous (blocking) methods -- see BaseOrchestrator in tau2's own source. But
`inspect_ai.model.Model.generate()` is async, and critically, `inspect_trace`'s Hooks only fire
inside inspect_ai's own Sample-execution context (`sample_active()` must be set by
inspect_ai's `emit_sample_event()` guard).
So the `generate()` call has to happen on the same asyncio/anyio task that inspect_ai's own eval
loop is running -- not in a detached thread with its own fresh state.

The fix: the whole synchronous `Orchestrator.run()` call is dispatched to a worker thread via
`anyio.to_thread.run_sync()` (see solver.py). Any code running inside that worker thread --
including this agent's `generate_next_message()`, called deep inside `Orchestrator.step()` -- can
use `anyio.from_thread.run()` to hop back onto the *original* event loop/task and await the real
`model.generate()` there. That's what makes `sample_active()` resolve correctly.

System prompt construction (`SYSTEM_PROMPT`/`AGENT_INSTRUCTION`) and state handling
(`LLMAgentState`) are imported directly from tau2's own `LLMAgent`, not reimplemented, so this
agent is a faithful stand-in for it -- the only thing that changes is which code actually issues
the model call.
"""

from __future__ import annotations

import logging

import anyio.from_thread
from inspect_ai.model import Model, get_model
from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT, LLMAgentState
from tau2.data_model.message import MultiToolMessage, SystemMessage
from tau2.environment.tool import Tool

from tau2_adapter.convert import (
    EmptyAssistantResponseError,
    model_output_to_tau2_assistant_message,
    tau2_message_to_chat_message,
    tau2_tool_to_tool_info,
)


logger = logging.getLogger(__name__)


class InspectAIAgent(HalfDuplexAgent[LLMAgentState]):
    """Drop-in replacement for `tau2.agent.llm_agent.LLMAgent` that generates via
    `inspect_ai.model.Model.generate()` instead of tau2's own `litellm`-based `generate()`
    utility, so `inspect_trace`'s Hooks observe the agent-under-test's calls.
    """

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        empty_response_retries: int = 3,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy)
        if empty_response_retries < 0:
            raise ValueError("empty_response_retries must be non-negative")
        self._empty_response_retries = empty_response_retries
        self._tool_infos = [tau2_tool_to_tool_info(t) for t in tools]
        # No model/base_url/api_key here on purpose: get_model() with no arguments returns
        # inspect_ai's *active* model -- whatever --model/-M flags eval() was invoked with. That
        # keeps model selection entirely on the standard inspect_ai CLI surface, the same as every
        # other Task in this project (run_bfcl_benchmark.sh's --model/-M pattern), instead of a
        # second, adapter-specific way to point at a model.
        self._model: Model = get_model()

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION
        )

    def get_init_state(self, message_history: list | None = None) -> LLMAgentState:
        if message_history is None:
            message_history = []
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> tuple:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        tau2_history = state.system_messages + state.messages
        chat_messages = [tau2_message_to_chat_message(m) for m in tau2_history]

        # The actual bridge: hop back to the event loop that's running inside inspect_ai's own
        # Sample execution (see module docstring). This call blocks this worker thread until the
        # real model.generate() completes on the other side.
        assistant_message = anyio.from_thread.run(
            self._generate_assistant, chat_messages
        )
        state.messages.append(assistant_message)
        return assistant_message, state

    async def _generate_assistant(self, chat_messages: list):
        """Generate a tau2-valid assistant message, retrying only empty responses.

        An OpenAI-compatible backend can occasionally return a successful response with
        usage metadata but neither content nor tool calls. Inspect treats that as a completed
        model request, while tau2 correctly rejects it as an invalid AssistantMessage. Retrying
        here preserves all normal provider errors and conversion failures while recovering from
        that one safe-to-repeat, side-effect-free boundary case.
        """
        attempts = self._empty_response_retries + 1
        for attempt in range(1, attempts + 1):
            output = await self._model.generate(
                input=chat_messages, tools=self._tool_infos
            )
            try:
                return model_output_to_tau2_assistant_message(output)
            except EmptyAssistantResponseError as exc:
                if attempt == attempts:
                    raise EmptyAssistantResponseError(
                        "Model returned an empty assistant response after "
                        f"{attempts} attempts."
                    ) from exc
                logger.warning(
                    "Model returned an empty assistant response; retrying (%d/%d).",
                    attempt,
                    self._empty_response_retries,
                )

        raise AssertionError("unreachable")
